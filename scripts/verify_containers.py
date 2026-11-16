"""Build and verify a temporary Compose stack, then remove only its own resources."""

import json
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kubeai import benchmark
from kubeai.transport import get_json, open_url


def command(args, **kwargs):
    return subprocess.run(args, cwd=ROOT, text=True, check=True, **kwargs)


def main():
    project = "kubeai-verify-" + uuid.uuid4().hex[:8]
    output = ROOT / "results" / project
    output.mkdir(parents=True)
    compose = ["docker", "compose", "--project-name", project, "--file", str(ROOT / "compose.yaml")]
    with tempfile.TemporaryDirectory(prefix="kubeai-") as temporary:
        override = Path(temporary) / "ports.yaml"
        override.write_text('services:\n  router:\n    ports: !override ["127.0.0.1::8080"]\n  prometheus:\n    ports: !override ["127.0.0.1::9090"]\n')
        compose += ["--file", str(override)]
        try:
            command(compose + ["build"])
            command(compose + ["up", "--detach", "--wait", "--wait-timeout", "90"])

            def address(service, port):
                value = command(compose + ["port", service, str(port)], capture_output=True).stdout.strip()
                return "http://" + value

            router, prometheus = address("router", 8080), address("prometheus", 9090)
            deadline = time.monotonic() + 30
            while True:
                data = get_json(prometheus + "/api/v1/query?" + urlencode({"query": 'up{job="vllm-emulator"}'}))
                targets = data["data"]["result"]
                if len(targets) == 2 and all(float(t["value"][1]) == 1 for t in targets):
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("Prometheus did not successfully scrape both emulator replicas")
                time.sleep(1)
            state = get_json(router + "/api/state")
            if sum(r["healthy"] for r in state["replicas"]) != 2:
                raise RuntimeError("router did not discover two healthy replicas")
            for path in ("/", "/app.js", "/style.css"):
                with open_url(router + path) as response:
                    if response.status != 200:
                        raise RuntimeError("dashboard resource unavailable: " + path)
            image_id = command(["docker", "image", "inspect", "kubeai:local", "--format", "{{.Id}}"], capture_output=True).stdout.strip()
            directory, summary = benchmark.run(router, output=output, duration=3, concurrency=4, environment="emulator",
                                               policy="kubeai", prometheus_url=prometheus, image_digest=image_id,
                                               model_revision="synthetic-emulator")
            if summary["errors"] or not summary["completed"]:
                raise RuntimeError("container benchmark reported errors")
            captured = json.loads((directory / "prometheus.json").read_text())["queries"]
            for metric in ("vllm:request_queue_time_seconds_bucket", "vllm:time_to_first_token_seconds_bucket"):
                if not captured[metric].get("data", {}).get("result"):
                    raise RuntimeError("Prometheus did not capture histogram series: " + metric)
            benchmark.write_json(output / "verification.json", {"status": "passed", "two_prometheus_targets_up": True,
                                  "healthy_replicas": 2, "benchmark_run": directory.name, "image_id": image_id,
                                  "dashboard_http_verified": True, "histograms_captured": True, "browser_visual_qa": "not performed"})
            print(f"Container verification passed: {output}", flush=True)
        finally:
            command(compose + ["down", "--volumes"], stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
