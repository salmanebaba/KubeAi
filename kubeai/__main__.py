"""Run with python -m kubeai; no dependency installation is required."""

import argparse
import json
import logging
import os
from pathlib import Path

from . import benchmark, demo, emulator, gpu, placement, workload
from .config import Config, POLICIES
from .router import Router, make_server


def main():
    parser = argparse.ArgumentParser(description="KubeAI inference orchestration prototype")
    sub = parser.add_subparsers(dest="command", required=True)
    router = sub.add_parser("router", help="serve inference proxy and dashboard")
    router.add_argument("--config", default="config/local.json")
    router.add_argument("--host", default="127.0.0.1")
    router.add_argument("--port", type=int, default=8080)
    router.add_argument("--results", default="results")
    emu = sub.add_parser("emulator", help="serve a synthetic replica; no GPU used")
    emu.add_argument("--host", default="127.0.0.1")
    emu.add_argument("--port", type=int, default=8001)
    emu.add_argument("--model", default="Qwen/Qwen3-0.6B")
    emu.add_argument("--capacity", type=int, default=2)
    emu.add_argument("--token-seconds", type=float, default=.003)
    emu.add_argument("--prefill-seconds", type=float, default=.01)
    exporter = sub.add_parser("gpu-exporter", help="expose physical telemetry for exactly one NVIDIA GPU")
    exporter.add_argument("--host", default="127.0.0.1")
    exporter.add_argument("--port", type=int, default=9401)
    exporter.add_argument("--gpu-id", default="0", help="physical GPU UUID or NVIDIA index")
    trace = sub.add_parser("trace", help="write a seeded fixed-arrival JSONL workload")
    trace.add_argument("--output", default="results/trace.jsonl")
    trace.add_argument("--requests", type=int, default=100)
    trace.add_argument("--rate", type=float, default=20)
    trace.add_argument("--seed", type=int, default=42)
    trace.add_argument("--scenario", choices=["steady", "burst", "mixed"], default="steady")
    trace.add_argument("--models", nargs="+", default=["Qwen/Qwen3-0.6B"])
    bench = sub.add_parser("benchmark", help="replay a trace or run closed-loop workers")
    bench.add_argument("--url", default="http://127.0.0.1:8080")
    bench.add_argument("--output", default="results")
    bench.add_argument("--trace", dest="trace_path")
    bench.add_argument("--request", dest="request_path", help="fixed chat JSON payload for a closed-loop run")
    bench.add_argument("--concurrency", type=int, default=4)
    bench.add_argument("--duration", type=float, default=30)
    bench.add_argument("--seed", type=int, default=42)
    bench.add_argument("--warmup", type=int, default=2)
    bench.add_argument("--no-stream", dest="stream", action="store_false")
    bench.add_argument("--metrics-url", dest="metrics_urls", action="append")
    bench.add_argument("--prometheus-url")
    bench.add_argument("--gpu", action="store_true")
    bench.add_argument("--environment", choices=["emulator", "hardware", "unverified"], default="unverified")
    bench.add_argument("--policy", choices=[*POLICIES, "direct"], default="direct")
    bench.add_argument("--model", default="Qwen/Qwen3-0.6B")
    bench.add_argument("--model-revision", default="unknown")
    bench.add_argument("--image-digest", default="unknown")
    bench.add_argument("--hourly-cost", type=float)
    bench.add_argument("--timeout", type=float, default=60)
    bench.add_argument("--max-inflight", type=int, default=128)
    compare = sub.add_parser("demo", help="compare all policies with fresh HTTP emulator replicas")
    compare.add_argument("--output", default="results")
    compare.add_argument("--repeats", type=int, default=3)
    compare.add_argument("--requests", type=int, default=48)
    compare.add_argument("--seed", type=int, default=42)
    place = sub.add_parser("placement", help="recommend placement and optionally write a Pod manifest; never applies")
    place.add_argument("--inventory", required=True)
    place.add_argument("--request", required=True)
    place.add_argument("--policy", choices=["first_fit", "best_fit"], default="best_fit")
    place.add_argument("--at", type=float, help="replay a historical inventory at this Unix timestamp")
    place.add_argument("--stale-after", type=float, default=30)
    place.add_argument("--pod-output")
    place.add_argument("--image", help="verified vLLM image digest, required with --pod-output")
    place.add_argument("--model-revision")
    args = vars(parser.parse_args())
    command = args.pop("command")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if command == "trace":
            output = args.pop("output")
            args["count"] = args.pop("requests")
            workload.save(output, workload.generate(**args))
            print(output)
        elif command == "benchmark":
            directory, summary = benchmark.run(**args, api_key=os.environ.get("KUBEAI_API_KEY", ""))
            print(json.dumps(summary, indent=2))
            print(f"Saved: {directory}")
        elif command == "demo":
            logging.getLogger("kubeai").setLevel(logging.WARNING)
            print(demo.run(**args))
        elif command == "placement":
            inventory = json.loads(Path(args["inventory"]).read_text())
            request = json.loads(Path(args["request"]).read_text())
            decision = placement.recommend(inventory, request, args["policy"], args["at"], args["stale_after"])
            if args["pod_output"]:
                if not args["image"]:
                    raise ValueError("--pod-output requires --image")
                benchmark.write_json(args["pod_output"], placement.pod_manifest(decision, args["image"], model_revision=args["model_revision"]))
            print(json.dumps(decision, indent=2))
        else:
            service = None
            if command == "router":
                service = Router(Config.load(args["config"]), args["config"], args["results"])
                service.start()
                server = make_server(service, args["host"], args["port"])
            elif command == "gpu-exporter":
                server = gpu.make_server(**args)
            else:
                host, port = args.pop("host"), args.pop("port")
                server = emulator.make_server(emulator.Emulator(**args), host, port)
            print(f"KubeAI {command}: http://{server.server_address[0]}:{server.server_port}", flush=True)
            try:
                server.serve_forever()
            finally:
                server.server_close()
                if service:
                    service.stop()
    except KeyboardInterrupt:
        pass
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        parser.exit(1, f"kubeai: {error}\n")


if __name__ == "__main__":
    main()
