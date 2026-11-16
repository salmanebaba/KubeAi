"""Run identical traces against actual HTTP emulators for all routing policies."""

import json
import statistics
import threading
from contextlib import ExitStack
from pathlib import Path

from . import benchmark, emulator, workload
from .config import Config, POLICIES
from .router import Router, make_server


def serve(stack, server):
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .05}, daemon=True)
    thread.start()

    def close():
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    stack.callback(close)
    return f"http://127.0.0.1:{server.server_port}"


def report(output, records):
    groups = {}
    for record in records:
        groups.setdefault((record["scenario"], record["policy"]), []).append(record)
    lines = ["# KubeAI emulator comparison", "", "These are synthetic HTTP emulator results, not GPU performance measurements.", "",
             "All policies receive the identical seeded trace for a scenario. Each repeat starts fresh replicas and a fresh router. "
             "Service-time estimates are known emulator parameters, not calibrated real-model estimates. "
             "HTTP overhead, OS scheduling and metric sampling add variation.", "",
             "P95 is the nearest-rank percentile across successful requests. Throughput includes final draining time. "
             "A completed response includes `stop`, `length` or `tool_calls`; truncations are recorded separately. "
             "TTFT begins at client dispatch and ends at the first generated content/reasoning/tool delta. "
             "Scheduled latency additionally includes load-generator delays. Warm-up requests are excluded.", "",
             "| Scenario | Policy | Repeats | Mean P95 latency (s) | P95 range (s) | Mean TTFT P95 (s) | Mean completed/s | Errors |",
             "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |"]
    for (scenario, policy), runs in groups.items():
        values = [r["latency_p95_seconds"] for r in runs if r["latency_p95_seconds"] is not None]
        ttft = [r["ttft_p95_seconds"] for r in runs if r["ttft_p95_seconds"] is not None]
        if not values or not ttft:
            lines.append(f"| {scenario} | {policy} | {len(runs)} | unavailable | unavailable | unavailable | 0 | {sum(r['errors'] for r in runs)} |")
            continue
        lines.append(f"| {scenario} | {policy} | {len(runs)} | {statistics.mean(values):.4f} | {min(values):.4f}–{max(values):.4f} | {statistics.mean(ttft):.4f} | {statistics.mean(r['completed_rps'] for r in runs):.2f} | {sum(r['errors'] for r in runs)} |")
    lines += ["", "Variation is shown as the observed range across repeats; it is not a confidence interval. "
              "The balanced case can favor a simple baseline. A benefit with unequal emulated speeds demonstrates policy logic only. "
              "It does not establish hardware efficiency, dollar savings or tenant scheduling fairness.", "",
              "Raw request CSV, copied trace, telemetry, run manifest and summary are in each run directory. "
              "`comparison.json` indexes every run and its scenario. No physical GPU utilization or cost was inferred.", "",
              "Placement is a separate recommendation policy. Whole-GPU count, per-device memory, model compatibility, "
              "CPU/RAM, node readiness and telemetry age are checked before producing node affinity. "
              "Live Kubernetes and multi-GPU validation remain necessary.", ""]
    (Path(output) / "REPORT.md").write_text("\n".join(lines))


def run(output="results", repeats=3, requests=48, seed=42):
    if repeats < 1 or requests < 3:
        raise ValueError("repeats must be positive and requests >= 3")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for scenario in ("balanced", "uneven", "mixed_models"):
        model = "Qwen/Qwen3-0.6B"
        models = [model, "demo/second-model"] if scenario == "mixed_models" else [model]
        trace = output / f"trace-{scenario}.jsonl"
        workload.save(trace, workload.generate(requests, rate=32, seed=seed, scenario="burst", models=models))
        for repeat in range(repeats):
            # Rotate policy order to reduce systematic run-order effects.
            for policy in POLICIES[repeat % 3:] + POLICIES[:repeat % 3]:
                with ExitStack() as stack:
                    replicas, metrics_urls = [], []
                    speeds = [.0015, .0015] if scenario == "balanced" else [.004, .001]
                    for index, speed in enumerate(speeds):
                        backend = emulator.Emulator(model, capacity=2, token_seconds=speed)
                        url = serve(stack, emulator.make_server(backend, port=0))
                        metrics_urls.append(url + "/metrics")
                        replicas.append({"name": f"replica-{index}", "url": url, "models": [model], "capacity": 2,
                                         "token_seconds": speed, "prefill_seconds": .01})
                    if scenario == "mixed_models":
                        backend = emulator.Emulator(models[1], capacity=1, token_seconds=.002)
                        url = serve(stack, emulator.make_server(backend, port=0))
                        metrics_urls.append(url + "/metrics")
                        replicas.append({"name": "second-model", "url": url, "models": [models[1]], "capacity": 1, "token_seconds": .002})
                    config = Config.parse({"replicas": replicas, "policy": policy, "poll_interval": .1, "stale_after": 2})
                    router = Router(config, results_dir=output)
                    router.start()
                    stack.callback(router.stop)
                    url = serve(stack, make_server(router, port=0))
                    directory, summary = benchmark.run(url, output=output, trace_path=trace, warmup=1,
                                                       metrics_urls=metrics_urls + [url + "/metrics"],
                                                       environment="emulator", policy=policy, seed=seed)
                    summary.update(scenario=scenario, repeat=repeat, replica_configuration=replicas,
                                   score_weights=config.weights)
                    benchmark.write_json(directory / "summary.json", summary)
                    manifest = json.loads((directory / "manifest.json").read_text())
                    manifest.update(scenario=scenario, repeat=repeat, router_configuration={"replicas": replicas, "policy": policy, "weights": config.weights})
                    benchmark.write_json(directory / "manifest.json", manifest)
                    records.append(summary)
                    print(f"{scenario} repeat={repeat + 1} {policy}: {summary['completed']}/{summary['requests']} completed, P95={summary['latency_p95_seconds']}", flush=True)
    benchmark.write_json(output / "comparison.json", {"environment": "emulator", "seed": seed, "repeats": repeats, "runs": records})
    report(output, records)
    return output / "REPORT.md"
