"""HTTP replay, closed-loop baseline, streaming TTFT and time-aligned telemetry."""

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import threading
import time
import urllib.error
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlsplit

from . import __version__, energy, workload
from .metrics import signals
from .transport import get_json, open_url, validate_chat

GPU_BASE_FIELDS = "timestamp,uuid,name,driver_version,memory.total,memory.used,utilization.gpu"
GPU_FIELDS = GPU_BASE_FIELDS + ",power.draw"


def source_digest():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and (path.suffix == ".py" or path.parent.name == "static"):
            digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()

FIELDS = ["id", "tenant", "model", "scheduled_unix", "started_unix", "ended_unix", "http_code", "completed",
          "latency_seconds", "scheduled_latency_seconds", "ttft_seconds", "scheduled_ttft_seconds", "finish_reason",
          "output_tokens", "replica", "policy", "error"]


def utc(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def request(url, entry, scheduled, stream=True, timeout=60, api_key=""):
    body = dict(entry["body"])
    body["stream"] = stream
    if stream:
        body["stream_options"] = {"include_usage": True}
    else:
        body.pop("stream_options", None)
    started, tick = time.time(), time.perf_counter()
    row = dict.fromkeys(FIELDS, "")
    row.update(id=entry["id"], tenant=entry.get("tenant", "default"), model=body["model"], scheduled_unix=scheduled,
               started_unix=started, http_code=0, completed=False)
    headers = {"Content-Type": "application/json", "X-Tenant-ID": entry.get("tenant", "default")}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    try:
        with open_url(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(), headers, timeout) as response:
            row.update(http_code=response.status, replica=response.headers.get("X-KubeAI-Replica", "direct"),
                       policy=response.headers.get("X-KubeAI-Policy", "direct"))
            done = False
            if stream:
                if "text/event-stream" not in response.headers.get("Content-Type", ""):
                    raise ValueError("expected text/event-stream")
                data_lines = []
                for raw in response:
                    line = raw.decode().rstrip("\r\n")
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                    elif not line and data_lines:
                        data = "\n".join(data_lines)
                        data_lines = []
                        if data == "[DONE]":
                            done = True
                            break
                        event = json.loads(data)
                        if event.get("error"):
                            raise ValueError("upstream stream error")
                        for choice in event.get("choices", []):
                            delta = choice.get("delta", {})
                            if any(delta.get(key) for key in ("content", "reasoning_content", "reasoning", "tool_calls")) and row["ttft_seconds"] == "":
                                row["ttft_seconds"] = time.perf_counter() - tick
                                row["scheduled_ttft_seconds"] = time.time() - scheduled
                            if choice.get("finish_reason"):
                                row["finish_reason"] = choice["finish_reason"]
                        if event.get("usage") and event["usage"].get("completion_tokens") is not None:
                            row["output_tokens"] = event["usage"]["completion_tokens"]
            else:
                result = json.load(response)
                row["finish_reason"] = result["choices"][0]["finish_reason"]
                row["output_tokens"] = result.get("usage", {}).get("completion_tokens", "")
                done = True
            row["completed"] = done and row["finish_reason"] in ("stop", "length", "tool_calls")
            if not row["completed"]:
                row["error"] = "incomplete_or_filtered_response"
    except urllib.error.HTTPError as error:
        row.update(http_code=error.code, error=f"HTTP_{error.code}")
    except Exception as error:
        row["error"] = type(error).__name__ + ": " + str(error)
    row.update(ended_unix=time.time(), latency_seconds=time.perf_counter() - tick,
               scheduled_latency_seconds=time.time() - scheduled)
    return row


def quantile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    return values[max(0, math.ceil(len(values) * fraction) - 1)]


def summarize(rows, elapsed, **metadata):
    completed = [r for r in rows if r["completed"]]
    latency = [r["latency_seconds"] for r in completed]
    ttft = [r["ttft_seconds"] for r in completed if r["ttft_seconds"] != ""]
    scheduled = [r["scheduled_latency_seconds"] for r in completed]
    offered_by_tenant = Counter(r["tenant"] for r in rows)
    delivered_by_tenant = Counter(r["tenant"] for r in completed)
    ratios = [delivered_by_tenant[t] / n for t, n in offered_by_tenant.items()]
    squares = sum(r * r for r in ratios)
    fairness = sum(ratios) ** 2 / (len(ratios) * squares) if squares else 0.0
    return {**metadata, "requests": len(rows), "completed": len(completed), "errors": len(rows) - len(completed),
            "error_rate": (len(rows) - len(completed)) / len(rows) if rows else 0,
            "dropped": sum(r["error"] == "client_inflight_limit" for r in rows),
            "truncated": sum(r["finish_reason"] == "length" for r in completed),
            "elapsed_seconds": elapsed, "completed_rps": len(completed) / elapsed if elapsed else 0,
            "latency_p50_seconds": quantile(latency, .5), "latency_p95_seconds": quantile(latency, .95),
            "scheduled_latency_p95_seconds": quantile(scheduled, .95),
            "ttft_p50_seconds": quantile(ttft, .5), "ttft_p95_seconds": quantile(ttft, .95),
            "output_tokens": sum(r["output_tokens"] for r in completed if r["output_tokens"] != ""),
            "token_usage_missing": sum(r["output_tokens"] == "" for r in completed),
            "tenant_completion_ratios": {t: delivered_by_tenant[t] / n for t, n in offered_by_tenant.items()},
            "jain_completion_ratio": fairness,
            "by_replica": dict(Counter(r["replica"] for r in completed))}


class Telemetry:
    def __init__(self, directory, urls, api_key="", interval=1.0, gpu=False, auth_origin="", metrics_headers=None):
        self.directory, self.urls, self.api_key = Path(directory), urls, api_key
        self.interval, self.gpu = interval, gpu
        parsed = urlsplit(auth_origin)
        self.auth_origin = (parsed.scheme, parsed.netloc)
        self.metrics_headers = metrics_headers or {}
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.primed = False
        self.gpu_program = None

    def sample_gpu(self):
        wall, tick = time.time(), time.monotonic()
        record = {"source": "nvidia-smi"}
        if self.gpu_program:
            for fields in (GPU_FIELDS, GPU_BASE_FIELDS):
                try:
                    result = subprocess.run([self.gpu_program, "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
                                            capture_output=True, text=True, timeout=2, check=True)
                    record.update(csv=result.stdout, fields=fields.split(","))
                    break
                except (subprocess.SubprocessError, OSError) as error:
                    if fields == GPU_FIELDS:
                        record["power_query_error"] = type(error).__name__
                    else:
                        record["error"] = type(error).__name__
        else:
            record["error"] = "nvidia-smi unavailable"
        record.update(timestamp_unix=(wall + time.time()) / 2,
                      sample_monotonic_seconds=(tick + time.monotonic()) / 2,
                      query_duration_seconds=time.monotonic() - tick)
        return record

    def loop(self):
        with (self.directory / "telemetry.jsonl").open("a" if self.primed else "w") as file:
            while not self.stop_event.is_set():
                for url in self.urls:
                    record = {"timestamp_unix": time.time(), "source": url}
                    try:
                        parsed = urlsplit(url)
                        same_origin = (parsed.scheme, parsed.netloc) == self.auth_origin
                        headers = dict(self.metrics_headers.get(url, {}))
                        if self.api_key and same_origin:
                            headers["Authorization"] = "Bearer " + self.api_key
                        with open_url(url, headers=headers, timeout=1) as response:
                            record["raw_metrics"] = response.read(4 * 1024 * 1024).decode()
                        record["signals"] = signals(record["raw_metrics"])
                    except Exception as error:
                        record["error"] = type(error).__name__
                    file.write(json.dumps(record) + "\n")
                if self.gpu:
                    file.write(json.dumps(self.sample_gpu()) + "\n")
                file.flush()
                self.stop_event.wait(self.interval)

    def __enter__(self):
        if self.gpu:
            self.gpu_program = shutil.which("nvidia-smi")
            # Before the measurement starts, so integration can bracket its first instant.
            with (self.directory / "telemetry.jsonl").open("w") as file:
                file.write(json.dumps(self.sample_gpu()) + "\n")
            self.primed = True
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join()
        if self.gpu:
            # After the measurement ends; interpolate to the boundary without counting shutdown time.
            with (self.directory / "telemetry.jsonl").open("a") as file:
                file.write(json.dumps(self.sample_gpu()) + "\n")


def capture_prometheus(url, start, end, directory):
    result = {"start_unix": start, "end_unix": end, "step_seconds": 2, "queries": {}}
    for query in ["vllm:num_requests_waiting", "vllm:num_requests_running", "vllm:kv_cache_usage_perc",
                  "vllm:time_to_first_token_seconds_bucket", "vllm:request_queue_time_seconds_bucket",
                  "vllm:time_to_first_token_seconds_sum", "vllm:time_to_first_token_seconds_count",
                  "vllm:request_queue_time_seconds_sum", "vllm:request_queue_time_seconds_count",
                  "DCGM_FI_DEV_GPU_UTIL", "DCGM_FI_DEV_FB_USED", "DCGM_FI_DEV_FB_FREE"]:
        try:
            data = get_json(url.rstrip("/") + "/api/v1/query_range?" + urlencode({"query": query, "start": start, "end": end, "step": 2}), timeout=5)
            result["queries"][query] = data
        except Exception as error:
            result["queries"][query] = {"error": type(error).__name__}
    write_json(Path(directory) / "prometheus.json", result)


def run(url, output="results", trace_path=None, concurrency=4, duration=30.0, seed=42, warmup=2, stream=True,
        metrics_urls=None, prometheus_url=None, gpu=False, environment="unverified", policy="direct",
        model="Qwen/Qwen3-0.6B", model_revision="unknown", image_digest="unknown", hourly_cost=None,
        api_key="", timeout=60, max_inflight=128, request_path=None, stop_event=None, metrics_headers=None):
    stop_event = stop_event or threading.Event()
    if concurrency < 1 or duration <= 0 or max_inflight < 1 or warmup < 0 or timeout <= 0:
        raise ValueError("concurrency, duration, max_inflight and timeout must be positive; warmup nonnegative")
    if not all(math.isfinite(value) for value in (duration, timeout)):
        raise ValueError("duration and timeout must be finite")
    if hourly_cost is not None and (hourly_cost < 0 or not math.isfinite(hourly_cost)):
        raise ValueError("hourly_cost must be finite and nonnegative")
    if trace_path and request_path:
        raise ValueError("choose either --trace or --request")
    router_before = None
    try:
        headers = {"Authorization": "Bearer " + api_key} if api_key else {}
        router_before = get_json(url.rstrip("/") + "/api/state", timeout=2, headers=headers)
    except Exception:
        pass  # A direct vLLM endpoint has no KubeAI state API.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + policy + "-" + uuid.uuid4().hex[:8]
    directory = Path(output) / run_id
    directory.mkdir(parents=True)
    if trace_path:
        entries, digest = workload.load(trace_path)
        (directory / "trace.jsonl").write_bytes(Path(trace_path).read_bytes())
        mode = "fixed_arrival"
    else:
        if request_path:
            body = json.loads(Path(request_path).read_text())
            validate_chat(body)
            entries = [{"id": "0", "offset_seconds": 0, "tenant": "default", "body": body}]
        else:
            entries = workload.generate(max(32, concurrency), seed=seed, models=[model])
        workload.save(directory / "inputs.jsonl", entries)
        _, digest = workload.load(directory / "inputs.jsonl")
        mode = "closed_loop"
    # Warm every requested model. Failed warm-up aborts instead of contaminating a run.
    for entry in {e["body"]["model"]: e for e in entries}.values():
        for _ in range(warmup):
            if stop_event.is_set():
                raise RuntimeError("Benchmark interrupted during warm-up")
            row = request(url, entry, time.time(), stream, timeout, api_key)
            if not row["completed"]:
                raise RuntimeError("warm-up failed: " + row["error"])
    rows = []
    interval = .25 if environment == "emulator" else 1.0
    manifest = {"schema_version": 1, "kubeai_version": __version__, "source_sha256": source_digest(), "run_id": run_id, "environment": environment,
                "policy": policy, "mode": mode, "url": url, "seed": seed, "trace_sha256": digest,
                "stream": stream, "warmup_per_model": warmup,
                "concurrency": concurrency if mode == "closed_loop" else None, "client_max_inflight": max_inflight,
                "requested_duration_seconds": duration if mode == "closed_loop" else None,
                "requested_models": sorted({entry["body"]["model"] for entry in entries}),
                "model_revision": model_revision, "image_digest": image_digest,
                "platform": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count(),
                "metrics_urls": metrics_urls or [url.rstrip("/") + "/metrics"], "gpu_telemetry_requested": gpu,
                "gpu_energy_measurement": energy.specification(interval) if gpu else None,
                "prometheus_url": prometheus_url, "timeout_seconds": timeout}
    manifest["router_state_before"] = router_before
    with Telemetry(directory, manifest["metrics_urls"], api_key, interval=interval,
                   gpu=gpu, auth_origin=url, metrics_headers=metrics_headers):
        started, tick = time.time(), time.monotonic()
        manifest.update(start_unix=started, start_utc=utc(started), start_monotonic_seconds=tick)
        write_json(directory / "manifest.json", manifest)
        if trace_path:
            slots = threading.BoundedSemaphore(max_inflight)

            def execute(entry):
                try:
                    return request(url, entry, started + entry["offset_seconds"], stream, timeout, api_key)
                finally:
                    slots.release()

            futures = []
            with ThreadPoolExecutor(max_workers=max_inflight) as pool:
                for entry in entries:
                    if stop_event.is_set():
                        break
                    wait = tick + entry["offset_seconds"] - time.monotonic()
                    if wait > 0 and stop_event.wait(wait):
                        break
                    if slots.acquire(blocking=False):
                        futures.append(pool.submit(execute, entry))
                    else:
                        row = dict.fromkeys(FIELDS, "")
                        row.update(id=entry["id"], tenant=entry.get("tenant", "default"), model=entry["body"]["model"],
                                   scheduled_unix=started + entry["offset_seconds"], started_unix=time.time(), ended_unix=time.time(),
                                   http_code=0, completed=False, error="client_inflight_limit")
                        rows.append(row)
                rows.extend(f.result() for f in futures)
        else:
            def worker(worker_id):
                batch, count = [], 0
                while not stop_event.is_set() and time.monotonic() - tick < duration:
                    entry = dict(entries[(worker_id + count * concurrency) % len(entries)])
                    entry["id"] = f"{worker_id}-{count}"
                    batch.append(request(url, entry, time.time(), stream, timeout, api_key))
                    count += 1
                return batch

            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for batch in pool.map(worker, range(concurrency)):
                    rows.extend(batch)
        # Exclude telemetry shutdown and post-run Prometheus queries from latency/throughput.
        ended, end_tick = time.time(), time.monotonic()
        elapsed = end_tick - tick
    rows.sort(key=lambda r: (r["scheduled_unix"], r["id"]))
    with (directory / "requests.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    observed = sorted({r["policy"] for r in rows if r["policy"]})
    actual_policy = observed[0] if len(observed) == 1 else ("mixed" if observed else policy)
    summary = summarize(rows, elapsed, environment=environment, policy=actual_policy, mode=mode, trace_sha256=digest,
                        start_utc=utc(started), run_id=run_id, source_sha256=manifest["source_sha256"])
    if stop_event.is_set():
        summary.update(interrupted=True, valid_for_comparison=False)
        manifest.update(interrupted=True, planned_requests=len(entries) if trace_path else None)
    if hourly_cost is not None:
        summary["cost_model_hourly_usd"] = hourly_cost
        summary["estimated_cost_per_completed_usd"] = hourly_cost * elapsed / 3600 / summary["completed"] if summary["completed"] else None
    # Largest sampled value, never presented as the actual continuous peak.
    peaks, gpu_records = {}, []
    for line in (directory / "telemetry.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record.get("source") == "nvidia-smi":
            gpu_records.append(record)
        for key, value in record.get("signals", {}).items():
            if key in ("waiting", "running", "kv"):
                peaks[key] = max(peaks.get(key, 0), value)
    summary["largest_sampled_signals"] = peaks
    manifest.update(end_unix=ended, end_utc=utc(ended), end_monotonic_seconds=end_tick, elapsed_seconds=elapsed, requests=len(rows),
                    requested_policy_label=policy, policy=actual_policy, observed_policies=observed,
                    policy_label_mismatch=bool(observed and policy != "direct" and observed != [policy]))
    try:
        manifest["router_state_after"] = get_json(url.rstrip("/") + "/api/state", timeout=2, headers=headers)
    except Exception:
        manifest["router_state_after"] = None
    summary["gpu_energy"] = energy.estimate(gpu_records, manifest, summary)
    write_json(directory / "manifest.json", manifest)
    write_json(directory / "summary.json", summary)
    if prometheus_url:
        capture_prometheus(prometheus_url, started, ended, directory)
    return directory, summary
