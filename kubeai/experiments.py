"""One operator-started benchmark at a time, using the CLI measurement engine."""

import copy
import json
import logging
import math
import os
import threading
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path

from . import benchmark, endpoints, workload
from .config import POLICIES
from .transport import upstream_headers

LOG = logging.getLogger("kubeai")
MAX_TRACE_BYTES = 8 * 1024 * 1024
MAX_TRACE_REQUESTS = 20000
MAX_TRACE_SECONDS = 600


def number(value, name, minimum, maximum, integer=False):
    if type(value) not in (int, float) or not minimum <= value <= maximum or not math.isfinite(value):
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    if integer and type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


class Benchmarks:
    def __init__(self, router):
        self.router = router
        self.root = router.results_dir / "dashboard-benchmarks"
        self.trace_root = router.results_dir / "workloads"
        self.active = False
        self.thread = None
        self.stop_event = threading.Event()
        self.interrupt_reason = ""
        self.job = None
        self.trace_cache = {}
        self.trace_lock = threading.Lock()
        self._recover_latest()

    def _recover_latest(self):
        for path in sorted(self.root.glob("*/job.json"), reverse=True):
            try:
                job = json.loads(path.read_text())
                if job["status"] in ("queued", "running"):
                    job.update(status="failed", finished_unix=None,
                               message="The router restarted before this benchmark finished. Partial files may remain.")
                self.job = job
                break
            except (OSError, ValueError, KeyError, TypeError):
                continue

    def status(self):
        with self.router.control_lock:
            job = copy.deepcopy(self.job)
            if job:
                end = time.time() if self.active else job.get("finished_unix")
                start = job.get("started_unix")
                job["elapsed_seconds"] = max(0, end - start) if end is not None and start is not None else None
            return {"active": self.active, "job": job, "results_directory": str(self.router.results_dir)}

    def _writable(self):
        path = self.router.results_dir
        if path.exists() and not path.is_dir():
            return False
        while not path.exists() and path != path.parent:
            path = path.parent
        return path.is_dir() and os.access(path, os.W_OK)

    def _trace(self, name):
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".jsonl"):
            raise ValueError("Choose a saved trace from the workload list")
        path = self.trace_root / name
        if path.is_symlink() or path.resolve().parent != self.trace_root.resolve() or not path.is_file():
            raise ValueError("Saved trace is unavailable")
        if path.stat().st_size > MAX_TRACE_BYTES:
            raise ValueError("Trace exceeds the dashboard's 8 MiB limit")
        raw = path.read_bytes()
        if len(raw) > MAX_TRACE_BYTES:
            raise ValueError("Trace exceeds the dashboard's 8 MiB limit")
        entries, digest = workload.load(path)
        # load() and the saved copy must describe the same file contents.
        if endpoints.revision(raw) != digest:
            raise endpoints.Conflict("Trace changed while loading. Reload the benchmark inputs.")
        if len(entries) > MAX_TRACE_REQUESTS or entries[-1]["offset_seconds"] > MAX_TRACE_SECONDS:
            raise ValueError("Dashboard traces are limited to 20,000 requests and 600 seconds")
        models = sorted({entry["body"]["model"] for entry in entries})
        return raw, {"id": name, "sha256": digest, "requests": len(entries),
                     "arrival_seconds": entries[-1]["offset_seconds"], "models": models}

    def _traces(self):
        traces, errors = [], []
        with self.trace_lock:
            for path in sorted(self.trace_root.glob("*.jsonl"))[:128]:
                try:
                    info = path.stat()
                    stamp = (info.st_mtime_ns, info.st_size)
                    cached = self.trace_cache.get(path.name)
                    if cached and cached[0] == stamp and not path.is_symlink():
                        view = cached[1]
                    else:
                        _, view = self._trace(path.name)
                        self.trace_cache[path.name] = (stamp, view)
                    traces.append(copy.deepcopy(view))
                except (OSError, ValueError, KeyError, TypeError) as error:
                    errors.append({"id": path.name, "message": "Cannot use this trace: " + type(error).__name__})
        return traces, errors

    def options(self):
        state = self.router.state()
        models = sorted({model for replica in state["replicas"] if replica["healthy"] for model in replica["models"]})
        reason = ""
        if state["configuration"]["discovery"]:
            reason = "Dashboard benchmarks currently require static endpoint configuration."
        elif not self._writable():
            reason = "The results directory must be writable to run dashboard benchmarks."
        elif state["control_error"]:
            reason = "Resolve the router configuration error before starting a benchmark."
        elif not models:
            reason = "No verified model is available on healthy replicas."
        traces, errors = self._traces()
        return {"can_run": not reason, "reason": reason, "models": models, "policies": list(POLICIES),
                "traces": traces, "trace_errors": errors,
                "defaults": {"mode": "closed_loop", "policy": state["policy"], "concurrency": 4,
                             "duration": 30, "seed": 42, "warmup": 2, "max_inflight": 128,
                             "environment": "unverified", "gpu": False}}

    def _settings(self, body):
        allowed = {"mode", "policy", "model", "concurrency", "duration", "seed", "warmup", "max_inflight",
                   "trace_id", "trace_sha256", "environment", "gpu", "model_revision", "image_digest"}
        if set(body) - allowed:
            raise ValueError("Unknown benchmark setting")
        mode = body.get("mode", "closed_loop")
        if mode not in ("closed_loop", "fixed_arrival"):
            raise ValueError("Choose closed_loop or fixed_arrival")
        policy = body.get("policy")
        if policy not in POLICIES:
            raise ValueError("Choose round_robin, capacity or kubeai")
        environment = body.get("environment", "unverified")
        if environment not in ("unverified", "hardware", "emulator"):
            raise ValueError("Invalid environment label")
        gpu = body.get("gpu", False)
        if type(gpu) is not bool or (gpu and environment != "hardware"):
            raise ValueError("Local GPU sampling requires the hardware environment label")
        settings = {"mode": mode, "policy": policy, "environment": environment, "gpu": gpu,
                    "warmup": number(body.get("warmup", 2), "warmup", 0, 20, True)}
        for key in ("model_revision", "image_digest"):
            value = body.get(key, "")
            if not isinstance(value, str) or len(value) > 512:
                raise ValueError(f"{key} must be a string of at most 512 characters")
            settings[key] = value.strip() or "unknown"
        raw_trace = None
        if mode == "closed_loop":
            if any(key in body for key in ("trace_id", "trace_sha256", "max_inflight")):
                raise ValueError("Trace and max-inflight settings only apply to fixed-arrival replay")
            settings.update(model=body.get("model"),
                            concurrency=number(body.get("concurrency", 4), "workers", 1, 128, True),
                            duration=number(body.get("duration", 30), "duration", .05, 300),
                            seed=number(body.get("seed", 42), "seed", 0, 2147483647, True))
            if not isinstance(settings["model"], str) or not settings["model"]:
                raise ValueError("Choose a model")
            models = [settings["model"]]
        else:
            if any(key in body for key in ("concurrency", "duration", "seed", "model")):
                raise ValueError("Workers, duration, seed and model are defined differently for trace replay; choose a saved trace")
            raw_trace, trace = self._trace(body.get("trace_id"))
            if body.get("trace_sha256") != trace["sha256"]:
                raise endpoints.Conflict("Trace changed. Reload the benchmark inputs before starting.")
            settings.update(trace_id=trace["id"], trace_sha256=trace["sha256"],
                            requests=trace["requests"], arrival_seconds=trace["arrival_seconds"], models=trace["models"],
                            max_inflight=number(body.get("max_inflight", 128), "max_inflight", 1, 128, True))
            models = trace["models"]
        available = {model for replica in self.router.policy.snapshot() if replica["healthy"] for model in replica["models"]}
        if set(models) - available:
            raise ValueError("A requested model is unavailable on healthy replicas")
        return settings, raw_trace

    def _persist(self):
        root = self.root / self.job["job_id"]
        temporary = root / "job.next.json"
        benchmark.write_json(temporary, self.job)
        os.replace(temporary, root / "job.json")

    def start(self, body, url):
        with self.router.control_lock:
            if self.active:
                raise endpoints.Conflict("A benchmark is already running. Wait for it to finish.")
            self.router.reconcile()
            if self.router.stop_event.is_set():
                raise endpoints.Conflict("The router is stopping")
            if self.router.policy.config.discovery:
                raise ValueError("Dashboard benchmarks currently require static endpoints")
            if self.router.control_error:
                raise ValueError("Resolve the router configuration error before benchmarking")
            if self.router.policy.active:
                raise endpoints.Conflict("Wait for current inference requests to finish before benchmarking")
            settings, raw_trace = self._settings(body)
            if not self._writable():
                raise ValueError("The results directory is not writable")
            original = self.router.policy.config
            job_id = time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
            root = self.root / job_id
            root.mkdir(parents=True)
            if raw_trace is not None:
                (root / "trace-input.jsonl").write_bytes(raw_trace)
            self.job = {"job_id": job_id, "status": "queued", "created_utc": benchmark.utc(time.time()),
                        "started_unix": None, "finished_unix": None, "settings": settings,
                        "message": "Benchmark queued", "run_id": None, "directory": None, "summary": None}
            self._persist()
            self.stop_event = threading.Event()
            self.interrupt_reason = ""
            self.active = True
            with self.router.policy.lock:
                self.router.policy.config = replace(original, policy=settings["policy"])
            self.thread = threading.Thread(target=self._run, args=(url, settings, root, original), daemon=True)
            try:
                self.thread.start()
            except Exception:
                self.active = False
                with self.router.policy.lock:
                    self.router.policy.config = original
                self.job.update(status="failed", finished_unix=time.time(), message="Could not start benchmark worker")
                self._persist()
                raise
            return self.status()

    def _run(self, url, settings, root, original):
        directory, summary, failure = None, None, None
        try:
            with self.router.control_lock:
                self.job.update(status="running", started_unix=time.time(), message="Benchmark running")
                context = asdict(self.router.policy.config)
                self._persist()
            urls, headers = [url + "/metrics"], {}
            for replica in original.replicas:
                metric_url = replica.url.rstrip("/") + "/metrics"
                urls.append(metric_url)
                try:
                    headers[metric_url] = upstream_headers(replica)
                except ValueError:
                    # Missing credentials on an unused/unhealthy replica are a
                    # telemetry failure, not a reason to discard other endpoints.
                    headers[metric_url] = {}
                if replica.gpu_metrics_url:
                    urls.append(replica.gpu_metrics_url)
            args = {"url": url, "output": root, "warmup": settings["warmup"], "environment": settings["environment"],
                    "policy": settings["policy"], "gpu": settings["gpu"], "model_revision": settings["model_revision"],
                    "image_digest": settings["image_digest"], "metrics_urls": list(dict.fromkeys(urls)),
                    "metrics_headers": headers, "api_key": os.environ.get("KUBEAI_API_KEY", ""), "stop_event": self.stop_event}
            if settings["mode"] == "closed_loop":
                args.update(model=settings["model"], concurrency=settings["concurrency"],
                            duration=settings["duration"], seed=settings["seed"])
            else:
                args.update(trace_path=root / "trace-input.jsonl", max_inflight=settings["max_inflight"], seed=None)
            directory, summary = benchmark.run(**args)
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["dashboard_experiment"] = {"job_id": self.job["job_id"], "settings": settings,
                                                 "router_configuration": context,
                                                 "gpu_collection_scope": "router_host" if settings["gpu"] else None}
            summary.update(dashboard_job_id=self.job["job_id"], benchmark_settings=settings,
                           router_configuration=context)
            with self.router.control_lock:
                self.router.reconcile()
            if self.stop_event.is_set() or summary["policy"] != settings["policy"]:
                summary["valid_for_comparison"] = False
                failure = self.interrupt_reason or "Benchmark interrupted or observed policy did not match the selection"
                manifest["dashboard_experiment"].update(valid_for_comparison=False, failure=failure)
            benchmark.write_json(manifest_path, manifest)
            benchmark.write_json(directory / "summary.json", summary)
        except Exception as error:
            failure = self.interrupt_reason or f"{type(error).__name__}: {str(error)[:400]}"
            LOG.warning("dashboard benchmark failed: %s", failure)
        finally:
            with self.router.control_lock:
                with self.router.policy.lock:
                    self.router.policy.config = original
                self.active = False
                self.router.reconcile()
                self.job.update(status="failed" if failure else "completed", finished_unix=time.time(),
                                message=failure or "Benchmark completed. Review request errors and token-limit truncations in the saved results.",
                                run_id=directory.name if directory else None,
                                directory=str(directory.relative_to(self.router.results_dir)) if directory else None,
                                summary=summary)
                try:
                    self._persist()
                except OSError as error:
                    LOG.warning("could not persist benchmark status: %s", type(error).__name__)

    def interrupt(self, reason):
        with self.router.control_lock:
            if self.active:
                self.interrupt_reason = reason
                self.stop_event.set()

    def stop(self):
        self.interrupt("The router stopped before the benchmark completed")
        if self.thread:
            self.thread.join(timeout=5)
