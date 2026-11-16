"""Read saved measurements and compare policies only on matching recorded inputs."""

import copy
import csv
import hashlib
import io
import json
import math
import statistics
import threading
from pathlib import Path

from . import energy


def finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def known(value):
    return value is not None and value != "" and value != "unknown" and value != "unverified"


def read_json(path):
    if path.is_symlink():
        raise ValueError("Artifact must be a regular file")
    def reject_constant(value):
        raise ValueError("Non-finite JSON number: " + value)
    value = json.loads(path.read_text(), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError("Artifact must contain an object")
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def replica_breakdown(path, configured_names=()):
    """Read client observations on demand; never invent a replica for a failed request."""
    def unavailable(message):
        return {"status": "unavailable", "message": message, "rows": None, "replicas": []}

    def number(value):
        try:
            parsed = float(value)
            return parsed if finite(parsed) and parsed >= 0 else None
        except (ValueError, TypeError, OverflowError):
            return None

    def percentile(values, fraction, completed):
        if not values or len(values) != completed:
            return None
        return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]

    def bucket(name):
        return {"replica": name, "requests": 0, "completed": 0, "errors": 0, "dropped": 0,
                "truncated": 0, "latencies": [], "ttfts": []}

    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
            return unavailable("Request CSV is missing, linked or exceeds the 64 MiB inspection limit.")
        groups = {name: bucket(name) for name in configured_names}
        total = 0
        with path.open(newline="") as file:
            reader = csv.DictReader(file)
            required = {"replica", "completed", "latency_seconds", "ttft_seconds", "error", "finish_reason"}
            if not required.issubset(reader.fieldnames or []):
                return unavailable("Request CSV does not contain the required measurement columns.")
            for row in reader:
                total += 1
                if total > 200000:
                    return unavailable("Request CSV exceeds the 200,000-row inspection limit.")
                if row.get("completed") not in ("True", "False", "true", "false") or None in row:
                    return unavailable("Request CSV contains an invalid measurement row.")
                name = row["replica"] or None
                group = groups.setdefault(name, bucket(name))
                group["requests"] += 1
                completed = row["completed"].lower() == "true"
                group["completed"] += int(completed)
                group["errors"] += int(not completed)
                group["dropped"] += int(row["error"] == "client_inflight_limit")
                if completed:
                    group["truncated"] += int(row["finish_reason"] == "length")
                    latency, ttft = number(row["latency_seconds"]), number(row["ttft_seconds"])
                    if latency is not None:
                        group["latencies"].append(latency)
                    if ttft is not None:
                        group["ttfts"].append(ttft)
        rows = []
        for group in sorted(groups.values(), key=lambda value: (value["replica"] is None, value["replica"] or "")):
            latency, ttft = group.pop("latencies"), group.pop("ttfts")
            group.update(request_share_percent=100 * group["requests"] / total if total else None,
                         latency_samples=len(latency), ttft_samples=len(ttft),
                         latency_p50_seconds=percentile(latency, .5, group["completed"]),
                         latency_p95_seconds=percentile(latency, .95, group["completed"]),
                         ttft_p95_seconds=percentile(ttft, .95, group["completed"]))
            rows.append(group)
        return {"status": "recorded", "rows": total, "replicas": rows,
                "message": "Client-observed responses. Percentiles use completed requests and require complete metric coverage. Requests without replica metadata are listed separately; this is not GPU attribution."}
    except (OSError, ValueError, TypeError, KeyError, csv.Error):
        return unavailable("Request CSV could not be read as valid measurements.")


def input_context(directory, mode, digest):
    path = directory / ("trace.jsonl" if mode == "fixed_arrival" else "inputs.jsonl")
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        return None
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Saved input hash does not match the manifest")
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if not rows or any(not isinstance(row.get("body", {}).get("model"), str) or not row["body"]["model"] for row in rows):
        raise ValueError("Saved inputs have no valid models")
    return {"models": sorted({row["body"]["model"] for row in rows}), "input_rows": len(rows),
            "arrival_seconds": rows[-1].get("offset_seconds") if rows and mode == "fixed_arrival" else None}


def gpu_context(directory, manifest, summary=None):
    requested = manifest.get("gpu_telemetry_requested")
    result = {"status": "disabled" if requested is False else "unknown", "scope": "benchmark host", "devices": [],
              "energy": energy.estimate([], manifest, summary or {})}
    if requested is not True:
        return result
    path = directory / "telemetry.jsonl"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        result["status"] = "unavailable"
        return result
    devices, power_records = {}, []
    with path.open() as file:
        for line in file:
            # Metrics lines may be large. Decode only the small NVIDIA records.
            if '"nvidia-smi"' not in line[:256]:
                continue
            try:
                record = json.loads(line)
                if record.get("source") != "nvidia-smi":
                    continue
                power_records.append(record)
                if not record.get("csv"):
                    continue
                fields = record.get("fields", ["timestamp", "uuid", "name", "driver_version", "memory.total", "memory.used", "utilization.gpu"])
                for row in csv.DictReader(io.StringIO(record["csv"]), fieldnames=fields):
                    identity = row["uuid"].strip()
                    total, used, utilization = (float(row[key].strip()) for key in ("memory.total", "memory.used", "utilization.gpu"))
                    if not identity or not all(finite(value) for value in (total, used, utilization)) or not 0 <= used <= total or not 0 <= utilization <= 100:
                        continue
                    device = devices.setdefault(identity, {"uuid": identity, "name": row["name"].strip(),
                                                          "driver_version": row["driver_version"].strip(),
                                                          "memory_total_mib": total, "memory": [], "utilization": []})
                    device["memory"].append(used)
                    device["utilization"].append(utilization)
            except (ValueError, TypeError, KeyError, AttributeError, csv.Error):
                continue
    for device in sorted(devices.values(), key=lambda item: item["uuid"]):
        memory, utilization = device.pop("memory"), device.pop("utilization")
        result["devices"].append({**device, "samples": len(memory), "memory_used_min_mib": min(memory),
                                  "memory_used_max_mib": max(memory), "utilization_mean_percent": statistics.mean(utilization),
                                  "utilization_max_percent": max(utilization)})
    result["status"] = "recorded" if result["devices"] else "unavailable"
    result["energy"] = energy.estimate(power_records, manifest, summary or {})
    return result


def routing_context(manifest, environment):
    before = (manifest.get("router_state_before") or {}).get("configuration")
    after = (manifest.get("router_state_after") or {}).get("configuration")
    if before is not None:
        if after is None or before != after:
            raise ValueError("Router configuration changed or its final snapshot is missing")
        value = copy.deepcopy(before)
        value.pop("policy", None)
        if value.get("discovery"):
            raise ValueError("Dynamic endpoint discovery needs a separately controlled comparison")
        replicas = value.get("replicas")
        if not isinstance(replicas, list) or not replicas:
            raise ValueError("Replica configuration is missing")
        if environment == "emulator":
            # Fresh emulator repeats use different temporary loopback ports.
            for replica in replicas:
                replica.pop("url", None)
        return {"kind": "router", "configuration": value}, len(replicas)
    if manifest.get("url"):
        return {"kind": "direct", "url": manifest["url"]}, 1
    return None, None


def enrich(summary, directory, manifest):
    missing, warnings = [], []
    mode = manifest.get("mode", summary.get("mode"))
    environment = summary.get("environment", manifest.get("environment"))
    settings = (manifest.get("dashboard_experiment") or {}).get("settings", summary.get("benchmark_settings", {}))
    duration = manifest.get("requested_duration_seconds", settings.get("duration"))
    workers = manifest.get("concurrency", settings.get("concurrency")) if mode == "closed_loop" else None
    digest = manifest.get("trace_sha256", summary.get("trace_sha256"))
    source = manifest.get("source_sha256", summary.get("source_sha256"))
    context = {"mode": mode, "workers": workers, "duration_seconds": duration if mode == "closed_loop" else None,
               "warmup_per_model": manifest.get("warmup_per_model"), "stream": manifest.get("stream"),
               "client_max_inflight": manifest.get("client_max_inflight"), "trace_sha256": digest,
               "source_sha256": source, "models": [], "input_rows": None, "arrival_seconds": None,
               "endpoint_count": None, "target_kind": None, "routing": None, "model_revision": manifest.get("model_revision"),
               "image_digest": manifest.get("image_digest"), "comparison_key": None,
               "exploratory_comparison_key": None, "exploratory_missing": [],
               "missing": missing, "warnings": warnings}
    invalid = bool(summary.get("interrupted") or summary.get("valid_for_comparison") is False or
                   manifest.get("interrupted") or manifest.get("policy_label_mismatch") or summary.get("policy") == "mixed")
    if invalid:
        warnings.append("Interrupted, invalid or mixed-policy run; excluded from policy comparisons.")
    for key in ("environment", "mode", "policy"):
        if summary.get(key) is not None and manifest.get(key) is not None and summary[key] != manifest[key]:
            invalid = True
            warnings.append("Summary and manifest disagree about " + key + ".")
    if not manifest:
        missing.append("run manifest")
    for label, value in (("workload mode", mode), ("input hash", digest), ("source hash", source),
                         ("warm-up count", context["warmup_per_model"]), ("stream setting", context["stream"]),
                         ("client in-flight limit", context["client_max_inflight"]), ("request timeout", manifest.get("timeout_seconds"))):
        if not known(value):
            missing.append(label)
    if mode == "closed_loop":
        if not finite(workers) or workers < 1:
            missing.append("workers")
        if not finite(duration) or duration <= 0:
            missing.append("requested duration")
    elif mode != "fixed_arrival":
        missing.append("supported workload mode")
    if environment not in ("hardware", "emulator"):
        missing.append("environment label")
    if environment == "hardware":
        for key in ("model_revision", "image_digest"):
            if not known(context[key]):
                missing.append(key.replace("_", " "))
        if any(label in missing for label in ("model revision", "image digest")):
            warnings.append("Hardware provenance is incomplete. Deployed revisions or images may differ even when the recorded workload matches.")
        else:
            warnings.append("Hardware provenance is recorded, not independently attested. Unrecorded serving flags and GPU sharing must also remain fixed.")
    try:
        inputs = input_context(directory, mode, digest)
        if inputs:
            context.update(inputs)
        else:
            missing.append("saved input snapshot")
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError):
        missing.append("verified input snapshot")
        warnings.append("Saved inputs are missing, invalid or do not match the recorded hash.")
    try:
        routing, count = routing_context(manifest, environment)
        if routing and routing["kind"] == "direct" and summary.get("policy") != "direct":
            routing, count = None, None
        context.update(endpoint_count=count, target_kind=routing["kind"] if routing else None, routing=routing)
        if routing is None:
            missing.append("endpoint setup")
    except (ValueError, TypeError, AttributeError):
        routing = None
        invalid = True
        warnings.append("Endpoint configuration was not stable or fully recorded.")
    for key, value in (("trace_sha256", digest), ("source_sha256", source)):
        if summary.get(key) and manifest.get(key) and summary[key] != manifest[key]:
            invalid = True
            warnings.append("Summary and manifest hashes disagree.")
    try:
        context["gpu"] = gpu_context(directory, manifest, summary)
    except (OSError, ValueError, UnicodeError):
        context["gpu"] = {"status": "unavailable", "scope": "benchmark host", "devices": [],
                          "energy": energy.estimate([], manifest, summary)}
    gpu = context["gpu"]
    if manifest.get("gpu_telemetry_requested") is True and gpu["status"] != "recorded":
        missing.append("requested GPU samples")
    hardware = [{key: device[key] for key in ("uuid", "name", "driver_version", "memory_total_mib")} for device in gpu["devices"]]
    job_path = directory.parent / "job.json"
    if job_path.is_file():
        try:
            if read_json(job_path).get("status") != "completed":
                invalid = True
                warnings.append("Dashboard job did not finish its final validation.")
        except (OSError, ValueError):
            invalid = True
            warnings.append("Dashboard job status is unreadable.")
    provenance = {"model revision", "image digest"}
    exploratory = environment == "hardware" and bool(missing) and set(missing) <= provenance
    if not invalid and (not missing or exploratory):
        signature = {"environment": environment, "scenario": summary.get("scenario"),
            "mode": mode, "workers": workers, "duration": duration if mode == "closed_loop" else None,
            "input_hash": digest, "source_hash": source, "warmup": context["warmup_per_model"],
            "stream": context["stream"], "client_limit": context["client_max_inflight"],
            "timeout": manifest.get("timeout_seconds"), "routing": routing,
            "model_revision": context["model_revision"] if environment == "hardware" else None,
            "image_digest": context["image_digest"] if environment == "hardware" else None,
            "gpu_sampling": manifest.get("gpu_telemetry_requested"), "observed_host_gpus": hardware,
            "gpu_energy_measurement": manifest.get("gpu_energy_measurement"),
            "benchmark_host": {key: manifest.get(key) for key in ("platform", "python", "cpu_count")}}
        if exploratory:
            # Only missing provenance is relaxed. Known, conflicting values still separate groups.
            for key in ("model_revision", "image_digest"):
                if not known(signature[key]):
                    signature[key] = None
            context["exploratory_comparison_key"] = fingerprint({"kind": "exploratory", "settings": signature})
            context["exploratory_missing"] = list(missing)
        else:
            context["comparison_key"] = fingerprint(signature)
    context["status"] = "excluded" if invalid else "matched_settings" if context["comparison_key"] else "individual_only"
    return context


class RunCatalog:
    def __init__(self, root):
        self.root = Path(root)
        self.cache = {}
        self.detail_cache = {}
        self.lock = threading.Lock()

    def _paths(self):
        return sorted(self.root.rglob("summary.json"), key=lambda path: path.parent.name)[-200:]

    def details(self, run_id):
        matches = [run for run in self.list() if run["run_id"] == run_id]
        if len(matches) != 1:
            return None
        run = matches[0]
        names = tuple(replica["name"] for replica in
                      (run["context"].get("routing") or {}).get("configuration", {}).get("replicas", []))
        with self.lock:
            paths = [path.parent / "requests.csv" for path in self._paths() if path.parent.name == run_id]
            if len(paths) != 1:
                return None
            path = paths[0]
            try:
                info = path.lstat()
                stamp = (info.st_mtime_ns, info.st_size, info.st_mode, names)
            except OSError:
                stamp = None
            cached = self.detail_cache.get(path)
            if cached is None or cached[0] != stamp:
                value = replica_breakdown(path, names)
                self.detail_cache[path] = (stamp, value)
            value = copy.deepcopy(self.detail_cache[path][1])
        if value["status"] == "recorded" and finite(run.get("requests")) and value["rows"] != run["requests"]:
            value.update(status="inconsistent", message="Request CSV row count differs from the saved summary. Inspect the raw artifacts before comparing this breakdown.")
        return {"run_id": run_id, "replica_breakdown": value}

    def list(self):
        with self.lock:
            runs = []
            paths = self._paths()
            for path in paths:
                try:
                    files = [path, path.parent / "manifest.json", path.parent / "inputs.jsonl",
                             path.parent / "trace.jsonl", path.parent / "telemetry.jsonl", path.parent.parent / "job.json"]
                    stamp = []
                    for file in files:
                        try:
                            info = file.stat()
                            stamp.append((info.st_mtime_ns, info.st_size))
                        except OSError:
                            stamp.append(None)
                    previous = self.cache.get(path)
                    if previous and previous[0] == stamp:
                        value = previous[1]
                    else:
                        value = read_json(path)
                        try:
                            manifest = read_json(path.parent / "manifest.json")
                        except (OSError, ValueError):
                            manifest = {}
                        value["run_id"] = path.parent.name
                        value["context"] = enrich(value, path.parent, manifest)
                        self.cache[path] = (stamp, value)
                    runs.append(copy.deepcopy(value))
                except (OSError, ValueError, TypeError, KeyError, AttributeError):
                    continue
            self.cache = {path: value for path, value in self.cache.items() if path in paths}
            active = {path.parent / "requests.csv" for path in paths}
            self.detail_cache = {path: value for path, value in self.detail_cache.items() if path in active}
            return runs


def comparisons(runs, mode="recorded"):
    if mode not in ("recorded", "exploratory"):
        raise ValueError("Comparison mode must be recorded or exploratory")
    field = "comparison_key" if mode == "recorded" else "exploratory_comparison_key"
    groups = {}
    for run in runs:
        context = run.get("context", {})
        key = context.get(field)
        if not key or run.get("policy") not in ("round_robin", "capacity", "kubeai", "direct"):
            continue
        group = groups.setdefault(key, {"id": key, "kind": mode, "environment": run.get("environment"),
                                       "scenario": run.get("scenario", run.get("mode")), "context": context,
                                       "missing_provenance": context.get("exploratory_missing", []) if mode == "exploratory" else [],
                                       "runs": []})
        group["runs"].append(run)
    output = []
    for group in groups.values():
        policies = []
        for policy in ("round_robin", "capacity", "kubeai", "direct"):
            selected = [run for run in group["runs"] if run["policy"] == policy]
            if not selected:
                continue
            values = [run.get("latency_p95_seconds") for run in selected]
            complete_p95 = all(finite(value) for value in values)
            def mean(key):
                items = [run[key] for run in selected if finite(run.get(key))]
                return statistics.mean(items) if len(items) == len(selected) else None
            estimates = [(run.get("context", {}).get("gpu") or {}).get("energy", {}) for run in selected]
            def energy_mean(key):
                values = [item.get(key) for item in estimates if item.get("status") == "estimated"]
                return statistics.mean(values) if len(values) == len(selected) and all(finite(value) for value in values) else None
            totals = {}
            for key in ("requests", "completed", "errors", "dropped", "truncated"):
                items = [run.get(key) for run in selected]
                totals[key] = sum(items) if all(finite(item) for item in items) else None
            policies.append({"policy": policy, "runs": len(selected), "latency_p50_mean_seconds": mean("latency_p50_seconds"),
                             "latency_p95_mean_seconds": statistics.mean(values) if complete_p95 else None,
                             "latency_p95_min_seconds": min(values) if complete_p95 else None,
                             "latency_p95_max_seconds": max(values) if complete_p95 else None, "ttft_p95_mean_seconds": mean("ttft_p95_seconds"),
                             "completed_rps_mean": mean("completed_rps"),
                             "energy_runs": sum(item.get("status") == "estimated" for item in estimates),
                             "energy_joules_mean": energy_mean("energy_joules"),
                             "power_mean_watts_mean": energy_mean("power_mean_watts"),
                             "joules_per_completed_request_mean": energy_mean("joules_per_completed_request"),
                             "joules_per_1000_output_tokens_mean": energy_mean("joules_per_1000_output_tokens"),
                             **totals})
        output.append({key: value for key, value in group.items() if key != "runs"} |
                      {"policies": policies, "run_ids": [run["run_id"] for run in group["runs"]]})
    return output
