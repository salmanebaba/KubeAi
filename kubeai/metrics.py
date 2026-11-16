"""Prometheus exposition parsing and small bounded-cardinality router counters."""

import math
import re
import threading
from collections import Counter

SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^\n]*\})?\s+([^\s]+)(?:\s+[^\s]+)?$')


def parse_metrics(text):
    """Sum engine-labelled gauges; ignore nonfinite and malformed samples."""
    values = {}
    for line in text.splitlines():
        match = SAMPLE.match(line)
        if not match:
            continue
        try:
            value = float(match[2])
        except ValueError:
            continue
        if math.isfinite(value):
            values.setdefault(match[1], []).append(value)
    return values


def signals(text):
    values = parse_metrics(text)
    result = {}
    for signal, aliases, aggregate in (
        ("waiting", ["vllm:num_requests_waiting"], sum),
        ("running", ["vllm:num_requests_running"], sum),
        ("kv", ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"], max),
        ("memory_free", ["kubeai_gpu_memory_free_mib"], min),
        ("memory_total", ["kubeai_gpu_memory_total_mib"], min),
    ):
        for alias in aliases:
            if alias in values:
                result[signal] = max(0.0, aggregate(values[alias]))
                break
    if "kv" in result:
        result["kv"] = min(1.0, result["kv"])
    return result


def label(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class Counters:
    buckets = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, float("inf"))

    def __init__(self):
        self.lock = threading.Lock()
        self.requests = Counter()
        self.latency = [0] * len(self.buckets)
        self.total = 0.0

    def record(self, policy, replica, status, duration):
        with self.lock:
            self.requests[policy, replica, status] += 1
            self.total += duration
            for i, bound in enumerate(self.buckets):
                self.latency[i] += duration <= bound

    def render(self):
        with self.lock:
            lines = ["# TYPE kubeai_requests_total counter"]
            for (policy, replica, status), count in sorted(self.requests.items()):
                lines.append(f'kubeai_requests_total{{policy="{label(policy)}",replica="{label(replica)}",status="{status}"}} {count}')
            lines.append("# TYPE kubeai_request_duration_seconds histogram")
            for bound, count in zip(self.buckets, self.latency):
                lines.append(f'kubeai_request_duration_seconds_bucket{{le="{bound if math.isfinite(bound) else "+Inf"}"}} {count}')
            lines.extend([f"kubeai_request_duration_seconds_sum {self.total}", f"kubeai_request_duration_seconds_count {self.latency[-1]}"])
            return "\n".join(lines) + "\n"
