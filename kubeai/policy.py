"""Atomic selection/reservation shared by all three policies."""

import threading
import time
from dataclasses import dataclass, field

from .config import Config, ReplicaConfig


@dataclass
class Replica:
    config: ReplicaConfig
    models: set = field(default_factory=set)
    healthy_at: float = 0.0
    metrics_at: float = 0.0
    metrics: dict = field(default_factory=dict)
    gpu_metrics: dict = field(default_factory=dict)
    gpu_metrics_at: float = 0.0
    inflight: int = 0
    blocked_until: float = 0.0
    last_error: str = "not yet probed"

    def healthy(self, now, ttl):
        return self.healthy_at > 0 and now - self.healthy_at <= ttl and now >= self.blocked_until


class Unavailable(Exception):
    pass


class Overloaded(Exception):
    pass


def score(replica, config, tokens, now):
    fresh = replica.metrics_at > 0 and now - replica.metrics_at <= config.stale_after
    metrics = dict(replica.metrics) if fresh else {}
    gpu_fresh = replica.gpu_metrics_at > 0 and now - replica.gpu_metrics_at <= config.stale_after
    if gpu_fresh:
        metrics.update(replica.gpu_metrics)
    # Local reservations overlap observed vLLM work: max avoids double counting.
    load = max(replica.inflight, metrics.get("waiting", 0) + metrics.get("running", 0)) / replica.config.capacity
    service = replica.config.prefill_seconds + tokens * replica.config.token_seconds
    memory = 0.0
    if metrics.get("memory_total", 0) > 0 and "memory_free" in metrics:
        memory = max(0.0, min(1.0, 1 - metrics["memory_free"] / metrics["memory_total"]))
    terms = {
        "queue": config.weights["queue"] * load * service,
        "service": config.weights["service"] * service,
        "kv": config.weights["kv"] * metrics.get("kv", 0),
        "memory": config.weights["memory"] * memory,
    }
    return sum(terms.values()), {"terms": terms, "metrics_fresh": fresh, "gpu_metrics_fresh": gpu_fresh, "load": load}


class Policy:
    def __init__(self, config: Config):
        self.config = config
        self.lock = threading.RLock()
        self.replicas = {r.name: Replica(r) for r in config.replicas}
        self.cursor = 0
        self.active = 0

    def reconcile(self, configs):
        with self.lock:
            old = self.replicas
            self.replicas = {}
            for config in configs:
                existing = old.get(config.name)
                if existing and existing.config == config:
                    self.replicas[config.name] = existing
                else:
                    self.replicas[config.name] = Replica(config)

    def reserve(self, model, tokens):
        now = time.monotonic()
        with self.lock:
            if self.active >= self.config.max_inflight:
                raise Overloaded("router in-flight limit reached")
            eligible = [r for r in self.replicas.values() if r.healthy(now, self.config.stale_after)
                        and model in r.models and model in r.config.models and tokens <= r.config.max_model_len]
            if not eligible:
                raise Unavailable("no healthy, compatible replica for this model and output budget")
            # Rotate ties so identical estimates do not starve later replicas.
            offset = self.cursor % len(eligible)
            eligible = eligible[offset:] + eligible[:offset]
            self.cursor += 1
            details = {}
            if self.config.policy == "round_robin":
                chosen = eligible[0]
            elif self.config.policy == "capacity":
                chosen = min(eligible, key=lambda r: (r.inflight + 1) / r.config.capacity)
            else:
                ranked = [(score(r, self.config, tokens, now), r) for r in eligible]
                (value, details), chosen = min(ranked, key=lambda entry: entry[0][0])
                details["score"] = value
            chosen.inflight += 1
            self.active += 1
            return chosen, {"policy": self.config.policy, "replica": chosen.config.name, **details}

    def release(self, replica, failed=False):
        with self.lock:
            replica.inflight -= 1
            self.active -= 1
            if failed:
                replica.blocked_until = time.monotonic() + self.config.circuit_seconds

    def snapshot(self):
        now = time.monotonic()
        with self.lock:
            return [{"name": r.config.name, "url": r.config.url, "models": sorted(r.models & set(r.config.models)),
                     "capacity": r.config.capacity, "inflight": r.inflight,
                     "healthy": r.healthy(now, self.config.stale_after),
                     "metrics_fresh": r.metrics_at > 0 and now - r.metrics_at <= self.config.stale_after,
                     "gpu_metrics_fresh": r.gpu_metrics_at > 0 and now - r.gpu_metrics_at <= self.config.stale_after,
                     "metrics": r.metrics, "gpu_metrics": r.gpu_metrics, "last_error": r.last_error}
                    for r in self.replicas.values()]
