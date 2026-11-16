"""Validated configuration. Endpoint URLs are operator-controlled, never client input."""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

POLICIES = ("round_robin", "capacity", "kubeai")


def positive(value, name, allow_zero=False):
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value


@dataclass(frozen=True)
class ReplicaConfig:
    name: str
    url: str
    models: tuple[str, ...]
    capacity: int = 1
    token_seconds: float = 0.01
    prefill_seconds: float = 0.02
    max_model_len: int = 2048
    api_key_env: str = ""
    gpu_metrics_url: str = ""

    @classmethod
    def parse(cls, data):
        if not isinstance(data, dict):
            raise ValueError("replica must be an object")
        item = dict(data)
        if not isinstance(item.get("models", ()), (list, tuple)):
            raise ValueError("models must be a list of model names")
        item["models"] = tuple(item.get("models", ()))
        obj = cls(**item)
        if not isinstance(obj.url, str):
            raise ValueError("replica URL must be a string")
        if not isinstance(obj.api_key_env, str) or not isinstance(obj.gpu_metrics_url, str):
            raise ValueError("credential variable and GPU metrics URL must be strings")
        url = urlsplit(obj.url)
        if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password:
            raise ValueError("replica URL must be http(s), without credentials")
        if url.path not in ("", "/") or url.query or url.fragment:
            raise ValueError("replica URL must be an origin, without a path/query")
        if url.port is not None and not 1 <= url.port <= 65535:
            raise ValueError("replica URL needs a valid port")
        if obj.gpu_metrics_url:
            gpu_url = urlsplit(obj.gpu_metrics_url)
            if gpu_url.scheme not in ("http", "https") or not gpu_url.hostname or gpu_url.username or gpu_url.password:
                raise ValueError("gpu_metrics_url must be an http(s) URL without credentials")
        if not isinstance(obj.name, str) or not obj.name.strip() or len(obj.name) > 128 or not obj.models or any(not isinstance(m, str) or not m.strip() for m in obj.models):
            raise ValueError("replicas need a name and nonempty model names")
        for key in ("capacity", "max_model_len"):
            value = getattr(obj, key)
            positive(value, key)
            if type(value) is not int:
                raise ValueError(f"{key} must be an integer")
        positive(obj.token_seconds, "token_seconds")
        positive(obj.prefill_seconds, "prefill_seconds", allow_zero=True)
        return obj


@dataclass(frozen=True)
class Config:
    replicas: tuple[ReplicaConfig, ...]
    policy: str = "kubeai"
    poll_interval: float = 1.0
    stale_after: float = 5.0
    upstream_timeout: float = 60.0
    probe_timeout: float = 1.0
    max_inflight: int = 128
    circuit_seconds: float = 3.0
    weights: dict = field(default_factory=lambda: {"queue": 1.0, "service": 1.0, "kv": 0.1, "memory": 0.05})
    discovery: dict | None = None

    @classmethod
    def parse(cls, data):
        item = dict(data)
        item["replicas"] = tuple(ReplicaConfig.parse(r) for r in item.get("replicas", []))
        obj = cls(**item)
        if obj.policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        names = [r.name for r in obj.replicas]
        if len(names) != len(set(names)) or (not names and not obj.discovery):
            raise ValueError("need unique replica names, or Kubernetes discovery")
        for key in ("poll_interval", "stale_after", "upstream_timeout", "probe_timeout", "max_inflight", "circuit_seconds"):
            positive(getattr(obj, key), key)
        if type(obj.max_inflight) is not int or obj.stale_after <= obj.poll_interval:
            raise ValueError("max_inflight must be an integer; stale_after must exceed poll_interval")
        if set(obj.weights) != {"queue", "service", "kv", "memory"}:
            raise ValueError("weights must contain queue, service, kv, memory")
        for key, value in obj.weights.items():
            positive(value, key, allow_zero=True)
        if obj.discovery:
            if not obj.discovery.get("namespace") or not obj.discovery.get("service"):
                raise ValueError("discovery requires namespace and service")
            # Validate discovery defaults exactly like static endpoints.
            ReplicaConfig.parse({"name": "discovery", "url": "http://localhost", **obj.discovery.get("replica", {})})
        return obj

    @classmethod
    def load(cls, path):
        return cls.parse(json.loads(Path(path).read_text()))
