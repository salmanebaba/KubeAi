"""Operator-managed vLLM endpoints and versioned, atomic configuration writes."""

import hashlib
import ipaddress
import json
import os
import re
import stat
import tempfile
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

from .config import Config, ReplicaConfig
from .metrics import signals
from .transport import open_url, upstream_headers


class Conflict(ValueError):
    pass


class ReadOnly(ValueError):
    pass


def loopback(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def parse_replica(value, router_port=None):
    if not isinstance(value, dict):
        raise ValueError("replica must be an object")
    try:
        replica = ReplicaConfig.parse(value)
        url = urlsplit(replica.url)
        port = url.port or (443 if url.scheme == "https" else 80)
        if router_port == port and loopback(url.hostname):
            raise ValueError("Use a vLLM server URL, not this router's URL")
        if replica.api_key_env and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", replica.api_key_env):
            raise ValueError("Credential environment variable must be a valid variable name")
        return replica
    except (TypeError, AttributeError, KeyError) as error:
        raise ValueError("Invalid replica configuration") from error


def revision(raw):
    return hashlib.sha256(raw).hexdigest()


def document(config, raw=None, reason=""):
    return {"editable": not reason, "reason": reason,
            "revision": revision(raw) if raw is not None else None,
            "replicas": [asdict(replica) for replica in config.replicas]}


def read(path, current):
    if path is None:
        return document(current, reason="Start the router with a configuration file to save endpoints.")
    path = Path(path)
    if path.is_symlink():
        return document(current, reason="Endpoint editing requires a regular configuration file, not a symbolic link.")
    raw = path.read_bytes()
    saved = Config.parse(json.loads(raw))
    if saved.discovery:
        return document(saved, raw, "Kubernetes discovery manages these endpoints. Configure the Service instead.")
    return document(saved, raw)


def save(path, current, values, expected_revision, router_port=None):
    view = read(path, current)
    if not view["editable"]:
        raise ReadOnly(view["reason"])
    if not isinstance(expected_revision, str) or expected_revision != view["revision"]:
        raise Conflict("Configuration changed. Reload the saved endpoints before applying your draft.")
    if not isinstance(values, list) or not 1 <= len(values) <= 128:
        raise ValueError("Provide between 1 and 128 endpoints")
    replicas = [parse_replica(value, router_port) for value in values]
    path = Path(path)
    original = path.read_bytes()
    if revision(original) != expected_revision:
        raise Conflict("Configuration changed. Reload the saved endpoints before applying your draft.")
    data = json.loads(original)
    data["replicas"] = [asdict(replica) for replica in replicas]
    config = Config.parse(data)
    raw = (json.dumps(data, indent=2, allow_nan=False) + "\n").encode()
    temporary = None
    try:
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        with os.fdopen(descriptor, "wb") as file:
            os.fchmod(file.fileno(), stat.S_IMODE(path.stat().st_mode))
            file.write(raw)
            file.flush()
            os.fsync(file.fileno())
        if revision(path.read_bytes()) != expected_revision:
            raise Conflict("Configuration changed while saving. Reload the saved endpoints.")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return config, path.stat().st_mtime_ns, document(config, raw)


def test_connection(replica, timeout=5):
    """Probe models and optional metrics without saving or sending inference work."""
    try:
        headers = upstream_headers(replica)
        with open_url(replica.url.rstrip("/") + "/v1/models", headers=headers, timeout=timeout) as response:
            raw = response.read(256 * 1024 + 1)
        if len(raw) > 256 * 1024:
            raise ValueError("Model list is too large")
        payload = json.loads(raw)
        models = sorted({item["id"] for item in payload["data"]
                         if isinstance(item.get("id"), str) and item["id"]})
    except Exception as error:
        return {"ok": False, "models": [], "metrics_available": False,
                "message": "Could not verify /v1/models (" + type(error).__name__ + "). Check the URL and server credentials."}
    missing = sorted(set(replica.models) - set(models))
    metrics_available = False
    try:
        with open_url(replica.url.rstrip("/") + "/metrics", headers=headers, timeout=timeout) as response:
            metrics_available = bool(signals(response.read(2 * 1024 * 1024).decode()))
    except Exception:
        pass
    return {"ok": not missing, "models": models, "metrics_available": metrics_available,
            "message": ("Configured models are unavailable: " + ", ".join(missing)) if missing else
                       ("Models verified. Metrics available." if metrics_available else
                        "Models verified. Metrics missing or unavailable; routing can still use local reservations.")}
