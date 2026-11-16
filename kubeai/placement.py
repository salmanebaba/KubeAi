"""Conservative whole-GPU placement recommendations; no cluster writes."""

import math
import time


def number(value, name, integer=False, minimum=0):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    if integer and type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def recommend(inventory, request, policy="best_fit", now=None, stale_after=30):
    if policy not in ("first_fit", "best_fit"):
        raise ValueError("placement policy must be first_fit or best_fit")
    now = time.time() if now is None else now
    number(stale_after, "stale_after", minimum=.001)
    count = number(request.get("gpus", 1), "gpus", integer=True, minimum=1)
    memory = number(request["gpu_memory_mib"], "gpu_memory_mib", minimum=1)
    headroom = number(request.get("headroom_mib", 512), "headroom_mib")
    cpu = number(request.get("cpu_millis", 1000), "cpu_millis", integer=True, minimum=1)
    ram = number(request.get("ram_mib", 4096), "ram_mib", integer=True, minimum=1)
    if not isinstance(request.get("model"), str) or not request["model"]:
        raise ValueError("model must be nonempty")
    eligible, excluded, seen = [], {}, set()
    for node in inventory["nodes"]:
        name = node["name"]
        if name in seen:
            raise ValueError("duplicate node names")
        seen.add(name)
        reasons = []
        observed = number(node.get("observed_at_unix", 0), "observed_at_unix")
        if observed <= 0 or now - observed > stale_after or observed > now + 1:
            reasons.append("stale_or_invalid_telemetry")
        if node.get("ready") is not True or node.get("unschedulable", True) is not False or node.get("untolerated_taints", ["unknown"]):
            reasons.append("node_not_schedulable")
        if request["model"] not in node.get("compatible_models", []):
            reasons.append("incompatible_model")
        if number(node.get("free_cpu_millis", 0), "free_cpu_millis") < cpu or number(node.get("free_ram_mib", 0), "free_ram_mib") < ram:
            reasons.append("insufficient_cpu_or_ram")
        gpu_ids, available = set(), []
        for gpu in node.get("gpus", []):
            if gpu["id"] in gpu_ids:
                raise ValueError("duplicate GPU IDs on node")
            gpu_ids.add(gpu["id"])
            total = number(gpu["total_mib"], "total_mib", minimum=1)
            free = number(gpu["free_mib"], "free_mib")
            if free > total:
                raise ValueError("GPU free memory exceeds total memory")
            if not gpu.get("allocated", True):
                available.append(gpu)
        # Device plugins select devices. Require ALL unallocated GPUs on this node
        # to fit, rather than pretending node affinity binds specific GPU UUIDs.
        if len(available) < count or any(g["free_mib"] < memory + headroom for g in available):
            reasons.append("insufficient_gpu_capacity_or_memory")
        if reasons:
            excluded[name] = reasons
            continue
        residual = sum(g["free_mib"] - memory - headroom for g in available)
        warm = request["model"] in node.get("cached_models", [])
        eligible.append((name, (not warm, len(available) - count, residual, name)))
    if not eligible:
        return {"policy": policy, "node": None, "excluded": excluded, "reason": "no eligible node", "applied": False}
    chosen = eligible[0][0] if policy == "first_fit" else min(eligible, key=lambda n: n[1])[0]
    return {"policy": policy, "node": chosen, "model": request["model"], "gpus": count,
            "gpu_memory_mib": memory, "headroom_mib": headroom, "cpu_millis": cpu, "ram_mib": ram,
            "eligible": [n[0] for n in eligible], "excluded": excluded,
            "reason": "first eligible node" if policy == "first_fit" else "prefer cached model, then pack available GPU slots and memory",
            "applied": False, "observed_at_unix": now}


def pod_manifest(decision, image, name="kubeai-model", namespace="kubeai", model_revision=None):
    if not decision.get("node"):
        raise ValueError("cannot build a Pod without an eligible node")
    if "@sha256:" not in image or len(image.rsplit("@sha256:", 1)[1]) != 64:
        raise ValueError("use a verified image@sha256:<64 hex digits> for reproducibility")
    int(image.rsplit("@sha256:", 1)[1], 16)
    args = ["--model", decision["model"], "--max-model-len", "2048", "--gpu-memory-utilization", "0.75",
            "--tensor-parallel-size", str(decision["gpus"])]
    if model_revision:
        args += ["--revision", model_revision]
    return {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": name, "namespace": namespace,
            "labels": {"app": "kubeai-vllm"}}, "spec": {"automountServiceAccountToken": False,
            "affinity": {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {"nodeSelectorTerms": [
                {"matchFields": [{"key": "metadata.name", "operator": "In", "values": [decision["node"]]}]}]}}},
            "containers": [{"name": "vllm", "image": image, "args": args, "ports": [{"name": "http", "containerPort": 8000}],
                            "resources": {"requests": {"cpu": f"{decision['cpu_millis']}m", "memory": f"{decision['ram_mib']}Mi", "nvidia.com/gpu": decision["gpus"]},
                                          "limits": {"cpu": f"{decision['cpu_millis']}m", "memory": f"{decision['ram_mib']}Mi", "nvidia.com/gpu": decision["gpus"]}},
                            "volumeMounts": [{"name": "shm", "mountPath": "/dev/shm"}],
                            "startupProbe": {"httpGet": {"path": "/v1/models", "port": "http"}, "failureThreshold": 120, "periodSeconds": 10},
                            "readinessProbe": {"httpGet": {"path": "/v1/models", "port": "http"}, "periodSeconds": 5}}],
            "volumes": [{"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "1Gi"}}]}}
