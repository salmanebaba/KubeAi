"""Seeded, reusable, fixed-arrival workload traces."""

import hashlib
import json
import math
import random
from pathlib import Path

from .transport import validate_chat

PROMPTS = [
    "Explain why GPUs help with inference in two sentences.",
    "Compare request routing with model placement. Give a concrete example.",
    "Summarize the tradeoffs between batching, queueing and response latency. " * 8,
    "Describe a reproducible experiment for inference serving. " * 24,
]


def generate(count=100, rate=20.0, seed=42, scenario="steady", models=None):
    if count < 1 or not math.isfinite(rate) or rate <= 0:
        raise ValueError("count and rate must be positive")
    if scenario not in ("steady", "burst", "mixed"):
        raise ValueError("scenario must be steady, burst or mixed")
    models = models or ["Qwen/Qwen3-0.6B"]
    rng, offset, rows = random.Random(seed), 0.0, []
    for index in range(count):
        if index:
            multiplier = 4.0 if scenario == "burst" and count // 3 <= index < 2 * count // 3 else 1.0
            offset += rng.expovariate(rate * multiplier)
        body = {"model": models[index % len(models)], "messages": [{"role": "user", "content": rng.choice(PROMPTS)}],
                "max_tokens": rng.choice([8, 16, 32, 64]), "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}}
        rows.append({"id": str(index), "offset_seconds": round(offset, 6), "tenant": f"tenant-{index % 3}", "body": body})
    return rows


def save(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def load(path):
    raw = Path(path).read_bytes()
    rows, previous, ids = [], -1.0, set()
    for line in raw.decode().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        offset = float(row["offset_seconds"])
        if not math.isfinite(offset) or offset < 0 or offset < previous:
            raise ValueError("trace arrival offsets must be finite, nonnegative, and sorted")
        if not isinstance(row.get("id"), str) or row["id"] in ids:
            raise ValueError("trace IDs must be unique strings")
        validate_chat(row["body"])
        if not isinstance(row.get("tenant", "default"), str):
            raise ValueError("tenant must be a string")
        row["offset_seconds"] = offset
        ids.add(row["id"])
        previous = offset
        rows.append(row)
    if not rows:
        raise ValueError("trace is empty")
    return rows, hashlib.sha256(raw).hexdigest()
