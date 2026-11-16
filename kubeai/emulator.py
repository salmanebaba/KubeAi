"""Synthetic OpenAI-compatible replica; sleeps and metrics are NOT GPU measurements."""

import json
import threading
import time
import uuid

from .transport import Handler, Server, validate_chat


class Emulator:
    buckets = (.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, float("inf"))

    def __init__(self, model="Qwen/Qwen3-0.6B", capacity=1, token_seconds=0.003, prefill_seconds=0.01):
        if capacity < 1 or token_seconds <= 0 or prefill_seconds < 0:
            raise ValueError("invalid emulator capacity/timing")
        self.model, self.capacity = model, capacity
        self.token_seconds, self.prefill_seconds = token_seconds, prefill_seconds
        self.slots = threading.BoundedSemaphore(capacity)
        self.lock = threading.Lock()
        self.waiting = self.running = self.completed = 0
        self.queue_sum = self.ttft_sum = 0.0
        self.queue_buckets = [0] * len(self.buckets)
        self.ttft_buckets = [0] * len(self.buckets)

    def metrics(self):
        with self.lock:
            lines = [
                "# Synthetic emulator metrics; no physical GPU is used.",
                f"vllm:num_requests_waiting {self.waiting}", f"vllm:num_requests_running {self.running}",
                f"vllm:kv_cache_usage_perc {min(1, self.running / (self.capacity * 2))}",
                "kubeai_gpu_memory_total_mib 8192", f"kubeai_gpu_memory_free_mib {6144 - self.running * 128}"]
            for name, counts, total in (("vllm:request_queue_time_seconds", self.queue_buckets, self.queue_sum),
                                        ("vllm:time_to_first_token_seconds", self.ttft_buckets, self.ttft_sum)):
                lines.append(f"# TYPE {name} histogram")
                for bound, count in zip(self.buckets, counts):
                    le = "+Inf" if bound == float("inf") else str(bound)
                    lines.append(f'{name}_bucket{{le="{le}"}} {count}')
                lines.extend([f"{name}_sum {total}", f"{name}_count {self.completed}"])
            return "\n".join(lines) + "\n"


def make_server(emulator, host="127.0.0.1", port=8001):
    class EmulatorHandler(Handler):
        def do_GET(self):
            if self.path in ("/healthz", "/readyz"):
                return self.reply(200, {"status": "ok", "synthetic": True})
            if self.path == "/v1/models":
                return self.reply(200, {"object": "list", "data": [{"id": emulator.model, "object": "model", "max_model_len": 2048, "created": 0, "owned_by": "emulator"}]})
            if self.path == "/metrics":
                return self.reply(200, emulator.metrics(), "text/plain; version=0.0.4")
            self.error(404, "unknown endpoint")

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self.error(404, "unknown endpoint")
            try:
                body = self.read_json()
                tokens = validate_chat(body)
                if tokens > 2048:
                    raise ValueError("emulator max_model_len is 2048")
            except (ValueError, UnicodeError) as error:
                return self.error(400, str(error))
            if body["model"] != emulator.model:
                return self.error(404, "model unavailable")
            queued_at = time.monotonic()
            with emulator.lock:
                emulator.waiting += 1
            acquired = emulator.slots.acquire(timeout=60)
            with emulator.lock:
                emulator.waiting -= 1
                emulator.running += int(acquired)
            if not acquired:
                return self.error(503, "emulator queue timeout")
            queue_time = time.monotonic() - queued_at
            # Length-dependent prefill; deterministic, no random server-side noise.
            prefill = emulator.prefill_seconds + len(json.dumps(body["messages"])) * 0.00001
            identifier = "chatcmpl-" + uuid.uuid4().hex
            ttft = None
            try:
                time.sleep(prefill)
                prompt_tokens = max(1, len(json.dumps(body["messages"])) // 4)
                usage = {"prompt_tokens": prompt_tokens, "completion_tokens": tokens, "total_tokens": prompt_tokens + tokens}
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()

                    def event(choices, usage_value=None):
                        data = {"id": identifier, "object": "chat.completion.chunk", "model": emulator.model,
                                "created": int(time.time()), "choices": choices}
                        if usage_value is not None:
                            data["usage"] = usage_value
                        self.wfile.write(("data: " + json.dumps(data) + "\n\n").encode())
                        self.wfile.flush()

                    event([{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}])
                    for _ in range(tokens):
                        time.sleep(emulator.token_seconds)
                        if ttft is None:
                            ttft = time.monotonic() - queued_at
                        event([{"index": 0, "delta": {"content": " synthetic"}, "finish_reason": None}])
                    event([{"index": 0, "delta": {}, "finish_reason": "length"}])
                    if body.get("stream_options", {}).get("include_usage"):
                        event([], usage)
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                else:
                    time.sleep(tokens * emulator.token_seconds)
                    ttft = queue_time + prefill + emulator.token_seconds
                    self.reply(200, {"id": identifier, "object": "chat.completion", "created": int(time.time()), "model": emulator.model,
                                     "choices": [{"index": 0, "message": {"role": "assistant", "content": " synthetic" * tokens}, "finish_reason": "length"}], "usage": usage})
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with emulator.lock:
                    emulator.running -= 1
                    emulator.completed += 1
                    emulator.queue_sum += queue_time
                    observed_ttft = ttft if ttft is not None else time.monotonic() - queued_at
                    emulator.ttft_sum += observed_ttft
                    for i, bound in enumerate(emulator.buckets):
                        emulator.queue_buckets[i] += queue_time <= bound
                        emulator.ttft_buckets[i] += observed_ttft <= bound
                emulator.slots.release()

    return Server((host, port), EmulatorHandler)
