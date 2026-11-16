import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from kubeai import benchmark, emulator, workload
from kubeai.config import Config
from kubeai.demo import serve
from kubeai.router import Router, make_server
from kubeai.transport import get_json, open_url


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.backend = emulator.Emulator("model", capacity=1, token_seconds=.01)
        self.backend_url = serve(self.stack, emulator.make_server(self.backend, port=0))
        self.config = Config.parse({"replicas": [{"name": "a", "url": self.backend_url, "models": ["model"], "token_seconds": .01}],
                                    "policy": "kubeai", "max_inflight": 1, "poll_interval": .1, "stale_after": 2})
        self.router = Router(self.config, results_dir=self.directory)
        self.router.start()
        self.stack.callback(self.router.stop)
        self.url = serve(self.stack, make_server(self.router, port=0))
        self.entry = {"id": "1", "tenant": "a", "body": {"model": "model", "messages": [{"role": "user", "content": "test"}], "max_tokens": 20}}

    def test_streaming_ttft_precedes_completion_and_records_usage(self):
        row = benchmark.request(self.url, self.entry, time.time())
        self.assertTrue(row["completed"], row)
        self.assertEqual(row["replica"], "a")
        self.assertEqual(row["output_tokens"], 20)
        self.assertLess(row["ttft_seconds"] + .1, row["latency_seconds"])
        # The client can observe [DONE] just before the proxy's finally block.
        deadline = time.monotonic() + 1
        while self.router.policy.active and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertEqual(self.router.policy.active, 0)

    def test_json_request_and_model_list(self):
        row = benchmark.request(self.url, self.entry, time.time(), stream=False)
        self.assertTrue(row["completed"])
        self.assertEqual(row["ttft_seconds"], "")
        self.assertEqual(get_json(self.url + "/v1/models")["data"][0]["id"], "model")

    def test_unknown_model_returns_503_without_dispatch(self):
        self.entry["body"]["model"] = "other"
        row = benchmark.request(self.url, self.entry, time.time())
        self.assertEqual(row["http_code"], 503)
        self.assertEqual(self.backend.completed, 0)

    def test_unhealthy_endpoint_stops_receiving_requests(self):
        self.router.stop()
        for replica in self.router.policy.replicas.values():
            replica.healthy_at = 0
        row = benchmark.request(self.url, self.entry, time.time())
        self.assertEqual(row["http_code"], 503)
        with self.assertRaises(urllib.error.HTTPError) as raised:
            get_json(self.url + "/readyz")
        self.assertEqual(raised.exception.code, 503)

    def test_inflight_limit_returns_429(self):
        replica, _ = self.router.policy.reserve("model", 8)
        try:
            row = benchmark.request(self.url, self.entry, time.time())
            self.assertEqual(row["http_code"], 429)
        finally:
            self.router.policy.release(replica)

    def test_validation_and_authentication(self):
        with self.assertRaises(urllib.error.HTTPError) as raised:
            open_url(self.url + "/v1/chat/completions", b"[]")
        self.assertEqual(raised.exception.code, 400)
        with patch.dict(os.environ, {"KUBEAI_API_KEY": "test-key"}):
            self.assertEqual(benchmark.request(self.url, self.entry, time.time())["http_code"], 401)
            self.assertTrue(benchmark.request(self.url, self.entry, time.time(), api_key="test-key")["completed"])

    def test_dashboard_and_metrics(self):
        with open_url(self.url) as response:
            self.assertIn(b"Inference observatory", response.read())
            self.assertIn("script-src 'self'", response.headers["Content-Security-Policy"])
        self.assertEqual(get_json(self.url + "/api/runs"), {"runs": []})
        with open_url(self.url + "/metrics") as response:
            self.assertIn(b'kubeai_replica_healthy{replica="a"} 1', response.read())

    def test_trace_replay_retains_raw_artifacts_and_hash(self):
        path = Path(self.directory) / "input.jsonl"
        entries = workload.generate(3, rate=2, models=["model"])
        for entry in entries:
            entry["body"]["max_tokens"] = 2
        workload.save(path, entries)
        directory, summary = benchmark.run(self.url, output=self.directory, trace_path=path, warmup=0, environment="emulator")
        self.assertEqual(summary["completed"], 3)
        self.assertEqual((directory / "trace.jsonl").read_bytes(), path.read_bytes())
        manifest = json.loads((directory / "manifest.json").read_text())
        self.assertGreater(manifest["end_unix"], manifest["start_unix"])
        self.assertEqual(manifest["trace_sha256"], summary["trace_sha256"])
        self.assertEqual(manifest["policy"], "kubeai")
        self.assertEqual(manifest["router_state_before"]["configuration"]["replicas"][0]["name"], "a")
        self.assertTrue((directory / "requests.csv").exists())
        self.assertTrue((directory / "telemetry.jsonl").exists())

    def test_closed_loop_can_reuse_original_request_payload(self):
        path = Path(self.directory) / "request.json"
        body = {**self.entry["body"], "max_tokens": 1}
        path.write_text(json.dumps(body))
        directory, summary = benchmark.run(self.url, output=self.directory, duration=.08, concurrency=1, request_path=path, warmup=0)
        self.assertGreater(summary["completed"], 0)
        self.assertEqual(summary["errors"], 0)
        inputs = json.loads((directory / "inputs.jsonl").read_text())
        self.assertEqual(inputs["body"], body)

    def test_energy_window_excludes_warmup_and_matches_saved_viewer_results(self):
        # Real temporary CPU inference, with all physical NVIDIA access replaced.
        observations, client_completions = [], []
        real_request = benchmark.request

        def observed_request(*args, **kwargs):
            row = real_request(*args, **kwargs)
            if row["completed"]:
                client_completions.append(time.monotonic())
            return row

        def fake_power(monitor):
            observations.append(len(client_completions))
            return {"source": "nvidia-smi", "sample_monotonic_seconds": time.monotonic(),
                    "timestamp_unix": time.time(), "fields": benchmark.GPU_FIELDS.split(","),
                    "csv": "2026/11/15 00:00:00, GPU-fake, Test GPU, test-driver, 8192, 6142, 50, 100\n"}

        request_path = Path(self.directory) / "energy-request.json"
        request_path.write_text(json.dumps({**self.entry["body"], "max_tokens": 2}))
        with patch.object(benchmark.Telemetry, "sample_gpu", fake_power), \
             patch("kubeai.benchmark.request", side_effect=observed_request), \
             patch("kubeai.benchmark.shutil.which", return_value="/fake/nvidia-smi"), \
             patch("kubeai.benchmark.subprocess.run", side_effect=AssertionError("No physical GPU access")):
            directory, summary = benchmark.run(self.url, output=self.directory, duration=.08, concurrency=1,
                                               request_path=request_path, warmup=1, gpu=True, environment="hardware")
        manifest = json.loads((directory / "manifest.json").read_text())
        records = [json.loads(line) for line in (directory / "telemetry.jsonl").read_text().splitlines()]
        power = [row for row in records if row["source"] == "nvidia-smi"]
        self.assertGreaterEqual(observations[0], 1, "GPU window must start after warm-up")
        self.assertLess(power[0]["sample_monotonic_seconds"], manifest["start_monotonic_seconds"])
        self.assertGreater(power[-1]["sample_monotonic_seconds"], manifest["end_monotonic_seconds"])
        report = summary["gpu_energy"]
        self.assertEqual(report["status"], "estimated")
        self.assertAlmostEqual(report["energy_joules"], 100 * summary["elapsed_seconds"], places=5)
        self.assertAlmostEqual(report["joules_per_completed_request"], report["energy_joules"] / summary["completed"])
        saved = json.loads((directory / "summary.json").read_text())
        self.assertEqual(saved["gpu_energy"], report)
        viewed = next(run for run in self.router.runs() if run["run_id"] == directory.name)
        self.assertEqual(viewed["context"]["gpu"]["energy"], report)

    def test_telemetry_does_not_forward_router_key_to_backend(self):
        import io
        observed = []
        monitor = benchmark.Telemetry(self.directory, [self.url + "/metrics", self.backend_url + "/metrics"],
                                      api_key="router-secret", auth_origin=self.url)

        def capture(url, **kwargs):
            observed.append((url, kwargs["headers"]))
            if len(observed) == 2:
                monitor.stop_event.set()
            return io.BytesIO(b"vllm:num_requests_running 0\n")

        with patch("kubeai.benchmark.open_url", side_effect=capture):
            monitor.loop()
        self.assertIn("Authorization", observed[0][1])
        self.assertNotIn("Authorization", observed[1][1])

    def test_upstream_failure_releases_reservation(self):
        with patch("kubeai.router.upstream_connection", side_effect=OSError("offline")):
            row = benchmark.request(self.url, self.entry, time.time())
        self.assertEqual(row["http_code"], 502)
        # HTTPError can be raised as soon as response headers arrive, before the
        # handler finishes sending its error body and releases its reservation.
        deadline = time.monotonic() + 1
        while self.router.policy.active and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertEqual(self.router.policy.active, 0)
        self.assertGreater(next(iter(self.router.policy.replicas.values())).blocked_until, time.monotonic())


if __name__ == "__main__":
    unittest.main()
