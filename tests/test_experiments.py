"""Dashboard benchmark jobs with real temporary HTTP emulators, never GPUs."""

import io
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from kubeai import benchmark, emulator, experiments, workload
from kubeai.config import Config
from kubeai.demo import serve
from kubeai.router import Router, make_server


class BenchmarkJobTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.backend = emulator.Emulator("model", capacity=4, token_seconds=.001)
        backend_url = serve(self.stack, emulator.make_server(self.backend, port=0))
        self.replica = {"name": "a", "url": backend_url, "models": ["model"], "capacity": 4}
        self.data = {"replicas": [self.replica], "policy": "round_robin", "poll_interval": .05,
                     "stale_after": 2, "max_inflight": 16}
        self.path = self.directory / "router.json"
        self.path.write_text(json.dumps(self.data))
        self.router = Router(Config.load(self.path), self.path, self.directory)
        self.router.start()
        self.stack.callback(self.router.stop)
        self.url = serve(self.stack, make_server(self.router, port=0))
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.body = {"mode": "closed_loop", "policy": "capacity", "model": "model", "concurrency": 2,
                     "duration": .12, "seed": 42, "warmup": 0, "environment": "emulator", "gpu": False}
        self.trace_path = self.directory / "workloads" / "sample.jsonl"
        entries = workload.generate(3, rate=20, seed=42, models=["model"])
        for entry in entries:
            entry["body"]["max_tokens"] = 8
        workload.save(self.trace_path, entries)

    def api(self, path="/api/benchmark", method="GET", body=None, headers=None):
        request = urllib.request.Request(self.url + path, method=method,
                                         headers={"Content-Type": "application/json", **(headers or {})},
                                         data=json.dumps(body).encode() if body is not None else None)
        with self.opener.open(request, timeout=8) as response:
            return response.status, json.load(response)

    def wait(self):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            _, result = self.api()
            if not result["active"]:
                return result["job"]
            time.sleep(.01)
        self.fail("benchmark did not finish")

    def trace_body(self):
        _, options = self.api("/api/benchmark/options")
        trace = options["traces"][0]
        return {"mode": "fixed_arrival", "policy": "kubeai", "trace_id": trace["id"],
                "trace_sha256": trace["sha256"], "max_inflight": 8, "warmup": 0, "environment": "emulator"}

    def test_options_list_verified_models_and_saved_trace_metadata(self):
        _, options = self.api("/api/benchmark/options")
        self.assertTrue(options["can_run"])
        self.assertEqual(options["models"], ["model"])
        self.assertEqual(options["traces"][0]["requests"], 3)
        self.assertEqual(options["traces"][0]["id"], "sample.jsonl")
        self.assertEqual(len(options["traces"][0]["sha256"]), 64)
        self.assertFalse(self.router.benchmarks.root.exists())
        self.assertEqual(self.backend.completed, 0)

    def test_closed_loop_job_uses_shared_engine_and_persists_results(self):
        original = self.path.read_bytes()
        status, _ = self.api(method="POST", body=self.body)
        self.assertEqual(status, 202)
        job = self.wait()
        self.assertEqual(job["status"], "completed", job)
        self.assertGreater(job["summary"]["completed"], 0)
        self.assertEqual(job["summary"]["errors"], 0)
        self.assertEqual(job["summary"]["policy"], "capacity")
        self.assertEqual(self.router.policy.config.policy, "round_robin")
        self.assertEqual(self.path.read_bytes(), original)
        directory = self.directory / job["directory"]
        for name in ("requests.csv", "manifest.json", "summary.json", "inputs.jsonl", "telemetry.jsonl"):
            self.assertTrue((directory / name).is_file(), name)
        manifest = json.loads((directory / "manifest.json").read_text())
        self.assertEqual(manifest["concurrency"], 2)
        self.assertEqual(manifest["requested_duration_seconds"], .12)
        self.assertEqual(manifest["requested_models"], ["model"])
        self.assertEqual(job["summary"]["gpu_energy"]["status"], "disabled")
        self.assertEqual(manifest["dashboard_experiment"]["settings"]["duration"], .12)
        self.assertFalse(manifest["policy_label_mismatch"])
        persisted = json.loads((self.router.benchmarks.root / job["job_id"] / "job.json").read_text())
        self.assertEqual(persisted["status"], "completed")
        _, runs = self.api("/api/runs")
        self.assertEqual(runs["runs"][0]["run_id"], job["run_id"])
        self.assertEqual(runs["runs"][0]["context"]["workers"], 2)
        _, comparison = self.api("/api/comparisons")
        self.assertEqual(comparison["groups"][0]["policies"][0]["policy"], "capacity")
        self.assertEqual(comparison["groups"][0]["policies"][0]["runs"], 1)
        _, detail = self.api("/api/runs/" + job["run_id"])
        breakdown = detail["replica_breakdown"]
        self.assertEqual(breakdown["status"], "recorded")
        self.assertEqual(breakdown["rows"], job["summary"]["requests"])
        self.assertEqual(breakdown["replicas"][0]["replica"], "a")
        self.assertEqual(breakdown["replicas"][0]["errors"], 0)

    def test_fixed_arrival_job_keeps_exact_trace_and_policy(self):
        body = self.trace_body()
        self.api(method="POST", body=body)
        job = self.wait()
        self.assertEqual(job["status"], "completed", job)
        self.assertEqual(job["summary"]["requests"], 3)
        self.assertEqual(job["summary"]["completed"], 3)
        self.assertEqual(job["summary"]["policy"], "kubeai")
        directory = self.directory / job["directory"]
        self.assertEqual((directory / "trace.jsonl").read_bytes(), self.trace_path.read_bytes())
        manifest = json.loads((directory / "manifest.json").read_text())
        self.assertEqual(manifest["trace_sha256"], body["trace_sha256"])
        self.assertIsNone(manifest["concurrency"])
        self.assertIsNone(manifest["seed"])

    def test_missing_provenance_requires_explicit_exploratory_api_mode(self):
        self.api(method="POST", body={**self.body, "environment": "hardware"})
        job = self.wait()
        self.assertEqual(job["status"], "completed")
        _, recorded = self.api("/api/comparisons")
        self.assertEqual(recorded["mode"], "recorded")
        self.assertEqual(recorded["groups"], [])
        _, exploratory = self.api("/api/comparisons?mode=exploratory")
        self.assertEqual(exploratory["mode"], "exploratory")
        self.assertEqual(len(exploratory["groups"]), 1)
        group = exploratory["groups"][0]
        self.assertEqual(group["kind"], "exploratory")
        self.assertEqual(set(group["missing_provenance"]), {"model revision", "image digest"})
        self.assertEqual(group["run_ids"], [job["run_id"]])
        for query in ("?mode=invalid", "?mode=", "?mode=recorded&mode=exploratory"):
            with self.assertRaises(urllib.error.HTTPError) as raised:
                self.api("/api/comparisons" + query)
            self.assertEqual(raised.exception.code, 400)

    def test_invalid_settings_cannot_start_work_or_change_policy(self):
        invalid = [{"concurrency": 0}, {"concurrency": True}, {"concurrency": 129}, {"duration": float("nan")},
                   {"duration": 301}, {"duration": 10 ** 400}, {"seed": -1}, {"warmup": 21},
                   {"warmup": .5}, {"model": "absent"}, {"mode": "wrong"}, {"policy": "wrong"},
                   {"environment": "wrong"}, {"gpu": "yes"}, {"gpu": True}, {"arbitrary_command": "ignored"},
                   {"max_inflight": 2}, {"trace_id": "sample.jsonl"}]
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    self.api(method="POST", body={**self.body, **values})
                self.assertEqual(raised.exception.code, 400)
                self.assertFalse(self.router.benchmarks.active)
                self.assertEqual(self.router.policy.config.policy, "round_robin")
        self.assertFalse(self.router.benchmarks.root.exists())
        self.assertEqual(self.backend.completed, 0)

    def test_trace_mode_rejects_irrelevant_worker_parameters(self):
        body = self.trace_body()
        for extra in ({"concurrency": 4}, {"duration": 30}, {"seed": 42}, {"model": "model"}, {"max_inflight": 129}):
            with self.subTest(extra=extra):
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    self.api(method="POST", body={**body, **extra})
                self.assertEqual(raised.exception.code, 400)

    def test_changed_trace_requires_reloading_selected_inputs(self):
        body = self.trace_body()
        self.trace_path.write_bytes(self.trace_path.read_bytes() + b"\n")
        with self.assertRaises(urllib.error.HTTPError) as raised:
            self.api(method="POST", body=body)
        self.assertEqual(raised.exception.code, 409)
        self.assertFalse(self.router.benchmarks.active)
        self.assertFalse(self.router.benchmarks.root.exists())

    def test_trace_paths_cannot_escape_workloads_directory(self):
        body = self.trace_body()
        outside = self.directory / "outside.jsonl"
        outside.write_bytes(self.trace_path.read_bytes())
        (self.trace_path.parent / "linked.jsonl").symlink_to(outside)
        for name in ("../outside.jsonl", str(outside), "linked.jsonl"):
            with self.subTest(name=name):
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    self.api(method="POST", body={**body, "trace_id": name})
                self.assertEqual(raised.exception.code, 400)
        _, options = self.api("/api/benchmark/options")
        self.assertEqual(len(options["trace_errors"]), 1)

    def test_active_job_blocks_other_jobs_and_endpoint_writes(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def blocked(**kwargs):
            entered.set()
            release.wait(4)
            raise RuntimeError("deliberate test failure")

        with patch("kubeai.experiments.benchmark.run", side_effect=blocked):
            self.api(method="POST", body=self.body)
            self.assertTrue(entered.wait(2))
            _, view = self.api("/api/endpoints")
            self.assertTrue(view["locked"])
            self.assertEqual(self.router.policy.config.policy, "capacity")
            for path, method, body in [("/api/benchmark", "POST", self.body),
                                       ("/api/endpoints", "PUT", {"replicas": [self.replica], "revision": view["revision"]})]:
                with self.subTest(path=path):
                    with self.assertRaises(urllib.error.HTTPError) as raised:
                        self.api(path, method, body)
                    self.assertEqual(raised.exception.code, 409)
            release.set()
            job = self.wait()
        self.assertEqual(job["status"], "failed")
        self.assertEqual(self.router.policy.config.policy, "round_robin")
        self.assertFalse(self.api("/api/endpoints")[1]["locked"])

    def test_external_configuration_edit_interrupts_instead_of_mixing_runs(self):
        self.api(method="POST", body={**self.body, "duration": 2})
        deadline = time.monotonic() + 2
        while self.router.benchmarks.job["status"] == "queued" and time.monotonic() < deadline:
            time.sleep(.005)
        self.path.write_text(json.dumps({**self.data, "policy": "kubeai"}))
        self.router.poll()
        job = self.wait()
        self.router.poll()
        self.assertEqual(job["status"], "failed", job)
        self.assertIn("Configuration file changed", job["message"])
        self.assertEqual(self.router.policy.config.policy, "kubeai")
        self.assertEqual(json.loads(self.path.read_text())["policy"], "kubeai")
        if job["summary"]:
            self.assertFalse(job["summary"]["valid_for_comparison"])

    def test_failed_worker_restores_policy_and_records_failure(self):
        with patch("kubeai.experiments.benchmark.run", side_effect=RuntimeError("planned failure")):
            self.api(method="POST", body=self.body)
            job = self.wait()
        self.assertEqual(job["status"], "failed")
        self.assertIn("planned failure", job["message"])
        self.assertEqual(self.router.policy.config.policy, "round_robin")

    def test_previous_job_status_survives_manager_restart(self):
        self.api(method="POST", body=self.body)
        job = self.wait()
        recovered = experiments.Benchmarks(self.router).status()
        self.assertFalse(recovered["active"])
        self.assertEqual(recovered["job"]["job_id"], job["job_id"])
        self.assertEqual(recovered["job"]["status"], "completed")

    def test_interrupted_job_is_not_restarted_or_reported_as_completed(self):
        root = self.router.benchmarks.root / "99999999-interrupted"
        root.mkdir(parents=True)
        benchmark.write_json(root / "job.json", {"job_id": "99999999-interrupted", "status": "running",
                                               "started_unix": time.time() - 1, "finished_unix": None})
        recovered = experiments.Benchmarks(self.router).status()
        self.assertEqual(recovered["job"]["status"], "failed")
        self.assertFalse(recovered["active"])
        self.assertEqual(self.backend.completed, 0)

    def test_operator_authentication_and_origin_rules_apply_to_benchmarks(self):
        with patch.dict(os.environ, {"KUBEAI_API_KEY": "operator-secret"}):
            for path, method, body in [("/api/benchmark", "GET", None),
                                       ("/api/benchmark/options", "GET", None),
                                       ("/api/comparisons", "GET", None),
                                       ("/api/comparisons?mode=exploratory", "GET", None),
                                       ("/api/runs/absent", "GET", None),
                                       ("/api/benchmark", "POST", self.body)]:
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    self.api(path, method, body)
                self.assertEqual(raised.exception.code, 401)
        for headers, code in [({"Origin": "https://other.example"}, 403),
                               ({"Host": "other.example"}, 403), ({"Content-Type": "text/plain"}, 415)]:
            with self.subTest(headers=headers):
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    self.api(method="POST", body=self.body, headers=headers)
                self.assertEqual(raised.exception.code, code)

    def test_local_gpu_sampling_is_never_enabled_by_default(self):
        _, options = self.api("/api/benchmark/options")
        self.assertFalse(options["defaults"]["gpu"])
        self.assertEqual(options["defaults"]["environment"], "unverified")

    def test_read_only_results_disable_jobs_without_changing_policy(self):
        with patch.object(self.router.benchmarks, "_writable", return_value=False):
            _, options = self.api("/api/benchmark/options")
            self.assertFalse(options["can_run"])
            with self.assertRaises(urllib.error.HTTPError) as raised:
                self.api(method="POST", body=self.body)
            self.assertEqual(raised.exception.code, 400)
        self.assertEqual(self.router.policy.config.policy, "round_robin")
        self.assertFalse(self.router.benchmarks.root.exists())

    def test_router_shutdown_interrupts_job_and_restores_policy(self):
        self.api(method="POST", body={**self.body, "duration": 2})
        deadline = time.monotonic() + 2
        while self.router.benchmarks.job["status"] == "queued" and time.monotonic() < deadline:
            time.sleep(.005)
        self.router.stop()
        job = self.wait()
        self.assertEqual(job["status"], "failed", job)
        self.assertIn("router stopped", job["message"])
        self.assertEqual(self.router.policy.config.policy, "round_robin")


class TelemetryCredentialTests(unittest.TestCase):
    def test_explicit_backend_credentials_do_not_leak_to_other_sources_or_records(self):
        with tempfile.TemporaryDirectory() as directory:
            router = "http://router.example/metrics"
            backend = "http://backend.example/metrics"
            public = "http://public.example/metrics"
            telemetry = benchmark.Telemetry(directory, [router, backend, public], api_key="router-secret",
                                            auth_origin="http://router.example", metrics_headers={
                                                backend: {"Authorization": "Bearer backend-secret"}})
            observed = []

            def capture(url, **kwargs):
                observed.append((url, kwargs["headers"]))
                if len(observed) == 3:
                    telemetry.stop_event.set()
                return io.BytesIO(b"vllm:num_requests_running 0\n")

            with patch("kubeai.benchmark.open_url", side_effect=capture):
                telemetry.loop()
            self.assertEqual(observed[0][1]["Authorization"], "Bearer router-secret")
            self.assertEqual(observed[1][1]["Authorization"], "Bearer backend-secret")
            self.assertNotIn("Authorization", observed[2][1])
            raw = (Path(directory) / "telemetry.jsonl").read_text()
            self.assertNotIn("router-secret", raw)
            self.assertNotIn("backend-secret", raw)


if __name__ == "__main__":
    unittest.main()
