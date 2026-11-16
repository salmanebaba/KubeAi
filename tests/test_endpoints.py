"""Endpoint management against temporary HTTP replicas and configuration files."""

import io
import json
import os
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from kubeai import benchmark, emulator, endpoints
from kubeai.config import Config
from kubeai.demo import serve
from kubeai.router import Router, make_server


class EndpointManagementTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.backend = emulator.Emulator("model", capacity=2, token_seconds=.01)
        self.backend_url = serve(self.stack, emulator.make_server(self.backend, port=0))
        self.other = emulator.Emulator("model", capacity=2, token_seconds=.001)
        self.other_url = serve(self.stack, emulator.make_server(self.other, port=0))
        self.replica = {"name": "a", "url": self.backend_url, "models": ["model"],
                        "capacity": 2, "token_seconds": .01, "prefill_seconds": .025}
        self.data = {"replicas": [self.replica], "policy": "round_robin", "poll_interval": .05,
                     "stale_after": 2, "max_inflight": 16}
        self.path = self.directory / "router.json"
        self.path.write_text(json.dumps(self.data))
        self.router = Router(Config.load(self.path), self.path, self.directory)
        self.router.start()
        self.stack.callback(self.router.stop)
        self.url = serve(self.stack, make_server(self.router, port=0))
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def api(self, path="/api/endpoints", method="GET", body=None, headers=None):
        headers = {"Content-Type": "application/json", **(headers or {})}
        request = urllib.request.Request(self.url + path, method=method, headers=headers,
                                         data=json.dumps(body).encode() if body is not None else None)
        with self.opener.open(request, timeout=5) as response:
            return json.load(response)

    def apply(self, replicas, revision=None, headers=None):
        if revision is None:
            revision = self.api()["revision"]
        return self.api(method="PUT", body={"revision": revision, "replicas": replicas}, headers=headers)

    def test_saved_endpoints_expose_urls_and_preserve_estimates(self):
        view = self.api()
        self.assertTrue(view["editable"])
        self.assertEqual(len(view["revision"]), 64)
        self.assertEqual(view["replicas"][0]["url"], self.backend_url)
        self.assertEqual(view["replicas"][0]["capacity"], 2)
        self.assertEqual(view["replicas"][0]["prefill_seconds"], .025)
        state = self.api("/api/state")
        self.assertEqual(state["replicas"][0]["url"], self.backend_url)

    def test_connection_probe_does_not_save_or_run_inference(self):
        original = self.path.read_bytes()
        result = self.api("/api/endpoints/test", "POST", {"replica": self.replica})
        self.assertTrue(result["ok"])
        self.assertTrue(result["metrics_available"])
        self.assertEqual(result["models"], ["model"])
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(self.backend.completed, 0)

    def test_connection_probe_reports_missing_models(self):
        result = self.api("/api/endpoints/test", "POST", {"replica": {**self.replica, "models": ["absent"]}})
        self.assertFalse(result["ok"])
        self.assertIn("absent", result["message"])
        self.assertEqual(result["models"], ["model"])

    def test_unreachable_probe_returns_failure_without_exposing_error_details(self):
        with patch("kubeai.endpoints.open_url", side_effect=OSError("private-secret")):
            result = self.api("/api/endpoints/test", "POST", {"replica": self.replica})
        self.assertFalse(result["ok"])
        self.assertIn("OSError", result["message"])
        self.assertNotIn("private-secret", json.dumps(result))

    def test_credentials_stay_server_side_and_only_reach_the_backend(self):
        observed = []

        def response(url, **kwargs):
            observed.append((url, kwargs["headers"]))
            if url.endswith("/v1/models"):
                return io.BytesIO(b'{"data":[{"id":"model"}]}')
            return io.BytesIO(b"vllm:num_requests_running 0\n")

        with patch.dict(os.environ, {"ENDPOINT_TEST_SECRET": "private-secret"}), \
                patch("kubeai.endpoints.open_url", side_effect=response):
            result = self.api("/api/endpoints/test", "POST", {
                "replica": {**self.replica, "api_key_env": "ENDPOINT_TEST_SECRET"}})
        self.assertTrue(result["ok"])
        self.assertEqual(observed[0][1]["Authorization"], "Bearer private-secret")
        self.assertNotIn("private-secret", json.dumps(result))

    def test_apply_persists_and_activates_endpoints_without_changing_policy(self):
        view = self.api()
        second = {**view["replicas"][0], "name": "b", "url": self.other_url}
        saved = self.apply([*view["replicas"], second], view["revision"])
        self.assertNotEqual(saved["revision"], view["revision"])
        self.assertEqual(set(self.router.policy.replicas), {"a", "b"})
        self.router.poll()
        self.assertTrue(all(replica["healthy"] for replica in self.router.state()["replicas"]))
        data = json.loads(self.path.read_text())
        self.assertEqual(data["policy"], self.data["policy"])
        self.assertEqual(data["max_inflight"], self.data["max_inflight"])
        self.assertEqual(data["replicas"][0]["token_seconds"], .01)
        self.assertEqual(data["replicas"][0]["prefill_seconds"], .025)
        self.assertEqual(Config.load(self.path).replicas[1].url, self.other_url)

    def test_invalid_edits_leave_disk_and_live_endpoints_unchanged(self):
        original = self.path.read_bytes()
        revision = self.api()["revision"]
        invalid = [[], [self.replica, self.replica],
                   [{**self.replica, "url": self.backend_url + "/v1"}],
                   [{**self.replica, "url": "http://127.0.0.1:99999"}],
                   [{**self.replica, "url": "http://127.0.0.1:0"}],
                   [{**self.replica, "url": "http://user:secret@localhost"}],
                   [{**self.replica, "models": "model"}],
                   [{**self.replica, "name": 4}],
                   [{**self.replica, "capacity": 0}]]
        for replicas in invalid:
            with self.subTest(replicas=replicas):
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    self.apply(replicas, revision)
                self.assertEqual(raised.exception.code, 400)
                self.assertEqual(self.path.read_bytes(), original)
                self.assertEqual(set(self.router.policy.replicas), {"a"})

    def test_router_cannot_be_configured_as_its_own_backend(self):
        with self.assertRaises(urllib.error.HTTPError) as raised:
            self.api("/api/endpoints/test", "POST", {"replica": {**self.replica, "url": self.url}})
        self.assertEqual(raised.exception.code, 400)
        with self.assertRaises(urllib.error.HTTPError) as raised:
            self.apply([{**self.replica, "url": self.url}])
        self.assertEqual(raised.exception.code, 400)

    def test_stale_draft_cannot_overwrite_external_configuration_changes(self):
        view = self.api()
        data = {**self.data, "policy": "capacity"}
        self.path.write_text(json.dumps(data))
        with self.assertRaises(urllib.error.HTTPError) as raised:
            self.apply(view["replicas"], view["revision"])
        self.assertEqual(raised.exception.code, 409)
        self.assertEqual(json.loads(self.path.read_text())["policy"], "capacity")

    def test_failed_atomic_write_leaves_original_configuration_active(self):
        original = self.path.read_bytes()
        view = self.api()
        with patch("kubeai.endpoints.os.replace", side_effect=PermissionError("read only")):
            with self.assertRaises(urllib.error.HTTPError) as raised:
                self.apply([{**self.replica, "name": "b", "url": self.other_url}], view["revision"])
        self.assertEqual(raised.exception.code, 500)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(set(self.router.policy.replicas), {"a"})
        self.assertEqual(list(self.directory.glob(".router.json.*")), [])

    def test_removed_endpoint_finishes_its_existing_request(self):
        entry = {"id": "pending", "body": {"model": "model", "messages": [{"role": "user", "content": "test"}], "max_tokens": 40}}
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(benchmark.request, self.url, entry, time.time())
            deadline = time.monotonic() + 2
            while not self.router.policy.active and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertEqual(self.router.policy.active, 1)
            self.apply([{**self.replica, "name": "b", "url": self.other_url}])
            self.assertNotIn("a", self.router.policy.replicas)
            result = pending.result(timeout=3)
        self.assertTrue(result["completed"], result)
        self.assertEqual(result["replica"], "a")
        self.router.poll()
        result = benchmark.request(self.url, {**entry, "id": "new"}, time.time())
        self.assertTrue(result["completed"], result)
        self.assertEqual(result["replica"], "b")

    def test_non_file_router_is_explicitly_read_only(self):
        with self.router.control_lock:
            self.router.config_path = None
        self.assertFalse(self.api()["editable"])
        with self.assertRaises(urllib.error.HTTPError) as raised:
            self.apply([self.replica], "not-a-revision")
        self.assertEqual(raised.exception.code, 409)

    def test_discovery_configuration_is_read_only(self):
        path = self.directory / "discovery.json"
        data = {**self.data, "replicas": [], "discovery": {
            "namespace": "kubeai", "service": "models", "replica": {"models": ["model"]}}}
        path.write_text(json.dumps(data))
        self.assertFalse(endpoints.read(path, self.router.policy.config)["editable"])
        with self.assertRaises(endpoints.ReadOnly):
            endpoints.save(path, self.router.policy.config, [self.replica], "unused")

    def test_management_obeys_api_authentication(self):
        with patch.dict(os.environ, {"KUBEAI_API_KEY": "operator-key"}):
            for path, method, body in [("/api/endpoints", "GET", None),
                                       ("/api/endpoints/test", "POST", {"replica": self.replica}),
                                       ("/api/endpoints", "PUT", {"replicas": [self.replica]})]:
                with self.subTest(path=path, method=method):
                    with self.assertRaises(urllib.error.HTTPError) as raised:
                        self.api(path, method, body)
                    self.assertEqual(raised.exception.code, 401)
            result = self.api("/api/endpoints/test", "POST", {"replica": self.replica},
                              {"Authorization": "Bearer operator-key"})
            self.assertTrue(result["ok"])

    def test_cross_site_writes_and_untrusted_hosts_are_rejected(self):
        for headers, status in [({"Origin": "https://other.example"}, 403),
                                ({"Sec-Fetch-Site": "cross-site"}, 403),
                                ({"Host": "other.example"}, 403),
                                ({"Content-Type": "text/plain"}, 415)]:
            with self.subTest(headers=headers):
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    self.api("/api/endpoints/test", "POST", {"replica": self.replica}, headers)
                self.assertEqual(raised.exception.code, status)


if __name__ == "__main__":
    unittest.main()
