import copy
import json
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from kubeai import benchmark, discovery, gpu, placement, workload
from kubeai.config import Config
from kubeai.metrics import signals
from kubeai.policy import Overloaded, Policy, Unavailable, score


def configuration(policy="kubeai", **extra):
    return Config.parse({"policy": policy, "replicas": [
        {"name": "a", "url": "http://localhost:8001", "models": ["model"], "capacity": 1, "token_seconds": .01},
        {"name": "b", "url": "http://localhost:8002", "models": ["model"], "capacity": 2, "token_seconds": .01}], **extra})


def healthy(policy):
    now = time.monotonic()
    for replica in policy.replicas.values():
        replica.healthy_at = replica.metrics_at = now
        replica.models = {"model"}


class RoutingTests(unittest.TestCase):
    def test_round_robin_is_balanced(self):
        policy = Policy(configuration("round_robin"))
        healthy(policy)
        names = []
        for _ in range(8):
            replica, _ = policy.reserve("model", 8)
            names.append(replica.config.name)
            policy.release(replica)
        self.assertEqual(names, ["a", "b"] * 4)

    def test_capacity_considers_slots(self):
        policy = Policy(configuration("capacity"))
        healthy(policy)
        replica, _ = policy.reserve("model", 8)
        self.assertEqual(replica.config.name, "b")
        policy.release(replica)

    def test_busy_replica_is_avoided(self):
        policy = Policy(configuration())
        healthy(policy)
        policy.replicas["a"].metrics = {"waiting": 20, "running": 1, "kv": .9}
        replica, decision = policy.reserve("model", 8)
        self.assertEqual(replica.config.name, "b")
        self.assertTrue(decision["metrics_fresh"])
        self.assertEqual(set(decision["terms"]), {"queue", "service", "kv", "memory"})
        policy.release(replica)

    def test_stale_metrics_use_local_reservations(self):
        policy = Policy(configuration())
        healthy(policy)
        a = policy.replicas["a"]
        a.metrics = {"waiting": 1000, "kv": 1}
        a.metrics_at = time.monotonic() - 100
        a.inflight = 1
        _, details = score(a, policy.config, 8, time.monotonic())
        self.assertFalse(details["metrics_fresh"])
        self.assertEqual(details["load"], 1)
        self.assertEqual(details["terms"]["kv"], 0)

    def test_local_and_observed_work_are_not_added(self):
        policy = Policy(configuration())
        healthy(policy)
        replica = policy.replicas["a"]
        replica.inflight = 5
        replica.metrics = {"running": 3, "waiting": 2}
        _, details = score(replica, policy.config, 8, time.monotonic())
        self.assertEqual(details["load"], 5)

    def test_healthy_probe_does_not_override_model_compatibility(self):
        policy = Policy(configuration())
        healthy(policy)
        with self.assertRaises(Unavailable):
            policy.reserve("other-model", 8)
        with self.assertRaises(Unavailable):
            policy.reserve("model", 4096)

    def test_expired_health_fails_closed(self):
        policy = Policy(configuration())
        healthy(policy)
        for replica in policy.replicas.values():
            replica.healthy_at -= 100
        with self.assertRaises(Unavailable):
            policy.reserve("model", 8)

    def test_atomic_limit_under_parallel_selection(self):
        policy = Policy(configuration(max_inflight=8))
        healthy(policy)

        def reserve(_):
            try:
                return policy.reserve("model", 8)[0]
            except Overloaded:
                return None

        with ThreadPoolExecutor(max_workers=16) as pool:
            selected = [r for r in pool.map(reserve, range(50)) if r]
        self.assertEqual(len(selected), 8)
        self.assertEqual(policy.active, 8)
        for replica in selected:
            policy.release(replica)
        self.assertEqual(policy.active, 0)
        self.assertEqual(sum(r.inflight for r in policy.replicas.values()), 0)

    def test_failure_temporarily_blocks_replica(self):
        policy = Policy(configuration("round_robin"))
        healthy(policy)
        replica, _ = policy.reserve("model", 8)
        policy.release(replica, failed=True)
        for _ in range(3):
            chosen, _ = policy.reserve("model", 8)
            self.assertNotEqual(chosen.config.name, replica.config.name)
            policy.release(chosen)

    def test_removed_endpoint_finishes_without_being_reselected(self):
        policy = Policy(configuration("round_robin"))
        healthy(policy)
        old, _ = policy.reserve("model", 8)
        policy.reconcile([policy.replicas["b"].config])
        policy.release(old)
        self.assertEqual(policy.active, 0)
        chosen, _ = policy.reserve("model", 8)
        self.assertEqual(chosen.config.name, "b")
        policy.release(chosen)

    def test_bad_config_rejected(self):
        for values in ({"policy": "random"}, {"poll_interval": 0}, {"stale_after": .5}, {"max_inflight": 1.5}, {"upstream_timeout": float("nan")}, {"poll_interval": "1"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                configuration(**values)


class MetricsTests(unittest.TestCase):
    def test_gpu_exporter_rejects_multiple_devices(self):
        with patch("kubeai.gpu.subprocess.run") as run:
            run.return_value.stdout = "8192, 4000, 50\n8192, 2000, 60\n"
            with self.assertRaises(ValueError):
                gpu.sample("0,1")

    def test_gpu_exporter_reports_physical_units(self):
        with patch("kubeai.gpu.subprocess.run") as run:
            run.return_value.stdout = "8192, 4000, 50\n"
            self.assertEqual(gpu.sample("0")["kubeai_gpu_memory_free_mib"], 4000)

    def test_separate_gpu_sample_expires_independently(self):
        policy = Policy(configuration())
        healthy(policy)
        replica = policy.replicas["a"]
        replica.gpu_metrics = {"memory_free": 0, "memory_total": 8192}
        replica.gpu_metrics_at = time.monotonic()
        _, first = score(replica, policy.config, 8, time.monotonic())
        self.assertEqual(first["terms"]["memory"], .05)
        replica.gpu_metrics_at -= 100
        _, stale = score(replica, policy.config, 8, time.monotonic())
        self.assertEqual(stale["terms"]["memory"], 0)

    def test_engine_labels_and_nonfinite_values(self):
        data = signals('# HELP comment\nvllm:num_requests_waiting{engine="0"} 2\nvllm:num_requests_waiting{engine="1"} 3\nvllm:kv_cache_usage_perc{engine="0"} .2\nvllm:kv_cache_usage_perc{engine="1"} .8\nvllm:num_requests_running NaN\nbad line\n')
        self.assertEqual(data, {"waiting": 5, "kv": .8})

    def test_old_cache_name_is_supported(self):
        self.assertEqual(signals("vllm:gpu_cache_usage_perc .5\n")["kv"], .5)

    def test_no_metrics_is_explicitly_missing(self):
        self.assertEqual(signals("# no data\n"), {})


class PlacementTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.inventory = json.loads((root / "config/inventory.example.json").read_text())
        self.request = json.loads((root / "config/placement-request.json").read_text())

    def test_baseline_and_best_fit_select_different_safe_nodes(self):
        baseline = placement.recommend(self.inventory, self.request, "first_fit", now=1000)
        smart = placement.recommend(self.inventory, self.request, "best_fit", now=1000)
        self.assertEqual(baseline["node"], "gpu-large")
        self.assertEqual(smart["node"], "gpu-small")
        self.assertFalse(smart["applied"])

    def test_stale_and_incompatible_nodes_are_excluded(self):
        self.assertIsNone(placement.recommend(self.inventory, self.request, now=1100)["node"])
        self.request["model"] = "unknown"
        self.assertIsNone(placement.recommend(self.inventory, self.request, now=1000)["node"])

    def test_does_not_sum_memory_across_gpus(self):
        self.inventory["nodes"] = [self.inventory["nodes"][1]]
        node = self.inventory["nodes"][0]
        node["gpus"] = [{"id": str(i), "allocated": False, "total_mib": 4096, "free_mib": 4096} for i in range(2)]
        self.assertIsNone(placement.recommend(self.inventory, self.request, now=1000)["node"])

    def test_does_not_assume_device_plugin_binds_uuid(self):
        self.inventory["nodes"] = [self.inventory["nodes"][1]]
        self.inventory["nodes"][0]["gpus"].append({"id": "small", "allocated": False, "total_mib": 1024, "free_mib": 1024})
        self.assertIsNone(placement.recommend(self.inventory, self.request, now=1000)["node"])

    def test_allocated_gpus_are_not_available(self):
        for node in self.inventory["nodes"]:
            node["gpus"][0]["allocated"] = True
        self.assertIsNone(placement.recommend(self.inventory, self.request, now=1000)["node"])

    def test_insufficient_host_resources_and_taints(self):
        self.inventory["nodes"][0]["free_ram_mib"] = 1
        self.inventory["nodes"][1]["untolerated_taints"] = ["reserved"]
        self.assertIsNone(placement.recommend(self.inventory, self.request, now=1000)["node"])

    def test_invalid_capacity_is_rejected(self):
        self.request["gpus"] = -1
        with self.assertRaises(ValueError):
            placement.recommend(self.inventory, self.request, now=1000)

    def test_generated_pod_preserves_scheduler_resource_checks(self):
        decision = placement.recommend(self.inventory, self.request, now=1000)
        pod = placement.pod_manifest(decision, "example/vllm@sha256:" + "a" * 64)
        self.assertNotIn("nodeName", pod["spec"])
        fields = pod["spec"]["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchFields"]
        self.assertEqual(fields[0]["values"], ["gpu-small"])
        resources = pod["spec"]["containers"][0]["resources"]
        self.assertEqual(resources["requests"]["nvidia.com/gpu"], resources["limits"]["nvidia.com/gpu"])
        with self.assertRaises(ValueError):
            placement.pod_manifest(decision, "example/vllm:latest")


class TraceAndMeasurementTests(unittest.TestCase):
    def test_seeded_trace_is_identical_and_replayable(self):
        first = workload.generate(20, seed=31, scenario="burst")
        self.assertEqual(first, workload.generate(20, seed=31, scenario="burst"))
        self.assertNotEqual(first, workload.generate(20, seed=32, scenario="burst"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            workload.save(path, first)
            actual, digest = workload.load(path)
        self.assertEqual(actual, first)
        self.assertEqual(len(digest), 64)

    def test_unsorted_trace_rejected(self):
        rows = workload.generate(3)
        rows[0]["offset_seconds"] = 100
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            workload.save(path, rows)
            with self.assertRaises(ValueError):
                workload.load(path)

    def test_empty_statistics_are_not_fabricated(self):
        value = benchmark.summarize([], 1)
        self.assertIsNone(value["latency_p95_seconds"])
        self.assertIsNone(value["ttft_p95_seconds"])
        self.assertEqual(value["completed_rps"], 0)

    def test_filtered_success_and_token_length_are_distinct(self):
        row = {"id": "a", "tenant": "t", "completed": True, "latency_seconds": .2,
               "ttft_seconds": .05, "scheduled_latency_seconds": .21, "error": "", "finish_reason": "length", "output_tokens": 8, "replica": "a"}
        failed = {**row, "completed": False, "error": "client_inflight_limit", "finish_reason": ""}
        value = benchmark.summarize([row, failed], 1)
        self.assertEqual(value["completed"], 1)
        self.assertEqual(value["truncated"], 1)
        self.assertEqual(value["dropped"], 1)
        self.assertEqual(value["error_rate"], .5)

    def test_discovery_excludes_unready_and_deduplicates(self):
        data = {"items": [{"ports": [{"name": "http", "port": 8000}], "endpoints": [
            {"addresses": ["10.0.0.1"], "conditions": {"ready": True}},
            {"addresses": ["10.0.0.2"], "conditions": {"ready": False}},
            {"addresses": ["10.0.0.3"], "conditions": {"ready": True, "terminating": True}},
            {"addresses": ["10.0.0.1"], "conditions": {"ready": True}}]}]}
        result = discovery.parse_slices(data, {"service": "demo", "replica": {"models": ["model"]}})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].url, "http://10.0.0.1:8000")

    def test_discovery_supports_ipv6(self):
        data = {"items": [{"ports": [{"name": "http", "port": 8000}], "endpoints": [{"addresses": ["::1"], "conditions": {"ready": True}}]}]}
        result = discovery.parse_slices(data, {"service": "demo", "replica": {"models": ["model"]}})
        self.assertEqual(result[0].url, "http://[::1]:8000")


if __name__ == "__main__":
    unittest.main()
