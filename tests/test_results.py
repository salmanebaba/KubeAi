"""Saved-artifact comparisons: no sockets, inference or GPU access."""

import copy
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from kubeai import results, workload


class ResultComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.catalog = results.RunCatalog(self.root)
        self.sequence = 0

    def saved(self, policy="round_robin", mode="closed_loop", environment="emulator", change=None, summary=None):
        self.sequence += 1
        directory = self.root / f"run-{self.sequence:03}"
        directory.mkdir()
        name = "trace.jsonl" if mode == "fixed_arrival" else "inputs.jsonl"
        workload.save(directory / name, workload.generate(4, seed=42, models=["model"]))
        digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        config = {"policy": policy, "discovery": None, "replicas": [
            {"name": "a", "url": "http://127.0.0.1:8001", "models": ["model"], "capacity": 4},
            {"name": "b", "url": "http://127.0.0.1:8002", "models": ["model"], "capacity": 4}]}
        manifest = {"mode": mode, "environment": environment, "policy": policy,
                    "source_sha256": "source-a", "trace_sha256": digest,
                    "url": "http://127.0.0.1:8080", "concurrency": 4 if mode == "closed_loop" else None,
                    "requested_duration_seconds": 30 if mode == "closed_loop" else None,
                    "warmup_per_model": 2, "stream": True, "client_max_inflight": 128,
                    "timeout_seconds": 60, "gpu_telemetry_requested": False,
                    "model_revision": "revision-a", "image_digest": "image-a",
                    "platform": "test-host", "python": "3.12", "cpu_count": 8,
                    "router_state_before": {"configuration": copy.deepcopy(config)},
                    "router_state_after": {"configuration": copy.deepcopy(config)}}
        if change:
            change(manifest)
        record = {"policy": policy, "mode": mode, "environment": environment,
                  "trace_sha256": manifest.get("trace_sha256"), "source_sha256": manifest.get("source_sha256"),
                  "requests": 10, "completed": 8, "errors": 2, "dropped": 1, "truncated": 3,
                  "latency_p50_seconds": .1, "latency_p95_seconds": .2, "ttft_p95_seconds": .05,
                  "completed_rps": 4}
        record.update(summary or {})
        self.write(directory / "manifest.json", manifest)
        self.write(directory / "summary.json", record)
        return directory

    def write(self, path, value):
        path.write_text(json.dumps(value))

    def alter(self, directory, filename, **values):
        path = directory / filename
        value = json.loads(path.read_text())
        value.update(values)
        self.write(path, value)

    def context(self, directory):
        return next(run["context"] for run in self.catalog.list() if run["run_id"] == directory.name)

    def gpu(self, directory, identity="GPU-a", requested=True):
        self.alter(directory, "manifest.json", gpu_telemetry_requested=requested)
        fields = ["timestamp", "uuid", "name", "driver_version", "memory.total", "memory.used", "utilization.gpu"]
        samples = [{"source": "nvidia-smi", "fields": fields,
                    "csv": f"2026/11/14 12:00:00, {identity}, GPU model, driver-a, 8192, {used}, {util}\n"}
                   for used, util in ((2048, 20), (4096, 80))]
        (directory / "telemetry.jsonl").write_text("".join(json.dumps(row) + "\n" for row in samples))

    def requests(self, directory, rows):
        fields = ["replica", "completed", "latency_seconds", "ttft_seconds", "error", "finish_reason"]
        with (directory / "requests.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({"replica": "a", "completed": True, "latency_seconds": .1,
                                 "ttft_seconds": .02, "error": "", "finish_reason": "stop", **row})

    def exploratory(self, policy="round_robin", change=None, summary=None):
        def incomplete(manifest):
            manifest.update(model_revision="unknown", image_digest="unknown")
            if change:
                change(manifest)
        return self.saved(policy, environment="hardware", change=incomplete, summary=summary)

    def test_exploratory_comparison_is_explicit_and_keeps_original_artifacts(self):
        directories = [self.exploratory(policy) for policy in ("round_robin", "capacity", "kubeai")]
        original = {(directory / name): (directory / name).read_bytes()
                    for directory in directories for name in ("summary.json", "manifest.json", "inputs.jsonl")}
        runs = self.catalog.list()
        self.assertEqual(results.comparisons(runs), [])
        groups = results.comparisons(runs, mode="exploratory")
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["kind"], "exploratory")
        self.assertEqual(set(groups[0]["missing_provenance"]), {"model revision", "image digest"})
        self.assertEqual({row["policy"] for row in groups[0]["policies"]}, {"round_robin", "capacity", "kubeai"})
        self.assertTrue(all(run["context"]["comparison_key"] is None for run in runs))
        self.assertEqual(original, {path: path.read_bytes() for path in original})
        with self.assertRaises(ValueError):
            results.comparisons(runs, mode="invalid")

    def test_exploratory_normalizes_missing_provenance_markers_only(self):
        for policy, marker in zip(("round_robin", "capacity", "kubeai"), ("unknown", None, "")):
            self.exploratory(policy, change=lambda value: value.update(model_revision=marker, image_digest=marker))
        groups = results.comparisons(self.catalog.list(), mode="exploratory")
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]["policies"]), 3)

    def test_exploratory_does_not_mix_known_conflicting_or_complete_provenance(self):
        self.exploratory()
        self.exploratory("capacity", change=lambda value: value.update(model_revision="revision-a"))
        self.exploratory("kubeai", change=lambda value: value.update(model_revision="revision-b"))
        self.exploratory("capacity", change=lambda value: value.update(image_digest="image-b"))
        complete = self.saved(environment="hardware")
        rows = self.catalog.list()
        self.assertEqual(len(results.comparisons(rows, mode="exploratory")), 4)
        recorded = results.comparisons(rows)
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0]["run_ids"], [complete.name])
        self.assertIsNone(self.context(complete)["exploratory_comparison_key"])

    def test_exploratory_preserves_workload_code_limits_and_endpoint_matching(self):
        baseline = self.exploratory()
        changes = [{"concurrency": 1}, {"requested_duration_seconds": 60}, {"source_sha256": "other-code"},
                   {"stream": False}, {"warmup_per_model": 3}, {"client_max_inflight": 32}, {"timeout_seconds": 30}]
        for change in changes:
            self.exploratory(change=lambda value: value.update(change))
        def changed_capacity(value):
            for key in ("router_state_before", "router_state_after"):
                value[key]["configuration"]["replicas"][0]["capacity"] = 8
        self.exploratory(change=changed_capacity)
        different_input = self.exploratory()
        workload.save(different_input / "inputs.jsonl", workload.generate(4, seed=9, models=["model"]))
        digest = hashlib.sha256((different_input / "inputs.jsonl").read_bytes()).hexdigest()
        self.alter(different_input, "manifest.json", trace_sha256=digest)
        self.alter(different_input, "summary.json", trace_sha256=digest)
        groups = results.comparisons(self.catalog.list(), mode="exploratory")
        self.assertEqual(len(groups), 3 + len(changes))
        self.assertTrue(all(len(group["run_ids"]) == 1 for group in groups))
        self.assertIsNotNone(self.context(baseline)["exploratory_comparison_key"])

    def test_exploratory_cannot_relax_missing_context_beyond_deployment_provenance(self):
        for key in ("source_sha256", "requested_duration_seconds", "warmup_per_model", "stream", "client_max_inflight", "timeout_seconds"):
            with self.subTest(key=key):
                directory = self.exploratory(change=lambda value: value.pop(key))
                if key == "source_sha256":
                    self.alter(directory, "summary.json", source_sha256=None)
                self.assertIsNone(self.context(directory)["exploratory_comparison_key"])
        no_inputs = self.exploratory()
        (no_inputs / "inputs.jsonl").unlink()
        no_manifest = self.exploratory()
        (no_manifest / "manifest.json").unlink()
        no_gpu = self.exploratory(change=lambda value: value.update(gpu_telemetry_requested=True))
        unverified = self.saved(environment="unverified")
        for directory in (no_inputs, no_manifest, no_gpu, unverified):
            self.assertIsNone(self.context(directory)["exploratory_comparison_key"])
        self.assertEqual(results.comparisons(self.catalog.list(), mode="exploratory"), [])

    def test_exploratory_excludes_interrupted_unstable_tampered_and_failed_jobs(self):
        def unstable(value):
            value["router_state_after"]["configuration"]["replicas"].pop()
        directories = [self.exploratory(change=unstable), self.exploratory(summary={"interrupted": True}),
                       self.exploratory(summary={"valid_for_comparison": False}),
                       self.exploratory(change=lambda value: value.update(policy_label_mismatch=True)),
                       self.exploratory(summary={"source_sha256": "disagrees"})]
        for directory in directories:
            self.assertEqual(self.context(directory)["status"], "excluded")
            self.assertIsNone(self.context(directory)["exploratory_comparison_key"])
        tampered = self.exploratory()
        (tampered / "inputs.jsonl").write_text("{}\n")
        self.assertIsNone(self.context(tampered)["exploratory_comparison_key"])
        failed = self.exploratory()
        self.write(self.root / "job.json", {"status": "failed"})
        self.assertIsNone(self.context(failed)["exploratory_comparison_key"])
        self.assertEqual(results.comparisons(self.catalog.list(), mode="exploratory"), [])

    def test_exploratory_keeps_gpu_sampling_and_physical_devices_separate(self):
        disabled = self.exploratory()
        first = self.exploratory()
        second = self.exploratory("capacity")
        third = self.exploratory("kubeai")
        self.gpu(first, "GPU-a")
        self.gpu(second, "GPU-b")
        self.gpu(third, "GPU-a")
        groups = results.comparisons(self.catalog.list(), mode="exploratory")
        self.assertEqual(len(groups), 3)
        matching = next(group for group in groups if first.name in group["run_ids"])
        self.assertEqual(set(matching["run_ids"]), {first.name, third.name})
        self.assertNotEqual(self.context(disabled)["exploratory_comparison_key"], self.context(first)["exploratory_comparison_key"])

    def test_exploratory_cache_refreshes_when_provenance_becomes_recorded(self):
        directory = self.exploratory()
        original = (directory / "summary.json").read_bytes()
        before = self.context(directory)["exploratory_comparison_key"]
        self.alter(directory, "manifest.json", model_revision="revision-a")
        self.assertNotEqual(self.context(directory)["exploratory_comparison_key"], before)
        self.alter(directory, "manifest.json", image_digest="image-a")
        self.assertIsNotNone(self.context(directory)["comparison_key"])
        self.assertIsNone(self.context(directory)["exploratory_comparison_key"])
        self.assertEqual((directory / "summary.json").read_bytes(), original)

    def test_policies_share_context_and_repeats_keep_mean_and_range(self):
        self.saved(summary={"latency_p95_seconds": .2})
        self.saved(summary={"latency_p95_seconds": .4})
        self.saved("capacity")
        self.saved("kubeai")
        groups = results.comparisons(self.catalog.list())
        self.assertEqual(len(groups), 1)
        policies = {row["policy"]: row for row in groups[0]["policies"]}
        self.assertEqual(set(policies), {"round_robin", "capacity", "kubeai"})
        rr = policies["round_robin"]
        self.assertEqual(rr["runs"], 2)
        self.assertAlmostEqual(rr["latency_p95_mean_seconds"], .3)
        self.assertEqual((rr["latency_p95_min_seconds"], rr["latency_p95_max_seconds"]), (.2, .4))
        self.assertEqual((rr["requests"], rr["completed"], rr["errors"], rr["dropped"], rr["truncated"]), (20, 16, 4, 2, 6))

    def test_different_workers_and_requested_durations_never_average(self):
        for workers, duration in ((1, 30), (16, 30), (1, 60)):
            self.saved(change=lambda value: value.update(concurrency=workers, requested_duration_seconds=duration))
        groups = results.comparisons(self.catalog.list())
        self.assertEqual(len(groups), 3)
        self.assertEqual({(row["context"]["workers"], row["context"]["duration_seconds"]) for row in groups},
                         {(1, 30), (16, 30), (1, 60)})

    def test_metadata_differences_keep_separate_comparison_groups(self):
        baseline = self.saved(environment="hardware")
        changes = [{"source_sha256": "other-source"}, {"model_revision": "other-revision"},
                   {"image_digest": "other-image"}, {"warmup_per_model": 3}, {"stream": False},
                   {"client_max_inflight": 16}, {"timeout_seconds": 30}, {"platform": "other-host"}]
        for change in changes:
            self.saved(environment="hardware", change=lambda value: value.update(change))

        def other_endpoints(value):
            for key in ("router_state_before", "router_state_after"):
                value[key]["configuration"]["replicas"][0]["url"] = "http://remote.example:8001"
        self.saved(environment="hardware", change=other_endpoints)
        groups = results.comparisons(self.catalog.list())
        self.assertEqual(len(groups), 2 + len(changes))
        self.assertEqual(self.context(baseline)["status"], "matched_settings")

    def test_fixed_arrival_matches_exact_inputs_and_not_closed_loop(self):
        self.saved(mode="fixed_arrival")
        self.saved("capacity", mode="fixed_arrival")
        changed = self.saved("kubeai", mode="fixed_arrival")
        workload.save(changed / "trace.jsonl", workload.generate(5, seed=9, models=["model"]))
        digest = hashlib.sha256((changed / "trace.jsonl").read_bytes()).hexdigest()
        self.alter(changed, "manifest.json", trace_sha256=digest)
        self.alter(changed, "summary.json", trace_sha256=digest)
        self.saved()
        groups = results.comparisons(self.catalog.list())
        self.assertEqual(len(groups), 3)
        original = next(row for row in groups if len(row["policies"]) == 2)
        self.assertEqual(original["context"]["input_rows"], 4)
        self.assertIsNone(original["context"]["workers"])
        self.assertIsNotNone(original["context"]["arrival_seconds"])

    def test_unknown_hardware_provenance_and_old_duration_remain_individual(self):
        hardware = self.saved(environment="hardware", change=lambda value: value.update(model_revision="unknown", image_digest="unknown"))
        old = self.saved(change=lambda value: value.pop("requested_duration_seconds"))
        self.assertEqual(self.context(hardware)["status"], "individual_only")
        self.assertIn("model revision", self.context(hardware)["missing"])
        self.assertIn("requested duration", self.context(old)["missing"])
        self.assertEqual(results.comparisons(self.catalog.list()), [])

    def test_dashboard_settings_can_supply_recorded_requested_duration(self):
        def change(value):
            value.pop("requested_duration_seconds")
            value["dashboard_experiment"] = {"settings": {"duration": 5}}
        directory = self.saved(change=change)
        self.assertEqual(self.context(directory)["duration_seconds"], 5)
        self.assertIsNotNone(self.context(directory)["comparison_key"])

    def test_unverified_environment_missing_manifest_and_inputs_stay_visible(self):
        unverified = self.saved(environment="unverified")
        no_manifest = self.saved()
        (no_manifest / "manifest.json").unlink()
        no_inputs = self.saved()
        (no_inputs / "inputs.jsonl").unlink()
        self.assertEqual(len(self.catalog.list()), 3)
        for directory in (unverified, no_manifest, no_inputs):
            self.assertEqual(self.context(directory)["status"], "individual_only")
        self.assertEqual(results.comparisons(self.catalog.list()), [])

    def test_changed_router_interruption_and_hash_disagreement_are_excluded(self):
        def unstable(value):
            value["router_state_after"]["configuration"]["replicas"].pop()
        directories = [self.saved(change=unstable), self.saved(summary={"interrupted": True}),
                       self.saved(summary={"valid_for_comparison": False}),
                       self.saved(change=lambda value: value.update(policy_label_mismatch=True)),
                       self.saved(summary={"source_sha256": "disagrees"})]
        for directory in directories:
            self.assertEqual(self.context(directory)["status"], "excluded")
        self.assertEqual(results.comparisons(self.catalog.list()), [])

    def test_tampered_saved_inputs_cannot_enter_comparison(self):
        directory = self.saved()
        self.assertIsNotNone(self.context(directory)["comparison_key"])
        (directory / "inputs.jsonl").write_text("{}\n")
        self.assertIsNone(self.context(directory)["comparison_key"])
        self.assertIn("verified input snapshot", self.context(directory)["missing"])

    def test_all_failed_policy_and_missing_metrics_are_not_hidden_or_zero_filled(self):
        self.saved()
        directory = self.saved(summary={"latency_p95_seconds": None, "ttft_p95_seconds": None})
        record = json.loads((directory / "summary.json").read_text())
        record.pop("truncated")
        self.write(directory / "summary.json", record)
        self.saved("capacity", summary={"completed": 0, "errors": 10, "dropped": 3,
                                      "latency_p95_seconds": None, "latency_p50_seconds": None,
                                      "ttft_p95_seconds": None, "completed_rps": 0})
        policies = {row["policy"]: row for row in results.comparisons(self.catalog.list())[0]["policies"]}
        self.assertEqual(policies["round_robin"]["runs"], 2)
        self.assertIsNone(policies["round_robin"]["latency_p95_mean_seconds"])
        self.assertIsNone(policies["round_robin"]["truncated"])
        self.assertEqual(policies["capacity"]["errors"], 10)
        self.assertIsNone(policies["capacity"]["latency_p95_mean_seconds"])
        self.assertEqual(policies["capacity"]["completed_rps_mean"], 0)

    def test_gpu_samples_are_per_device_and_missing_is_not_zero(self):
        directory = self.saved(environment="hardware")
        self.gpu(directory)
        context = self.context(directory)
        self.assertEqual(context["endpoint_count"], 2)
        devices = context["gpu"]["devices"]
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0]["samples"], 2)
        self.assertEqual(devices[0]["memory_used_max_mib"], 4096)
        self.assertEqual(devices[0]["utilization_mean_percent"], 50)
        self.assertEqual(devices[0]["utilization_max_percent"], 80)
        (directory / "telemetry.jsonl").write_text(json.dumps({"source": "nvidia-smi", "error": "unavailable"}) + "\n")
        self.assertEqual(self.context(directory)["gpu"]["status"], "unavailable")
        self.assertIsNone(self.context(directory)["comparison_key"])
        self.alter(directory, "manifest.json", gpu_telemetry_requested=False)
        self.assertEqual(self.context(directory)["gpu"]["status"], "disabled")

    def test_different_observed_physical_gpus_do_not_mix(self):
        first = self.saved(environment="hardware")
        second = self.saved("capacity", environment="hardware")
        self.gpu(first, "GPU-a")
        self.gpu(second, "GPU-b")
        self.assertEqual(len(results.comparisons(self.catalog.list())), 2)

    def test_temporary_emulator_ports_can_change_but_capacities_cannot(self):
        first = self.saved()
        def different_port(value):
            for key in ("router_state_before", "router_state_after"):
                value[key]["configuration"]["replicas"][0]["url"] = "http://127.0.0.1:9999"
        second = self.saved("capacity", change=different_port)
        self.assertEqual(self.context(first)["comparison_key"], self.context(second)["comparison_key"])
        def different_capacity(value):
            for key in ("router_state_before", "router_state_after"):
                value[key]["configuration"]["replicas"][0]["capacity"] = 8
        third = self.saved("kubeai", change=different_capacity)
        self.assertNotEqual(self.context(first)["comparison_key"], self.context(third)["comparison_key"])

    def test_catalog_cache_tracks_job_completion_and_protects_saved_artifacts(self):
        directory = self.saved()
        original = (directory / "summary.json").read_bytes()
        self.write(self.root / "job.json", {"status": "running"})
        self.assertEqual(self.context(directory)["status"], "excluded")
        self.write(self.root / "job.json", {"status": "completed"})
        self.assertEqual(self.context(directory)["status"], "matched_settings")
        rows = self.catalog.list()
        rows[0]["context"]["models"].clear()
        self.assertEqual(self.context(directory)["models"], ["model"])
        self.assertEqual((directory / "summary.json").read_bytes(), original)

    def test_missing_router_snapshot_does_not_turn_a_routed_run_into_direct(self):
        def without_router(value):
            value.pop("router_state_before")
            value.pop("router_state_after")
        routed = self.saved(change=without_router)
        self.assertIn("endpoint setup", self.context(routed)["missing"])
        direct = self.saved("direct", change=without_router)
        self.assertEqual(self.context(direct)["target_kind"], "direct")
        self.assertEqual(len(results.comparisons(self.catalog.list())), 1)

    def test_nonfinite_summary_is_skipped_without_breaking_valid_runs(self):
        self.saved()
        self.saved(summary={"latency_p95_seconds": float("nan")})
        rows = self.catalog.list()
        self.assertEqual(len(rows), 1)
        json.dumps(results.comparisons(rows), allow_nan=False)

    def test_replica_breakdown_retains_counts_percentiles_and_unused_endpoints(self):
        directory = self.saved(summary={"requests": 3})
        self.requests(directory, [{"latency_seconds": .1}, {"latency_seconds": .3, "finish_reason": "length"},
                                  {"completed": False, "latency_seconds": 5, "error": "HTTP_500", "finish_reason": ""}])
        details = self.catalog.details(directory.name)["replica_breakdown"]
        self.assertEqual(details["status"], "recorded")
        rows = {row["replica"]: row for row in details["replicas"]}
        self.assertEqual((rows["a"]["requests"], rows["a"]["completed"], rows["a"]["errors"], rows["a"]["truncated"]), (3, 2, 1, 1))
        self.assertEqual(rows["a"]["latency_p50_seconds"], .1)
        self.assertEqual(rows["a"]["latency_p95_seconds"], .3)
        self.assertEqual(rows["a"]["request_share_percent"], 100)
        self.assertEqual(rows["b"]["requests"], 0)
        self.assertEqual(rows["b"]["request_share_percent"], 0)
        self.assertIsNone(rows["b"]["latency_p95_seconds"])

    def test_unassigned_failures_and_drops_are_not_attributed_to_a_replica(self):
        directory = self.saved(summary={"requests": 3})
        self.requests(directory, [{}, {"replica": "b"}, {"replica": "", "completed": False,
                                  "error": "client_inflight_limit", "finish_reason": ""}])
        rows = self.catalog.details(directory.name)["replica_breakdown"]["replicas"]
        self.assertEqual(sum(row["requests"] for row in rows), 3)
        self.assertAlmostEqual(sum(row["request_share_percent"] for row in rows), 100)
        unassigned = next(row for row in rows if row["replica"] is None)
        self.assertEqual((unassigned["requests"], unassigned["errors"], unassigned["dropped"]), (1, 1, 1))
        self.assertIsNone(unassigned["latency_p95_seconds"])

    def test_missing_replica_measurements_are_unknown_with_coverage_counts(self):
        directory = self.saved(summary={"requests": 2})
        self.requests(directory, [{}, {"latency_seconds": "nan", "ttft_seconds": ""}])
        data = self.catalog.details(directory.name)
        row = data["replica_breakdown"]["replicas"][0]
        self.assertEqual((row["latency_samples"], row["ttft_samples"], row["completed"]), (1, 1, 2))
        self.assertIsNone(row["latency_p95_seconds"])
        self.assertIsNone(row["ttft_p95_seconds"])
        json.dumps(data, allow_nan=False)

    def test_missing_or_malformed_request_csv_leaves_summary_visible(self):
        directory = self.saved()
        self.assertEqual(self.catalog.details(directory.name)["replica_breakdown"]["status"], "unavailable")
        (directory / "requests.csv").write_text("replica,completed\na,True\n")
        self.assertEqual(self.catalog.details(directory.name)["replica_breakdown"]["status"], "unavailable")
        self.requests(directory, [{"completed": "unknown"}])
        self.assertEqual(self.catalog.details(directory.name)["replica_breakdown"]["status"], "unavailable")
        self.assertEqual(len(self.catalog.list()), 1)

    def test_linked_and_oversized_request_csv_are_not_read(self):
        directory = self.saved()
        outside = self.root / "outside.csv"
        outside.write_text("not a measurement")
        path = directory / "requests.csv"
        path.symlink_to(outside)
        self.assertEqual(self.catalog.details(directory.name)["replica_breakdown"]["status"], "unavailable")
        path.unlink()
        with path.open("wb") as file:
            file.truncate(64 * 1024 * 1024 + 1)
        self.assertEqual(self.catalog.details(directory.name)["replica_breakdown"]["status"], "unavailable")

    def test_empty_request_csv_keeps_undefined_shares_unknown(self):
        directory = self.saved(summary={"requests": 0})
        self.requests(directory, [])
        data = self.catalog.details(directory.name)["replica_breakdown"]
        self.assertEqual((data["status"], data["rows"]), ("recorded", 0))
        self.assertTrue(all(row["request_share_percent"] is None for row in data["replicas"]))

    def test_detail_cache_refreshes_and_marks_csv_summary_disagreements(self):
        directory = self.saved(summary={"requests": 1})
        self.requests(directory, [{}])
        data = self.catalog.details(directory.name)
        data["replica_breakdown"]["replicas"].clear()
        self.assertEqual(len(self.catalog.details(directory.name)["replica_breakdown"]["replicas"]), 2)
        self.requests(directory, [{}, {"replica": "b"}])
        self.assertEqual(self.catalog.details(directory.name)["replica_breakdown"]["status"], "inconsistent")
        self.alter(directory, "summary.json", requests=2)
        self.assertEqual(self.catalog.details(directory.name)["replica_breakdown"]["status"], "recorded")

    def test_detail_lookup_rejects_paths_and_ambiguous_run_identifiers(self):
        directory = self.saved()
        for name in ("../run-001", str(directory), "absent", "run-001/../run-001"):
            self.assertIsNone(self.catalog.details(name))
        duplicate = self.root / "duplicate" / directory.name
        duplicate.mkdir(parents=True)
        (duplicate / "summary.json").write_bytes((directory / "summary.json").read_bytes())
        self.assertIsNone(self.catalog.details(directory.name))


if __name__ == "__main__":
    unittest.main()
