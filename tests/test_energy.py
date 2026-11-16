"""Known power curves and fake NVIDIA subprocesses; never query physical hardware."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from kubeai import benchmark, energy, results


def frame(timestamp, powers, fields=None):
    rows = []
    values = powers.items() if isinstance(powers, dict) else powers
    for identity, watts in values:
        rows.append(f"2026/11/15 00:00:00, {identity}, Test GPU, driver, 8192, 6142, 50, {watts}\n")
    return {"source": "nvidia-smi", "timestamp_unix": timestamp + 1000, "sample_monotonic_seconds": timestamp,
            "fields": fields or benchmark.GPU_FIELDS.split(","), "csv": "".join(rows)}


class EnergyTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {"gpu_telemetry_requested": True, "gpu_energy_measurement": energy.specification(1),
                         "start_monotonic_seconds": 100, "end_monotonic_seconds": 104, "elapsed_seconds": 4}
        self.summary = {"completed": 4, "output_tokens": 100, "token_usage_missing": 0}
        self.frames = [frame(timestamp, {"GPU-a": 50}) for timestamp in (99, 101, 103, 105)]

    def estimate(self, records=None, manifest=None, summary=None):
        report = energy.estimate(self.frames if records is None else records,
                                 self.manifest if manifest is None else manifest,
                                 self.summary if summary is None else summary)
        json.dumps(report, allow_nan=False)
        return report

    def test_constant_power_integrates_duration_and_valid_denominators(self):
        report = self.estimate()
        self.assertEqual(report["status"], "estimated")
        self.assertEqual(report["energy_joules"], 200)
        self.assertEqual(report["power_mean_watts"], 50)
        self.assertEqual(report["power_max_sampled_watts"], 50)
        self.assertEqual(report["coverage_percent"], 100)
        self.assertEqual(report["joules_per_completed_request"], 50)
        self.assertEqual(report["joules_per_1000_output_tokens"], 2000)

    def test_irregular_ramp_clips_boundaries_and_excludes_outside_maxima(self):
        manifest = {**self.manifest, "end_monotonic_seconds": 105, "elapsed_seconds": 5}
        records = [frame(timestamp, {"GPU-a": watts}) for timestamp, watts in ((99, 10), (101, 30), (104, 60), (106, 80))]
        report = self.estimate(records, manifest)
        self.assertEqual(report["energy_joules"], 225)
        self.assertEqual(report["power_mean_watts"], 45)
        self.assertEqual(report["power_max_sampled_watts"], 60)
        self.assertEqual(report["power_samples_in_window"], 2)

    def test_short_window_can_be_bracketed_without_a_sample_inside(self):
        report = self.estimate([frame(99, {"GPU-a": 40}), frame(101, {"GPU-a": 40})],
                               {**self.manifest, "end_monotonic_seconds": 100.1, "elapsed_seconds": .1})
        self.assertAlmostEqual(report["energy_joules"], 4)
        self.assertEqual(report["status"], "estimated")
        self.assertIsNone(report["power_max_sampled_watts"])

    def test_unbracketed_edges_stay_partial_without_extrapolation(self):
        report = self.estimate([frame(timestamp, {"GPU-a": 50}) for timestamp in (101, 102, 103)])
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["coverage_percent"], 50)
        self.assertEqual(report["covered_energy_joules"], 100)
        for key in ("energy_joules", "power_mean_watts", "joules_per_completed_request", "joules_per_1000_output_tokens"):
            self.assertIsNone(report[key])

    def test_long_gaps_are_not_bridged(self):
        report = self.estimate([frame(timestamp, {"GPU-a": 50}) for timestamp in (100, 101, 109, 110)],
                               {**self.manifest, "end_monotonic_seconds": 110, "elapsed_seconds": 10})
        self.assertEqual(report["coverage_percent"], 20)
        self.assertEqual(report["covered_energy_joules"], 100)
        self.assertIsNone(report["energy_joules"])

    def test_invalid_power_breaks_continuity_instead_of_becoming_zero(self):
        for invalid in ("N/A", "nan", "inf", -1):
            with self.subTest(power=invalid):
                report = self.estimate([frame(100, {"GPU-a": 50}), frame(101, {"GPU-a": invalid}),
                                        frame(102, {"GPU-a": 50}), frame(103, {"GPU-a": 50})],
                                       {**self.manifest, "end_monotonic_seconds": 103, "elapsed_seconds": 3})
                self.assertEqual(report["covered_seconds"], 1)
                self.assertEqual(report["covered_energy_joules"], 50)
                self.assertIsNone(report["energy_joules"])

    def test_failed_query_breaks_continuity(self):
        error = {"source": "nvidia-smi", "sample_monotonic_seconds": 101, "error": "TimeoutExpired"}
        report = self.estimate([frame(100, {"GPU-a": 50}), error, frame(102, {"GPU-a": 50}), frame(104, {"GPU-a": 50})])
        self.assertEqual(report["covered_seconds"], 2)
        self.assertIsNone(report["energy_joules"])

    def test_each_physical_uuid_counts_once_including_duplicate_identical_rows(self):
        records = [frame(timestamp, [("GPU-a", 50), ("GPU-a", 50), ("GPU-b", 100)]) for timestamp in (99, 101, 103, 105)]
        report = self.estimate(records)
        self.assertEqual(len(report["devices"]), 2)
        self.assertEqual(report["energy_joules"], 600)
        self.assertEqual(report["power_mean_watts"], 150)
        self.assertEqual([device["energy_joules"] for device in report["devices"]], [200, 400])

    def test_conflicting_duplicate_gpu_rows_do_not_create_a_power_sum(self):
        records = [frame(timestamp, [("GPU-a", 50), ("GPU-a", 100)]) for timestamp in (99, 101, 103, 105)]
        self.assertIsNone(self.estimate(records)["energy_joules"])

    def test_missing_power_on_one_gpu_prevents_a_partial_fleet_total(self):
        records = [frame(timestamp, {"GPU-a": 50, "GPU-b": "N/A"}) for timestamp in (99, 101, 103, 105)]
        report = self.estimate(records)
        self.assertIsNone(report["energy_joules"])
        self.assertEqual(report["devices"][0]["energy_joules"], 200)
        self.assertIsNone(report["devices"][1]["energy_joules"])

    def test_timestamp_corruption_and_inconsistent_windows_are_unavailable(self):
        records = [self.frames, list(reversed(self.frames)), self.frames + [self.frames[-1]],
                   [{key: value for key, value in row.items() if key != "sample_monotonic_seconds"} for row in self.frames]]
        for bad in records[1:]:
            self.assertEqual(self.estimate(bad)["status"], "unavailable")
        self.assertEqual(self.estimate(manifest={**self.manifest, "elapsed_seconds": 8})["status"], "unavailable")

    def test_disabled_legacy_and_unsupported_measurements_do_not_invent_energy(self):
        self.assertEqual(self.estimate(manifest={"gpu_telemetry_requested": False})["status"], "disabled")
        self.assertEqual(self.estimate(manifest={"gpu_telemetry_requested": True})["status"], "unavailable")
        records = [{**row, "fields": benchmark.GPU_BASE_FIELDS.split(","),
                    "csv": "2026/11/15 00:00:00, GPU-a, Test GPU, driver, 8192, 6142, 50\n"} for row in self.frames]
        self.assertIsNone(self.estimate(records)["energy_joules"])

    def test_missing_usage_and_zero_denominators_do_not_produce_normalized_energy(self):
        for change in ({"completed": 0, "output_tokens": 0}, {"completed": None},
                       {"output_tokens": 0}, {"token_usage_missing": 1}, {"token_usage_missing": None}):
            report = self.estimate(summary={**self.summary, **change})
            self.assertEqual(report["energy_joules"], 200)
            self.assertIsNone(report["joules_per_1000_output_tokens"])
        self.assertIsNone(self.estimate(summary={**self.summary, "completed": 0})["joules_per_completed_request"])

    def test_zero_power_is_distinct_from_missing_power(self):
        report = self.estimate([frame(timestamp, {"GPU-a": 0}) for timestamp in (99, 101, 103, 105)])
        self.assertEqual((report["status"], report["energy_joules"], report["joules_per_completed_request"]), ("estimated", 0, 0))

    def test_zero_or_one_sample_never_estimates_full_energy(self):
        for records in ([], [frame(102, {"GPU-a": 50})]):
            report = self.estimate(records)
            self.assertEqual(report["status"], "unavailable")
            self.assertIsNone(report["energy_joules"])

    def test_extreme_corrupt_readings_never_emit_nonfinite_json(self):
        report = self.estimate([frame(timestamp, {"GPU-a": 1e308}) for timestamp in (99, 101, 103, 105)])
        self.assertIsNone(report["energy_joules"])

    def test_comparison_means_require_every_run_to_have_complete_energy(self):
        def run(identity, report):
            return {"run_id": identity, "policy": "kubeai", "context": {"comparison_key": "same",
                    "exploratory_comparison_key": "exploratory-same", "exploratory_missing": ["model revision"], "gpu": {"energy": report}}}
        first = self.estimate()
        second = self.estimate([frame(timestamp, {"GPU-a": 100}) for timestamp in (99, 101, 103, 105)])
        data = [run("a", first), run("b", second)]
        row = results.comparisons(data)[0]["policies"][0]
        self.assertEqual(row["energy_joules_mean"], 300)
        self.assertEqual(row["joules_per_completed_request_mean"], 75)
        data.append(run("c", self.estimate([])))
        row = results.comparisons(data)[0]["policies"][0]
        self.assertEqual(row["energy_runs"], 2)
        self.assertIsNone(row["energy_joules_mean"])
        exploratory = results.comparisons(data, mode="exploratory")[0]["policies"][0]
        self.assertEqual(exploratory["energy_runs"], 2)
        self.assertIsNone(exploratory["energy_joules_mean"])

    def test_viewer_derives_energy_from_raw_data_without_rewriting_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "run-1"
            path.mkdir()
            manifest = {**self.manifest, "mode": "closed_loop", "environment": "hardware"}
            summary = {**self.summary, "policy": "kubeai", "mode": "closed_loop", "environment": "hardware", "gpu_energy": {"energy_joules": 999}}
            (path / "manifest.json").write_text(json.dumps(manifest))
            (path / "summary.json").write_text(json.dumps(summary))
            (path / "telemetry.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.frames))
            original = (path / "summary.json").read_bytes()
            catalog = results.RunCatalog(root)
            self.assertEqual(catalog.list()[0]["context"]["gpu"]["energy"]["energy_joules"], 200)
            (path / "telemetry.jsonl").write_text("".join(json.dumps(frame(timestamp, {"GPU-a": 100})) + "\n" for timestamp in (99, 101, 103, 105)))
            self.assertEqual(catalog.list()[0]["context"]["gpu"]["energy"]["energy_joules"], 400)
            self.assertEqual((path / "summary.json").read_bytes(), original)


class PowerCollectorTests(unittest.TestCase):
    def test_power_query_failure_preserves_base_gpu_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = benchmark.Telemetry(directory, [], gpu=True)
            monitor.gpu_program = "/fake/nvidia-smi"
            with patch("kubeai.benchmark.subprocess.run", side_effect=[
                subprocess.CalledProcessError(1, "fake"), Mock(stdout="base GPU metrics\n")]) as query:
                report = monitor.sample_gpu()
            self.assertIn("power.draw", query.call_args_list[0].args[0][1])
            self.assertNotIn("power.draw", query.call_args_list[1].args[0][1])
            self.assertEqual(report["fields"], benchmark.GPU_BASE_FIELDS.split(","))
            self.assertEqual(report["power_query_error"], "CalledProcessError")
            self.assertGreater(report["sample_monotonic_seconds"], 0)

    def test_sampling_disabled_never_calls_nvidia(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(benchmark.Telemetry, "sample_gpu", side_effect=AssertionError("GPU must not be queried")):
                with benchmark.Telemetry(directory, [], gpu=False):
                    pass


if __name__ == "__main__":
    unittest.main()
