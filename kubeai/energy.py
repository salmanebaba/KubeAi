"""Gross GPU energy estimates from timestamped board-power samples."""

import csv
import io
import math


METHOD = "trapezoidal-monotonic-v1"
SCOPE = "all physical GPUs on the benchmark host"


def number(value):
    try:
        parsed = float(value)
        return parsed if not isinstance(value, bool) and math.isfinite(parsed) and parsed >= 0 else None
    except (ValueError, TypeError, OverflowError):
        return None


def specification(interval):
    return {"method": METHOD, "scope": SCOPE, "interval_seconds": interval,
            "max_gap_seconds": max(3 * interval, 1.0), "boundary_samples": True}


def integrate(points, start, end, max_gap):
    """Clip valid adjacent sample pairs to the measurement window; never extrapolate."""
    covered, joules = 0.0, 0.0
    for (left, first), (right, last) in zip(points, points[1:]):
        if first is None or last is None or not 0 < right - left <= max_gap:
            continue
        low, high = max(start, left), min(end, right)
        if high <= low:
            continue
        first_clipped = first + (last - first) * (low - left) / (right - left)
        last_clipped = first + (last - first) * (high - left) / (right - left)
        covered += high - low
        joules += (first_clipped / 2 + last_clipped / 2) * (high - low)
    duration = end - start
    valid = math.isfinite(joules)
    complete = valid and covered > 0 and abs(covered - duration) <= max(1e-7, duration * 1e-7)
    inside = [power for timestamp, power in points if start <= timestamp <= end and power is not None]
    return {"status": "estimated" if complete else "partial" if covered > 0 and valid else "unavailable",
            "covered_seconds": covered, "coverage_percent": min(100.0, 100 * covered / duration),
            "covered_energy_joules": joules if covered > 0 and valid else None,
            "energy_joules": joules if complete else None,
            "power_mean_watts": joules / duration if complete else None,
            "power_max_sampled_watts": max(inside) if inside else None,
            "power_samples_in_window": len(inside)}


def estimate(records, manifest, summary):
    requested = manifest.get("gpu_telemetry_requested")
    report = {"status": "disabled" if requested is False else "unavailable", "method": METHOD,
              "scope": SCOPE, "duration_seconds": None, "covered_seconds": 0.0, "coverage_percent": None,
              "energy_joules": None, "covered_energy_joules": None, "power_mean_watts": None,
              "power_max_sampled_watts": None, "power_samples_in_window": 0,
              "joules_per_completed_request": None, "joules_per_1000_output_tokens": None, "devices": [],
              "message": "GPU power sampling was disabled." if requested is False else "GPU power samples are unavailable."}
    if requested is not True:
        return report
    start, end = number(manifest.get("start_monotonic_seconds")), number(manifest.get("end_monotonic_seconds"))
    spec = manifest.get("gpu_energy_measurement") or {}
    if not isinstance(spec, dict):
        spec = {}
    max_gap = number(spec.get("max_gap_seconds"))
    if spec.get("method") != METHOD or start is None or end is None or end <= start or not max_gap:
        report["message"] = "This run has no recorded GPU energy measurement window and method. Historical power is not inferred."
        return report
    report.update(duration_seconds=end - start, coverage_percent=0.0, max_gap_seconds=max_gap)
    elapsed = number(manifest.get("elapsed_seconds"))
    if elapsed is not None and not math.isclose(elapsed, end - start, rel_tol=1e-6, abs_tol=1e-6):
        report["message"] = "Energy and benchmark timing windows disagree."
        return report
    frames, identities = [], set()
    invalid_clock, previous = False, None
    for record in records:
        if not isinstance(record, dict) or record.get("source") != "nvidia-smi":
            continue
        timestamp = number(record.get("sample_monotonic_seconds"))
        if timestamp is None or previous is not None and timestamp <= previous:
            invalid_clock = True
            continue
        previous = timestamp
        powers = {}
        try:
            fields = record.get("fields", [])
            if record.get("csv") and "uuid" in fields:
                for row in csv.DictReader(io.StringIO(record["csv"]), fieldnames=fields):
                    identity = (row.get("uuid") or "").strip()
                    if not identity:
                        continue
                    value = number(row.get("power.draw"))
                    # Repeated identical UUID rows count once; conflicting rows invalidate that sample.
                    powers[identity] = value if identity not in powers or powers[identity] == value else None
                    identities.add(identity)
        except (ValueError, TypeError, csv.Error):
            powers = {}
        frames.append((timestamp, powers))
    if invalid_clock:
        report["message"] = "GPU sample timestamps are missing, repeated or out of order."
        return report
    if not identities:
        report["message"] = "No physical GPU identities were recorded for energy estimation."
        return report
    if not any(value is not None for _, powers in frames for value in powers.values()):
        report["message"] = "Board power was unsupported or missing; memory/utilization samples are not power measurements."
        return report
    for identity in sorted(identities):
        points = [(timestamp, powers.get(identity)) for timestamp, powers in frames]
        report["devices"].append({"uuid": identity, **integrate(points, start, end, max_gap)})
    # An interval contributes to the total only when every observed physical GPU has valid power.
    total = [(timestamp, number(sum(powers[identity] for identity in identities))
              if all(powers.get(identity) is not None for identity in identities) else None)
             for timestamp, powers in frames]
    report.update(integrate(total, start, end, max_gap))
    if report["status"] == "estimated":
        report["message"] = "Gross GPU energy estimated over the complete benchmark window. Includes GPU idle/other-process power; no system CPU/RAM or per-replica attribution."
        completed = number(summary.get("completed"))
        tokens = number(summary.get("output_tokens"))
        if completed and completed.is_integer():
            report["joules_per_completed_request"] = report["energy_joules"] / completed
        if (completed and completed.is_integer() and tokens and tokens.is_integer()
                and type(summary.get("token_usage_missing")) is int and summary["token_usage_missing"] == 0):
            report["joules_per_1000_output_tokens"] = number(report["energy_joules"] / tokens * 1000)
    else:
        report["message"] = "Power coverage is incomplete. Full-window energy and normalized energy are unavailable; covered energy is only a partial observation."
    return report
