#!/usr/bin/env python3
"""Evaluate torque utilization metrics near saturation for a CSV log."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple


def _read_metadata(path: Path) -> Tuple[float, float]:
    limit1 = None
    limit2 = None
    with path.open("r", encoding="utf-8", newline="") as f:
        for line in f:
            if not line.startswith("#"):
                break
            if line.startswith("# TorqueLimits="):
                match = re.search(r"\[([^\]]+)\]", line)
                if match:
                    parts = [p.strip() for p in match.group(1).split(",")]
                    if len(parts) >= 2:
                        try:
                            limit1 = float(parts[0])
                            limit2 = float(parts[1])
                        except ValueError:
                            pass
                break
    return (limit1 or 1.0, limit2 or 1.0)


def _iter_rows(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(line for line in f if line.strip() and not line.startswith("#"))
        for row in reader:
            yield row


def _resolve_value(row: dict, candidates: Iterable[str]) -> float:
    for name in candidates:
        if name in row:
            return float(row[name])
    raise KeyError(f"None of the columns {list(candidates)} were found in the CSV.")


def _time_weighted_mean(time: list[float], values: list[float], mask: list[bool]) -> float:
    if not time:
        return math.nan
    if len(time) == 1:
        return values[0] if mask[0] else math.nan
    total = 0.0
    duration = 0.0
    for i in range(1, len(time)):
        if mask[i] and mask[i - 1]:
            dt = time[i] - time[i - 1]
            if dt <= 0.0:
                continue
            total += 0.5 * dt * (values[i] + values[i - 1])
            duration += dt
    if duration <= 0.0:
        return math.nan
    return total / duration


def evaluate(
    path: Path,
    limits: Optional[Tuple[float, float]] = None,
    pre_min: float = 0.8,
    pre_max: float = 0.98,
    sat_threshold: float = 0.98,
) -> dict:
    limit1, limit2 = limits if limits is not None else _read_metadata(path)
    time: list[float] = []
    mu0: list[float] = []
    mu1: list[float] = []
    mu_max: list[float] = []

    for row in _iter_rows(path):
        t = float(row["time"])
        tau_0 = _resolve_value(row, ("tau_1", "motor0_torque", "motor1_torque"))
        tau_1 = _resolve_value(row, ("tau_2", "motor1_torque", "motor2_torque"))
        u0 = abs(tau_0) / (limit1 or 1.0)
        u1 = abs(tau_1) / (limit2 or 1.0)
        time.append(t)
        mu0.append(u0)
        mu1.append(u1)
        mu_max.append(max(u0, u1))

    if not time:
        raise ValueError(f"{path} does not contain any data rows.")

    mask = [(pre_min <= u < pre_max) for u in mu_max]
    abs_diff = [abs(a - b) for a, b in zip(mu0, mu1)]

    j_early = _time_weighted_mean(time, abs_diff, mask)
    pre_values = [1.0 - u for u, m in zip(mu_max, mask) if m]
    j_margin = min(pre_values) if pre_values else math.nan

    mu0_max = max(mu0)
    mu1_max = max(mu1)
    margin0_min = min(1.0 - u for u in mu0)
    margin1_min = min(1.0 - u for u in mu1)

    t_sat = math.nan
    for t, u in zip(time, mu_max):
        if u >= sat_threshold:
            t_sat = t
            break

    pre_duration = 0.0
    for i in range(1, len(time)):
        if mask[i] and mask[i - 1]:
            dt = time[i] - time[i - 1]
            if dt > 0.0:
                pre_duration += dt

    return {
        "file": str(path),
        "limit1": limit1,
        "limit2": limit2,
        "pre_min": pre_min,
        "pre_max": pre_max,
        "sat_threshold": sat_threshold,
        "pre_duration": pre_duration,
        "j_early": j_early,
        "j_margin": j_margin,
        "t_sat": t_sat,
        "mu0_max": mu0_max,
        "mu1_max": mu1_max,
        "margin0_min": margin0_min,
        "margin1_min": margin1_min,
    }


def _match_input_type(path: Path, input_type: Optional[str]) -> bool:
    if not input_type:
        return True
    name = path.stem.lower()
    if input_type == "chirp":
        return "chirp" in name
    if input_type == "sin":
        return "sine" in name or "sin" in name
    return True


def _collect_csv_paths(csv_args: list[Path], csv_dir: Optional[Path]) -> list[Path]:
    paths: list[Path] = []
    for path in csv_args:
        if path.is_dir():
            paths.extend(sorted(path.glob("*.csv")))
        else:
            paths.append(path)
    if csv_dir is not None:
        paths.extend(sorted(csv_dir.glob("*.csv")))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate torque utilization metrics.")
    parser.add_argument("csv", type=Path, nargs="*", help="Path to a CSV log.")
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Directory that contains CSV logs.",
    )
    parser.add_argument(
        "--input-type",
        choices=("sin", "chirp"),
        default=None,
        help="Filter CSVs by input type in filename.",
    )
    parser.add_argument(
        "--limits",
        type=float,
        nargs=2,
        metavar=("LIMIT1", "LIMIT2"),
        help="Override torque limits for motor1 and motor2.",
    )
    parser.add_argument("--pre-min", type=float, default=0.8, help="Lower bound of pre-sat range.")
    parser.add_argument("--pre-max", type=float, default=0.98, help="Upper bound of pre-sat range.")
    parser.add_argument("--sat-threshold", type=float, default=0.98, help="Saturation threshold.")
    parser.add_argument(
        "--format",
        choices=("text", "csv", "json"),
        default="text",
        help="Output format.",
    )
    args = parser.parse_args()

    paths = _collect_csv_paths(args.csv, args.dir)
    paths = [p for p in paths if _match_input_type(p, args.input_type)]
    if not paths:
        raise SystemExit("No CSV logs matched the given inputs.")

    limits = tuple(args.limits) if args.limits else None
    metrics_list = [
        evaluate(
            path,
            limits=limits,
            pre_min=args.pre_min,
            pre_max=args.pre_max,
            sat_threshold=args.sat_threshold,
        )
        for path in paths
    ]

    if args.format == "json":
        print(json.dumps(metrics_list, indent=2))
        return 0

    if args.format == "csv":
        headers = list(metrics_list[0].keys())
        print(",".join(headers))
        for metrics in metrics_list:
            row = []
            for h in headers:
                value = metrics[h]
                if isinstance(value, float) and math.isnan(value):
                    row.append("")
                else:
                    row.append(str(value))
            print(",".join(row))
        return 0

    for idx, metrics in enumerate(metrics_list):
        if idx:
            print("")
        print(f"file: {metrics['file']}")
        print(f"limits: motor1={metrics['limit1']}, motor2={metrics['limit2']}")
        print(f"pre_range: {metrics['pre_min']} <= mu_max < {metrics['pre_max']}")
        print(f"pre_duration: {metrics['pre_duration']:.6f} s")
        if math.isnan(metrics["j_early"]):
            print("j_early: NaN")
        else:
            print(f"j_early: {metrics['j_early']:.6f}")
        if math.isnan(metrics["j_margin"]):
            print("j_margin: NaN")
        else:
            print(f"j_margin: {metrics['j_margin']:.6f}")
        if math.isnan(metrics["t_sat"]):
            print("t_sat: NaN")
        else:
            print(f"t_sat: {metrics['t_sat']:.6f} s")
        print(f"mu0_max: {metrics['mu0_max']:.6f}")
        print(f"mu1_max: {metrics['mu1_max']:.6f}")
        print(f"margin0_min: {metrics['margin0_min']:.6f}")
        print(f"margin1_min: {metrics['margin1_min']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
