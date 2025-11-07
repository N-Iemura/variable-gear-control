#!/usr/bin/env python3
"""Quick plotting utility for controller CSV logs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _read_log(path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", newline="") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            rows.append(line)
    if not rows:
        raise ValueError(f"{path} does not contain any data rows.")

    reader = csv.DictReader(rows)
    columns: Dict[str, List[float]] = {name: [] for name in reader.fieldnames or []}
    for row in reader:
        for key, value in row.items():
            columns[key].append(float(value))

    time = np.asarray(columns["time"], dtype=float)
    series = {name: np.asarray(values, dtype=float) for name, values in columns.items()}
    return time, series


def _resolve_series(series: Dict[str, np.ndarray], candidates: Iterable[str]) -> Tuple[str, np.ndarray]:
    for name in candidates:
        if name in series:
            return name, series[name]
    raise KeyError(f"None of the columns {list(candidates)} were found in the CSV.")


def plot_csv(csv_path: Path, save_path: Path | None = None, show: bool = False) -> Path:
    time, series = _read_log(csv_path)

    _, theta_ref = _resolve_series(series, ("theta_ref",))
    _, output_pos = _resolve_series(series, ("output_pos",))
    _, output_vel = _resolve_series(series, ("output_vel",))
    name_tau1, tau_1 = _resolve_series(series, ("tau_1", "motor1_torque", "motor0_torque"))
    name_tau2, tau_2 = _resolve_series(series, ("tau_2", "motor2_torque", "motor1_torque"))

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(time, theta_ref * 360.0, "--", label="θ_ref [deg]")
    axes[0].plot(time, output_pos * 360.0, "-", label="θ_out [deg]")
    axes[0].set_ylabel("Position [deg]")
    axes[0].legend(loc="upper right")

    axes[1].plot(time, output_vel * 360.0, "-", label="ω_out [deg/s]")
    axes[1].set_ylabel("Velocity [deg/s]")
    axes[1].legend(loc="upper right")

    label_tau1 = "τ_1" if name_tau1.startswith("tau") else ("τ_motor1" if "motor1" in name_tau1 else "τ_motor0")
    label_tau2 = "τ_2" if name_tau2.startswith("tau") else ("τ_motor2" if "motor2" in name_tau2 else "τ_motor1")
    axes[2].plot(time, tau_1, label=label_tau1)
    axes[2].plot(time, tau_2, label=label_tau2)
    axes[2].set_ylabel("Torque [Nm]")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend(loc="upper right")

    for ax in axes:
        ax.grid(False)
        ax.tick_params(axis="both", direction="in", length=6, width=0.8)

    fig.tight_layout()

    if save_path is None:
        save_path = csv_path.with_suffix(csv_path.suffix + "_plot.pdf")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot controller CSV logs.")
    parser.add_argument("csv_path", type=Path, help="Path to a generated CSV log.")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the figure (default: <csv>_plot.pdf).",
    )
    parser.add_argument("--show", action="store_true", help="Display the figure interactively.")
    args = parser.parse_args()

    output_path = plot_csv(args.csv_path, save_path=args.save, show=args.show)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
