#!/usr/bin/env python3
"""
Compare torque utilization between two CSV logs.
Plots torque waveforms and utilization ratios.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 14,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
    }
)


def _read_log(path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    import csv
    rows = []
    with path.open("r", newline="") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            rows.append(line)
    
    reader = csv.DictReader(rows)
    columns = {name: [] for name in reader.fieldnames or []}
    for row in reader:
        for key, value in row.items():
            columns[key].append(float(value))

    time = np.asarray(columns["time"], dtype=float)
    series = {name: np.asarray(values, dtype=float) for name, values in columns.items()}
    return time, series


def plot_comparison(
    csv_raw: Path,
    csv_prop: Path,
    limits: Tuple[float, float] = (2.0, 0.5),
    save_path: Path | None = None
) -> None:
    t_raw, s_raw = _read_log(csv_raw)
    t_prop, s_prop = _read_log(csv_prop)

    # Calculate Utilization
    lim1, lim2 = limits
    
    # Raw (Fixed)
    tau1_raw = np.abs(s_raw["tau_1"])
    tau2_raw = np.abs(s_raw["tau_2"])
    util1_raw = tau1_raw / lim1 * 100.0
    util2_raw = tau2_raw / lim2 * 100.0
    
    # Proposed (Utilization)
    tau1_prop = np.abs(s_prop["tau_1"])
    tau2_prop = np.abs(s_prop["tau_2"])
    util1_prop = tau1_prop / lim1 * 100.0
    util2_prop = tau2_prop / lim2 * 100.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    
    # Plot 1: Torque Waveforms (Raw)
    ax = axes[0, 0]
    ax.set_title("Torque (Fixed Ratio / Conventional)")
    ax.plot(t_raw, s_raw["tau_1"], label="Motor1 (Limit=2.0)")
    ax.plot(t_raw, s_raw["tau_2"], label="Motor2 (Limit=0.5)")
    ax.axhline(lim1, color="r", linestyle="--", alpha=0.3)
    ax.axhline(-lim1, color="r", linestyle="--", alpha=0.3)
    ax.axhline(lim2, color="g", linestyle="--", alpha=0.3)
    ax.axhline(-lim2, color="g", linestyle="--", alpha=0.3)
    ax.set_ylabel("Torque [Nm]")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Plot 2: Torque Waveforms (Proposed)
    ax = axes[0, 1]
    ax.set_title("Torque (Utilization Opt / Proposed)")
    ax.plot(t_prop, s_prop["tau_1"], label="Motor1")
    ax.plot(t_prop, s_prop["tau_2"], label="Motor2")
    ax.axhline(lim1, color="r", linestyle="--", alpha=0.3)
    ax.axhline(-lim1, color="r", linestyle="--", alpha=0.3)
    ax.axhline(lim2, color="g", linestyle="--", alpha=0.3)
    ax.axhline(-lim2, color="g", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Plot 3: Utilization (Raw)
    ax = axes[1, 0]
    ax.set_title("Utilization % (Fixed Ratio)")
    ax.plot(t_raw, util1_raw, label="Motor1 Util")
    ax.plot(t_raw, util2_raw, label="Motor2 Util")
    ax.axhline(100, color="k", linestyle="--", label="Limit")
    ax.set_ylabel("Utilization [%]")
    ax.set_xlabel("Time [s]")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Plot 4: Utilization (Proposed)
    ax = axes[1, 1]
    ax.set_title("Utilization % (Proposed)")
    ax.plot(t_prop, util1_prop, label="Motor1 Util")
    ax.plot(t_prop, util2_prop, label="Motor2 Util")
    ax.axhline(100, color="k", linestyle="--", label="Limit")
    ax.set_xlabel("Time [s]")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare torque utilization.")
    parser.add_argument("csv_raw", type=Path, help="Path to raw/fixed log CSV")
    parser.add_argument("csv_prop", type=Path, help="Path to proposed/utilization log CSV")
    parser.add_argument("--out", type=Path, default=Path("fig/comparison_utilization.pdf"), help="Output PDF path")
    args = parser.parse_args()
    
    plot_comparison(args.csv_raw, args.csv_prop, save_path=args.out)
