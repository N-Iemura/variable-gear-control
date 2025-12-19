#!/usr/bin/env python3
"""Plot torque command/measurement and utilization against limits."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def _load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _torque_limits(controller_cfg: Dict[str, object]) -> tuple[float, float]:
    limits = controller_cfg.get("torque_limits", {}) if isinstance(controller_cfg, dict) else {}
    m1 = float(limits.get("motor1", 1.0))
    m2 = float(limits.get("motor2", 1.0))
    return abs(m1), abs(m2)


def _default_output_path(input_csv: Path) -> Path:
    return input_csv.with_name(input_csv.stem + "_torque_margin.pdf")


def _maybe_get(
    df: pd.DataFrame, columns: Sequence[str]
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if all(col in df.columns for col in columns):
        values = df[list(columns)].to_numpy(dtype=float)
        if np.any(np.abs(values) > 1e-9):
            return values[:, 0], values[:, 1]
    return None


def plot_margin(
    df: pd.DataFrame,
    limits: tuple[float, float],
    output_path: Path,
    show: bool = False,
) -> None:
    time = df["time"].to_numpy(dtype=float)
    tau_cmd = df[["tau_1", "tau_2"]].to_numpy(dtype=float)
    tau_meas = _maybe_get(df, ("tau_meas_1", "tau_meas_2"))

    limit1, limit2 = limits
    limits_vec = np.array([limit1, limit2], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(time, tau_cmd[:, 0], label="tau_1 (cmd)", color="tab:blue")
    axes[0].plot(time, tau_cmd[:, 1], label="tau_2 (cmd)", color="tab:red")
    if tau_meas is not None:
        axes[0].plot(time, tau_meas[0], "--", label="tau_1 (meas)", color="tab:blue", alpha=0.7)
        axes[0].plot(time, tau_meas[1], "--", label="tau_2 (meas)", color="tab:red", alpha=0.7)
    axes[0].axhline(limit1, color="k", linestyle=":", linewidth=1, label="limit1")
    axes[0].axhline(-limit1, color="k", linestyle=":", linewidth=1)
    axes[0].axhline(limit2, color="gray", linestyle=":", linewidth=1, label="limit2")
    axes[0].axhline(-limit2, color="gray", linestyle=":", linewidth=1)
    axes[0].set_ylabel("Torque [Nm]")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    util_cmd = np.abs(tau_cmd) / limits_vec
    axes[1].plot(time, util_cmd[:, 0], label="motor1 utilization", color="tab:blue")
    axes[1].plot(time, util_cmd[:, 1], label="motor2 utilization", color="tab:red")
    if tau_meas is not None:
        util_meas = np.abs(np.vstack(tau_meas).T) / limits_vec
        axes[1].plot(
            time,
            util_meas[:, 0],
            "--",
            color="tab:blue",
            alpha=0.7,
            label="motor1 utilization (meas)",
        )
        axes[1].plot(
            time,
            util_meas[:, 1],
            "--",
            color="tab:red",
            alpha=0.7,
            label="motor2 utilization (meas)",
        )
    axes[1].axhline(1.0, color="k", linestyle=":", linewidth=1, label="limit")
    axes[1].set_ylabel("Utilization |tau|/limit")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot torque headroom relative to limits from a control log CSV.",
    )
    parser.add_argument("csv", type=Path, help="CSV file produced by the controller/logger.")
    parser.add_argument(
        "--controller",
        type=Path,
        default=Path("config/controller.yaml"),
        help="Controller config (torque_limits).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PDF path (default: <csv>_torque_margin.pdf).",
    )
    parser.add_argument("--show", action="store_true", help="Show plot window as well.")
    args = parser.parse_args()

    controller_cfg = _load_yaml(args.controller)
    limits = _torque_limits(controller_cfg)
    df = pd.read_csv(args.csv, comment="#")
    output_path = args.output if args.output is not None else _default_output_path(args.csv)

    plot_margin(df, limits, output_path, show=args.show)
    print(f"Saved torque margin plot to {output_path}")


if __name__ == "__main__":
    main()
