#!/usr/bin/env python3
"""Quick diagnostics for tracking error, torque outputs, and saturation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


def _load_csv(path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, str]]:
    metadata: Dict[str, str] = {}
    with path.open("r") as f:
        rows = []
        for line in f:
            if not line.strip():
                continue
            if line.startswith("#"):
                parts = line[1:].split("=", 1)
                if len(parts) == 2:
                    metadata[parts[0].strip()] = parts[1].strip()
                continue
            rows.append(line)
    if not rows:
        raise ValueError(f"{path} has no data rows.")
    reader = csv.DictReader(rows)
    if not reader.fieldnames:
        raise ValueError(f"{path} does not define any header column.")
    columns: Dict[str, list[float]] = {name: [] for name in reader.fieldnames}
    for row in reader:
        for key, value in row.items():
            columns[key].append(float(value))
    series = {name: np.asarray(values, dtype=float) for name, values in columns.items()}
    time = series.get("time")
    if time is None:
        raise KeyError("CSV is missing required column 'time'.")
    return time, series, metadata


def _load_limits(config_path: Optional[Path]) -> Dict[str, float]:
    if config_path is None or yaml is None:
        return {}
    if not config_path.exists():
        return {}
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)
    limits_cfg = cfg.get("torque_limits", {}) if isinstance(cfg, dict) else {}
    return {
        "motor1": float(limits_cfg.get("motor1")) if "motor1" in limits_cfg else None,
        "motor2": float(limits_cfg.get("motor2")) if "motor2" in limits_cfg else None,
    }


def analyze(csv_path: Path, config_path: Optional[Path] = None, show: bool = False) -> Path:
    time, series, metadata = _load_csv(csv_path)
    command_type = metadata.get("CommandType", "position").lower()
    if command_type not in {"position", "velocity"}:
        command_type = "position"
    if command_type == "velocity" and "omega_ref" in series:
        ref = series["omega_ref"]
        feedback = series.get("output_vel", series.get("vel_1"))
        error = ref - feedback
        ref_label = "Velocity [turn/s]"
        error_label = "Velocity error [turn/s]"
    else:
        ref = series["theta_ref"]
        feedback = series.get("output_pos", series.get("pos_1"))
        error = ref - feedback
        ref_label = "Position [turn]"
        error_label = "Position error [turn]"
    tau_pid = series.get("tau_pid", series.get("theta_ctrl"))
    tau_dob = series.get("tau_dob", series.get("tau_out"))
    tau_out = series.get("tau_out", tau_dob)
    tau_1 = series.get("tau_1")
    tau_2 = series.get("tau_2")

    limits = _load_limits(config_path)
    limit1 = limits.get("motor1")
    limit2 = limits.get("motor2")

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Tracking
    axes[0].plot(time, ref, "--", label="command")
    axes[0].plot(time, feedback, "-", label="feedback")
    axes[0].set_ylabel(ref_label)
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Tracking (error shown below)")
    ax_err = axes[0].twinx()
    ax_err.plot(time, error, color="tab:red", alpha=0.4, label="error")
    ax_err.set_ylabel(error_label, color="tab:red")
    ax_err.tick_params(axis="y", labelcolor="tab:red")

    # Torque time series with limits
    axes[1].plot(time, tau_pid, label="tau_pid", color="tab:blue")
    axes[1].plot(time, tau_dob, label="tau_dob", color="tab:orange")
    axes[1].plot(time, tau_out, label="tau_out", color="tab:green", alpha=0.7)
    if tau_1 is not None:
        axes[1].plot(time, tau_1, label="tau_1", color="tab:purple", alpha=0.7)
        if limit1 is not None:
            axes[1].axhline(limit1, color="tab:purple", ls="--", alpha=0.5)
            axes[1].axhline(-limit1, color="tab:purple", ls="--", alpha=0.5)
    if tau_2 is not None:
        axes[1].plot(time, tau_2, label="tau_2", color="tab:brown", alpha=0.7)
        if limit2 is not None:
            axes[1].axhline(limit2, color="tab:brown", ls="--", alpha=0.5)
            axes[1].axhline(-limit2, color="tab:brown", ls="--", alpha=0.5)
    axes[1].set_ylabel("Torque [Nm]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    # Scatter: error vs tau_pid (slope ≈ kp if Ki=0)
    axes[2].scatter(error, tau_pid, s=4, alpha=0.5, label="tau_pid vs error")
    axes[2].set_xlabel(error_label)
    axes[2].set_ylabel("tau_pid [Nm]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    fig.tight_layout()

    out_dir = csv_path.parent / "fig"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{csv_path.stem}_analysis.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize tracking error and torque saturation.")
    parser.add_argument("csv_path", type=Path, help="Path to controller CSV log.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to controller.yaml for torque limits (optional).",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot interactively.")
    args = parser.parse_args()

    out = analyze(args.csv_path, config_path=args.config, show=args.show)
    print(f"Saved analysis plot to {out}")


if __name__ == "__main__":
    main()
