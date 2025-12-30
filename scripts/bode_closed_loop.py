#!/usr/bin/env python3
"""Generate a closed-loop Bode plot from the current controller config."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import yaml

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None


def _load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _pid_transfer(kp: float, ki: float, kd: float, s: np.ndarray) -> np.ndarray:
    return kp + ki / s + kd * s


def _closed_loop(
    mode: str,
    kp: float,
    ki: float,
    kd: float,
    inertia: float,
    damping: float,
    use_feedforward: bool,
    omega: np.ndarray,
) -> np.ndarray:
    s = 1j * omega
    if mode == "velocity":
        plant = 1.0 / (inertia * s + damping)
        ff = inertia * s + damping if use_feedforward else 0.0
    else:
        plant = 1.0 / (inertia * s * s + damping * s)
        ff = inertia * s * s + damping * s if use_feedforward else 0.0

    controller = _pid_transfer(kp, ki, kd, s)
    open_loop = controller * plant
    closed_loop = (controller + ff) * plant / (1.0 + open_loop)
    return closed_loop


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Closed-loop Bode plot for the outer-loop controller."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "config" / "controller.yaml",
    )
    parser.add_argument("--fmin", type=float, default=0.1, help="Min frequency [Hz].")
    parser.add_argument("--fmax", type=float, default=50.0, help="Max frequency [Hz].")
    parser.add_argument("--points", type=int, default=400, help="Number of log-spaced points.")
    parser.add_argument(
        "--mode",
        choices=["position", "velocity"],
        default=None,
        help="Override command type (default: from config).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional output path for PDF plot.",
    )
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    plant = cfg.get("plant", {}) if isinstance(cfg.get("plant"), dict) else {}
    inertia = float(plant.get("inertia", 0.0))
    damping = float(plant.get("damping", 0.0))

    mode = args.mode or str(cfg.get("command_type", "position")).lower()
    if mode not in {"position", "velocity"}:
        raise ValueError("command_type must be 'position' or 'velocity'.")

    if mode == "velocity":
        pid_cfg = cfg.get("velocity_pid", {}) if isinstance(cfg.get("velocity_pid"), dict) else {}
    else:
        pid_cfg = cfg.get("outer_pid", {}) if isinstance(cfg.get("outer_pid"), dict) else {}

    kp = float(pid_cfg.get("kp", 0.0))
    ki = float(pid_cfg.get("ki", 0.0))
    kd = float(pid_cfg.get("kd", 0.0))
    use_ff = bool(pid_cfg.get("use_feedforward", True))

    if inertia <= 0.0 and damping <= 0.0:
        raise ValueError("plant.inertia or plant.damping must be positive for Bode analysis.")

    freqs = np.logspace(np.log10(args.fmin), np.log10(args.fmax), args.points)
    omega = 2.0 * np.pi * freqs
    response = _closed_loop(mode, kp, ki, kd, inertia, damping, use_ff, omega)
    mag = 20.0 * np.log10(np.maximum(np.abs(response), 1e-12))
    phase = np.unwrap(np.angle(response)) * 180.0 / np.pi

    if plt is None:
        raise RuntimeError("matplotlib is required to plot the Bode diagram.")

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].semilogx(freqs, mag)
    axes[0].set_ylabel("Magnitude [dB]")
    axes[0].grid(True, which="both", linestyle="--", alpha=0.4)

    axes[1].semilogx(freqs, phase)
    axes[1].set_ylabel("Phase [deg]")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].grid(True, which="both", linestyle="--", alpha=0.4)

    title = f"Closed-loop Bode ({mode})"
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))

    if args.save is None:
        fig_dir = Path("fig")
        fig_dir.mkdir(parents=True, exist_ok=True)
        args.save = fig_dir / f"bode_closed_{mode}.pdf"

    fig.savefig(args.save, dpi=300, bbox_inches="tight")
    print(f"Saved Bode plot to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
