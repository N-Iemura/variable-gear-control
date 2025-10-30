from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class IdentificationResult:
    inertia: float
    damping: float
    residual_norm: float


def lowpass_filter(data: np.ndarray, alpha: float) -> np.ndarray:
    """Simple first-order low-pass filter."""
    filtered = np.empty_like(data, dtype=float)
    acc = float(data[0])
    for idx, value in enumerate(data):
        acc += alpha * (float(value) - acc)
        filtered[idx] = acc
    return filtered


def estimate_inertia_damping(
    time: np.ndarray,
    velocity: np.ndarray,
    torque: np.ndarray,
    filter_alpha: float = 0.1,
) -> IdentificationResult:
    time = np.asarray(time, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    torque = np.asarray(torque, dtype=float)

    dt = np.diff(time)
    if np.any(dt <= 0):
        raise ValueError("Time vector must be strictly increasing.")
    dt_mean = float(np.mean(dt))

    omega = lowpass_filter(velocity, filter_alpha)
    omega_dot = np.gradient(omega, dt_mean)

    Phi = np.column_stack((omega_dot, omega))
    theta, residuals, _, _ = np.linalg.lstsq(Phi, torque, rcond=None)
    inertia, damping = theta
    residual_norm = float(residuals[0]) if residuals.size else float(
        np.linalg.norm(Phi @ theta - torque)
    )
    return IdentificationResult(float(inertia), float(damping), residual_norm)


def identify_from_csv(path: Path, velocity_column: str = "output_vel", torque_column: str = "tau_out") -> IdentificationResult:
    df = pd.read_csv(path, comment="#")
    if "time" not in df.columns:
        raise KeyError("CSV file must contain 'time' column.")
    if velocity_column not in df.columns or torque_column not in df.columns:
        raise KeyError("CSV file missing required columns for identification.")
    return estimate_inertia_damping(df["time"].to_numpy(), df[velocity_column].to_numpy(), df[torque_column].to_numpy())
