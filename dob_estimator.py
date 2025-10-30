from __future__ import annotations

import math
from typing import Dict, Tuple


class DisturbanceObserver:
    """Simple discrete-time disturbance observer for torque estimation."""

    def __init__(self, inertia: float, damping: float, dt: float, cutoff_hz: float) -> None:
        self.inertia = float(inertia)
        self.damping = float(damping)
        self.dt = float(dt)
        self.cutoff_hz = float(cutoff_hz)
        self.alpha = 1.0 - math.exp(-2.0 * math.pi * self.cutoff_hz * self.dt)

        self.prev_velocity = 0.0
        self.estimate = 0.0

    def reset(self) -> None:
        self.prev_velocity = 0.0
        self.estimate = 0.0

    def update(self, velocity: float, torque_command: float) -> Tuple[float, Dict[str, float]]:
        omega = float(velocity)
        torque_command = float(torque_command)
        omega_dot = (omega - self.prev_velocity) / self.dt
        self.prev_velocity = omega

        equivalent_torque = self.inertia * omega_dot + self.damping * omega
        raw_disturbance = equivalent_torque - torque_command

        self.estimate += self.alpha * (raw_disturbance - self.estimate)
        augmented_torque = torque_command - self.estimate

        diagnostics = {
            "omega_dot": omega_dot,
            "equivalent_torque": equivalent_torque,
            "raw_disturbance": raw_disturbance,
            "filtered_disturbance": self.estimate,
            "augmented_torque": augmented_torque,
        }
        return augmented_torque, diagnostics
