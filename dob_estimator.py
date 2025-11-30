from __future__ import annotations

import math
from typing import Dict, Tuple


class DisturbanceObserver:
    """Simple discrete-time disturbance observer for torque estimation."""

    def __init__(
        self,
        inertia: float,
        damping: float,
        dt: float,
        cutoff_hz: float,
        use_damping: bool = True,
    ) -> None:
        self.inertia = float(inertia)
        self.damping = float(damping)
        self.dt = float(dt)
        self.cutoff_hz = float(cutoff_hz)
        self.alpha = 1.0 - math.exp(-2.0 * math.pi * self.cutoff_hz * self.dt)
        self.use_damping = bool(use_damping)

        self.prev_velocity = 0.0
        self.prev_applied_torque = 0.0
        self.estimate = 0.0

    def reset(self) -> None:
        self.prev_velocity = 0.0
        self.prev_applied_torque = 0.0
        self.estimate = 0.0

    def update(self, velocity: float, torque_command: float) -> Tuple[float, Dict[str, float]]:
        omega = float(velocity)
        torque_command = float(torque_command)
        omega_dot = (omega - self.prev_velocity) / self.dt
        self.prev_velocity = omega

        damping_term = self.damping * omega if self.use_damping else 0.0
        equivalent_torque = self.inertia * omega_dot + damping_term

        # Disturbance is the torque required to explain the measured acceleration
        # minus the torque we actually applied on the previous cycle.
        raw_disturbance = equivalent_torque - self.prev_applied_torque

        self.estimate += self.alpha * (raw_disturbance - self.estimate)
        augmented_torque = torque_command - self.estimate
        self.prev_applied_torque = augmented_torque

        diagnostics = {
            "omega_dot": omega_dot,
            "equivalent_torque": equivalent_torque,
            "raw_disturbance": raw_disturbance,
            "filtered_disturbance": self.estimate,
            "augmented_torque": augmented_torque,
            "use_damping": self.use_damping,
        }
        return augmented_torque, diagnostics
