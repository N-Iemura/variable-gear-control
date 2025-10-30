from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class PositionCommand:
    """Reference signals for the outer-loop position controller."""

    position: float
    velocity: float = 0.0
    acceleration: float = 0.0


@dataclass
class PositionFeedback:
    """Measured signals required by the position controller."""

    position: float
    velocity: float


class PositionController:
    """PID + feedforward outer-loop controller for output torque generation."""

    def __init__(
        self,
        gains: Dict[str, float],
        plant_inertia: float,
        plant_damping: float,
        dt: float,
        derivative_mode: str = "error",
        derivative_filter_alpha: float = 1.0,
    ) -> None:
        self.kp = float(gains.get("kp", 0.0))
        self.ki = float(gains.get("ki", 0.0))
        self.kd = float(gains.get("kd", 0.0))
        self.max_output = abs(float(gains.get("max_output", math.inf)))
        self.inertia = float(plant_inertia)
        self.damping = float(plant_damping)
        self.dt = float(dt)
        self.derivative_mode = derivative_mode.lower()
        if self.derivative_mode not in {"error", "measurement"}:
            raise ValueError("derivative_mode must be 'error' or 'measurement'")
        self.alpha = max(0.0, min(1.0, float(derivative_filter_alpha)))

        self.integral = 0.0
        self.integral_limit = (
            self.max_output / max(abs(self.ki), 1e-9) if self.ki != 0.0 else math.inf
        )
        self.prev_error = 0.0
        self.prev_feedback_vel = 0.0
        self.derivative_state = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_feedback_vel = 0.0
        self.derivative_state = 0.0

    def update(
        self, command: PositionCommand, feedback: PositionFeedback
    ) -> Tuple[float, Dict[str, float]]:
        error = float(command.position - feedback.position)
        error_rate = self._compute_derivative(error, feedback.velocity, command.velocity)

        # Integrator with clamping for basic anti-wind-up.
        self.integral += error * self.dt
        if self.integral_limit < math.inf:
            self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))

        feedforward = self.inertia * float(command.acceleration) + self.damping * float(
            command.velocity
        )

        unsaturated = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * self.derivative_state
            + feedforward
        )
        saturated = self._saturate(unsaturated)

        # Simple anti-wind-up: if saturated, back-calculate integral correction.
        if self.ki != 0.0 and saturated != unsaturated:
            excess = saturated - unsaturated
            self.integral += excess / self.ki
            if self.integral_limit < math.inf:
                self.integral = max(
                    -self.integral_limit, min(self.integral_limit, self.integral)
                )

        diagnostics = {
            "error": error,
            "error_rate": error_rate,
            "integral": self.integral,
            "derivative": self.derivative_state,
            "feedforward": feedforward,
            "unsaturated_output": unsaturated,
            "output": saturated,
        }
        return saturated, diagnostics

    def _compute_derivative(
        self, error: float, feedback_velocity: float, command_velocity: float
    ) -> float:
        if self.derivative_mode == "measurement":
            raw = float(command_velocity - feedback_velocity)
        else:
            raw = (error - self.prev_error) / self.dt
            self.prev_error = error

        # First order low-pass filter.
        self.derivative_state += self.alpha * (raw - self.derivative_state)
        self.prev_feedback_vel = feedback_velocity
        return raw

    def _saturate(self, value: float) -> float:
        if not math.isfinite(self.max_output):
            return value
        return max(-self.max_output, min(self.max_output, value))
