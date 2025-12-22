from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


def min_norm_velocity_split(
    kinematic_matrix: np.ndarray, omega_out: float, weights: np.ndarray
) -> np.ndarray:
    """Weighted minimum-norm velocity split (cost = sum_i w_i * omega_i^2)."""
    A = np.asarray(kinematic_matrix, dtype=float).reshape(1, 2)
    weights = np.asarray(weights, dtype=float).reshape(2)
    if np.any(weights <= 0.0):
        raise ValueError("weights must be strictly positive")
    W_inv = np.diag(1.0 / weights)
    At = A.T
    denom = float(A @ W_inv @ At)
    if denom < 1e-9:
        raise ValueError("Kinematic matrix is ill-conditioned for allocation.")
    scaling = float(omega_out) / denom
    omega = (W_inv @ At * scaling).reshape(2)
    return omega


def _solve_with_limits(
    kinematic_matrix: np.ndarray,
    omega_desired: float,
    velocity_limits: np.ndarray,
    weights: np.ndarray,
    omega_preferred: np.ndarray,
) -> np.ndarray:
    """Project the velocity vector onto the feasible polytope while respecting the equality."""
    A = np.asarray(kinematic_matrix, dtype=float).reshape(1, 2)
    weights = np.asarray(weights, dtype=float).reshape(2)
    omega_preferred = np.asarray(omega_preferred, dtype=float).reshape(2)
    limits = np.asarray(velocity_limits, dtype=float).reshape(2)
    if np.any(limits <= 0.0):
        raise ValueError("Velocity limits must be positive.")

    omega_candidate = min_norm_velocity_split(A, omega_desired, weights)
    if np.all(np.abs(omega_candidate) <= limits + 1e-9):
        return omega_candidate

    # Project along nullspace direction (A @ n = 0).
    a1, a2 = float(A[0, 0]), float(A[0, 1])
    null_vec = np.array([a2, -a1], dtype=float)
    null_norm_sq = float(np.dot(null_vec, null_vec))
    if null_norm_sq < 1e-12:
        return np.clip(omega_candidate, -limits, limits)

    def project(preference: np.ndarray) -> np.ndarray | None:
        alpha_opt = float(np.dot(null_vec, preference - omega_candidate) / null_norm_sq)
        alpha_min, alpha_max = -math.inf, math.inf
        for idx in range(2):
            n_i = null_vec[idx]
            if abs(n_i) < 1e-12:
                if abs(omega_candidate[idx]) <= limits[idx] + 1e-9:
                    continue
                return None
            lower = (-limits[idx] - omega_candidate[idx]) / n_i
            upper = (limits[idx] - omega_candidate[idx]) / n_i
            low, high = (lower, upper) if lower <= upper else (upper, lower)
            alpha_min = max(alpha_min, low)
            alpha_max = min(alpha_max, high)
            if alpha_min > alpha_max:
                return None
        alpha = min(max(alpha_opt, alpha_min), alpha_max)
        projected = omega_candidate + alpha * null_vec
        if np.all(np.abs(projected) <= limits + 1e-9):
            return projected
        return None

    for ref in (omega_preferred, omega_candidate):
        projected = project(ref)
        if projected is not None:
            return projected

    # Fall back to exhaustive search on box corners.
    corners = np.array(
        [
            [limits[0], limits[1]],
            [limits[0], -limits[1]],
            [-limits[0], limits[1]],
            [-limits[0], -limits[1]],
        ],
        dtype=float,
    )
    omega_out_values = corners @ A.T
    idx = int(np.argmin(np.abs(omega_out_values.flatten() - omega_desired)))
    return corners[idx]


class VelocityAllocator:
    """Manages velocity distribution across two motors with safety constraints."""

    def __init__(
        self,
        kinematic_matrix: np.ndarray,
        velocity_limits: Dict[str, float],
        dt: float,
        rate_limits: Dict[str, float] | None = None,
        preferred_motor: str | None = None,
        sign_enforcement: bool = True,
        weight_mode: str = "raw",
        preference_mode: str = "primary",
    ) -> None:
        self.A = np.asarray(kinematic_matrix, dtype=float).reshape(1, 2)
        self.velocity_limits = np.array(
            [
                abs(float(velocity_limits.get("motor1", math.inf))),
                abs(float(velocity_limits.get("motor2", math.inf))),
            ],
            dtype=float,
        )
        self.dt = float(dt)
        rate_limits = rate_limits or {}
        self.rate_limits = np.array(
            [
                max(0.0, float(rate_limits.get("motor1", math.inf))),
                max(0.0, float(rate_limits.get("motor2", math.inf))),
            ],
            dtype=float,
        )
        self.preferred_motor = preferred_motor
        self.sign_enforcement = bool(sign_enforcement)
        self.weight_mode = str(weight_mode).lower()
        self.preference_mode = str(preference_mode).lower()
        self.prev_omega = np.zeros(2, dtype=float)

    def reset(self) -> None:
        self.prev_omega[:] = 0.0

    def allocate(
        self,
        omega_out: float,
        weights: np.ndarray,
        secondary_gain: float = 1.0,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        weights = np.asarray(weights, dtype=float).reshape(2)
        eff_weights = self._effective_weights(weights)
        omega_preferred = min_norm_velocity_split(self.A, omega_out, eff_weights)
        omega_preferred = self._apply_preference(omega_preferred, secondary_gain)
        omega_solution = _solve_with_limits(
            self.A, omega_out, self.velocity_limits, eff_weights, omega_preferred
        )
        omega_solution = self._enforce_rate_limits(omega_solution)
        if self.sign_enforcement:
            omega_solution = self._enforce_sign(omega_solution, omega_out)

        self.prev_omega = omega_solution.copy()
        diagnostics = {
            "omega_out": omega_out,
            "allocated": omega_solution.copy(),
        }
        return omega_solution, diagnostics

    def _apply_preference(self, omega: np.ndarray, secondary_gain: float) -> np.ndarray:
        omega = np.asarray(omega, dtype=float).reshape(2).copy()
        if self.preference_mode != "primary" or self.preferred_motor is None:
            return omega
        secondary_idx = 1 if self.preferred_motor == "motor1" else 0
        gain = max(0.0, min(1.0, float(secondary_gain)))
        omega[secondary_idx] *= gain
        return omega

    def _effective_weights(self, weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=float).reshape(2)
        if np.any(weights <= 0.0):
            raise ValueError("weights must be strictly positive")
        if self.weight_mode != "raw":
            raise ValueError(f"Unsupported weight_mode: {self.weight_mode}")
        return weights

    def _enforce_rate_limits(self, omega: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega, dtype=float).reshape(2).copy()
        delta = omega - self.prev_omega
        max_delta = self.rate_limits * self.dt
        for idx in range(2):
            limit = max_delta[idx]
            if not math.isfinite(limit) or limit <= 0.0:
                continue
            delta[idx] = max(-limit, min(limit, delta[idx]))
        return self.prev_omega + delta

    def _enforce_sign(self, omega: np.ndarray, omega_out: float) -> np.ndarray:
        omega = np.asarray(omega, dtype=float).reshape(2).copy()
        sign_out = math.copysign(1.0, omega_out) if abs(omega_out) > 1e-9 else 0.0
        if sign_out == 0.0:
            return np.zeros_like(omega)
        expected_signs = np.sign(self.A).flatten() * sign_out
        for idx in range(2):
            if expected_signs[idx] == 0.0:
                continue
            omega[idx] = abs(omega[idx]) * expected_signs[idx]
        return omega
