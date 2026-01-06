from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


def min_norm_torque_split(
    mechanism_matrix: np.ndarray, tau_out: float, weights: np.ndarray
) -> np.ndarray:
    """Weighted minimum-norm torque split (cost = sum_i w_i * tau_i^2)."""
    A = np.asarray(mechanism_matrix, dtype=float).reshape(1, 2)
    weights = np.asarray(weights, dtype=float).reshape(2)
    if np.any(weights <= 0.0):
        raise ValueError("weights must be strictly positive")
    W_inv = np.diag(1.0 / weights)
    At = A.T
    denom = float(A @ W_inv @ At)
    if denom < 1e-9:
        raise ValueError("Mechanism matrix is ill-conditioned for allocation.")
    scaling = float(tau_out) / denom
    tau = (W_inv @ At * scaling).reshape(2)
    return tau


def _solve_with_limits(
    mechanism_matrix: np.ndarray,
    tau_desired: float,
    torque_limits: np.ndarray,
    weights: np.ndarray,
    tau_preferred: np.ndarray,
) -> np.ndarray:
    """Project the torque vector onto the feasible polytope while respecting the equality."""
    A = np.asarray(mechanism_matrix, dtype=float).reshape(1, 2)
    weights = np.asarray(weights, dtype=float).reshape(2)
    tau_preferred = np.asarray(tau_preferred, dtype=float).reshape(2)
    limits = np.asarray(torque_limits, dtype=float).reshape(2)
    if np.any(limits <= 0.0):
        raise ValueError("Torque limits must be positive.")

    tau_candidate = min_norm_torque_split(A, tau_desired, weights)
    if np.all(np.abs(tau_candidate) <= limits + 1e-9):
        return tau_candidate

    # Project along nullspace direction (A @ n = 0).
    a1, a2 = float(A[0, 0]), float(A[0, 1])
    null_vec = np.array([a2, -a1], dtype=float)
    null_norm_sq = float(np.dot(null_vec, null_vec))
    if null_norm_sq < 1e-12:
        return np.clip(tau_candidate, -limits, limits)

    def project(preference: np.ndarray) -> np.ndarray | None:
        alpha_opt = float(np.dot(null_vec, preference - tau_candidate) / null_norm_sq)
        alpha_min, alpha_max = -math.inf, math.inf
        for idx in range(2):
            n_i = null_vec[idx]
            if abs(n_i) < 1e-12:
                if abs(tau_candidate[idx]) <= limits[idx] + 1e-9:
                    continue
                return None
            lower = (-limits[idx] - tau_candidate[idx]) / n_i
            upper = (limits[idx] - tau_candidate[idx]) / n_i
            low, high = (lower, upper) if lower <= upper else (upper, lower)
            alpha_min = max(alpha_min, low)
            alpha_max = min(alpha_max, high)
            if alpha_min > alpha_max:
                return None
        alpha = min(max(alpha_opt, alpha_min), alpha_max)
        projected = tau_candidate + alpha * null_vec
        if np.all(np.abs(projected) <= limits + 1e-9):
            return projected
        return None

    for ref in (tau_preferred, tau_candidate):
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
    tau_out_values = corners @ A.T
    idx = int(np.argmin(np.abs(tau_out_values.flatten() - tau_desired)))
    return corners[idx]


class TorqueAllocator:
    """Manages torque distribution across two motors with safety constraints."""

    def __init__(
        self,
        mechanism_matrix: np.ndarray,
        torque_limits: Dict[str, float],
        dt: float,
        rate_limits: Dict[str, float] | None = None,
        preferred_motor: str | None = None,
        sign_enforcement: bool = True,
        weight_mode: str = "raw",
        preference_mode: str = "primary",
        dynamic_utilization: bool = True,
        weight_filter_tau: float | None = None,
        weight_filter_alpha: float | None = None,
    ) -> None:
        self.A = np.asarray(mechanism_matrix, dtype=float).reshape(1, 2)
        self.torque_limits = np.array(
            [abs(float(torque_limits.get("motor1", 1.0))), abs(float(torque_limits.get("motor2", 1.0)))],
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
        self.dynamic_utilization = bool(dynamic_utilization)
        self.weight_filter_alpha = self._resolve_weight_filter_alpha(
            weight_filter_alpha, weight_filter_tau
        )
        self.filtered_weights: np.ndarray | None = None
        self.prev_tau = np.zeros(2, dtype=float)

    def reset(self) -> None:
        self.prev_tau[:] = 0.0
        self.filtered_weights = None

    def allocate(
        self,
        tau_out: float,
        weights: np.ndarray,
        secondary_gain: float = 1.0,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        weights = np.asarray(weights, dtype=float).reshape(2)
        eff_weights = self._effective_weights(weights)
        eff_weights = self._smooth_weights(eff_weights)
        tau_preferred = min_norm_torque_split(self.A, tau_out, eff_weights)
        tau_preferred = self._apply_preference(tau_preferred, secondary_gain)
        tau_solution = _solve_with_limits(
            self.A, tau_out, self.torque_limits, eff_weights, tau_preferred
        )
        tau_solution = self._enforce_rate_limits(tau_solution)
        if self.sign_enforcement:
            tau_solution = self._enforce_sign(tau_solution, tau_out)

        self.prev_tau = tau_solution.copy()
        diagnostics = {
            "tau_out": tau_out,
            "allocated": tau_solution.copy(),
        }
        return tau_solution, diagnostics

    def _apply_preference(self, tau: np.ndarray, secondary_gain: float) -> np.ndarray:
        tau = np.asarray(tau, dtype=float).reshape(2).copy()
        if self.preference_mode != "primary" or self.preferred_motor is None:
            return tau
        secondary_idx = 1 if self.preferred_motor == "motor1" else 0
        gain = max(0.0, min(1.0, float(secondary_gain)))
        tau[secondary_idx] *= gain
        return tau

    def _effective_weights(self, weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=float).reshape(2)
        if np.any(weights <= 0.0):
            raise ValueError("weights must be strictly positive")
        
        if self.weight_mode == "torque_utilization":
            if np.any(self.torque_limits <= 0.0):
                raise ValueError("Torque limits must be positive for utilization weighting.")
            
            base_scale = self.torque_limits ** 2

            if not self.dynamic_utilization:
                return weights / base_scale

            # Calculate current utilization based on previous torque command
            # util = |tau| / limit
            util = np.abs(self.prev_tau) / (self.torque_limits + 1e-9)
            # Clip utilization to [0, 0.99] to avoid division by zero in barrier
            util = np.clip(util, 0.0, 0.99)
            
            # Barrier function: weight increases as utilization approaches 1.0
            # Base weight: 1 / Limit^2 (Standard utilization balancing)
            # Dynamic factor: 1 / (1 - util)^2
            barrier_factor = 1.0 / ((1.0 - util) ** 2 + 1e-6)
            
            return weights / base_scale * barrier_factor
            
        return weights

    def _resolve_weight_filter_alpha(
        self, alpha: float | None, tau: float | None
    ) -> float | None:
        if alpha is not None:
            alpha = float(alpha)
            if not math.isfinite(alpha):
                raise ValueError("weight_filter_alpha must be finite")
            if alpha <= 0.0:
                return None
            return min(1.0, alpha)
        if tau is None:
            return None
        tau = float(tau)
        if not math.isfinite(tau):
            raise ValueError("weight_filter_tau must be finite")
        if tau <= 0.0:
            return None
        return self.dt / (tau + self.dt)

    def _smooth_weights(self, weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=float).reshape(2)
        alpha = self.weight_filter_alpha
        if alpha is None:
            return weights
        if self.filtered_weights is None:
            self.filtered_weights = weights.copy()
            return weights
        self.filtered_weights += alpha * (weights - self.filtered_weights)
        return self.filtered_weights.copy()

    def _enforce_rate_limits(self, tau: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau, dtype=float).reshape(2).copy()
        delta_tau = tau - self.prev_tau
        max_delta = self.rate_limits * self.dt
        for idx in range(2):
            limit = max_delta[idx]
            if not math.isfinite(limit) or limit <= 0.0:
                continue
            delta_tau[idx] = max(-limit, min(limit, delta_tau[idx]))
        return self.prev_tau + delta_tau

    def _enforce_sign(self, tau: np.ndarray, tau_out: float) -> np.ndarray:
        tau = np.asarray(tau, dtype=float).reshape(2).copy()
        sign_out = math.copysign(1.0, tau_out) if abs(tau_out) > 1e-9 else 0.0
        if sign_out == 0.0:
            return np.zeros_like(tau)
        expected_signs = np.sign(self.A).flatten() * sign_out
        for idx in range(2):
            if expected_signs[idx] == 0.0:
                continue
            tau[idx] = abs(tau[idx]) * expected_signs[idx]
        return tau
