from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class AssistStatus:
    assist_enabled: bool
    blend: float
    load_ratio: float
    weights: np.ndarray
    secondary_gain: float


class AssistManager:
    """Assist-as-Needed controller that adapts torque weights based on primary load."""

    def __init__(
        self,
        dt: float,
        mechanism_matrix: np.ndarray,
        torque_limits: Dict[str, float],
        config: Dict[str, object],
    ) -> None:
        self.dt = float(dt)
        self.A = np.asarray(mechanism_matrix, dtype=float).reshape(2)
        self.mode = str(config.get("mode", "threshold")).lower()
        self.primary_motor = str(config.get("primary_motor", "motor1"))
        self.primary_index = 0 if self.primary_motor == "motor1" else 1
        self.secondary_index = 1 - self.primary_index
        torque_limits = dict(torque_limits)
        self.primary_limit = abs(float(torque_limits.get(self.primary_motor, 1.0)))
        self.primary_gain = abs(float(self.A[self.primary_index]))

        thresholds = config.get("torque_thresholds", {})
        self.activate_threshold = float(thresholds.get("activate", 0.65))
        self.deactivate_threshold = float(thresholds.get("deactivate", 0.55))

        self.release_start_ratio = float(config.get("release_start_ratio", 0.5))
        self.release_full_ratio = float(config.get("release_full_ratio", 0.9))

        weighting_cfg = config.get("weighting", {})
        self.weights_hold = np.array(
            [
                float(weighting_cfg.get("hold", {}).get("motor1", 1.0)),
                float(weighting_cfg.get("hold", {}).get("motor2", 1.0)),
            ],
            dtype=float,
        )
        self.weights_release = np.array(
            [
                float(weighting_cfg.get("release", {}).get("motor1", 1.0)),
                float(weighting_cfg.get("release", {}).get("motor2", 1.0)),
            ],
            dtype=float,
        )

        secondary_cfg = config.get("secondary_gain", {})
        self.secondary_hold = float(secondary_cfg.get("hold", 0.0))
        self.secondary_release = float(secondary_cfg.get("release", 1.0))

        weight_limits = config.get("weight_limits", {})
        self.w_off = float(weight_limits.get("w_off", self.weights_hold[1]))
        self.w_on = max(float(weight_limits.get("w_on", self.weights_release[1])), 1e-6)

        self.time_constant = max(float(config.get("time_constant", 0.15)), 1e-6)

        self.blend = 0.0
        self.assist_flag = False

    def reset(self) -> None:
        self.blend = 0.0
        self.assist_flag = False

    def update(self, tau_out_aug: float) -> AssistStatus:
        if self.mode == "balanced":
            weights = np.ones(2, dtype=float)
            return AssistStatus(
                assist_enabled=True,
                blend=1.0,
                load_ratio=0.0,
                weights=weights,
                secondary_gain=1.0,
            )

        load_ratio = self._compute_load_ratio(tau_out_aug)
        target_flag = self._compute_target_flag(load_ratio)
        target_blend = 1.0 if target_flag else self._release_progress(load_ratio)

        self.blend += (self.dt / self.time_constant) * (target_blend - self.blend)
        self.blend = max(0.0, min(1.0, self.blend))
        self.assist_flag = target_flag

        weights = self._geometric_blend(self.weights_hold, self.weights_release, self.blend)
        secondary_idx = self.secondary_index
        weights[secondary_idx] = max(self.w_on, min(self.w_off, weights[secondary_idx]))
        secondary_gain = self._geometric_scalar(
            self.secondary_hold, self.secondary_release, self.blend
        )

        status = AssistStatus(
            assist_enabled=self.assist_flag,
            blend=self.blend,
            load_ratio=load_ratio,
            weights=weights,
            secondary_gain=secondary_gain,
        )
        return status

    def _compute_load_ratio(self, tau_out_aug: float) -> float:
        if self.primary_limit <= 0.0 or self.primary_gain <= 0.0:
            return 0.0
        equivalent_primary = abs(float(tau_out_aug)) / max(self.primary_gain, 1e-9)
        return equivalent_primary / self.primary_limit

    def _compute_target_flag(self, load_ratio: float) -> bool:
        if self.assist_flag:
            return load_ratio >= self.deactivate_threshold
        return load_ratio >= self.activate_threshold

    def _release_progress(self, load_ratio: float) -> float:
        start_ratio = self.release_start_ratio
        full_ratio = max(self.release_full_ratio, start_ratio + 1e-6)
        if load_ratio <= start_ratio:
            return 0.0
        if load_ratio >= full_ratio:
            return 1.0
        span = full_ratio - start_ratio
        return (load_ratio - start_ratio) / span

    @staticmethod
    def _geometric_blend(vec_a: np.ndarray, vec_b: np.ndarray, alpha: float) -> np.ndarray:
        vec_a = np.asarray(vec_a, dtype=float)
        vec_b = np.asarray(vec_b, dtype=float)
        alpha = max(0.0, min(1.0, float(alpha)))
        return np.exp(np.log(vec_a) * (1.0 - alpha) + np.log(vec_b) * alpha)

    @staticmethod
    def _geometric_scalar(a: float, b: float, alpha: float) -> float:
        if a <= 0.0:
            a = 1e-9
        if b <= 0.0:
            b = 1e-9
        alpha = max(0.0, min(1.0, float(alpha)))
        return math.exp(math.log(a) * (1.0 - alpha) + math.log(b) * alpha)
