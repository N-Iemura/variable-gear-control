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
        torque_input_mode: str = "command",
    ) -> None:
        self.inertia = float(inertia)
        self.damping = float(damping)
        self.dt = float(dt)
        self.cutoff_hz = float(cutoff_hz)
        self.alpha = 1.0 - math.exp(-2.0 * math.pi * self.cutoff_hz * self.dt)
        self.use_damping = bool(use_damping)
        self.torque_input_mode = str(torque_input_mode)

        self.prev_velocity = 0.0
        # self.prev_applied_torque = 0.0
        self.estimate = 0.0

    def reset(self) -> None:
        self.prev_velocity = 0.0
        # self.prev_applied_torque = 0.0
        self.estimate = 0.0

    def update(
        self, velocity: float, torque_command: float, torque_applied: float | None = None
    ) -> Tuple[float, Dict[str, float]]:
        omega = float(velocity)
        torque_command = float(torque_command)
        omega_dot = (omega - self.prev_velocity) / self.dt
        self.prev_velocity = omega

        damping_term = self.damping * omega if self.use_damping else 0.0
        equivalent_torque = self.inertia * omega_dot + damping_term

        # raw_disturbanceは「どれだけトルクが足りなかったか」を推定する。
        # torque_input は「実際に入れた（前周期の）トルク」が取れればそれを使い、
        # 無ければ当周期コマンドを使う。
        if self.torque_input_mode == "applied" and torque_applied is not None:
            torque_input = float(torque_applied)
        else:
            torque_input = torque_command

        raw_disturbance = torque_input - equivalent_torque

        self.estimate += self.alpha * (raw_disturbance - self.estimate)

        # 新たに出すコマンドは常に「目標トルク = torque_command」基準で補償を足す。
        augmented_torque = torque_command + self.estimate

        diagnostics = {
            "omega_dot": omega_dot,
            "equivalent_torque": equivalent_torque,
            "raw_disturbance": raw_disturbance,
            "filtered_disturbance": self.estimate,
            "augmented_torque": augmented_torque,
            "use_damping": self.use_damping,
            "torque_input_mode": self.torque_input_mode,
            "torque_input": torque_input,
        }
        return augmented_torque, diagnostics
