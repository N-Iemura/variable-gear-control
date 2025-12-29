from __future__ import annotations

import argparse
import logging
import math
import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml

from assist_manager import AssistManager
from dob_estimator import DisturbanceObserver
from logger import DataLogger
from odrive_interface import ODriveInterface
from position_controller import (
    PositionCommand,
    PositionController,
    PositionFeedback,
    VelocityController,
)
from torque_distribution import TorqueAllocator
from velocity_distribution import VelocityAllocator


_LOGGER = logging.getLogger("main_control")


def _load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class ReferenceGenerator:
    config: Dict[str, object]

    def __post_init__(self) -> None:
        self.command_type = str(self.config.get("command_type", "position")).lower()
        if self.command_type not in {"position", "velocity"}:
            raise ValueError("command_type must be 'position' or 'velocity'")
        self.profile = str(self.config.get("active_profile", "step"))
        self.profiles = {
            key: value for key, value in self.config.items() if isinstance(value, dict)
        }
        if self.profile not in self.profiles:
            raise KeyError(f"Profile '{self.profile}' is not defined in reference config.")

    def sample(self, elapsed: float) -> PositionCommand:
        profile_cfg = self.profiles[self.profile]
        if self.profile == "step":
            base = self._step(profile_cfg, elapsed)
        elif self.profile == "sine":
            base = self._sine(profile_cfg, elapsed)
        elif self.profile == "chirp":
            base = self._chirp(profile_cfg, elapsed)
        elif self.profile == "ramp":
            base = self._ramp(profile_cfg, elapsed)
        elif self.profile == "ramp_b":
            base = self._ramp_b(profile_cfg, elapsed)
        elif self.profile == "ramp_step":
            base = self._ramp_step(profile_cfg, elapsed)
        elif self.profile == "pulse":
            base = self._pulse(profile_cfg, elapsed)
        elif self.profile == "nstep":
            base = self._nstep(profile_cfg, elapsed)
        else:
            raise KeyError(f"Unsupported profile: {self.profile}")
        return self._map_command(base)

    def get_active_profile_name(self) -> str:
        return self.profile

    def get_command_type(self) -> str:
        return self.command_type

    def get_profile_label(self) -> str:
        label = self.config.get("file_label")
        if label is None:
            return self.profile
        return str(label)

    @staticmethod
    def sanitize_label_for_filename(label: str) -> str:
        if not label:
            return "profile"
        safe = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(label))
        return safe or "profile"

    def _map_command(self, base: PositionCommand) -> PositionCommand:
        if self.command_type == "position":
            return base
        if self.command_type == "velocity":
            return PositionCommand(
                position=0.0,
                velocity=base.position,
                acceleration=base.velocity,
            )
        raise ValueError(f"Unsupported command_type: {self.command_type}")

    @staticmethod
    def _normalize_filename_values(values) -> Optional[np.ndarray]:
        if values is None:
            return None
        arr = np.asarray(values, dtype=float).ravel()
        return arr if arr.size > 0 else None

    def _infer_profile_filename_values(self, profile: str, cfg: Dict[str, object]) -> Optional[np.ndarray]:
        if profile == "step":
            offset = float(cfg.get("offset", 0.0))
            amp = float(cfg.get("output_amplitude", 0.0))
            if abs(amp) < 1e-12:
                return self._normalize_filename_values([offset])
            return self._normalize_filename_values([offset, offset + amp])
        if profile in ("sine", "chirp"):
            offset = float(cfg.get("offset", 0.0))
            amp = float(cfg.get("output_amplitude", 0.0))
            if abs(amp) < 1e-12:
                return self._normalize_filename_values([offset])
            return self._normalize_filename_values([offset - amp, offset + amp])
        if profile in ("ramp", "ramp_b"):
            start_value = float(cfg.get("start_value", 0.0))
            end_value = float(cfg.get("end_value", start_value))
            return self._normalize_filename_values([start_value, end_value])
        if profile == "ramp_step":
            start_value = float(cfg.get("start_value", 0.0))
            end_value = float(cfg.get("end_value", start_value))
            step_cfg = cfg.get("step", {}) if isinstance(cfg.get("step"), dict) else {}
            amplitude = float(step_cfg.get("amplitude", 0.0))
            candidates = [start_value, end_value]
            if abs(amplitude) >= 1e-12:
                candidates.append(end_value + amplitude)
                if not bool(step_cfg.get("align_to_ramp_end", True)):
                    candidates.append(start_value + amplitude)
            return self._normalize_filename_values(candidates)
        if profile == "nstep":
            values = list(cfg.get("values", []))
            wait_value = cfg.get("wait_value")
            if wait_value is not None:
                values = [wait_value] + values
            return self._normalize_filename_values(values)
        if profile == "pulse":
            start_value = float(cfg.get("start_value", 0.0))
            end_value = float(cfg.get("end_value", start_value))
            return self._normalize_filename_values([start_value, end_value])
        return None

    def resolve_command_values(
        self,
        base_command,
        theta_ref: Optional[object] = None,
        override: Optional[object] = None,
        decimals: int = 4,
    ) -> np.ndarray:
        profile = self.get_active_profile_name()
        cfg = self.profiles.get(profile, {})
        explicit = self._normalize_filename_values(cfg.get("filename_values"))
        if explicit is not None:
            values = explicit
        else:
            global_override = self._normalize_filename_values(override)
            if global_override is not None:
                values = global_override
            else:
                base_arr = np.asarray(base_command, dtype=float).ravel()
                if base_arr.size == 0:
                    base_arr = np.array([0.0], dtype=float)
                values = np.unique(np.round(base_arr, decimals=decimals))
                if theta_ref is not None:
                    theta_arr = np.asarray(theta_ref, dtype=float).ravel()
                else:
                    theta_arr = base_arr
                if values.size > 20 and theta_arr.size > 0:
                    values = np.unique(
                        np.round(
                            [np.min(theta_arr), np.max(theta_arr)],
                            decimals=decimals,
                        )
                    )
        return np.round(values, decimals=decimals)

    @staticmethod
    def format_command_value(value: float, decimals: int = 4) -> str:
        txt = f"{float(value):.{decimals}f}".rstrip("0").rstrip(".")
        return txt if txt else "0"

    def format_profile_settings(self, profile: str) -> str:
        cfg = self.profiles.get(profile, {})
        unit = "turn/s" if self.command_type == "velocity" else "turn"

        def fmt(value):
            if isinstance(value, float):
                return f"{value:g}"
            if isinstance(value, (list, tuple, np.ndarray)):
                arr = np.asarray(value, dtype=float).ravel()
                return "[" + ", ".join(f"{v:g}" for v in arr) + "]"
            return str(value)

        if profile == "step":
            return ", ".join(
                [
                    f"wait={fmt(cfg.get('initial_wait', 0.0))}s",
                    f"step={fmt(cfg.get('step_duration', 0.0))}s",
                    f"amp={fmt(cfg.get('output_amplitude', 0.0))}{unit}",
                    f"offset={fmt(cfg.get('offset', 0.0))}{unit}",
                ]
            )
        if profile == "sine":
            return ", ".join(
                [
                    f"wait={fmt(cfg.get('initial_wait', 0.0))}s",
                    f"amp={fmt(cfg.get('output_amplitude', 0.0))}{unit}",
                    f"freq={fmt(cfg.get('frequency_hz', 0.0))}Hz",
                    f"offset={fmt(cfg.get('offset', 0.0))}{unit}",
                ]
            )
        if profile == "chirp":
            return ", ".join(
                [
                    f"wait={fmt(cfg.get('initial_wait', 0.0))}s",
                    f"amp={fmt(cfg.get('output_amplitude', 0.0))}{unit}",
                    f"f0={fmt(cfg.get('start_frequency_hz', 0.0))}Hz",
                    f"f1={fmt(cfg.get('end_frequency_hz', 0.0))}Hz",
                    f"dur={fmt(cfg.get('duration', 0.0))}s",
                    f"offset={fmt(cfg.get('offset', 0.0))}{unit}",
                ]
            )
        if profile == "ramp":
            return ", ".join(
                [
                    f"wait={fmt(cfg.get('initial_wait', 0.0))}s",
                    f"start={fmt(cfg.get('start_value', 0.0))}{unit}",
                    f"end={fmt(cfg.get('end_value', 0.0))}{unit}",
                    f"ramp={fmt(cfg.get('ramp_duration', 0.0))}s",
                    f"hold={fmt(cfg.get('hold_duration', 0.0))}s",
                    f"return={fmt(cfg.get('return_duration', 0.0))}s",
                ]
            )
        if profile == "ramp_b":
            return ", ".join(
                [
                    f"wait={fmt(cfg.get('initial_wait', 0.0))}s",
                    f"start={fmt(cfg.get('start_value', 0.0))}{unit}",
                    f"end={fmt(cfg.get('end_value', 0.0))}{unit}",
                    f"ramp={fmt(cfg.get('ramp_duration', 0.0))}s",
                    f"hold={fmt(cfg.get('hold_duration', 0.0))}s",
                    f"return={fmt(cfg.get('return_duration', 0.0))}s",
                    f"start_hold={fmt(cfg.get('start_hold_duration', 0.0))}s",
                ]
            )
        if profile == "ramp_step":
            step_cfg = cfg.get("step", {}) if isinstance(cfg.get("step"), dict) else {}
            settings = [
                f"wait={fmt(cfg.get('initial_wait', 0.0))}s",
                f"start={fmt(cfg.get('start_value', 0.0))}{unit}",
                f"end={fmt(cfg.get('end_value', 0.0))}{unit}",
                f"ramp={fmt(cfg.get('ramp_duration', 0.0))}s",
                f"hold={fmt(cfg.get('hold_duration', 0.0))}s",
                f"return={fmt(cfg.get('return_duration', 0.0))}s",
                f"step_amp={fmt(step_cfg.get('amplitude', 0.0))}{unit}",
                f"step_dur={fmt(step_cfg.get('duration', 0.0))}s",
            ]
            if step_cfg:
                settings.append(f"step_period={fmt(step_cfg.get('period', 0.0))}s")
                settings.append(f"step_repeat={step_cfg.get('repeat', True)}")
            return ", ".join(settings)
        if profile == "nstep":
            return ", ".join(
                [
                    f"wait={fmt(cfg.get('initial_wait', 0.0))}s",
                    f"values={fmt(cfg.get('values', []))}{unit}",
                    f"hold={fmt(cfg.get('hold_duration', 0.0))}s",
                    f"repeat={cfg.get('repeat', True)}",
                ]
            )
        if profile == "pulse":
            return ", ".join(
                [
                    f"wait={fmt(cfg.get('initial_wait', 0.0))}s",
                    f"start={fmt(cfg.get('start_value', 0.0))}{unit}",
                    f"end={fmt(cfg.get('end_value', 0.0))}{unit}",
                    f"rise={fmt(cfg.get('ramp_duration', 0.0))}s",
                    f"hold={fmt(cfg.get('hold_duration', 0.0))}s",
                    f"fall={fmt(cfg.get('return_duration', 0.0))}s",
                    f"repeat={cfg.get('repeat', True)}",
                ]
            )
        return ""

    def _step(self, cfg: Dict[str, object], t: float) -> PositionCommand:
        wait = float(cfg.get("initial_wait", 0.0))
        amp = float(cfg.get("output_amplitude", 0.0))
        duration = float(cfg.get("step_duration", 0.0))
        offset = float(cfg.get("offset", 0.0))
        if t < wait:
            return PositionCommand(offset)
        period = duration * 2.0 if duration > 0.0 else math.inf
        if period == math.inf:
            return PositionCommand(offset + amp)
        phase = (t - wait) % period
        position = offset + (amp if phase < duration else 0.0)
        return PositionCommand(position)

    def _sine(self, cfg: Dict[str, object], t: float) -> PositionCommand:
        wait = float(cfg.get("initial_wait", 0.0))
        amp = float(cfg.get("output_amplitude", 0.0))
        freq = float(cfg.get("frequency_hz", 0.0))
        offset = float(cfg.get("offset", 0.0))
        if t < wait:
            return PositionCommand(offset)
        omega = 2.0 * math.pi * freq
        phase_time = t - wait
        position = offset + amp * math.sin(omega * phase_time)
        velocity = amp * omega * math.cos(omega * phase_time)
        acceleration = -amp * (omega ** 2) * math.sin(omega * phase_time)
        return PositionCommand(position, velocity, acceleration)

    def _chirp(self, cfg: Dict[str, object], t: float) -> PositionCommand:
        wait = float(cfg.get("initial_wait", 0.0))
        amp = float(cfg.get("output_amplitude", 0.0))
        f0 = float(cfg.get("start_frequency_hz", 0.0))
        f1 = float(cfg.get("end_frequency_hz", f0))
        duration = max(float(cfg.get("duration", 1.0)), 1e-3)
        offset = float(cfg.get("offset", 0.0))
        if t < wait:
            return PositionCommand(offset)
        tau = t - wait
        k = (f1 - f0) / duration
        if tau > duration:
            tau = duration
        phase = 2.0 * math.pi * (f0 * tau + 0.5 * k * tau**2)
        freq_inst = f0 + k * tau
        omega = 2.0 * math.pi * freq_inst
        position = offset + amp * math.sin(phase)
        velocity = amp * omega * math.cos(phase)
        acceleration = -amp * (omega**2) * math.sin(phase)
        return PositionCommand(position, velocity, acceleration)

    def _ramp(self, cfg: Dict[str, object], t: float) -> PositionCommand:
        wait = float(cfg.get("initial_wait", 0.0))
        start_value = float(cfg.get("start_value", 0.0))
        end_value = float(cfg.get("end_value", start_value))
        ramp_duration = max(float(cfg.get("ramp_duration", 0.0)), 1e-6)
        hold_duration = max(float(cfg.get("hold_duration", 0.0)), 0.0)
        return_duration = max(float(cfg.get("return_duration", 0.0)), 0.0)
        repeat = bool(cfg.get("repeat", True))

        if t < wait:
            return PositionCommand(start_value)
        phase_time = t - wait
        cycle = ramp_duration + hold_duration + return_duration
        if repeat and cycle > 0.0:
            phase = phase_time % cycle
        else:
            phase = min(phase_time, cycle)

        if phase < ramp_duration:
            ratio = phase / ramp_duration
            position = start_value + (end_value - start_value) * ratio
            velocity = (end_value - start_value) / ramp_duration
            return PositionCommand(position, velocity)
        phase -= ramp_duration
        if phase < hold_duration:
            return PositionCommand(end_value)
        phase -= hold_duration
        if return_duration > 0.0:
            ratio = min(phase / return_duration, 1.0)
            position = end_value + (start_value - end_value) * ratio
            velocity = (start_value - end_value) / return_duration
            return PositionCommand(position, velocity)
        return PositionCommand(start_value if repeat else end_value)

    def _ramp_b(self, cfg: Dict[str, object], t: float) -> PositionCommand:
        start_hold = max(float(cfg.get("start_hold_duration", 0.0)), 0.0)
        wait = float(cfg.get("initial_wait", 0.0))
        if t < wait:
            return PositionCommand(float(cfg.get("start_value", 0.0)))
        t_shift = t - wait
        base_cfg = dict(cfg)
        base_cfg["initial_wait"] = start_hold
        return self._ramp(base_cfg, t_shift)

    def _ramp_step(self, cfg: Dict[str, object], t: float) -> PositionCommand:
        base = self._ramp(cfg, t)
        step_cfg = cfg.get("step", {})
        amplitude = float(step_cfg.get("amplitude", 0.0))
        if abs(amplitude) < 1e-9:
            return base

        start_after = float(step_cfg.get("start_after", 0.0))
        duration = float(step_cfg.get("duration", 0.0))
        period = float(step_cfg.get("period", duration))
        repeat = bool(step_cfg.get("repeat", False))
        offset = float(step_cfg.get("offset_in_cycle", 0.0))
        align = bool(step_cfg.get("align_to_ramp_end", True))

        trigger_time = start_after
        if align:
            trigger_time = cfg.get("initial_wait", 0.0) + cfg.get("ramp_duration", 0.0)

        if t < trigger_time:
            return base
        local_time = t - trigger_time
        if repeat and period > 0.0:
            phase = (local_time + offset) % period
        else:
            phase = local_time
        if phase <= duration:
            return PositionCommand(base.position + amplitude, base.velocity, base.acceleration)
        return base

    def _pulse(self, cfg: Dict[str, object], t: float) -> PositionCommand:
        """Symmetric ramp-hold-return signal used for sharp torque pulses."""
        return self._ramp(cfg, t)

    def _nstep(self, cfg: Dict[str, object], t: float) -> PositionCommand:
        wait = float(cfg.get("initial_wait", 0.0))
        values = list(cfg.get("values", []))
        hold = max(float(cfg.get("hold_duration", 0.0)), 1e-6)
        repeat = bool(cfg.get("repeat", True))
        wait_value = float(cfg.get("wait_value", values[0] if values else 0.0))
        if t < wait:
            return PositionCommand(wait_value)
        if not values:
            return PositionCommand(0.0)
        index = int((t - wait) // hold)
        if repeat:
            index %= len(values)
        else:
            index = min(index, len(values) - 1)
        return PositionCommand(float(values[index]))


@dataclass
class TorqueToVelocityConverter:
    inertia: float
    damping: float
    dt: float
    velocity_limit: float = math.inf
    accel_limit: float = math.inf
    use_damping: bool = True
    use_friction: bool = True
    omega_ref: float = 0.0

    def reset(self, omega: float = 0.0) -> None:
        self.omega_ref = float(omega)

    def update(self, tau_out: float, measured_velocity: float, friction_torque: float = 0.0) -> float:
        if not math.isfinite(self.inertia) or abs(self.inertia) < 1e-9:
            return self.omega_ref
        torque_eff = float(tau_out)
        if self.use_friction:
            torque_eff -= float(friction_torque)
        if self.use_damping:
            torque_eff -= self.damping * float(measured_velocity)
        accel = torque_eff / self.inertia
        if math.isfinite(self.accel_limit):
            accel = max(-self.accel_limit, min(self.accel_limit, accel))
        omega_next = self.omega_ref + accel * self.dt
        if math.isfinite(self.velocity_limit):
            omega_next = max(-self.velocity_limit, min(self.velocity_limit, omega_next))
        self.omega_ref = omega_next
        return omega_next


def build_modules(config_dir: Path) -> Dict[str, object]:
    controller_cfg = _load_yaml(config_dir / "controller.yaml")
    hardware_cfg = _load_yaml(config_dir / "hardware.yaml")
    reference_cfg = _load_yaml(config_dir / "reference.yaml")
    logger_cfg = _load_yaml(config_dir / "logger.yaml")

    command_type = str(
        controller_cfg.get("command_type", reference_cfg.get("command_type", "position"))
    ).lower()
    if command_type not in {"position", "velocity"}:
        raise ValueError("command_type must be 'position' or 'velocity'")
    reference_cfg["command_type"] = command_type

    dt = float(controller_cfg.get("sample_time_s", 0.005))
    plant_cfg = controller_cfg.get("plant", {})
    mechanism_matrix = np.asarray(plant_cfg.get("mechanism_matrix", [-0.05, 0.0815]))
    motor_output_gains = np.asarray(
        plant_cfg.get("motor_output_gains", mechanism_matrix), dtype=float
    )
    velocity_cfg = controller_cfg.get("velocity_distribution", {})
    if not isinstance(velocity_cfg, dict):
        velocity_cfg = {}
    kinematic_matrix = np.asarray(
        velocity_cfg.get("kinematic_matrix", plant_cfg.get("kinematic_matrix", mechanism_matrix)),
        dtype=float,
    )
    inertia = float(plant_cfg.get("inertia", 0.015))
    damping = float(plant_cfg.get("damping", 0.002))

    outer_pid = controller_cfg.get("outer_pid", {})
    per_motor_cfg = controller_cfg.get("per_motor_pid", {})
    derivative_mode = per_motor_cfg.get("derivative_mode", "error")
    derivative_alpha = float(per_motor_cfg.get("derivative_filter_alpha", 1.0))

    if command_type == "velocity":
        velocity_pid = controller_cfg.get("velocity_pid", outer_pid)
        controller = VelocityController(
            gains=velocity_pid,
            plant_inertia=inertia,
            plant_damping=damping,
            dt=dt,
            derivative_mode=derivative_mode,
            derivative_filter_alpha=derivative_alpha,
            use_feedforward=bool(velocity_pid.get("use_feedforward", True)),
        )
    else:
        controller = PositionController(
            gains=outer_pid,
            plant_inertia=inertia,
            plant_damping=damping,
            dt=dt,
            derivative_mode=derivative_mode,
            derivative_filter_alpha=derivative_alpha,
            use_feedforward=bool(outer_pid.get("use_feedforward", True)),
        )

    dob_cfg = controller_cfg.get("dob", {})
    dob_enabled = bool(dob_cfg.get("enabled", True))
    dob_input_mode = str(dob_cfg.get("torque_input_mode", "command")).lower()
    dob_applied_sign = float(
        dob_cfg.get("applied_sign", -1.0 if dob_input_mode == "applied" else 1.0)
    )
    dob: Optional[DisturbanceObserver]
    if dob_enabled:
        cutoff_hz = float(dob_cfg.get("cutoff_hz", 20.0))
        dob_use_damping = bool(dob_cfg.get("use_damping", True))
        dob = DisturbanceObserver(
            inertia,
            damping,
            dt,
            cutoff_hz,
            use_damping=dob_use_damping,
            torque_input_mode=dob_input_mode,
        )
    else:
        dob = None

    torque_limits = controller_cfg.get("torque_limits", {})
    rate_limits = controller_cfg.get("torque_rate_limits", {})
    allocation_cfg = controller_cfg.get("torque_allocation", {})
    torque_dist_cfg = controller_cfg.get("torque_distribution", {})
    preference_cfg = controller_cfg.get("torque_preference", {})
    sign_cfg = controller_cfg.get("sign_enforcement", {})
    weight_mode = str(allocation_cfg.get("weight_mode", "raw")).lower()
    preference_mode = str(preference_cfg.get("mode", "primary")).lower()
    velocity_weight_mode = str(velocity_cfg.get("weight_mode", "raw")).lower()
    velocity_preference_mode = str(
        velocity_cfg.get("preference_mode", preference_mode)
    ).lower()
    velocity_preferred_motor = velocity_cfg.get(
        "preferred_motor", preference_cfg.get("preferred_motor")
    )
    velocity_sign_enforcement = bool(velocity_cfg.get("sign_enforcement", sign_cfg.get("enabled", True)))

    allocator = TorqueAllocator(
        mechanism_matrix=mechanism_matrix,
        torque_limits=torque_limits,
        dt=dt,
        rate_limits=rate_limits,
        preferred_motor=preference_cfg.get("preferred_motor"),
        sign_enforcement=sign_cfg.get("enabled", True),
        weight_mode=weight_mode,
        preference_mode=preference_mode,
    )
    velocity_allocator = VelocityAllocator(
        kinematic_matrix=kinematic_matrix,
        velocity_limits=velocity_cfg.get("velocity_limits", {}),
        dt=dt,
        rate_limits=velocity_cfg.get("velocity_rate_limits", {}),
        preferred_motor=velocity_preferred_motor,
        sign_enforcement=velocity_sign_enforcement,
        weight_mode=velocity_weight_mode,
        preference_mode=velocity_preference_mode,
    )
    direct_velocity = bool(velocity_cfg.get("direct_velocity", False))

    assist_cfg = controller_cfg.get("assist_manager", {})
    assist_enabled = bool(assist_cfg.get("enabled", True))
    if assist_enabled:
        assist_manager = AssistManager(
            dt=dt,
            mechanism_matrix=mechanism_matrix,
            torque_limits=torque_limits,
            config=assist_cfg,
        )
    else:
        assist_manager = None

    hardware = hardware_cfg
    reference = ReferenceGenerator(reference_cfg)
    friction_ff = controller_cfg.get("friction_ff", {})
    logger = DataLogger(
        logger_cfg,
        base_path=config_dir.parent,
        controller_config=controller_cfg,
        reference=reference,
        torque_constants=hardware_cfg.get("odrive", {}).get("torque_constants", {}),
    )

    odrive_iface = ODriveInterface(hardware)
    motor_control_mode = str(controller_cfg.get("motor_control_mode", "torque")).lower()
    odrive_velocity_gains = controller_cfg.get("odrive_velocity_gains", {})
    tau_to_velocity = TorqueToVelocityConverter(
        inertia=inertia,
        damping=damping,
        dt=dt,
        velocity_limit=float(velocity_cfg.get("output_velocity_limit", math.inf)),
        accel_limit=float(velocity_cfg.get("output_accel_limit", math.inf)),
        use_damping=bool(velocity_cfg.get("use_damping_comp", True)),
        use_friction=bool(velocity_cfg.get("use_friction_comp", True)),
    )

    return {
        "controller": controller,
        "dob": dob,
        "allocator": allocator,
        "velocity_allocator": velocity_allocator,
        "tau_to_velocity": tau_to_velocity,
        "assist": assist_manager,
        "reference": reference,
        "logger": logger,
        "hardware": hardware,
        "odrive": odrive_iface,
        "friction_ff": friction_ff,
        "config": controller_cfg,
        "dt": dt,
        "command_type": command_type,
        "mechanism_matrix": mechanism_matrix,
        "motor_output_gains": motor_output_gains,
        "kinematic_matrix": kinematic_matrix,
        "dob_enabled": dob_enabled,
        "dob_input_mode": dob_input_mode,
        "dob_applied_sign": dob_applied_sign,
        "torque_distribution": torque_dist_cfg,
        "motor_control_mode": motor_control_mode,
        "odrive_velocity_gains": odrive_velocity_gains,
        "direct_velocity": direct_velocity,
    }


def run_control_loop(modules: Dict[str, object], duration: Optional[float] = None) -> None:
    controller = modules["controller"]
    dob: Optional[DisturbanceObserver] = modules["dob"]
    dob_enabled: bool = modules.get("dob_enabled", True)
    dob_input_mode: str = modules.get("dob_input_mode", "command")
    dob_applied_sign: float = float(modules.get("dob_applied_sign", 1.0))
    allocator: TorqueAllocator = modules["allocator"]
    velocity_allocator: VelocityAllocator = modules["velocity_allocator"]
    tau_to_velocity: TorqueToVelocityConverter = modules["tau_to_velocity"]
    direct_velocity = bool(modules.get("direct_velocity", False))
    assist_manager: Optional[AssistManager] = modules["assist"]
    reference: ReferenceGenerator = modules["reference"]
    logger: DataLogger = modules["logger"]
    odrive_iface: ODriveInterface = modules["odrive"]
    dt: float = modules["dt"]
    command_type = str(modules.get("command_type", "position")).lower()
    motor_control_mode = str(modules.get("motor_control_mode", "torque")).lower()
    if motor_control_mode not in {"torque", "velocity"}:
        raise ValueError("motor_control_mode must be 'torque' or 'velocity'")
    mechanism_matrix = np.asarray(modules.get("mechanism_matrix"), dtype=float)
    motor_output_gains = np.asarray(
        modules.get("motor_output_gains", mechanism_matrix), dtype=float
    )
    torque_dist_cfg = modules.get("torque_distribution", {}) or {}
    dist_mode = str(torque_dist_cfg.get("mode", "dynamic")).lower()
    fixed_weights = np.asarray(torque_dist_cfg.get("fixed_weights", [1.0, 1.0]), dtype=float)
    fixed_secondary_gain = float(torque_dist_cfg.get("fixed_secondary_gain", 1.0))
    if fixed_weights.shape != (2,) or np.any(fixed_weights <= 0.0):
        raise ValueError("torque_distribution.fixed_weights must be two positive values.")
    friction_cfg = modules.get("friction_ff", {}) or {}
    use_measured_torque = dob_enabled and dob_input_mode == "applied"
    measured_tau = np.zeros(2, dtype=float)
    measured_iq = np.zeros(2, dtype=float)
    measured_tau_time = 0.0
    measured_tau_lock = threading.Lock()
    stop_measured_torque = threading.Event()
    torque_poll_thread: Optional[threading.Thread] = None

    def friction_feedforward(velocity: float, reference_velocity: float, error_hint: float) -> float:
        """Return output-axis friction feedforward torque."""
        coulomb = float(friction_cfg.get("coulomb", 0.0))
        if abs(coulomb) < 1e-12:
            return 0.0
        vel_deadband = abs(float(friction_cfg.get("vel_deadband", 0.0)))
        blend_width = max(float(friction_cfg.get("blend_width", 0.05)), 1e-6)
        vel = float(velocity)
        if abs(vel) < vel_deadband:
            direction = reference_velocity if abs(reference_velocity) > 1e-9 else error_hint
            if abs(direction) < 1e-9:
                direction = vel
            if abs(direction) < 1e-9:
                return 0.0
            return math.copysign(coulomb, direction)
        return coulomb * math.tanh(vel / blend_width)

    odrive_iface.connect(calibrate=False, control_mode=motor_control_mode)
    if motor_control_mode == "velocity":
        if command_type == "position":
            _LOGGER.warning(
                "motor_control_mode is velocity but command_type is position; "
                "verify outer-loop behavior."
            )
        if direct_velocity and command_type != "velocity":
            _LOGGER.warning(
                "direct_velocity is enabled but command_type is not velocity; "
                "falling back to torque-to-velocity conversion."
            )
        gains_cfg = modules.get("odrive_velocity_gains", {})
        if isinstance(gains_cfg, dict):
            for name, gains in gains_cfg.items():
                if not isinstance(gains, dict):
                    continue
                applied = odrive_iface.set_velocity_gains(
                    name,
                    vel_gain=gains.get("vel_gain"),
                    vel_integrator_gain=gains.get("vel_integrator_gain"),
                    vel_integrator_limit=gains.get("vel_integrator_limit"),
                )
                if applied:
                    _LOGGER.info("ODrive velocity gains applied on %s: %s", name, applied)
        if dob_enabled and dob_input_mode == "applied":
            _LOGGER.info("DOB is enabled with applied torque input in velocity mode.")
    odrive_iface.zero_positions()
    _LOGGER.info("Control loop started (dt=%.6f s)", dt)

    def _poll_measured_torque() -> None:
        nonlocal measured_tau_time
        poll_interval = max(0.0005, dt)  # do not spin too fast
        while not stop_measured_torque.is_set():
            try:
                poll_states = odrive_iface.read_states(fast=False)
                with measured_tau_lock:
                    measured_tau[0] = poll_states["motor1"].torque_measured
                    measured_tau[1] = poll_states["motor2"].torque_measured
                    measured_iq[0] = poll_states["motor1"].current_iq
                    measured_iq[1] = poll_states["motor2"].current_iq
                    measured_tau_time = time.time()
            except Exception:
                _LOGGER.exception("Failed to poll measured torque.")
            stop_measured_torque.wait(poll_interval)

    if use_measured_torque:
        torque_poll_thread = threading.Thread(
            target=_poll_measured_torque, name="torque_sampler", daemon=True
        )
        torque_poll_thread.start()

    start_time = time.time()
    running = True
    prev_loop_time = start_time
    loop_intervals: list[float] = []
    prev_tau_alloc = np.zeros(2, dtype=float)  # last commanded motor torques

    try:
        while running:
            now = time.time()
            elapsed = now - start_time
            if duration is not None and elapsed >= duration:
                break

            command = reference.sample(elapsed)

            t_read_start = time.perf_counter()
            states = odrive_iface.read_states(fast=True)
            t_read_end = time.perf_counter()
            output_state = states.get("output", states["motor1"])

            feedback = PositionFeedback(
                position=output_state.position,
                velocity=output_state.velocity,
            )
            tau_ctrl, ctrl_diag = controller.update(command, feedback)
            if command_type == "velocity":
                reference_position = 0.0
                reference_velocity = command.velocity
            else:
                reference_position = command.position
                reference_velocity = command.velocity
            error_hint = float(
                ctrl_diag.get(
                    "error",
                    reference_velocity if command_type == "velocity" else reference_position - feedback.position,
                )
            )
            friction_tau = friction_feedforward(
                output_state.velocity, reference_velocity, error_hint
            )
            tau_cmd = tau_ctrl + friction_tau

            if use_measured_torque:
                with measured_tau_lock:
                    tau_measured_snapshot = measured_tau.copy()
                    iq_measured_snapshot = measured_iq.copy()
                    has_measured_tau = measured_tau_time > 0.0
            else:
                tau_measured_snapshot = np.array(
                    [
                        states["motor1"].torque_measured,
                        states["motor2"].torque_measured,
                    ],
                    dtype=float,
                )
                iq_measured_snapshot = np.array(
                    [
                        states["motor1"].current_iq,
                        states["motor2"].current_iq,
                    ],
                    dtype=float,
                )
                has_measured_tau = True

            if dob_enabled and dob is not None:
                torque_applied_input = None
                if dob_input_mode == "applied":
                    if use_measured_torque and has_measured_tau:
                        applied_motor_tau = tau_measured_snapshot
                    else:
                        applied_motor_tau = prev_tau_alloc
                    applied_tau_out = dob_applied_sign * float(
                        np.dot(motor_output_gains, applied_motor_tau)
                    )
                    torque_applied_input = applied_tau_out
                tau_aug, dob_diag = dob.update(
                    feedback.velocity, tau_cmd, torque_applied=torque_applied_input
                )
            else:
                tau_aug = tau_cmd
                dob_diag = {"filtered_disturbance": 0.0}

            if dist_mode == "fixed":
                weights = fixed_weights
                secondary_gain = fixed_secondary_gain
            elif dist_mode == "dynamic":
                if assist_manager is not None:
                    assist_status = assist_manager.update(tau_aug)
                    weights = assist_status.weights
                    secondary_gain = assist_status.secondary_gain
                else:
                    weights = np.ones(2, dtype=float)
                    secondary_gain = 1.0
            else:
                raise ValueError("torque_distribution.mode must be 'fixed' or 'dynamic'.")

            tau_alloc, alloc_diag = allocator.allocate(tau_aug, weights, secondary_gain)
            # 次周期のDOB入力用に、今回送ったモータ指令を保存
            prev_tau_alloc = tau_alloc.copy()
            if motor_control_mode == "velocity":
                if direct_velocity and command_type == "velocity":
                    omega_out_ref = float(reference_velocity)
                    vel_limit = float(getattr(tau_to_velocity, "velocity_limit", math.inf))
                    if math.isfinite(vel_limit):
                        omega_out_ref = max(-vel_limit, min(vel_limit, omega_out_ref))
                else:
                    omega_out_ref = tau_to_velocity.update(
                        tau_aug, output_state.velocity, friction_torque=friction_tau
                    )
                omega_alloc, _ = velocity_allocator.allocate(
                    omega_out_ref, weights, secondary_gain
                )
                odrive_iface.command_velocities(float(omega_alloc[0]), float(omega_alloc[1]))
            else:
                odrive_iface.command_torques(float(tau_alloc[0]), float(tau_alloc[1]))
            t_ctrl_end = time.perf_counter()

            motor1_state = states["motor1"]
            motor2_state = states["motor2"]

            loop_dt_actual = now - prev_loop_time
            logger.log(
                elapsed,
                motor1_state.position,
                motor1_state.velocity,
                tau_alloc[0],
                float(iq_measured_snapshot[0]),
                float(tau_measured_snapshot[0]),
                motor2_state.position,
                motor2_state.velocity,
                tau_alloc[1],
                float(iq_measured_snapshot[1]),
                float(tau_measured_snapshot[1]),
                output_state.position,
                output_state.velocity,
                reference_position,
                reference_velocity=reference_velocity,
                reference_control=tau_cmd,
                tau_pid=tau_ctrl,
                tau_dob=tau_aug,
                dob_disturbance=dob_diag.get("filtered_disturbance", 0.0),
                tau_out=tau_aug,
                loop_dt=loop_dt_actual,
                diag_read=t_read_end - t_read_start,
                diag_ctrl=t_ctrl_end - t_read_end,
                diag_log=time.perf_counter() - t_ctrl_end,
            )
            loop_intervals.append(loop_dt_actual)
            prev_loop_time = now
    except KeyboardInterrupt:
        _LOGGER.info("Keyboard interrupt received, stopping control loop.")
        running = False
    finally:
        if use_measured_torque and torque_poll_thread is not None:
            stop_measured_torque.set()
            torque_poll_thread.join(timeout=1.0)
        odrive_iface.shutdown()
        metadata = {
            "DurationSec": f"{time.time() - start_time:.3f}",
            "SampleTime": f"{dt}",
            "MotorControlMode": motor_control_mode,
            "DirectVelocity": str(bool(direct_velocity)),
        }
        if loop_intervals:
            metadata.update(
                {
                    "LoopDtMean": f"{float(np.mean(loop_intervals)):.6f}",
                    "LoopDtStd": f"{float(np.std(loop_intervals)):.6f}",
                    "LoopDtMin": f"{float(np.min(loop_intervals)):.6f}",
                    "LoopDtMax": f"{float(np.max(loop_intervals)):.6f}",
                }
            )
        save_result = logger.save(metadata=metadata)
        csv_path = save_result.get("csv")
        fig_path = save_result.get("figure")
        if csv_path is not None:
            _LOGGER.info("Log saved to %s", csv_path)
        if fig_path is not None:
            _LOGGER.info("Figure saved to %s", fig_path)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Variable gear torque control (ODrive).")
    parser.add_argument("--config-dir", type=Path, default=Path(__file__).parent / "config")
    parser.add_argument(
        "--duration", type=float, default=None, help="Optional duration [s] to run the controller."
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    modules = build_modules(args.config_dir)
    run_control_loop(modules, duration=args.duration)
    return 0


if __name__ == "__main__":
    sys.exit(main())
