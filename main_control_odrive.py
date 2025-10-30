from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml

from assist_manager import AssistManager
from dob_estimator import DisturbanceObserver
from logger import DataLogger
from odrive_interface import ODriveInterface
from position_controller import PositionCommand, PositionController, PositionFeedback
from torque_distribution import TorqueAllocator


_LOGGER = logging.getLogger("main_control")


def _load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class ReferenceGenerator:
    config: Dict[str, object]

    def __post_init__(self) -> None:
        self.profile = str(self.config.get("active_profile", "step"))
        self.profiles = {
            key: value for key, value in self.config.items() if isinstance(value, dict)
        }
        if self.profile not in self.profiles:
            raise KeyError(f"Profile '{self.profile}' is not defined in reference config.")

    def sample(self, elapsed: float) -> PositionCommand:
        profile_cfg = self.profiles[self.profile]
        if self.profile == "step":
            return self._step(profile_cfg, elapsed)
        if self.profile == "sine":
            return self._sine(profile_cfg, elapsed)
        if self.profile == "chirp":
            return self._chirp(profile_cfg, elapsed)
        if self.profile == "ramp":
            return self._ramp(profile_cfg, elapsed)
        if self.profile == "ramp_b":
            return self._ramp_b(profile_cfg, elapsed)
        if self.profile == "ramp_step":
            return self._ramp_step(profile_cfg, elapsed)
        if self.profile == "nstep":
            return self._nstep(profile_cfg, elapsed)
        raise KeyError(f"Unsupported profile: {self.profile}")

    def get_active_profile_name(self) -> str:
        return self.profile

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
                    f"amp={fmt(cfg.get('output_amplitude', 0.0))}turn",
                    f"offset={fmt(cfg.get('offset', 0.0))}",
                ]
            )
        if profile == "sine":
            return ", ".join(
                [
                    f"wait={fmt(cfg.get('initial_wait', 0.0))}s",
                    f"amp={fmt(cfg.get('output_amplitude', 0.0))}turn",
                    f"freq={fmt(cfg.get('frequency_hz', 0.0))}Hz",
                    f"offset={fmt(cfg.get('offset', 0.0))}",
                ]
            )
        if profile == "chirp":
            return ", ".join(
                [
                    f"wait={fmt(cfg.get('initial_wait', 0.0))}s",
                    f"amp={fmt(cfg.get('output_amplitude', 0.0))}turn",
                    f"f0={fmt(cfg.get('start_frequency_hz', 0.0))}Hz",
                    f"f1={fmt(cfg.get('end_frequency_hz', 0.0))}Hz",
                    f"dur={fmt(cfg.get('duration', 0.0))}s",
                    f"offset={fmt(cfg.get('offset', 0.0))}",
                ]
            )
        if profile == "ramp":
            return ", ".join(
                [
                    f"wait={fmt(cfg.get('initial_wait', 0.0))}s",
                    f"start={fmt(cfg.get('start_value', 0.0))}",
                    f"end={fmt(cfg.get('end_value', 0.0))}",
                    f"ramp={fmt(cfg.get('ramp_duration', 0.0))}s",
                    f"hold={fmt(cfg.get('hold_duration', 0.0))}s",
                    f"return={fmt(cfg.get('return_duration', 0.0))}s",
                ]
            )
        if profile == "ramp_b":
            return ", ".join(
                [
                    f"wait={fmt(cfg.get('initial_wait', 0.0))}s",
                    f"start={fmt(cfg.get('start_value', 0.0))}",
                    f"end={fmt(cfg.get('end_value', 0.0))}",
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
                f"start={fmt(cfg.get('start_value', 0.0))}",
                f"end={fmt(cfg.get('end_value', 0.0))}",
                f"ramp={fmt(cfg.get('ramp_duration', 0.0))}s",
                f"hold={fmt(cfg.get('hold_duration', 0.0))}s",
                f"return={fmt(cfg.get('return_duration', 0.0))}s",
                f"step_amp={fmt(step_cfg.get('amplitude', 0.0))}",
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
                    f"values={fmt(cfg.get('values', []))}",
                    f"hold={fmt(cfg.get('hold_duration', 0.0))}s",
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


def build_modules(config_dir: Path) -> Dict[str, object]:
    controller_cfg = _load_yaml(config_dir / "controller.yaml")
    hardware_cfg = _load_yaml(config_dir / "hardware.yaml")
    reference_cfg = _load_yaml(config_dir / "reference.yaml")
    logger_cfg = _load_yaml(config_dir / "logger.yaml")

    dt = float(controller_cfg.get("sample_time_s", 0.005))
    plant_cfg = controller_cfg.get("plant", {})
    mechanism_matrix = np.asarray(plant_cfg.get("mechanism_matrix", [-0.05, 0.0815]))
    inertia = float(plant_cfg.get("inertia", 0.015))
    damping = float(plant_cfg.get("damping", 0.002))

    outer_pid = controller_cfg.get("outer_pid", {})
    per_motor_cfg = controller_cfg.get("per_motor_pid", {})
    derivative_mode = per_motor_cfg.get("derivative_mode", "error")
    derivative_alpha = float(per_motor_cfg.get("derivative_filter_alpha", 1.0))

    controller = PositionController(
        gains=outer_pid,
        plant_inertia=inertia,
        plant_damping=damping,
        dt=dt,
        derivative_mode=derivative_mode,
        derivative_filter_alpha=derivative_alpha,
    )

    dob_cfg = controller_cfg.get("dob", {})
    cutoff_hz = float(dob_cfg.get("cutoff_hz", 20.0))
    dob = DisturbanceObserver(inertia, damping, dt, cutoff_hz)

    torque_limits = controller_cfg.get("torque_limits", {})
    rate_limits = controller_cfg.get("torque_rate_limits", {})
    preference_cfg = controller_cfg.get("torque_preference", {})
    sign_cfg = controller_cfg.get("sign_enforcement", {})

    allocator = TorqueAllocator(
        mechanism_matrix=mechanism_matrix,
        torque_limits=torque_limits,
        dt=dt,
        rate_limits=rate_limits,
        preferred_motor=preference_cfg.get("preferred_motor"),
        sign_enforcement=sign_cfg.get("enabled", True),
    )

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
    logger = DataLogger(
        logger_cfg,
        base_path=config_dir.parent,
        controller_config=controller_cfg,
        reference=reference,
    )

    odrive_iface = ODriveInterface(hardware)

    return {
        "controller": controller,
        "dob": dob,
        "allocator": allocator,
        "assist": assist_manager,
        "reference": reference,
        "logger": logger,
        "hardware": hardware,
        "odrive": odrive_iface,
        "config": controller_cfg,
        "dt": dt,
        "mechanism_matrix": mechanism_matrix,
    }


def run_control_loop(modules: Dict[str, object], duration: Optional[float] = None) -> None:
    controller: PositionController = modules["controller"]
    dob: DisturbanceObserver = modules["dob"]
    allocator: TorqueAllocator = modules["allocator"]
    assist_manager: Optional[AssistManager] = modules["assist"]
    reference: ReferenceGenerator = modules["reference"]
    logger: DataLogger = modules["logger"]
    odrive_iface: ODriveInterface = modules["odrive"]
    dt: float = modules["dt"]

    odrive_iface.connect(calibrate=False)
    odrive_iface.zero_positions()
    _LOGGER.info("Control loop started (dt=%.6f s)", dt)

    start_time = time.time()
    loop_time = start_time
    running = True

    try:
        while running:
            now = time.time()
            elapsed = now - start_time
            if duration is not None and elapsed >= duration:
                break

            command = reference.sample(elapsed)

            states = odrive_iface.read_states()
            output_state = states.get("output", states["motor0"])

            feedback = PositionFeedback(
                position=output_state.position,
                velocity=output_state.velocity,
            )
            tau_cmd, ctrl_diag = controller.update(command, feedback)
            tau_aug, dob_diag = dob.update(feedback.velocity, tau_cmd)

            if assist_manager is not None:
                assist_status = assist_manager.update(tau_aug)
                weights = assist_status.weights
                secondary_gain = assist_status.secondary_gain
            else:
                weights = np.ones(2, dtype=float)
                secondary_gain = 1.0

            tau_alloc, alloc_diag = allocator.allocate(tau_aug, weights, secondary_gain)
            odrive_iface.command_torques(float(tau_alloc[0]), float(tau_alloc[1]))

            motor0_state = states["motor0"]
            motor1_state = states["motor1"]

            logger.log(
                elapsed,
                motor0={"pos": motor0_state.position, "vel": motor0_state.velocity, "torque": tau_alloc[0]},
                motor1={"pos": motor1_state.position, "vel": motor1_state.velocity, "torque": tau_alloc[1]},
                output={"pos": output_state.position, "vel": output_state.velocity},
                reference={"position": command.position, "control": tau_cmd},
                torques={"output": tau_aug},
            )

            # Maintain timing
            loop_time += dt
            sleep_time = loop_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                loop_time = time.time()
    except KeyboardInterrupt:
        _LOGGER.info("Keyboard interrupt received, stopping control loop.")
        running = False
    finally:
        odrive_iface.shutdown()
        metadata = {
            "DurationSec": f"{time.time() - start_time:.3f}",
            "SampleTime": f"{dt}",
        }
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
