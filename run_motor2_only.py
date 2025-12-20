#!/usr/bin/env python3
"""Run motor2-only experiments with torque or ODrive velocity control."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import yaml

from odrive_interface import ODriveInterface


_LOGGER = logging.getLogger("motor2_only")

RAD_PER_TURN = 2.0 * math.pi
TURN_PER_RAD = 1.0 / RAD_PER_TURN


@dataclass(frozen=True)
class VelocityStep:
    duration_s: float
    cmd_w2_rad_s: float
    steady_start_s: float
    label: str


@dataclass(frozen=True)
class TorqueStep:
    duration_s: float
    cmd_tau_2_Nm: float
    steady_start_s: float
    label: str


def _load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _as_list(value: object) -> List[float]:
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    if value is None:
        return []
    return [float(value)]


def _clamp(value: float, limit: float) -> float:
    if not math.isfinite(limit):
        return value
    if limit <= 0.0:
        return 0.0
    return max(-limit, min(limit, value))


def _ramp_value(current: float, target: float, rate_limit: float, dt: float) -> float:
    if not math.isfinite(rate_limit) or rate_limit <= 0.0:
        return float(target)
    max_delta = rate_limit * dt
    delta = float(target) - float(current)
    if abs(delta) <= max_delta:
        return float(target)
    return float(current) + math.copysign(max_delta, delta)


def _normalize_sign(value: float) -> float:
    return -1.0 if float(value) < 0.0 else 1.0


def _format_gain_map(values: Dict[str, float]) -> str:
    return ",".join(f"{key}={values[key]:g}" for key in sorted(values))


def _extract_odrive_velocity_gains(cfg: object, motor_name: str) -> Dict[str, float]:
    if not isinstance(cfg, dict):
        return {}
    keys = {"vel_gain", "vel_integrator_gain", "vel_integrator_limit"}
    if keys & cfg.keys():
        return {key: float(cfg[key]) for key in keys if key in cfg}
    motor_cfg = cfg.get(motor_name)
    if isinstance(motor_cfg, dict) and keys & motor_cfg.keys():
        return {key: float(motor_cfg[key]) for key in keys if key in motor_cfg}
    return {}


class VelocityPID:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        dt: float,
        output_limit: float = math.inf,
        integrator_limit: float | None = None,
        derivative_filter_alpha: float = 1.0,
    ) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.dt = float(dt)
        self.output_limit = abs(float(output_limit)) if math.isfinite(output_limit) else math.inf
        if integrator_limit is None:
            if self.ki != 0.0 and math.isfinite(self.output_limit):
                integrator_limit = self.output_limit / max(abs(self.ki), 1e-9)
            else:
                integrator_limit = math.inf
        self.integrator_limit = abs(float(integrator_limit))
        self.alpha = max(0.0, min(1.0, float(derivative_filter_alpha)))
        self.integral = 0.0
        self.prev_error = 0.0
        self.derivative_state = 0.0

    @classmethod
    def from_config(cls, cfg: Dict[str, object], dt: float, output_limit: float) -> "VelocityPID":
        cfg = cfg if isinstance(cfg, dict) else {}
        return cls(
            kp=float(cfg.get("kp", 0.0)),
            ki=float(cfg.get("ki", 0.0)),
            kd=float(cfg.get("kd", 0.0)),
            dt=dt,
            output_limit=output_limit,
            integrator_limit=cfg.get("integrator_limit"),
            derivative_filter_alpha=float(cfg.get("derivative_filter_alpha", 1.0)),
        )

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.derivative_state = 0.0

    def update(self, target: float, measured: float) -> float:
        error = float(target - measured)
        derivative = (error - self.prev_error) / self.dt if self.dt > 0.0 else 0.0
        self.prev_error = error
        self.derivative_state += self.alpha * (derivative - self.derivative_state)

        tentative_integral = self.integral + error * self.dt
        tentative_integral = _clamp(tentative_integral, self.integrator_limit)
        output = (
            self.kp * error
            + self.ki * tentative_integral
            + self.kd * self.derivative_state
        )
        output_limited = _clamp(output, self.output_limit)

        if self.ki == 0.0 or output_limited == output or output_limited * error < 0.0:
            self.integral = tentative_integral

        return output_limited

    def describe(self) -> str:
        return (
            f"kp={self.kp:g},ki={self.ki:g},kd={self.kd:g},"
            f"ilim={self.integrator_limit:g},alpha={self.alpha:g}"
        )


class CsvExperimentLogger:
    def __init__(
        self,
        path: Path,
        header: Sequence[str],
        metadata: Dict[str, str],
        step_lines: Iterable[str],
    ) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="")
        for key, value in metadata.items():
            self._file.write(f"# {key}={value}\n")
        for line in step_lines:
            self._file.write(f"# {line}\n")
        self._writer = csv.writer(self._file)
        self._writer.writerow(header)

    def log(self, row: Sequence[object]) -> None:
        self._writer.writerow(row)

    def close(self) -> None:
        if not self._file.closed:
            self._file.flush()
            self._file.close()


def _plot_results(csv_path: Path, enabled: bool) -> None:
    if not enabled:
        return
    try:
        import plot_prelim_identification
    except Exception:
        _LOGGER.exception("Failed to import plot_prelim_identification.")
        return
    try:
        plot_prelim_identification.main([str(csv_path), "--experiment", "motor2_only"])
    except SystemExit:
        pass
    except Exception:
        _LOGGER.exception("Failed to plot experiment results.")


def _build_velocity_steps(cfg: Dict[str, object]) -> List[VelocityStep]:
    base_hold = float(cfg.get("hold_time_s", 1.5))
    base_settle = float(cfg.get("settle_time_s", 0.5))
    base_rest = float(cfg.get("rest_time_s", 0.0))
    initial_wait = float(cfg.get("initial_wait_s", 0.0))
    cooldown = float(cfg.get("cooldown_s", 0.0))

    exp_cfg = cfg.get("motor2_only", {}) if isinstance(cfg.get("motor2_only"), dict) else {}
    if not exp_cfg:
        exp_cfg = cfg.get("exp5", {}) if isinstance(cfg.get("exp5"), dict) else {}
    hold = float(exp_cfg.get("hold_time_s", base_hold))
    settle = float(exp_cfg.get("settle_time_s", base_settle))
    rest = float(exp_cfg.get("rest_time_s", base_rest))
    vel_list = _as_list(exp_cfg.get("motor2_velocity_rad_s", [1, 2, -1, -2]))

    steps: List[VelocityStep] = []
    if initial_wait > 0.0:
        steps.append(
            VelocityStep(
                duration_s=initial_wait,
                cmd_w2_rad_s=0.0,
                steady_start_s=math.inf,
                label="initial_wait",
            )
        )

    for w2 in vel_list:
        steps.append(
            VelocityStep(
                duration_s=hold,
                cmd_w2_rad_s=float(w2),
                steady_start_s=settle,
                label=f"w2={w2:g}",
            )
        )
        if rest > 0.0:
            steps.append(
                VelocityStep(
                    duration_s=rest,
                    cmd_w2_rad_s=0.0,
                    steady_start_s=math.inf,
                    label="rest",
                )
            )

    if cooldown > 0.0:
        steps.append(
            VelocityStep(
                duration_s=cooldown,
                cmd_w2_rad_s=0.0,
                steady_start_s=math.inf,
                label="cooldown",
            )
        )

    return steps


def _build_torque_steps(
    cfg: Dict[str, object],
    torque_limit: float,
    torque_set: Sequence[float] | None = None,
) -> List[TorqueStep]:
    base_hold = float(cfg.get("hold_time_s", 1.5))
    base_settle = float(cfg.get("settle_time_s", 0.5))
    base_rest = float(cfg.get("rest_time_s", 0.0))
    initial_wait = float(cfg.get("initial_wait_s", 0.0))
    cooldown = float(cfg.get("cooldown_s", 0.0))

    exp_cfg = cfg.get("motor2_only_torque", {}) if isinstance(cfg.get("motor2_only_torque"), dict) else {}
    hold = float(exp_cfg.get("hold_time_s", base_hold))
    settle = float(exp_cfg.get("settle_time_s", base_settle))
    rest = float(exp_cfg.get("rest_time_s", base_rest))
    if torque_set is None:
        tau_list = _as_list(exp_cfg.get("torque_set_Nm", [0.1, 0.2, -0.1, -0.2]))
    else:
        tau_list = [float(t) for t in torque_set]
    tau_list = [_clamp(t, torque_limit) for t in tau_list]

    steps: List[TorqueStep] = []
    if initial_wait > 0.0:
        steps.append(
            TorqueStep(
                duration_s=initial_wait,
                cmd_tau_2_Nm=0.0,
                steady_start_s=math.inf,
                label="initial_wait",
            )
        )

    for tau in tau_list:
        steps.append(
            TorqueStep(
                duration_s=hold,
                cmd_tau_2_Nm=float(tau),
                steady_start_s=settle,
                label=f"tau2={tau:g}",
            )
        )
        if rest > 0.0:
            steps.append(
                TorqueStep(
                    duration_s=rest,
                    cmd_tau_2_Nm=0.0,
                    steady_start_s=math.inf,
                    label="rest",
                )
            )

    if cooldown > 0.0:
        steps.append(
            TorqueStep(
                duration_s=cooldown,
                cmd_tau_2_Nm=0.0,
                steady_start_s=math.inf,
                label="cooldown",
            )
        )

    return steps


def _summarize_velocity_steps(steps: Sequence[VelocityStep]) -> List[str]:
    lines: List[str] = []
    for idx, step in enumerate(steps):
        lines.append(
            "Step{:03d}=label:{},w2:{:.4g},dur:{:.4g},steady:{:.4g}".format(
                idx,
                step.label,
                step.cmd_w2_rad_s,
                step.duration_s,
                step.steady_start_s if math.isfinite(step.steady_start_s) else -1.0,
            )
        )
    return lines


def _summarize_torque_steps(steps: Sequence[TorqueStep]) -> List[str]:
    lines: List[str] = []
    for idx, step in enumerate(steps):
        lines.append(
            "Step{:03d}=label:{},tau2:{:.4g},dur:{:.4g},steady:{:.4g}".format(
                idx,
                step.label,
                step.cmd_tau_2_Nm,
                step.duration_s,
                step.steady_start_s if math.isfinite(step.steady_start_s) else -1.0,
            )
        )
    return lines


def _run_experiment(
    config_dir: Path,
    config_path: Path,
    calibrate: bool,
    dry_run: bool,
    plot_results: bool,
    mode: str,
    torque_set: Sequence[float] | None,
    torque_ramp_rate: float | None,
) -> None:
    config = _load_yaml(config_path)
    if not config:
        raise ValueError(f"Experiment config is empty: {config_path}")

    hardware_cfg = _load_yaml(config_dir / "hardware.yaml")
    logger_cfg = _load_yaml(config_dir / "logger.yaml")

    dt = float(config.get("sample_time_s", 0.01))

    controller_cfg: Dict[str, object] = {}
    controller_path = config_dir / "controller.yaml"
    if controller_path.exists():
        controller_cfg = _load_yaml(controller_path)

    torque_limits_cfg: Dict[str, object] = {}
    if isinstance(controller_cfg.get("torque_limits"), dict):
        torque_limits_cfg.update(controller_cfg.get("torque_limits", {}))
    if isinstance(config.get("torque_limits"), dict):
        torque_limits_cfg.update(config.get("torque_limits", {}))
    torque_limit_2 = abs(float(torque_limits_cfg.get("motor2", 1.0)))

    pid_cfg = config.get("velocity_pid", {}) if isinstance(config.get("velocity_pid"), dict) else {}
    pid2 = VelocityPID.from_config(pid_cfg.get("motor2", {}), dt, torque_limit_2)

    ramp_cfg = config.get("command_ramp_rate_rad_s_per_s")
    if isinstance(ramp_cfg, dict):
        ramp_rate_2 = float(ramp_cfg.get("motor2", math.inf))
    elif ramp_cfg is None:
        ramp_rate_2 = math.inf
    else:
        ramp_rate_2 = float(ramp_cfg)
    ramp_rate_2 = abs(ramp_rate_2) if math.isfinite(ramp_rate_2) and ramp_rate_2 > 0 else math.inf

    torque_ramp_cfg = config.get("command_ramp_rate_torque_Nm_s")
    if torque_ramp_rate is None:
        if torque_ramp_cfg is None:
            torque_ramp_rate = math.inf
        else:
            torque_ramp_rate = float(torque_ramp_cfg)
    torque_ramp_rate = (
        abs(float(torque_ramp_rate))
        if torque_ramp_rate is not None and math.isfinite(torque_ramp_rate) and torque_ramp_rate > 0
        else math.inf
    )

    sign_cfg = config.get("measurement_sign", {}) if isinstance(config.get("measurement_sign"), dict) else {}
    sign_motor2 = _normalize_sign(sign_cfg.get("motor2", 1.0))
    command_sign_cfg = (
        config.get("command_sign", {}) if isinstance(config.get("command_sign"), dict) else {}
    )
    cmd_sign_motor2 = _normalize_sign(command_sign_cfg.get("motor2", 1.0))
    zero_torque_on_zero_cmd = bool(config.get("zero_torque_on_zero_cmd", True))
    feedback_limit_cfg = config.get("velocity_feedback_limit_rad_s")
    if isinstance(feedback_limit_cfg, dict):
        feedback_limit = float(feedback_limit_cfg.get("motor2", math.inf))
    elif feedback_limit_cfg is None:
        feedback_limit = math.inf
    else:
        feedback_limit = float(feedback_limit_cfg)
    feedback_limit = abs(feedback_limit) if math.isfinite(feedback_limit) and feedback_limit > 0 else math.inf

    fast_read = bool(config.get("fast_read", True))
    iq_read_interval_s = float(config.get("iq_read_interval_s", 0.05))
    if not math.isfinite(iq_read_interval_s) or iq_read_interval_s <= 0.0:
        iq_read_interval_s = 0.0

    vel_filter_cfg = config.get("velocity_filter_alpha")
    if isinstance(vel_filter_cfg, dict):
        vel_filter_alpha = float(vel_filter_cfg.get("motor2", 1.0))
    elif vel_filter_cfg is None:
        vel_filter_alpha = 1.0
    else:
        vel_filter_alpha = float(vel_filter_cfg)
    vel_filter_alpha = max(0.0, min(1.0, vel_filter_alpha))

    torque_rate_cfg = config.get("torque_rate_limit_Nm_s")
    if isinstance(torque_rate_cfg, dict):
        torque_rate_limit = float(torque_rate_cfg.get("motor2", math.inf))
    elif torque_rate_cfg is None:
        torque_rate_limit = math.inf
    else:
        torque_rate_limit = float(torque_rate_cfg)
    torque_rate_limit = (
        abs(torque_rate_limit)
        if math.isfinite(torque_rate_limit) and torque_rate_limit > 0.0
        else math.inf
    )

    motor2_only_cfg = config.get("motor2_only", {}) if isinstance(config.get("motor2_only"), dict) else {}
    odrive_gains_cfg = (
        motor2_only_cfg.get("odrive_velocity_gains")
        if isinstance(motor2_only_cfg.get("odrive_velocity_gains"), dict)
        else config.get("odrive_velocity_gains")
    )
    odrive_velocity_gains = _extract_odrive_velocity_gains(odrive_gains_cfg, "motor2")

    base_dir = config_dir.parent
    csv_dir = base_dir / str(logger_cfg.get("csv_dir", "csv"))
    prefix = f"{config.get('log_prefix', 'prelim')}_motor2_only"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = csv_dir / f"{prefix}_{timestamp}.csv"

    header = [
        "time_s",
        "step_id",
        "step_time_s",
        "steady",
        "cmd_w2_rad_s",
        "cmd_w2_ramped_rad_s",
        "cmd_tau_2_Nm",
        "meas_w2_rad_s",
        "pos_2_rad",
        "iq_2_A",
        "tau_meas_2_Nm",
    ]

    if mode == "velocity":
        steps = _build_velocity_steps(config)
        total_duration = sum(step.duration_s for step in steps)
        if dry_run:
            _LOGGER.info("Dry-run: %d steps, total %.2f s", len(steps), total_duration)
            for idx, step in enumerate(steps):
                _LOGGER.info(
                    "  %02d: w2=%.3f rad/s, dur=%.2f s, steady=%.2f s, %s",
                    idx,
                    step.cmd_w2_rad_s,
                    step.duration_s,
                    step.steady_start_s if math.isfinite(step.steady_start_s) else -1.0,
                    step.label,
                )
            return

        odrive_iface = ODriveInterface(hardware_cfg)
        odrive_iface.connect(calibrate=calibrate, control_mode="torque", axes=["motor2"])
        _LOGGER.info("Experiment started (motor2_only:%s). Logging to %s", mode, csv_path)

        metadata = {
            "Experiment": "motor2_only",
            "Mode": "velocity",
            "SampleTime": f"{dt:g}",
            "TotalDuration": f"{total_duration:g}",
            "UnitsVelocity": "rad/s",
            "UnitsPosition": "rad",
            "TorqueLimits": f"motor2={torque_limit_2:g}",
            "VelocityPID_motor2": pid2.describe(),
            "CommandRampRate": f"motor2={ramp_rate_2:g}",
            "MeasurementSign": f"motor2={sign_motor2:g}",
            "CommandSign": f"motor2={cmd_sign_motor2:g}",
            "ZeroTorqueOnZeroCmd": str(bool(zero_torque_on_zero_cmd)),
            "VelocityFeedbackLimit": f"{feedback_limit:g}",
            "VelocityFilterAlpha": f"{vel_filter_alpha:g}",
            "TorqueRateLimit": f"{torque_rate_limit:g}",
            "FastRead": str(bool(fast_read)),
            "IqReadInterval": f"{iq_read_interval_s:g}",
        }
        step_lines = _summarize_velocity_steps(steps)
        data_logger = CsvExperimentLogger(csv_path, header, metadata, step_lines)

        start_time = time.time()
        loop_time = start_time
        cmd_w2_ramped = 0.0
        tau_cmd_ramped = 0.0
        last_iq_time = -math.inf
        cached_iq2 = 0.0
        cached_tau2 = 0.0
        meas_w2_filtered = 0.0
        try:
            for step_idx, step in enumerate(steps):
                _LOGGER.info(
                    "Step %d/%d: w2=%.3f rad/s, dur=%.2f s (%s)",
                    step_idx + 1,
                    len(steps),
                    step.cmd_w2_rad_s,
                    step.duration_s,
                    step.label,
                )
                pid2.reset()
                step_start = time.time()

                while True:
                    now = time.time()
                    elapsed = now - start_time
                    step_elapsed = now - step_start
                    if step_elapsed >= step.duration_s:
                        break

                    use_fast = False
                    if fast_read and iq_read_interval_s > 0.0:
                        use_fast = (now - last_iq_time) < iq_read_interval_s
                    states = odrive_iface.read_states(fast=use_fast)
                    motor2_state = states["motor2"]
                    if use_fast:
                        iq2 = cached_iq2
                        tau_meas_2 = cached_tau2
                    else:
                        cached_iq2 = motor2_state.current_iq
                        cached_tau2 = motor2_state.torque_measured
                        iq2 = cached_iq2
                        tau_meas_2 = cached_tau2
                        last_iq_time = now
                    meas_w2 = sign_motor2 * motor2_state.velocity * RAD_PER_TURN
                    if vel_filter_alpha < 1.0:
                        meas_w2_filtered = (
                            vel_filter_alpha * meas_w2 + (1.0 - vel_filter_alpha) * meas_w2_filtered
                        )
                    else:
                        meas_w2_filtered = meas_w2

                    cmd_w2_ramped = _ramp_value(cmd_w2_ramped, step.cmd_w2_rad_s, ramp_rate_2, dt)
                    meas_w2_ctrl = (
                        max(-feedback_limit, min(feedback_limit, meas_w2_filtered))
                        if math.isfinite(feedback_limit)
                        else meas_w2_filtered
                    )
                    if zero_torque_on_zero_cmd and abs(step.cmd_w2_rad_s) <= 1e-6:
                        tau_cmd_2 = 0.0
                    else:
                        tau_cmd_2 = cmd_sign_motor2 * pid2.update(cmd_w2_ramped, meas_w2_ctrl)
                    tau_cmd_ramped = _ramp_value(tau_cmd_ramped, tau_cmd_2, torque_rate_limit, dt)
                    odrive_iface.command_torques(0.0, tau_cmd_ramped)

                    reached_target = abs(cmd_w2_ramped - step.cmd_w2_rad_s) <= 1e-3
                    steady = 1 if (step_elapsed >= step.steady_start_s and reached_target) else 0

                    data_logger.log(
                        [
                            elapsed,
                            step_idx,
                            step_elapsed,
                            steady,
                            step.cmd_w2_rad_s,
                            cmd_w2_ramped,
                            tau_cmd_ramped,
                            meas_w2,
                            sign_motor2 * motor2_state.position * RAD_PER_TURN,
                            iq2,
                            tau_meas_2,
                        ]
                    )

                    loop_time += dt
                    # sleep = loop_time - time.time()
                    # if sleep > 0:
                    #     time.sleep(sleep)
                    # else:
                    #     loop_time = time.time()
        except KeyboardInterrupt:
            _LOGGER.info("Interrupted by user.")
        finally:
            try:
                odrive_iface.command_torques(0.0, 0.0)
                odrive_iface.shutdown()
            except Exception:
                _LOGGER.exception("Failed to safely shutdown ODrive.")
            data_logger.close()
            _LOGGER.info("Experiment finished. Log saved to %s", csv_path)
        _plot_results(csv_path, plot_results)
        return

    if mode == "odrive":
        steps = _build_velocity_steps(config)
        total_duration = sum(step.duration_s for step in steps)
        if dry_run:
            _LOGGER.info("Dry-run: %d steps, total %.2f s", len(steps), total_duration)
            for idx, step in enumerate(steps):
                _LOGGER.info(
                    "  %02d: w2=%.3f rad/s, dur=%.2f s, steady=%.2f s, %s",
                    idx,
                    step.cmd_w2_rad_s,
                    step.duration_s,
                    step.steady_start_s if math.isfinite(step.steady_start_s) else -1.0,
                    step.label,
                )
            return

        odrive_iface = ODriveInterface(hardware_cfg)
        odrive_iface.connect(calibrate=calibrate, control_mode="velocity", axes=["motor2"])
        gains_applied: Dict[str, float] = {}
        if odrive_velocity_gains:
            gains_applied = odrive_iface.set_velocity_gains(
                "motor2",
                vel_gain=odrive_velocity_gains.get("vel_gain"),
                vel_integrator_gain=odrive_velocity_gains.get("vel_integrator_gain"),
                vel_integrator_limit=odrive_velocity_gains.get("vel_integrator_limit"),
            )
            if gains_applied:
                _LOGGER.info(
                    "Applied ODrive velocity gains on motor2: %s",
                    _format_gain_map(gains_applied),
                )
            else:
                _LOGGER.warning(
                    "ODrive velocity gains configured but controller attributes missing; skipped."
                )
        _LOGGER.info("Experiment started (motor2_only:odrive). Logging to %s", csv_path)

        metadata = {
            "Experiment": "motor2_only",
            "Mode": "odrive_velocity",
            "SampleTime": f"{dt:g}",
            "TotalDuration": f"{total_duration:g}",
            "UnitsVelocity": "rad/s",
            "UnitsPosition": "rad",
            "CommandRampRate": f"motor2={ramp_rate_2:g}",
            "MeasurementSign": f"motor2={sign_motor2:g}",
            "CommandSign": f"motor2={cmd_sign_motor2:g}",
            "FastRead": str(bool(fast_read)),
            "IqReadInterval": f"{iq_read_interval_s:g}",
        }
        if odrive_velocity_gains:
            metadata["ODriveVelocityGainsRequested"] = _format_gain_map(odrive_velocity_gains)
        if gains_applied:
            metadata["ODriveVelocityGainsApplied"] = _format_gain_map(gains_applied)
        step_lines = _summarize_velocity_steps(steps)
        data_logger = CsvExperimentLogger(csv_path, header, metadata, step_lines)

        start_time = time.time()
        loop_time = start_time
        cmd_w2_ramped = 0.0
        last_iq_time = -math.inf
        cached_iq2 = 0.0
        cached_tau2 = 0.0
        try:
            for step_idx, step in enumerate(steps):
                _LOGGER.info(
                    "Step %d/%d: w2=%.3f rad/s, dur=%.2f s (%s)",
                    step_idx + 1,
                    len(steps),
                    step.cmd_w2_rad_s,
                    step.duration_s,
                    step.label,
                )
                step_start = time.time()

                while True:
                    now = time.time()
                    elapsed = now - start_time
                    step_elapsed = now - step_start
                    if step_elapsed >= step.duration_s:
                        break

                    use_fast = False
                    if fast_read and iq_read_interval_s > 0.0:
                        use_fast = (now - last_iq_time) < iq_read_interval_s
                    states = odrive_iface.read_states(fast=use_fast)
                    motor2_state = states["motor2"]
                    if use_fast:
                        iq2 = cached_iq2
                        tau_meas_2 = cached_tau2
                    else:
                        cached_iq2 = motor2_state.current_iq
                        cached_tau2 = motor2_state.torque_measured
                        iq2 = cached_iq2
                        tau_meas_2 = cached_tau2
                        last_iq_time = now
                    meas_w2 = sign_motor2 * motor2_state.velocity * RAD_PER_TURN

                    cmd_w2_ramped = _ramp_value(cmd_w2_ramped, step.cmd_w2_rad_s, ramp_rate_2, dt)
                    cmd_turn_s = cmd_sign_motor2 * cmd_w2_ramped * TURN_PER_RAD
                    odrive_iface.command_velocities(0.0, cmd_turn_s)

                    reached_target = abs(cmd_w2_ramped - step.cmd_w2_rad_s) <= 1e-3
                    steady = 1 if (step_elapsed >= step.steady_start_s and reached_target) else 0

                    data_logger.log(
                        [
                            elapsed,
                            step_idx,
                            step_elapsed,
                            steady,
                            step.cmd_w2_rad_s,
                            cmd_w2_ramped,
                            math.nan,
                            meas_w2,
                            sign_motor2 * motor2_state.position * RAD_PER_TURN,
                            iq2,
                            tau_meas_2,
                        ]
                    )

                    loop_time += dt
                    # sleep = loop_time - time.time()
                    # if sleep > 0:
                    #     time.sleep(sleep)
                    # else:
                    #     loop_time = time.time()
        except KeyboardInterrupt:
            _LOGGER.info("Interrupted by user.")
        finally:
            try:
                odrive_iface.command_velocities(0.0, 0.0)
                odrive_iface.shutdown()
            except Exception:
                _LOGGER.exception("Failed to safely shutdown ODrive.")
            data_logger.close()
            _LOGGER.info("Experiment finished. Log saved to %s", csv_path)
        _plot_results(csv_path, plot_results)
        return

    steps = _build_torque_steps(config, torque_limit_2, torque_set)
    total_duration = sum(step.duration_s for step in steps)
    if dry_run:
        _LOGGER.info("Dry-run: %d steps, total %.2f s", len(steps), total_duration)
        for idx, step in enumerate(steps):
            _LOGGER.info(
                "  %02d: tau2=%.3f Nm, dur=%.2f s, steady=%.2f s, %s",
                idx,
                step.cmd_tau_2_Nm,
                step.duration_s,
                step.steady_start_s if math.isfinite(step.steady_start_s) else -1.0,
                step.label,
            )
        return

    odrive_iface = ODriveInterface(hardware_cfg)
    odrive_iface.connect(calibrate=calibrate, control_mode="torque", axes=["motor2"])
    _LOGGER.info("Experiment started (motor2_only:%s). Logging to %s", mode, csv_path)

    metadata = {
        "Experiment": "motor2_only",
        "Mode": "torque",
        "SampleTime": f"{dt:g}",
        "TotalDuration": f"{total_duration:g}",
        "UnitsVelocity": "rad/s",
        "UnitsPosition": "rad",
        "TorqueLimits": f"motor2={torque_limit_2:g}",
        "CommandRampRate": f"motor2={ramp_rate_2:g}",
        "TorqueRampRate": f"motor2={torque_ramp_rate:g}",
        "MeasurementSign": f"motor2={sign_motor2:g}",
        "CommandSign": f"motor2={cmd_sign_motor2:g}",
        "FastRead": str(bool(fast_read)),
        "IqReadInterval": f"{iq_read_interval_s:g}",
    }
    step_lines = _summarize_torque_steps(steps)
    data_logger = CsvExperimentLogger(csv_path, header, metadata, step_lines)

    start_time = time.time()
    loop_time = start_time
    cmd_tau_ramped = 0.0
    last_iq_time = -math.inf
    cached_iq2 = 0.0
    cached_tau2 = 0.0
    try:
        for step_idx, step in enumerate(steps):
            _LOGGER.info(
                "Step %d/%d: tau2=%.3f Nm, dur=%.2f s (%s)",
                step_idx + 1,
                len(steps),
                step.cmd_tau_2_Nm,
                step.duration_s,
                step.label,
            )
            step_start = time.time()
            while True:
                now = time.time()
                elapsed = now - start_time
                step_elapsed = now - step_start
                if step_elapsed >= step.duration_s:
                    break

                use_fast = False
                if fast_read and iq_read_interval_s > 0.0:
                    use_fast = (now - last_iq_time) < iq_read_interval_s
                states = odrive_iface.read_states(fast=use_fast)
                motor2_state = states["motor2"]
                if use_fast:
                    iq2 = cached_iq2
                    tau_meas_2 = cached_tau2
                else:
                    cached_iq2 = motor2_state.current_iq
                    cached_tau2 = motor2_state.torque_measured
                    iq2 = cached_iq2
                    tau_meas_2 = cached_tau2
                    last_iq_time = now
                meas_w2 = sign_motor2 * motor2_state.velocity * RAD_PER_TURN

                cmd_tau_ramped = _ramp_value(
                    cmd_tau_ramped, step.cmd_tau_2_Nm, torque_ramp_rate, dt
                )
                cmd_tau_applied = cmd_sign_motor2 * cmd_tau_ramped
                odrive_iface.command_torques(0.0, cmd_tau_applied)

                reached_target = abs(cmd_tau_ramped - step.cmd_tau_2_Nm) <= 1e-4
                steady = 1 if (step_elapsed >= step.steady_start_s and reached_target) else 0

                data_logger.log(
                    [
                        elapsed,
                        step_idx,
                        step_elapsed,
                        steady,
                        math.nan,
                        math.nan,
                        cmd_tau_applied,
                        meas_w2,
                        sign_motor2 * motor2_state.position * RAD_PER_TURN,
                        iq2,
                        tau_meas_2,
                    ]
                )

                loop_time += dt
                # sleep = loop_time - time.time()
                # if sleep > 0:
                #     time.sleep(sleep)
                # else:
                #     loop_time = time.time()
    except KeyboardInterrupt:
        _LOGGER.info("Interrupted by user.")
    finally:
        try:
            odrive_iface.command_torques(0.0, 0.0)
            odrive_iface.shutdown()
        except Exception:
            _LOGGER.exception("Failed to safely shutdown ODrive.")
        data_logger.close()
        _LOGGER.info("Experiment finished. Log saved to %s", csv_path)
    _plot_results(csv_path, plot_results)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run motor2-only velocity experiments (torque control).")
    parser.add_argument("--config-dir", type=Path, default=Path(__file__).parent / "config")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to prelim_identification.yaml (defaults to config-dir).",
    )
    parser.add_argument("--calibrate", action="store_true", help="Run full calibration on connect.")
    parser.add_argument("--dry-run", action="store_true", help="Print the step table and exit.")
    parser.add_argument("--no-plot", action="store_true", help="Skip auto-plot after experiment.")
    parser.add_argument("--mode", choices=["velocity", "torque", "odrive"], default="velocity")
    parser.add_argument(
        "--torque-set",
        type=float,
        nargs="+",
        default=None,
        help="Torque steps for --mode torque (Nm). Overrides config.",
    )
    parser.add_argument(
        "--torque-ramp-rate",
        type=float,
        default=None,
        help="Torque ramp rate for --mode torque (Nm/s).",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config_path = args.config or (args.config_dir / "prelim_identification.yaml")
    _run_experiment(
        args.config_dir,
        config_path,
        args.calibrate,
        args.dry_run,
        not args.no_plot,
        args.mode,
        args.torque_set,
        args.torque_ramp_rate,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
