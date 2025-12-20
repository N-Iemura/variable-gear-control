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


_LOGGER = logging.getLogger("prelim_identification")

RAD_PER_TURN = 2.0 * math.pi
TURN_PER_RAD = 1.0 / RAD_PER_TURN


@dataclass(frozen=True)
class Step:
    duration_s: float
    cmd_w1_rad_s: float
    cmd_w2_rad_s: float
    steady_start_s: float
    label: str


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


def _plot_results(csv_path: Path, exp_name: str, enabled: bool) -> None:
    if not enabled:
        return
    try:
        import plot_prelim_identification
    except Exception:
        _LOGGER.exception("Failed to import plot_prelim_identification.")
        return
    try:
        plot_prelim_identification.main([str(csv_path), "--experiment", exp_name])
    except SystemExit:
        pass
    except Exception:
        _LOGGER.exception("Failed to plot experiment results.")


def _load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _as_list(value: object) -> List[float]:
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    if value is None:
        return []
    return [float(value)]


def _build_steps(exp_name: str, cfg: Dict[str, object]) -> List[Step]:
    base_hold = float(cfg.get("hold_time_s", 1.5))
    base_settle = float(cfg.get("settle_time_s", 0.5))
    base_rest = float(cfg.get("rest_time_s", 0.0))
    initial_wait = float(cfg.get("initial_wait_s", 0.0))
    cooldown = float(cfg.get("cooldown_s", 0.0))

    exp_cfg = cfg.get(exp_name, {}) if isinstance(cfg.get(exp_name), dict) else {}
    hold = float(exp_cfg.get("hold_time_s", base_hold))
    settle = float(exp_cfg.get("settle_time_s", base_settle))
    rest = float(exp_cfg.get("rest_time_s", base_rest))

    steps: List[Step] = []
    if initial_wait > 0.0:
        steps.append(
            Step(
                duration_s=initial_wait,
                cmd_w1_rad_s=0.0,
                cmd_w2_rad_s=0.0,
                steady_start_s=math.inf,
                label="initial_wait",
            )
        )

    if exp_name == "exp1":
        vel_set = _as_list(exp_cfg.get("velocity_set_rad_s", [-15, -10, -5, 5, 10, 15]))
        for w1 in vel_set:
            for w2 in vel_set:
                steps.append(
                    Step(
                        duration_s=hold,
                        cmd_w1_rad_s=float(w1),
                        cmd_w2_rad_s=float(w2),
                        steady_start_s=settle,
                        label=f"w1={w1:g},w2={w2:g}",
                    )
                )
                if rest > 0.0:
                    steps.append(
                        Step(
                            duration_s=rest,
                            cmd_w1_rad_s=0.0,
                            cmd_w2_rad_s=0.0,
                            steady_start_s=math.inf,
                            label="rest",
                        )
                    )
    elif exp_name == "exp2":
        w1_ref = float(exp_cfg.get("motor1_velocity_rad_s", 10.0))
        rho_list = _as_list(exp_cfg.get("rho_list", [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]))
        for rho in rho_list:
            w2_ref = float(rho) * w1_ref
            steps.append(
                Step(
                    duration_s=hold,
                    cmd_w1_rad_s=w1_ref,
                    cmd_w2_rad_s=w2_ref,
                    steady_start_s=settle,
                    label=f"rho={rho:g}",
                )
            )
            if rest > 0.0:
                steps.append(
                    Step(
                        duration_s=rest,
                        cmd_w1_rad_s=0.0,
                        cmd_w2_rad_s=0.0,
                        steady_start_s=math.inf,
                        label="rest",
                    )
                )
    elif exp_name == "exp3":
        vel_list = _as_list(exp_cfg.get("motor1_velocity_rad_s", [2, 5, 10, 15, -2, -5, -10, -15]))
        for w1 in vel_list:
            steps.append(
                Step(
                    duration_s=hold,
                    cmd_w1_rad_s=float(w1),
                    cmd_w2_rad_s=0.0,
                    steady_start_s=settle,
                    label=f"w1={w1:g}",
                )
            )
            if rest > 0.0:
                steps.append(
                    Step(
                        duration_s=rest,
                        cmd_w1_rad_s=0.0,
                        cmd_w2_rad_s=0.0,
                        steady_start_s=math.inf,
                        label="rest",
                    )
                )
    elif exp_name == "exp4":
        w1_ref = float(exp_cfg.get("motor1_velocity_rad_s", 0.3))
        cycles = int(exp_cfg.get("cycles", 4))
        pause = float(exp_cfg.get("pause_time_s", 0.0))
        settle = float(exp_cfg.get("settle_time_s", base_settle))
        hold = float(exp_cfg.get("hold_time_s", base_hold))
        for idx in range(cycles):
            steps.append(
                Step(
                    duration_s=hold,
                    cmd_w1_rad_s=w1_ref,
                    cmd_w2_rad_s=0.0,
                    steady_start_s=settle,
                    label=f"cycle{idx + 1}_plus",
                )
            )
            if pause > 0.0:
                steps.append(
                    Step(
                        duration_s=pause,
                        cmd_w1_rad_s=0.0,
                        cmd_w2_rad_s=0.0,
                        steady_start_s=math.inf,
                        label="pause",
                    )
                )
            steps.append(
                Step(
                    duration_s=hold,
                    cmd_w1_rad_s=-w1_ref,
                    cmd_w2_rad_s=0.0,
                    steady_start_s=settle,
                    label=f"cycle{idx + 1}_minus",
                )
            )
            if pause > 0.0:
                steps.append(
                    Step(
                        duration_s=pause,
                        cmd_w1_rad_s=0.0,
                        cmd_w2_rad_s=0.0,
                        steady_start_s=math.inf,
                        label="pause",
                    )
                )
    elif exp_name == "exp5":
        vel_list = _as_list(exp_cfg.get("motor2_velocity_rad_s", [2, 5, 10, 15, -2, -5, -10, -15]))
        for w2 in vel_list:
            steps.append(
                Step(
                    duration_s=hold,
                    cmd_w1_rad_s=0.0,
                    cmd_w2_rad_s=float(w2),
                    steady_start_s=settle,
                    label=f"w2={w2:g}",
                )
            )
            if rest > 0.0:
                steps.append(
                    Step(
                        duration_s=rest,
                        cmd_w1_rad_s=0.0,
                        cmd_w2_rad_s=0.0,
                        steady_start_s=math.inf,
                        label="rest",
                    )
                )
    else:
        raise ValueError(f"Unsupported experiment: {exp_name}")

    if cooldown > 0.0:
        steps.append(
            Step(
                duration_s=cooldown,
                cmd_w1_rad_s=0.0,
                cmd_w2_rad_s=0.0,
                steady_start_s=math.inf,
                label="cooldown",
            )
        )
    return steps


def _summarize_steps(steps: Sequence[Step]) -> List[str]:
    lines: List[str] = []
    for idx, step in enumerate(steps):
        lines.append(
            "Step{:03d}=label:{},w1:{:.4g},w2:{:.4g},dur:{:.4g},steady:{:.4g}".format(
                idx,
                step.label,
                step.cmd_w1_rad_s,
                step.cmd_w2_rad_s,
                step.duration_s,
                step.steady_start_s if math.isfinite(step.steady_start_s) else -1.0,
            )
        )
    return lines


def _run_experiment(
    exp_name: str,
    config_dir: Path,
    config_path: Path,
    calibrate: bool,
    dry_run: bool,
    plot_results: bool,
) -> None:
    config = _load_yaml(config_path)
    if not config:
        raise ValueError(f"Experiment config is empty: {config_path}")

    hardware_cfg = _load_yaml(config_dir / "hardware.yaml")
    logger_cfg = _load_yaml(config_dir / "logger.yaml")

    dt = float(config.get("sample_time_s", 0.01))
    steps = _build_steps(exp_name, config)
    total_duration = sum(step.duration_s for step in steps)

    if dry_run:
        _LOGGER.info("Dry-run: %d steps, total %.2f s", len(steps), total_duration)
        for idx, step in enumerate(steps):
            _LOGGER.info(
                "  %02d: w1=%.3f rad/s, w2=%.3f rad/s, dur=%.2f s, steady=%.2f s, %s",
                idx,
                step.cmd_w1_rad_s,
                step.cmd_w2_rad_s,
                step.duration_s,
                step.steady_start_s if math.isfinite(step.steady_start_s) else -1.0,
                step.label,
            )
        return

    controller_cfg: Dict[str, object] = {}
    controller_path = config_dir / "controller.yaml"
    if controller_path.exists():
        controller_cfg = _load_yaml(controller_path)

    torque_limits_cfg: Dict[str, object] = {}
    if isinstance(controller_cfg.get("torque_limits"), dict):
        torque_limits_cfg.update(controller_cfg.get("torque_limits", {}))
    if isinstance(config.get("torque_limits"), dict):
        torque_limits_cfg.update(config.get("torque_limits", {}))
    torque_limit_1 = abs(float(torque_limits_cfg.get("motor1", 1.0)))
    torque_limit_2 = abs(float(torque_limits_cfg.get("motor2", 1.0)))

    pid_cfg = config.get("velocity_pid", {}) if isinstance(config.get("velocity_pid"), dict) else {}
    pid1 = VelocityPID.from_config(pid_cfg.get("motor1", {}), dt, torque_limit_1)
    pid2 = VelocityPID.from_config(pid_cfg.get("motor2", {}), dt, torque_limit_2)

    ramp_cfg = config.get("command_ramp_rate_rad_s_per_s")
    if isinstance(ramp_cfg, dict):
        ramp_rate_1 = float(ramp_cfg.get("motor1", math.inf))
        ramp_rate_2 = float(ramp_cfg.get("motor2", ramp_rate_1))
    elif ramp_cfg is None:
        ramp_rate_1 = math.inf
        ramp_rate_2 = math.inf
    else:
        ramp_rate_1 = float(ramp_cfg)
        ramp_rate_2 = float(ramp_cfg)
    ramp_rate_1 = abs(ramp_rate_1) if math.isfinite(ramp_rate_1) and ramp_rate_1 > 0 else math.inf
    ramp_rate_2 = abs(ramp_rate_2) if math.isfinite(ramp_rate_2) and ramp_rate_2 > 0 else math.inf

    sign_cfg = config.get("measurement_sign", {}) if isinstance(config.get("measurement_sign"), dict) else {}
    sign_motor1 = _normalize_sign(sign_cfg.get("motor1", 1.0))
    sign_motor2 = _normalize_sign(sign_cfg.get("motor2", 1.0))
    sign_output = _normalize_sign(sign_cfg.get("output", 1.0))
    command_sign_cfg = (
        config.get("command_sign", {}) if isinstance(config.get("command_sign"), dict) else {}
    )
    cmd_sign_motor1 = _normalize_sign(command_sign_cfg.get("motor1", 1.0))
    cmd_sign_motor2 = _normalize_sign(command_sign_cfg.get("motor2", 1.0))

    fast_read = bool(config.get("fast_read", True))
    iq_read_interval_s = float(config.get("iq_read_interval_s", 0.05))
    if not math.isfinite(iq_read_interval_s) or iq_read_interval_s <= 0.0:
        iq_read_interval_s = 0.0

    base_dir = config_dir.parent
    csv_dir = base_dir / str(logger_cfg.get("csv_dir", "csv"))
    prefix = str(config.get("log_prefix", "prelim"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = csv_dir / f"{prefix}_{exp_name}_{timestamp}.csv"

    metadata = {
        "Experiment": exp_name,
        "SampleTime": f"{dt:g}",
        "TotalDuration": f"{total_duration:g}",
        "UnitsVelocity": "rad/s",
        "UnitsPosition": "rad",
        "TorqueLimits": f"motor1={torque_limit_1:g},motor2={torque_limit_2:g}",
        "VelocityPID_motor1": pid1.describe(),
        "VelocityPID_motor2": pid2.describe(),
        "CommandRampRate": f"motor1={ramp_rate_1:g},motor2={ramp_rate_2:g}",
        "MeasurementSign": f"motor1={sign_motor1:g},motor2={sign_motor2:g},output={sign_output:g}",
        "CommandSign": f"motor1={cmd_sign_motor1:g},motor2={cmd_sign_motor2:g}",
        "FastRead": str(bool(fast_read)),
        "IqReadInterval": f"{iq_read_interval_s:g}",
    }
    kinematic_cfg = config.get("kinematic", {}) if isinstance(config.get("kinematic"), dict) else {}
    if "a1" in kinematic_cfg:
        metadata["KinematicA1"] = f"{float(kinematic_cfg['a1']):g}"
    if "a2" in kinematic_cfg:
        metadata["KinematicA2"] = f"{float(kinematic_cfg['a2']):g}"
    step_lines = _summarize_steps(steps)

    header = [
        "time_s",
        "step_id",
        "step_time_s",
        "steady",
        "cmd_w1_rad_s",
        "cmd_w2_rad_s",
        "cmd_rho",
        "cmd_w1_ramped_rad_s",
        "cmd_w2_ramped_rad_s",
        "cmd_tau_1_Nm",
        "cmd_tau_2_Nm",
        "meas_w1_rad_s",
        "meas_w2_rad_s",
        "meas_w_out_rad_s",
        "pos_1_rad",
        "pos_2_rad",
        "pos_out_rad",
        "iq_1_A",
        "iq_2_A",
        "tau_meas_1_Nm",
        "tau_meas_2_Nm",
    ]

    data_logger = CsvExperimentLogger(csv_path, header, metadata, step_lines)

    odrive_iface = ODriveInterface(hardware_cfg)
    odrive_iface.connect(calibrate=calibrate, control_mode="torque")
    odrive_iface.zero_positions()
    _LOGGER.info("Experiment started (%s). Logging to %s", exp_name, csv_path)

    start_time = time.time()
    loop_time = start_time
    cmd_w1_ramped = 0.0
    cmd_w2_ramped = 0.0
    last_iq_time = -math.inf
    cached_iq1 = 0.0
    cached_iq2 = 0.0
    cached_tau1 = 0.0
    cached_tau2 = 0.0
    try:
        for step_idx, step in enumerate(steps):
            _LOGGER.info(
                "Step %d/%d: w1=%.3f rad/s, w2=%.3f rad/s, dur=%.2f s (%s)",
                step_idx + 1,
                len(steps),
                step.cmd_w1_rad_s,
                step.cmd_w2_rad_s,
                step.duration_s,
                step.label,
            )
            pid1.reset()
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
                motor1_state = states["motor1"]
                motor2_state = states["motor2"]
                output_state = states.get("output", motor1_state)
                if use_fast:
                    iq1 = cached_iq1
                    iq2 = cached_iq2
                    tau_meas_1 = cached_tau1
                    tau_meas_2 = cached_tau2
                else:
                    cached_iq1 = motor1_state.current_iq
                    cached_iq2 = motor2_state.current_iq
                    cached_tau1 = motor1_state.torque_measured
                    cached_tau2 = motor2_state.torque_measured
                    iq1 = cached_iq1
                    iq2 = cached_iq2
                    tau_meas_1 = cached_tau1
                    tau_meas_2 = cached_tau2
                    last_iq_time = now
                meas_w1 = sign_motor1 * motor1_state.velocity * RAD_PER_TURN
                meas_w2 = sign_motor2 * motor2_state.velocity * RAD_PER_TURN
                meas_w_out = sign_output * output_state.velocity * RAD_PER_TURN

                cmd_w1_ramped = _ramp_value(
                    cmd_w1_ramped, step.cmd_w1_rad_s, ramp_rate_1, dt
                )
                cmd_w2_ramped = _ramp_value(
                    cmd_w2_ramped, step.cmd_w2_rad_s, ramp_rate_2, dt
                )

                tau_cmd_1 = cmd_sign_motor1 * pid1.update(cmd_w1_ramped, meas_w1)
                tau_cmd_2 = cmd_sign_motor2 * pid2.update(cmd_w2_ramped, meas_w2)
                odrive_iface.command_torques(tau_cmd_1, tau_cmd_2)

                reached_target = (
                    abs(cmd_w1_ramped - step.cmd_w1_rad_s) <= 1e-3
                    and abs(cmd_w2_ramped - step.cmd_w2_rad_s) <= 1e-3
                )
                steady = 1 if (step_elapsed >= step.steady_start_s and reached_target) else 0
                cmd_rho = (
                    step.cmd_w2_rad_s / step.cmd_w1_rad_s
                    if abs(step.cmd_w1_rad_s) > 1e-9
                    else math.nan
                )

                data_logger.log(
                    [
                        elapsed,
                        step_idx,
                        step_elapsed,
                        steady,
                        step.cmd_w1_rad_s,
                        step.cmd_w2_rad_s,
                        cmd_rho,
                        cmd_w1_ramped,
                        cmd_w2_ramped,
                        tau_cmd_1,
                        tau_cmd_2,
                        meas_w1,
                        meas_w2,
                        meas_w_out,
                        sign_motor1 * motor1_state.position * RAD_PER_TURN,
                        sign_motor2 * motor2_state.position * RAD_PER_TURN,
                        sign_output * output_state.position * RAD_PER_TURN,
                        iq1,
                        iq2,
                        tau_meas_1,
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
    _plot_results(csv_path, exp_name, plot_results)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run preliminary identification experiments (kinematics, ratio, friction, backlash)."
    )
    parser.add_argument("experiment", choices=["exp1", "exp2", "exp3", "exp4", "exp5"])
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
        args.experiment,
        args.config_dir,
        config_path,
        args.calibrate,
        args.dry_run,
        not args.no_plot,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
