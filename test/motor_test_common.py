from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from odrive_interface import ODriveInterface
from position_controller import (
    PositionCommand,
    PositionController,
    PositionFeedback,
    VelocityController,
)

try:
    from odrive.enums import (
        AXIS_STATE_CLOSED_LOOP_CONTROL,
        CONTROL_MODE_POSITION_CONTROL,
        CONTROL_MODE_TORQUE_CONTROL,
        CONTROL_MODE_VELOCITY_CONTROL,
        INPUT_MODE_PASSTHROUGH,
    )
except ImportError:  # pragma: no cover - hardware dependency
    AXIS_STATE_CLOSED_LOOP_CONTROL = 8
    CONTROL_MODE_TORQUE_CONTROL = 1
    CONTROL_MODE_VELOCITY_CONTROL = 2
    CONTROL_MODE_POSITION_CONTROL = 3
    INPUT_MODE_PASSTHROUGH = 1


_LOGGER = logging.getLogger("motor_test")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _configure_odrive_position(handle, input_position: float) -> None:
    controller = getattr(handle.axis, "controller", None)
    if controller and hasattr(controller, "config"):
        controller.config.control_mode = CONTROL_MODE_POSITION_CONTROL
        if hasattr(controller.config, "input_mode"):
            controller.config.input_mode = INPUT_MODE_PASSTHROUGH
    handle.request_state(AXIS_STATE_CLOSED_LOOP_CONTROL)
    if controller is None:
        return
    if hasattr(controller, "input_pos"):
        controller.input_pos = float(input_position)
    elif hasattr(controller, "pos_setpoint"):
        controller.pos_setpoint = float(input_position)


def _configure_odrive_gains(handle, args) -> None:
    controller = getattr(handle.axis, "controller", None)
    config = getattr(controller, "config", None) if controller is not None else None
    if config is None:
        return
    if args.odrive_pos_gain is not None and hasattr(config, "pos_gain"):
        config.pos_gain = float(args.odrive_pos_gain)
    if args.odrive_vel_gain is not None and hasattr(config, "vel_gain"):
        config.vel_gain = float(args.odrive_vel_gain)
    if args.odrive_vel_integrator_gain is not None and hasattr(config, "vel_integrator_gain"):
        config.vel_integrator_gain = float(args.odrive_vel_integrator_gain)
    if args.odrive_vel_integrator_limit is not None and hasattr(config, "vel_integrator_limit"):
        config.vel_integrator_limit = float(args.odrive_vel_integrator_limit)


def _command_odrive_position(handle, position: float) -> None:
    controller = getattr(handle.axis, "controller", None)
    if controller is None:
        return
    if hasattr(controller, "input_pos"):
        controller.input_pos = float(position)
        return
    if hasattr(controller, "pos_setpoint"):
        controller.pos_setpoint = float(position)
        return
    raise AttributeError("ODrive controller does not expose a position setpoint attribute.")


def _command_odrive_velocity(handle, velocity: float) -> None:
    controller = getattr(handle.axis, "controller", None)
    if controller is None:
        return
    if hasattr(controller, "input_vel"):
        controller.input_vel = float(velocity)
        return
    if hasattr(controller, "vel_setpoint"):
        controller.vel_setpoint = float(velocity)
        return
    raise AttributeError("ODrive controller does not expose a velocity setpoint attribute.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-motor test controller.")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=_ROOT / "config",
        help="Config directory containing hardware.yaml.",
    )
    parser.add_argument(
        "--test-config",
        type=Path,
        default=None,
        help="YAML file for test parameters.",
    )
    parser.add_argument(
        "--trajectory-file",
        type=Path,
        default=None,
        help="YAML file describing target trajectory.",
    )
    parser.add_argument(
        "--trajectory-profile",
        type=str,
        default=None,
        help="Profile name to use inside trajectory file.",
    )
    parser.add_argument(
        "--mode",
        choices=["position", "velocity", "torque"],
        default="position",
        help="Control mode for the test.",
    )
    parser.add_argument(
        "--pid-location",
        choices=["pc", "odrive"],
        default="pc",
        help="Where PID is executed (pc or odrive).",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=0.0,
        help="Target command (position [turn], velocity [turn/s], torque [Nm]).",
    )
    parser.add_argument("--duration", type=float, default=None, help="Run time in seconds.")
    parser.add_argument("--dt", type=float, default=0.01, help="Control loop interval.")
    parser.add_argument("--kp", type=float, default=0.0, help="PID proportional gain.")
    parser.add_argument("--ki", type=float, default=0.0, help="PID integral gain.")
    parser.add_argument("--kd", type=float, default=0.0, help="PID derivative gain.")
    parser.add_argument(
        "--max-output",
        type=float,
        default=math.inf,
        help="Absolute torque limit for PC PID [Nm].",
    )
    parser.add_argument(
        "--inertia",
        type=float,
        default=0.0,
        help="Motor inertia used for feedforward [kg m^2].",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=0.0,
        help="Motor viscous damping used for feedforward [Nms].",
    )
    parser.add_argument(
        "--derivative-mode",
        choices=["error", "measurement"],
        default="error",
        help="Derivative term source for PC PID.",
    )
    parser.add_argument(
        "--derivative-filter",
        type=float,
        default=1.0,
        help="Low-pass filter alpha for derivative term (0-1).",
    )
    parser.add_argument(
        "--no-feedforward",
        action="store_true",
        help="Disable inertia/damping feedforward when using PC PID.",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Enable fast ODrive reads (position/velocity only).",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.2,
        help="Status print interval in seconds.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run full calibration sequence before control.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=_ROOT / "test" / "csv",
        help="Directory for CSV logs.",
    )
    parser.add_argument(
        "--log-prefix",
        type=str,
        default="motor_test",
        help="Prefix for CSV log filename.",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=_ROOT / "test" / "fig",
        help="Directory for figure outputs.",
    )
    parser.add_argument(
        "--fig-prefix",
        type=str,
        default="motor_test",
        help="Prefix for figure filename.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Generate plot after run.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_false",
        dest="plot",
        help="Disable plot generation after run.",
    )
    parser.add_argument(
        "--plot-show",
        action="store_true",
        default=True,
        help="Show plot window after run.",
    )
    parser.add_argument(
        "--no-plot-show",
        action="store_false",
        dest="plot_show",
        help="Disable showing plot window after run.",
    )
    parser.add_argument(
        "--odrive-pos-gain",
        type=float,
        default=None,
        help="ODrive position loop gain (if supported).",
    )
    parser.add_argument(
        "--odrive-vel-gain",
        type=float,
        default=None,
        help="ODrive velocity loop gain (if supported).",
    )
    parser.add_argument(
        "--odrive-vel-integrator-gain",
        type=float,
        default=None,
        help="ODrive velocity integrator gain (if supported).",
    )
    parser.add_argument(
        "--odrive-vel-integrator-limit",
        type=float,
        default=None,
        help="ODrive velocity integrator limit (if supported).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser


def _merge_config(args, defaults, cfg: dict, motor: str) -> None:
    if not cfg:
        return
    merged = {}
    merged.update(cfg.get("defaults", {}) if isinstance(cfg.get("defaults"), dict) else {})
    motor_cfg = cfg.get(motor, {})
    if isinstance(motor_cfg, dict):
        merged.update(motor_cfg)

    for key, value in merged.items():
        if not hasattr(args, key):
            continue
        if getattr(args, key) == getattr(defaults, key):
            setattr(args, key, value)


def _build_controller(args):
    gains = {"kp": args.kp, "ki": args.ki, "kd": args.kd, "max_output": args.max_output}
    use_ff = not args.no_feedforward
    if args.mode == "position":
        return PositionController(
            gains=gains,
            plant_inertia=args.inertia,
            plant_damping=args.damping,
            dt=args.dt,
            derivative_mode=args.derivative_mode,
            derivative_filter_alpha=args.derivative_filter,
            use_feedforward=use_ff,
        )
    if args.mode == "velocity":
        return VelocityController(
            gains=gains,
            plant_inertia=args.inertia,
            plant_damping=args.damping,
            dt=args.dt,
            derivative_mode=args.derivative_mode,
            derivative_filter_alpha=args.derivative_filter,
            use_feedforward=use_ff,
        )
    return None


def _prepare_log(file_handle) -> csv.writer:
    writer = csv.writer(file_handle)
    writer.writerow(
        [
            "time_s",
            "motor",
            "mode",
            "pid_location",
            "target",
            "command",
            "position",
            "velocity",
            "torque_measured",
            "iq_measured",
        ]
    )
    return writer


def _build_trajectory(args):
    def constant_position(t: float) -> PositionCommand:
        return PositionCommand(position=float(args.target))

    def constant_velocity(t: float) -> PositionCommand:
        return PositionCommand(position=0.0, velocity=float(args.target))

    def constant_torque(t: float) -> float:
        return float(args.target)

    if args.trajectory_file is None:
        if args.mode == "position":
            return constant_position
        if args.mode == "velocity":
            return constant_velocity
        return constant_torque

    cfg = _load_yaml(args.trajectory_file)
    if not isinstance(cfg, dict):
        raise ValueError("trajectory_file must be a YAML mapping.")

    profile_cfg = cfg
    profiles = cfg.get("profiles")
    if isinstance(profiles, dict):
        selected = args.trajectory_profile or cfg.get("active") or cfg.get("active_profile")
        if selected is None:
            raise ValueError("trajectory_file has profiles; specify trajectory_profile or active.")
        if selected not in profiles:
            raise KeyError(f"Unknown trajectory profile: {selected}")
        profile_cfg = profiles[selected]
        if not isinstance(profile_cfg, dict):
            raise ValueError("Selected trajectory profile must be a mapping.")

    traj_type = str(profile_cfg.get("type", profile_cfg.get("profile", "step"))).lower()
    mode = str(args.mode).lower()
    start_time = float(profile_cfg.get("start_time", 0.0))
    repeat = bool(profile_cfg.get("repeat", False))

    if traj_type == "step":
        start_value = float(profile_cfg.get("start_value", 0.0))
        target_value = float(profile_cfg.get("target", profile_cfg.get("target_value", args.target)))

        def step_fn(t: float):
            value = target_value if t >= start_time else start_value
            if mode == "position":
                return PositionCommand(position=value)
            if mode == "velocity":
                return PositionCommand(position=0.0, velocity=value)
            return float(value)

        return step_fn

    if traj_type == "sine":
        amp = float(profile_cfg.get("amplitude", 0.0))
        offset = float(profile_cfg.get("offset", 0.0))
        freq = float(profile_cfg.get("frequency_hz", 0.0))
        omega = 2.0 * math.pi * freq

        def sine_fn(t: float):
            if t < start_time:
                value = offset
                vel = 0.0
                acc = 0.0
            else:
                tau = t - start_time
                value = offset + amp * math.sin(omega * tau)
                vel = amp * omega * math.cos(omega * tau)
                acc = -amp * (omega ** 2) * math.sin(omega * tau)
            if mode == "position":
                return PositionCommand(position=value, velocity=vel, acceleration=acc)
            if mode == "velocity":
                return PositionCommand(position=0.0, velocity=value, acceleration=vel)
            return float(value)

        return sine_fn

    if traj_type == "ramp":
        start_value = float(profile_cfg.get("start_value", 0.0))
        end_value = float(profile_cfg.get("end_value", args.target))
        ramp_duration = max(float(profile_cfg.get("ramp_duration", 1.0)), 1e-6)
        hold_duration = max(float(profile_cfg.get("hold_duration", 0.0)), 0.0)
        cycle = ramp_duration + hold_duration
        slope = (end_value - start_value) / ramp_duration

        def ramp_fn(t: float):
            if t < start_time:
                value = start_value
                vel = 0.0
                acc = 0.0
            else:
                tau = t - start_time
                if repeat and cycle > 0.0:
                    tau = tau % cycle
                if tau <= ramp_duration:
                    value = start_value + slope * tau
                    vel = slope
                    acc = 0.0
                else:
                    value = end_value
                    vel = 0.0
                    acc = 0.0
            if mode == "position":
                return PositionCommand(position=value, velocity=vel, acceleration=acc)
            if mode == "velocity":
                return PositionCommand(position=0.0, velocity=value, acceleration=vel)
            return float(value)

        return ramp_fn

    raise ValueError(f"Unsupported trajectory type: {traj_type}")


def _plot_log(csv_path: Path, fig_path: Path, show: bool) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _LOGGER.warning("matplotlib is not available; skipping plot.")
        return

    data = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                data.setdefault(key, [])
                try:
                    data[key].append(float(value))
                except (TypeError, ValueError):
                    pass

    if "time_s" not in data:
        _LOGGER.warning("CSV missing time_s; skipping plot.")
        return

    t = data.get("time_s", [])
    target = data.get("target", [])
    position = data.get("position", [])
    velocity = data.get("velocity", [])
    command = data.get("command", [])
    torque_measured = data.get("torque_measured", [])

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, position, label="position")
    if target:
        axes[0].plot(t, target, "--", label="target")
    axes[0].set_ylabel("turn")
    axes[0].legend(loc="best")

    axes[1].plot(t, velocity, label="velocity")
    axes[1].set_ylabel("turn/s")
    axes[1].legend(loc="best")

    axes[2].plot(t, command, label="command")
    if torque_measured:
        axes[2].plot(t, torque_measured, label="torque_measured")
    axes[2].set_ylabel("Nm")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(loc="best")

    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _command_torque(iface: ODriveInterface, motor: str, torque: float) -> None:
    if motor == "motor1":
        iface.command_torques(torque, 0.0)
    else:
        iface.command_torques(0.0, torque)


def _command_velocity(iface: ODriveInterface, motor: str, velocity: float) -> None:
    if motor == "motor1":
        iface.command_velocities(velocity, 0.0)
    else:
        iface.command_velocities(0.0, velocity)


def run_motor_cli(motor: str, argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    defaults = parser.parse_args([])

    if args.test_config is not None and args.test_config.exists():
        cfg = _load_yaml(args.test_config)
        _merge_config(args, defaults, cfg, motor)
    if not isinstance(args.log_dir, Path):
        args.log_dir = Path(args.log_dir)
    if not isinstance(args.fig_dir, Path):
        args.fig_dir = Path(args.fig_dir)
    if args.trajectory_file is not None and not isinstance(args.trajectory_file, Path):
        args.trajectory_file = Path(args.trajectory_file)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    hw_cfg = _load_yaml(Path(args.config_dir) / "hardware.yaml")
    odrive_iface = ODriveInterface(hw_cfg)

    mode = args.mode
    pid_location = args.pid_location
    if mode == "torque":
        pid_location = "pc"
    if pid_location == "odrive" and mode == "torque":
        _LOGGER.warning("Torque mode ignores ODrive PID and uses direct torque command.")

    control_mode = "torque" if pid_location == "pc" or mode == "torque" else "velocity"
    odrive_iface.connect(calibrate=args.calibrate, control_mode=control_mode, axes=[motor])
    odrive_iface.zero_positions()
    handle = odrive_iface.devices[motor]
    trajectory = _build_trajectory(args)
    initial_ref = trajectory(0.0)

    if pid_location == "odrive":
        _configure_odrive_gains(handle, args)
        if mode == "position":
            if isinstance(initial_ref, PositionCommand):
                _configure_odrive_position(handle, initial_ref.position)
            else:
                _configure_odrive_position(handle, float(initial_ref))
        elif mode == "velocity":
            handle.prepare_for_velocity_control()

    controller = None
    if pid_location == "pc" and mode in {"position", "velocity"}:
        controller = _build_controller(args)
        if controller is None:
            raise ValueError("PC PID requested but controller could not be built.")

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_path = args.log_dir / f"{args.log_prefix}_{motor}_{timestamp}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", newline="", encoding="utf-8")
    csv_writer = _prepare_log(log_file)
    fig_path = args.fig_dir / f"{args.fig_prefix}_{motor}_{timestamp}.png"

    start_time = time.perf_counter()
    next_time = start_time
    last_print = start_time

    try:
        while True:
            now = time.perf_counter()
            if args.duration is not None and (now - start_time) >= args.duration:
                break
            if now < next_time:
                time.sleep(max(0.0, next_time - now))
                continue
            next_time += args.dt

            states = odrive_iface.read_states(fast=args.fast_mode)
            state = states[motor]
            ref = trajectory(now - start_time)

            if mode == "torque":
                target_value = float(ref)
                _command_torque(odrive_iface, motor, target_value)
                command_value = target_value
            elif pid_location == "odrive":
                if mode == "position":
                    target_value = (
                        ref.position if isinstance(ref, PositionCommand) else float(ref)
                    )
                    _command_odrive_position(handle, target_value)
                else:
                    target_value = (
                        ref.velocity if isinstance(ref, PositionCommand) else float(ref)
                    )
                    _command_odrive_velocity(handle, target_value)
                command_value = target_value
            else:
                feedback = PositionFeedback(position=state.position, velocity=state.velocity)
                if not isinstance(ref, PositionCommand):
                    if mode == "position":
                        command = PositionCommand(position=float(ref))
                    else:
                        command = PositionCommand(position=0.0, velocity=float(ref))
                else:
                    command = ref
                target_value = command.position if mode == "position" else command.velocity
                torque_cmd, _ = controller.update(command, feedback)
                _command_torque(odrive_iface, motor, torque_cmd)
                command_value = torque_cmd

            csv_writer.writerow(
                [
                    f"{now - start_time:.6f}",
                    motor,
                    mode,
                    pid_location,
                    f"{target_value:.6f}",
                    f"{command_value:.6f}",
                    f"{state.position:.6f}",
                    f"{state.velocity:.6f}",
                    f"{state.torque_measured:.6f}",
                    f"{state.current_iq:.6f}",
                ]
            )

            if (now - last_print) >= args.print_interval:
                last_print = now
                _LOGGER.info(
                    "%s mode=%s pid=%s target=%.4f pos=%.4f vel=%.4f cmd=%.4f",
                    motor,
                    mode,
                    pid_location,
                    args.target,
                    state.position,
                    state.velocity,
                    command_value,
                )
    except KeyboardInterrupt:
        _LOGGER.info("Keyboard interrupt received, stopping test loop.")
    finally:
        log_file.close()
        odrive_iface.shutdown()
        _LOGGER.info("Log saved to %s", log_path)
        if args.plot:
            _plot_log(log_path, fig_path, args.plot_show)
            _LOGGER.info("Figure saved to %s", fig_path)
    return 0
