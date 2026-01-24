from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from odrive_interface import ODriveInterface
from position_controller import PositionCommand, PositionFeedback, VelocityController


_LOGGER = logging.getLogger("dual_velocity_test")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge_config(args, defaults, cfg: dict) -> None:
    if not cfg:
        return
    merged = {}
    merged.update(cfg.get("defaults", {}) if isinstance(cfg.get("defaults"), dict) else {})
    for key, value in merged.items():
        if not hasattr(args, key):
            continue
        if getattr(args, key) == getattr(defaults, key):
            setattr(args, key, value)

    motor1_cfg = cfg.get("motor1", {}) if isinstance(cfg.get("motor1"), dict) else {}
    motor2_cfg = cfg.get("motor2", {}) if isinstance(cfg.get("motor2"), dict) else {}
    if "velocity" in motor1_cfg and args.motor1_velocity == defaults.motor1_velocity:
        args.motor1_velocity = float(motor1_cfg["velocity"])
    if "velocity" in motor2_cfg and args.motor2_velocity == defaults.motor2_velocity:
        args.motor2_velocity = float(motor2_cfg["velocity"])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run both motors at constant velocity and log output speed."
    )
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
        "--motor1-velocity",
        type=float,
        default=0.0,
        help="Motor1 velocity command [turn/s].",
    )
    parser.add_argument(
        "--motor2-velocity",
        type=float,
        default=0.0,
        help="Motor2 velocity command [turn/s].",
    )
    parser.add_argument(
        "--control-mode",
        choices=["velocity", "torque"],
        default="velocity",
        help="Control mode for motors (velocity via ODrive, torque via PC PID).",
    )
    parser.add_argument("--duration", type=float, default=5.0, help="Run time in seconds.")
    parser.add_argument("--dt", type=float, default=0.01, help="Control loop interval.")
    parser.add_argument("--kp", type=float, default=0.0, help="PC PID proportional gain.")
    parser.add_argument("--ki", type=float, default=0.0, help="PC PID integral gain.")
    parser.add_argument("--kd", type=float, default=0.0, help="PC PID derivative gain.")
    parser.add_argument(
        "--max-output",
        type=float,
        default=float("inf"),
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
        default="dual_velocity",
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
        default="dual_velocity",
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
        "--odrive-vel-gain",
        type=float,
        default=None,
        help="ODrive velocity loop gain (applied to both motors if supported).",
    )
    parser.add_argument(
        "--odrive-vel-integrator-gain",
        type=float,
        default=None,
        help="ODrive velocity integrator gain (applied to both motors if supported).",
    )
    parser.add_argument(
        "--odrive-vel-integrator-limit",
        type=float,
        default=None,
        help="ODrive velocity integrator limit (applied to both motors if supported).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser


def _build_velocity_controller(args, cfg: dict, motor: str) -> VelocityController:
    motor_cfg = cfg.get(motor, {}) if isinstance(cfg.get(motor), dict) else {}

    def _pick(key: str) -> float:
        if key in motor_cfg:
            return float(motor_cfg[key])
        return float(getattr(args, key))

    gains = {
        "kp": _pick("kp"),
        "ki": _pick("ki"),
        "kd": _pick("kd"),
        "max_output": _pick("max_output"),
    }
    use_ff = not bool(getattr(args, "no_feedforward"))
    return VelocityController(
        gains=gains,
        plant_inertia=_pick("inertia"),
        plant_damping=_pick("damping"),
        dt=float(args.dt),
        derivative_mode=str(args.derivative_mode),
        derivative_filter_alpha=float(args.derivative_filter),
        use_feedforward=use_ff,
    )


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

    t = data["time_s"]
    v1 = data.get("motor1_velocity", [])
    v2 = data.get("motor2_velocity", [])
    v_out = data.get("output_velocity", [])
    t1 = data.get("motor1_target_velocity", [])
    t2 = data.get("motor2_target_velocity", [])

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, v1, label="motor1_velocity")
    axes[0].plot(t, v2, label="motor2_velocity")
    if t1:
        axes[0].plot(t, t1, "--", label="motor1_target")
    if t2:
        axes[0].plot(t, t2, "--", label="motor2_target")
    axes[0].set_ylabel("turn/s")
    axes[0].legend(loc="best")

    axes[1].plot(t, v_out, label="output_velocity")
    axes[1].set_ylabel("turn/s")
    axes[1].set_xlabel("time [s]")
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    defaults = parser.parse_args([])

    cfg: dict = {}
    if args.test_config is not None and args.test_config.exists():
        cfg = _load_yaml(args.test_config)
        _merge_config(args, defaults, cfg)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    hw_cfg = _load_yaml(Path(args.config_dir) / "hardware.yaml")
    odrive_iface = ODriveInterface(hw_cfg)
    control_mode = str(args.control_mode).lower()
    odrive_iface.connect(
        calibrate=args.calibrate,
        control_mode="velocity" if control_mode == "velocity" else "torque",
        axes=["motor1", "motor2", "output"],
    )

    if control_mode == "velocity":
        for name in ("motor1", "motor2"):
            motor_cfg = cfg.get(name, {}) if isinstance(cfg.get(name), dict) else {}
            vel_gain = motor_cfg.get("odrive_vel_gain", args.odrive_vel_gain)
            vel_int_gain = motor_cfg.get(
                "odrive_vel_integrator_gain", args.odrive_vel_integrator_gain
            )
            vel_int_limit = motor_cfg.get(
                "odrive_vel_integrator_limit", args.odrive_vel_integrator_limit
            )
            odrive_iface.set_velocity_gains(
                name,
                vel_gain=vel_gain,
                vel_integrator_gain=vel_int_gain,
                vel_integrator_limit=vel_int_limit,
            )

    controllers = {}
    if control_mode == "torque":
        controllers["motor1"] = _build_velocity_controller(args, cfg, "motor1")
        controllers["motor2"] = _build_velocity_controller(args, cfg, "motor2")

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_path = Path(args.log_dir) / f"{args.log_prefix}_{timestamp}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fig_path = Path(args.fig_dir) / f"{args.fig_prefix}_{timestamp}.png"

    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(
            [
                "time_s",
                "motor1_target_velocity",
                "motor2_target_velocity",
                "motor1_velocity",
                "motor2_velocity",
                "output_velocity",
                "motor1_torque_command",
                "motor2_torque_command",
                "motor1_position",
                "motor2_position",
                "output_position",
            ]
        )

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
                if "output" not in states:
                    raise RuntimeError("Output axis is not connected; check hardware.yaml.")

                motor1 = states["motor1"]
                motor2 = states["motor2"]
                output = states["output"]

                if control_mode == "velocity":
                    odrive_iface.command_velocities(args.motor1_velocity, args.motor2_velocity)
                    motor1_torque = 0.0
                    motor2_torque = 0.0
                else:
                    cmd1 = PositionCommand(position=0.0, velocity=float(args.motor1_velocity))
                    cmd2 = PositionCommand(position=0.0, velocity=float(args.motor2_velocity))
                    fb1 = PositionFeedback(position=motor1.position, velocity=motor1.velocity)
                    fb2 = PositionFeedback(position=motor2.position, velocity=motor2.velocity)
                    motor1_torque, _ = controllers["motor1"].update(cmd1, fb1)
                    motor2_torque, _ = controllers["motor2"].update(cmd2, fb2)
                    odrive_iface.command_torques(motor1_torque, motor2_torque)

                writer.writerow(
                    [
                        f"{now - start_time:.6f}",
                        f"{args.motor1_velocity:.6f}",
                        f"{args.motor2_velocity:.6f}",
                        f"{motor1.velocity:.6f}",
                        f"{motor2.velocity:.6f}",
                        f"{output.velocity:.6f}",
                        f"{motor1_torque:.6f}",
                        f"{motor2_torque:.6f}",
                        f"{motor1.position:.6f}",
                        f"{motor2.position:.6f}",
                        f"{output.position:.6f}",
                    ]
                )

                if (now - last_print) >= args.print_interval:
                    last_print = now
                    _LOGGER.info(
                        "v1=%.4f v2=%.4f | out=%.4f (turn/s)",
                        motor1.velocity,
                        motor2.velocity,
                        output.velocity,
                    )
        except KeyboardInterrupt:
            _LOGGER.info("Keyboard interrupt received, stopping test loop.")
        finally:
            if control_mode == "velocity":
                odrive_iface.command_velocities(0.0, 0.0)
            else:
                odrive_iface.command_torques(0.0, 0.0)
            odrive_iface.shutdown()
            _LOGGER.info("Log saved to %s", log_path)
            if args.plot:
                _plot_log(log_path, fig_path, args.plot_show)
                _LOGGER.info("Figure saved to %s", fig_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
