#!/usr/bin/env python3
"""Simple torque-only functional test for each motor.

This script commands small positive/negative torque steps to motor1 and motor2
individually so you can confirm basic motion with nothing attached to the shafts.
All parameters are exposed via CLI flags to keep the test gentle.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import yaml

from logger import DataLogger
from odrive_interface import ODriveInterface


_LOGGER = logging.getLogger("torque_smoke_test")


def _load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_logger(
    enabled: bool,
    config_dir: Path,
    controller_cfg: Dict[str, object],
    hardware_cfg: Dict[str, object],
) -> Optional[DataLogger]:
    if not enabled:
        return None
    logger_cfg = _load_yaml(config_dir / "logger.yaml")
    torque_constants = hardware_cfg.get("odrive", {}).get("torque_constants", {})
    return DataLogger(
        logger_cfg,
        base_path=config_dir.parent,
        controller_config=controller_cfg,
        reference=None,
        torque_constants=torque_constants,
    )


def _check_velocity_limit(
    states: Dict[str, object],
    limit: float,
    monitored: Iterable[str],
) -> Optional[Tuple[str, float]]:
    if limit <= 0:
        return None
    for name in monitored:
        state = states.get(name)
        if state is None:
            continue
        if abs(state.velocity) > limit:
            return name, float(state.velocity)
    return None


def _log_state(
    logger_obj: DataLogger,
    states: Dict[str, object],
    tau_cmd1: float,
    tau_cmd2: float,
    elapsed: float,
    loop_dt: float,
) -> None:
    motor1_state = states["motor1"]
    motor2_state = states["motor2"]
    output_state = states.get("output", motor1_state)
    logger_obj.log(
        time_stamp=elapsed,
        pos_1=motor1_state.position,
        vel_1=motor1_state.velocity,
        tau_1=tau_cmd1,
        iq_1=motor1_state.current_iq,
        tau_meas_1=motor1_state.torque_measured,
        pos_2=motor2_state.position,
        vel_2=motor2_state.velocity,
        tau_2=tau_cmd2,
        iq_2=motor2_state.current_iq,
        tau_meas_2=motor2_state.torque_measured,
        output_pos=output_state.position,
        output_vel=output_state.velocity,
        reference_position=0.0,
        reference_velocity=0.0,
        reference_control=0.0,
        tau_pid=0.0,
        tau_dob=0.0,
        dob_disturbance=0.0,
        tau_out=0.0,
        loop_dt=loop_dt,
    )


def _hold_torque(
    iface: ODriveInterface,
    target_motor: str,
    torque: float,
    duration: float,
    dt: float,
    logger_obj: Optional[DataLogger],
    start_time: float,
    velocity_limit: float,
) -> None:
    end_time = time.time() + duration
    monitored = ("motor1", "motor2")
    while time.time() < end_time:
        loop_start = time.time()
        tau1 = torque if target_motor == "motor1" else 0.0
        tau2 = torque if target_motor == "motor2" else 0.0
        iface.command_torques(tau1, tau2)
        states = iface.read_states(fast=logger_obj is None)
        hit = _check_velocity_limit(states, velocity_limit, monitored)
        if hit is not None:
            name, vel = hit
            iface.command_torques(0.0, 0.0)
            raise RuntimeError(f"Velocity limit exceeded on {name}: {vel:.3f} turn/s")
        if logger_obj:
            elapsed = loop_start - start_time
            _log_state(logger_obj, states, tau1, tau2, elapsed, dt)
        sleep = dt - (time.time() - loop_start)
        if sleep > 0:
            time.sleep(sleep)
    iface.command_torques(0.0, 0.0)


def run_test(args: argparse.Namespace) -> None:
    config_dir = Path(args.config_dir)
    controller_cfg = _load_yaml(config_dir / "controller.yaml")
    hardware_cfg = _load_yaml(config_dir / "hardware.yaml")

    data_logger = _build_logger(args.log, config_dir, controller_cfg, hardware_cfg)

    iface = ODriveInterface(hardware_cfg)
    iface.connect(calibrate=False)
    iface.zero_positions()
    _LOGGER.info("Starting torque smoke test (motor=%s, torque=%.3f Nm)", args.motor, args.torque)

    start_time = time.time()
    motors = ["motor1", "motor2"] if args.motor == "both" else [args.motor]
    try:
        for motor in motors:
            _LOGGER.info("Testing %s: %d cycles, hold=%.2f s, settle=%.2f s", motor, args.cycles, args.hold, args.settle)
            for cycle in range(1, args.cycles + 1):
                _LOGGER.info("  Cycle %d/%d: +%.3f Nm", cycle, args.cycles, args.torque)
                _hold_torque(
                    iface=iface,
                    target_motor=motor,
                    torque=abs(args.torque),
                    duration=args.hold,
                    dt=args.sample_time,
                    logger_obj=data_logger,
                    start_time=start_time,
                    velocity_limit=args.velocity_limit,
                )
                if args.settle > 0:
                    _hold_torque(
                        iface=iface,
                        target_motor=motor,
                        torque=0.0,
                        duration=args.settle,
                        dt=args.sample_time,
                        logger_obj=data_logger,
                        start_time=start_time,
                        velocity_limit=args.velocity_limit,
                    )
                _LOGGER.info("  Cycle %d/%d: -%.3f Nm", cycle, args.cycles, args.torque)
                _hold_torque(
                    iface=iface,
                    target_motor=motor,
                    torque=-abs(args.torque),
                    duration=args.hold,
                    dt=args.sample_time,
                    logger_obj=data_logger,
                    start_time=start_time,
                    velocity_limit=args.velocity_limit,
                )
                if args.settle > 0:
                    _hold_torque(
                        iface=iface,
                        target_motor=motor,
                        torque=0.0,
                        duration=args.settle,
                        dt=args.sample_time,
                        logger_obj=data_logger,
                        start_time=start_time,
                        velocity_limit=args.velocity_limit,
                    )
            if args.rest > 0:
                _LOGGER.info("Resting at zero torque for %.2f s before next motor", args.rest)
                _hold_torque(
                    iface=iface,
                    target_motor=motor,
                    torque=0.0,
                    duration=args.rest,
                    dt=args.sample_time,
                    logger_obj=data_logger,
                    start_time=start_time,
                    velocity_limit=args.velocity_limit,
                )
        _LOGGER.info("Torque smoke test complete. Holding zero torque.")
    except KeyboardInterrupt:
        _LOGGER.info("Interrupted by user, idling axes.")
    finally:
        try:
            iface.command_torques(0.0, 0.0)
            iface.shutdown()
        finally:
            if data_logger is not None:
                metadata = {
                    "Experiment": "torque_smoke_test",
                    "Motor": args.motor,
                    "Torque": args.torque,
                    "Cycles": args.cycles,
                }
                save_result = data_logger.save(metadata=metadata)
                if save_result.get("csv"):
                    _LOGGER.info("Log saved to %s", save_result["csv"])
                if save_result.get("figure"):
                    _LOGGER.info("Figure saved to %s", save_result["figure"])


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply small torque steps to motor1/motor2 for functional checks.",
    )
    parser.add_argument("--config-dir", type=str, default="config", help="Directory containing hardware.yaml/logger.yaml.")
    parser.add_argument("--motor", choices=["motor1", "motor2", "both"], default="both", help="Select which motor(s) to exercise.")
    parser.add_argument("--torque", type=float, default=0.1, help="Peak torque command [Nm] for the test.")
    parser.add_argument("--hold", type=float, default=1.0, help="Duration [s] for each non-zero torque hold.")
    parser.add_argument("--settle", type=float, default=0.5, help="Duration [s] to stay at zero torque between + and - steps.")
    parser.add_argument("--cycles", type=int, default=2, help="Number of (+/-) cycles per motor.")
    parser.add_argument("--rest", type=float, default=1.0, help="Zero-torque rest time [s] before switching motors.")
    parser.add_argument("--sample-time", type=float, default=0.01, help="Loop period [s] for sending commands and reading states.")
    parser.add_argument("--velocity-limit", type=float, default=5.0, help="Stop if abs(velocity) exceeds this [turn/s]; set <=0 to disable.")
    parser.add_argument("--log", action="store_true", help="Enable CSV/plot logging using logger.yaml settings.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level (DEBUG, INFO, WARNING...).")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        run_test(args)
    except Exception as exc:  # pragma: no cover - hardware side effects
        _LOGGER.exception("Torque smoke test failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
