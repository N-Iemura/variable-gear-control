from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import yaml

from logger import DataLogger
from odrive_interface import ODriveInterface


_LOGGER = logging.getLogger("identification")


def _load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _compute_torque_profile(cfg: Dict[str, object], elapsed: float) -> Tuple[float, bool]:
    wait = float(cfg.get("initial_wait", 0.0))
    if elapsed < wait:
        return 0.0, False

    t = elapsed - wait
    target = float(cfg.get("target_torque", 1.0))
    ramp = max(float(cfg.get("ramp_duration", 1.0)), 1e-6)
    hold = max(float(cfg.get("hold_duration", 0.0)), 0.0)
    ret = max(float(cfg.get("return_duration", 1.0)), 1e-6)
    rest = max(float(cfg.get("rest_duration", 0.0)), 0.0)
    repeat = bool(cfg.get("repeat", False))

    cycle = ramp + hold + ret + rest
    if cycle <= 0.0:
        return 0.0, True

    if repeat:
        phase = t % cycle
        completed = False
    else:
        if t >= cycle:
            return 0.0, True
        phase = t
        completed = False

    if phase < ramp:
        torque = target * (phase / ramp)
    elif phase < ramp + hold:
        torque = target
    elif phase < ramp + hold + ret:
        ratio = (phase - ramp - hold) / ret
        torque = target * (1.0 - ratio)
    else:
        torque = 0.0
        if not repeat and phase >= ramp + hold + ret + rest:
            completed = True
    return torque, completed


def _solve_with_freeze(A: np.ndarray, tau_out: float, freeze_idx: int) -> np.ndarray:
    A = np.asarray(A, dtype=float).reshape(2)
    tau = np.zeros(2, dtype=float)
    if freeze_idx == 0:
        if abs(A[1]) < 1e-9:
            raise ValueError("Mechanism gain for motor1 is too small to solve torque.")
        tau[0] = 0.0
        tau[1] = float(tau_out) / A[1]
    else:
        if abs(A[0]) < 1e-9:
            raise ValueError("Mechanism gain for motor0 is too small to solve torque.")
        tau[1] = 0.0
        tau[0] = float(tau_out) / A[0]
    return tau


def run_identification(config_dir: Path, log_level: str) -> None:
    controller_cfg = _load_yaml(config_dir / "controller.yaml")
    hardware_cfg = _load_yaml(config_dir / "hardware.yaml")
    logger_cfg = _load_yaml(config_dir / "logger.yaml")

    identification_cfg = controller_cfg.get("identification", {})
    plant_cfg = controller_cfg.get("plant", {})
    A = np.asarray(plant_cfg.get("mechanism_matrix", [-0.05, 0.0815]), dtype=float)

    dt = float(controller_cfg.get("sample_time_s", 0.005))
    torque_limits_cfg = controller_cfg.get("torque_limits", {})
    torque_limits = np.array(
        [
            float(torque_limits_cfg.get("motor0", 6.0)),
            float(torque_limits_cfg.get("motor1", 1.0)),
        ],
        dtype=float,
    )

    freeze_motor = identification_cfg.get("motor_freeze", "motor1")
    freeze_idx = 0 if freeze_motor == "motor0" else 1

    odrive_iface = ODriveInterface(hardware_cfg)
    data_logger = DataLogger(
        logger_cfg,
        base_path=config_dir.parent,
        controller_config=controller_cfg,
        reference=None,
    )

    odrive_iface.connect(calibrate=False)
    odrive_iface.zero_positions()
    _LOGGER.info("Identification sequence started.")

    wait = float(identification_cfg.get("initial_wait", 0.0))
    ramp = max(float(identification_cfg.get("ramp_duration", 1.0)), 1e-6)
    hold = max(float(identification_cfg.get("hold_duration", 0.0)), 0.0)
    ret = max(float(identification_cfg.get("return_duration", 1.0)), 1e-6)
    rest = max(float(identification_cfg.get("rest_duration", 0.0)), 0.0)
    total_duration = wait + ramp + hold + ret + rest

    start_time = time.time()
    loop_time = start_time
    completed = False

    try:
        while True:
            now = time.time()
            elapsed = now - start_time

            tau_out_cmd, done = _compute_torque_profile(identification_cfg, elapsed)
            tau_cmd = _solve_with_freeze(A, tau_out_cmd, freeze_idx)
            tau_cmd = np.clip(tau_cmd, -torque_limits, torque_limits)

            odrive_iface.command_torques(float(tau_cmd[0]), float(tau_cmd[1]))

            states = odrive_iface.read_states()
            motor0_state = states["motor0"]
            motor1_state = states["motor1"]
            output_state = states.get("output", motor0_state)

            data_logger.log(
                elapsed,
                motor0={
                    "pos": motor0_state.position,
                    "vel": motor0_state.velocity,
                    "torque": tau_cmd[0],
                },
                motor1={
                    "pos": motor1_state.position,
                    "vel": motor1_state.velocity,
                    "torque": tau_cmd[1],
                },
                output={
                    "pos": output_state.position,
                    "vel": output_state.velocity,
                },
                reference={
                    "position": 0.0,
                    "control": tau_out_cmd,
                },
                torques={"output": tau_out_cmd},
            )

            if done and not completed:
                completed = True
                _LOGGER.info("Torque profile completed, holding zero torque.")
            if completed and elapsed >= total_duration:
                break

            loop_time += dt
            sleep = loop_time - time.time()
            if sleep > 0:
                time.sleep(sleep)
            else:
                loop_time = time.time()
    except KeyboardInterrupt:
        _LOGGER.info("Interrupted by user.")
    finally:
        try:
            odrive_iface.command_torques(0.0, 0.0)
            odrive_iface.shutdown()
        except Exception:
            _LOGGER.exception("Failed to safely shutdown ODrive.")

        metadata = {
            "Experiment": "identification",
            "FreezeMotor": freeze_motor,
            "TargetTorque": identification_cfg.get("target_torque", 0.0),
        }
        save_result = data_logger.save(metadata=metadata)
        csv_path = save_result.get("csv")
        fig_path = save_result.get("figure")
        if csv_path:
            _LOGGER.info("Log saved to %s", csv_path)
        if fig_path:
            _LOGGER.info("Figure saved to %s", fig_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run identification torque ramp experiment.")
    parser.add_argument("--config-dir", type=Path, default=Path(__file__).parent / "config")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_identification(args.config_dir, args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
