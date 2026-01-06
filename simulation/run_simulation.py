from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from assist_manager import AssistManager  # noqa: E402
from dob_estimator import DisturbanceObserver  # noqa: E402
from logger import DataLogger  # noqa: E402
from main_control_odrive import ReferenceGenerator  # noqa: E402
from position_controller import (  # noqa: E402
    PositionCommand,
    PositionController,
    PositionFeedback,
    VelocityController,
)
from torque_distribution import TorqueAllocator  # noqa: E402


def _load_yaml(path: Path) -> Dict[str, object]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to read config files. Install with: python -m pip install pyyaml"
        ) from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _default_duration(reference_cfg: Dict[str, object]) -> float:
    profile = str(reference_cfg.get("active_profile", "step"))
    cfg = reference_cfg.get(profile, {}) if isinstance(reference_cfg.get(profile), dict) else {}
    wait = float(cfg.get("initial_wait", 0.0))
    if profile == "chirp":
        return wait + float(cfg.get("duration", 10.0)) + float(cfg.get("post_duration", 0.0))
    if profile == "ramp":
        return wait + float(cfg.get("ramp_duration", 1.0)) + float(
            cfg.get("hold_duration", 0.0)
        ) + float(cfg.get("return_duration", 0.0)) + 1.0
    if profile == "ramp_b":
        return wait + float(cfg.get("ramp_duration", 1.0)) + float(
            cfg.get("hold_duration", 0.0)
        ) + float(cfg.get("return_duration", 0.0)) + float(
            cfg.get("start_hold_duration", 0.0)
        ) + 1.0
    if profile == "step":
        return wait + float(cfg.get("step_duration", 2.0)) * 2.0 + 1.0
    if profile == "sine":
        freq = max(float(cfg.get("frequency_hz", 0.2)), 1e-3)
        return wait + max(10.0 / freq, 5.0)
    return 20.0


def _friction_feedforward(
    friction_cfg: Dict[str, object],
    velocity: float,
    reference_velocity: float,
    error_hint: float,
) -> float:
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


def _build_controller(
    controller_cfg: Dict[str, object],
    command_type: str,
    dt: float,
    inertia: float,
    damping: float,
) -> PositionController | VelocityController:
    per_motor_cfg = controller_cfg.get("per_motor_pid", {})
    derivative_mode = per_motor_cfg.get("derivative_mode", "error")
    derivative_alpha = float(per_motor_cfg.get("derivative_filter_alpha", 1.0))
    if command_type == "velocity":
        gains = controller_cfg.get("velocity_pid", {})
        return VelocityController(
            gains=gains,
            plant_inertia=inertia,
            plant_damping=damping,
            dt=dt,
            derivative_mode=derivative_mode,
            derivative_filter_alpha=derivative_alpha,
            use_feedforward=bool(gains.get("use_feedforward", True)),
        )
    gains = controller_cfg.get("outer_pid", {})
    return PositionController(
        gains=gains,
        plant_inertia=inertia,
        plant_damping=damping,
        dt=dt,
        derivative_mode=derivative_mode,
        derivative_filter_alpha=derivative_alpha,
        use_feedforward=bool(gains.get("use_feedforward", True)),
    )


def _update_output_state(
    position: float,
    velocity: float,
    torque_out: float,
    inertia: float,
    damping: float,
    dt: float,
) -> tuple[float, float]:
    if abs(inertia) < 1e-9:
        return position, velocity
    if damping > 0.0:
        decay = math.exp(-damping * dt / inertia)
        vel_next = velocity * decay + (torque_out / damping) * (1.0 - decay)
    else:
        vel_next = velocity + (torque_out / inertia) * dt
    pos_next = position + 0.5 * (velocity + vel_next) * dt
    return pos_next, vel_next


def run_simulation(config_dir: Path, duration: float | None = None) -> Path:
    controller_cfg = _load_yaml(config_dir / "controller.yaml")
    reference_cfg = _load_yaml(config_dir / "reference.yaml")
    logger_cfg = _load_yaml(config_dir / "logger.yaml")

    command_type = str(
        controller_cfg.get("command_type", reference_cfg.get("command_type", "position"))
    ).lower()
    if command_type not in {"position", "velocity"}:
        raise ValueError("command_type must be 'position' or 'velocity'")
    sim_mode = str(controller_cfg.get("simulation_mode", "proposed")).lower()
    if sim_mode not in {"proposed", "conventional"}:
        raise ValueError("simulation_mode must be 'proposed' or 'conventional'")
    reference_cfg = dict(reference_cfg)
    reference_cfg["command_type"] = command_type
    reference = ReferenceGenerator(reference_cfg)

    dt = float(controller_cfg.get("sample_time_s", 0.0))
    if dt <= 0.0:
        hz = float(controller_cfg.get("control_frequency_hz", 100.0))
        dt = 1.0 / max(hz, 1e-6)

    plant_cfg = controller_cfg.get("plant", {})
    inertia = float(plant_cfg.get("inertia", 0.015))
    damping = float(plant_cfg.get("damping", 0.002))
    mechanism_matrix = np.asarray(plant_cfg.get("mechanism_matrix", [1.0, 1.0]), dtype=float)
    mechanism_vec = np.asarray(mechanism_matrix, dtype=float).reshape(2)
    motor_output_gains = np.asarray(
        plant_cfg.get("motor_output_gains", mechanism_matrix), dtype=float
    )

    per_motor_cfg = controller_cfg.get("per_motor_pid", {})
    derivative_mode = per_motor_cfg.get("derivative_mode", "error")
    derivative_alpha = float(per_motor_cfg.get("derivative_filter_alpha", 1.0))
    conventional_cfg = controller_cfg.get("conventional", {})
    if not isinstance(conventional_cfg, dict):
        conventional_cfg = {}
    if sim_mode == "conventional" and command_type == "position":
        motor1_pid = conventional_cfg.get("motor1_pid", controller_cfg.get("outer_pid", {}))
        controller = PositionController(
            gains=motor1_pid,
            plant_inertia=inertia,
            plant_damping=damping,
            dt=dt,
            derivative_mode=derivative_mode,
            derivative_filter_alpha=derivative_alpha,
            use_feedforward=bool(motor1_pid.get("use_feedforward", True)),
        )
    else:
        controller = _build_controller(controller_cfg, command_type, dt, inertia, damping)

    torque_limits = controller_cfg.get("torque_limits", {})
    rate_limits = controller_cfg.get("torque_rate_limits", {})
    allocation_cfg = controller_cfg.get("torque_allocation", {})
    torque_dist_cfg = controller_cfg.get("torque_distribution", {})
    preference_cfg = controller_cfg.get("torque_preference", {})
    sign_cfg = controller_cfg.get("sign_enforcement", {})

    allocator = TorqueAllocator(
        mechanism_matrix=mechanism_matrix,
        torque_limits=torque_limits,
        dt=dt,
        rate_limits=rate_limits,
        preferred_motor=preference_cfg.get("preferred_motor"),
        sign_enforcement=sign_cfg.get("enabled", True),
        weight_mode=str(allocation_cfg.get("weight_mode", "raw")).lower(),
        preference_mode=str(preference_cfg.get("mode", "primary")).lower(),
        dynamic_utilization=bool(allocation_cfg.get("dynamic_utilization", False)),
        weight_filter_tau=allocation_cfg.get("weight_filter_tau"),
        weight_filter_alpha=allocation_cfg.get("weight_filter_alpha"),
    )

    dist_mode = str(torque_dist_cfg.get("mode", "fixed")).lower()
    fixed_weights = np.asarray(torque_dist_cfg.get("fixed_weights", [1.0, 1.0]), dtype=float)
    fixed_secondary_gain = float(torque_dist_cfg.get("fixed_secondary_gain", 1.0))

    assist_manager = None
    if dist_mode == "dynamic":
        assist_cfg = controller_cfg.get("assist_manager", {})
        if bool(assist_cfg.get("enabled", True)):
            assist_manager = AssistManager(
                dt=dt,
                mechanism_matrix=mechanism_matrix,
                torque_limits=torque_limits,
                config=assist_cfg,
            )

    dob_cfg = controller_cfg.get("dob", {})
    if sim_mode == "conventional":
        dob_cfg = conventional_cfg.get("dob", dob_cfg) if isinstance(conventional_cfg, dict) else dob_cfg
    dob_enabled = bool(dob_cfg.get("enabled", True))
    dob_input_mode = str(dob_cfg.get("torque_input_mode", "command")).lower()
    dob_applied_sign = float(
        dob_cfg.get("applied_sign", -1.0 if dob_input_mode == "applied" else 1.0)
    )
    dob = None
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

    logger = DataLogger(
        logger_cfg,
        base_path=config_dir.parent,
        controller_config=controller_cfg,
        reference=reference,
    )

    friction_cfg = controller_cfg.get("friction_ff", {}) or {}

    if duration is None:
        duration = _default_duration(reference_cfg)
    duration = float(duration)
    steps = max(int(duration / dt), 1)

    output_pos = 0.0
    output_vel = 0.0
    prev_tau_alloc = np.zeros(2, dtype=float)
    motor2_pos = 0.0
    motor2_vel = 0.0
    motor2_active = False
    motor2_controller = None
    motor2_plant_cfg = conventional_cfg.get("motor2_plant", {}) if sim_mode == "conventional" else {}
    motor2_inertia = float(motor2_plant_cfg.get("inertia", inertia))
    motor2_damping = float(motor2_plant_cfg.get("damping", damping))
    if sim_mode == "conventional":
        motor2_pid_cfg = conventional_cfg.get("motor2_pid", {})
        m2_ctrl_gains = {
            "kp": float(motor2_pid_cfg.get("kp", 0.1)),
            "ki": float(motor2_pid_cfg.get("ki", 0.0)),
            "kd": float(motor2_pid_cfg.get("kd", 0.0)),
            "max_output": float(motor2_pid_cfg.get("max_output", torque_limits.get("motor2", 1.0))),
        }
        motor2_controller = VelocityController(
            gains=m2_ctrl_gains,
            plant_inertia=0.0,
            plant_damping=0.0,
            dt=dt,
            derivative_mode=derivative_mode,
            derivative_filter_alpha=derivative_alpha,
            use_feedforward=False,
        )
    position_guard_cfg = controller_cfg.get("position_guard", {}) or {}
    position_guard_enabled = bool(position_guard_cfg.get("enabled", False))
    position_guard_limit = abs(float(position_guard_cfg.get("limit_turns", 0.0)))
    position_guard_soft = abs(float(position_guard_cfg.get("soft_zone", 0.0)))
    position_guard_min_scale = float(position_guard_cfg.get("min_scale", 0.0))
    position_guard_min_scale = max(0.0, min(1.0, position_guard_min_scale))

    for step in range(steps + 1):
        t = step * dt
        command: PositionCommand = reference.sample(t)
        feedback = PositionFeedback(position=output_pos, velocity=output_vel)
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
                reference_velocity
                if command_type == "velocity"
                else reference_position - feedback.position,
            )
        )
        friction_tau = _friction_feedforward(
            friction_cfg, output_vel, reference_velocity, error_hint
        )
        tau_cmd = tau_ctrl + friction_tau

        if dob_enabled and dob is not None:
            torque_applied_input = None
            if dob_input_mode == "applied":
                applied_tau_out = dob_applied_sign * float(
                    np.dot(motor_output_gains, prev_tau_alloc)
                )
                torque_applied_input = applied_tau_out
            tau_aug, dob_diag = dob.update(
                feedback.velocity, tau_cmd, torque_applied=torque_applied_input
            )
        else:
            tau_aug = tau_cmd
            dob_diag = {"filtered_disturbance": 0.0}

        if position_guard_enabled and position_guard_limit > 0.0:
            pos_abs = abs(float(output_pos))
            pos_sign = 1.0 if output_pos >= 0.0 else -1.0
            if position_guard_soft > 0.0:
                start = max(0.0, position_guard_limit - position_guard_soft)
                if pos_abs >= position_guard_limit:
                    if tau_aug * pos_sign > 0.0:
                        tau_aug *= position_guard_min_scale
                elif pos_abs > start:
                    ratio = (position_guard_limit - pos_abs) / position_guard_soft
                    scale = position_guard_min_scale + (1.0 - position_guard_min_scale) * max(
                        0.0, min(1.0, ratio)
                    )
                    if tau_aug * pos_sign > 0.0:
                        tau_aug *= scale
            else:
                if pos_abs >= position_guard_limit and tau_aug * pos_sign > 0.0:
                    tau_aug *= position_guard_min_scale

        if sim_mode == "conventional":
            motor2_vel_cfg = conventional_cfg.get("motor2_velocity", {})
            motor2_mode = motor2_vel_cfg.get("mode", "constant")
            if motor2_mode == "constant":
                vel_cmd_m2 = float(motor2_vel_cfg.get("value", 0.0))
            elif motor2_mode == "adaptive":
                A1 = float(mechanism_vec[0])
                limit1 = float(torque_limits.get("motor1", 1.0))
                if abs(A1) > 1e-6:
                    tau1_req = tau_aug / A1
                    util1 = abs(tau1_req) / (limit1 + 1e-9)
                else:
                    util1 = 0.0
                threshold = float(motor2_vel_cfg.get("utilization_threshold", 0.6))
                target_vel = float(motor2_vel_cfg.get("target_velocity", 5.0))
                hysteresis = float(motor2_vel_cfg.get("hysteresis", 0.1))
                if motor2_active:
                    if util1 < (threshold - hysteresis):
                        motor2_active = False
                else:
                    if util1 > threshold:
                        motor2_active = True
                sign_direction = math.copysign(1.0, tau_aug) if abs(tau_aug) > 1e-9 else 0.0
                vel_cmd_m2 = target_vel * sign_direction if motor2_active else 0.0
            else:
                vel_cmd_m2 = 0.0

            if motor2_controller is None:
                tau2_cmd = 0.0
            else:
                m2_feedback = PositionFeedback(position=motor2_pos, velocity=motor2_vel)
                m2_command = PositionCommand(position=0.0, velocity=vel_cmd_m2, acceleration=0.0)
                tau2_cmd, _ = motor2_controller.update(m2_command, m2_feedback)

            A1 = float(mechanism_vec[0])
            A2 = float(mechanism_vec[1])
            tau2_measured = tau2_cmd
            if abs(A1) < 1e-6:
                tau1_cmd = 0.0
            else:
                tau1_cmd = (tau_aug - A2 * tau2_measured) / A1
            tau_cmds = np.array([tau1_cmd, tau2_cmd], dtype=float)
            limits_vec = np.array(
                [
                    abs(float(torque_limits.get("motor1", 1.0))),
                    abs(float(torque_limits.get("motor2", 1.0))),
                ],
                dtype=float,
            )
            tau_alloc = np.clip(tau_cmds, -limits_vec, limits_vec)
            prev_tau_alloc = tau_alloc.copy()
            tau_out = float(np.dot(mechanism_matrix, tau_alloc))
            output_pos, output_vel = _update_output_state(
                output_pos,
                output_vel,
                tau_out,
                inertia,
                damping,
                dt,
            )
            motor2_pos, motor2_vel = _update_output_state(
                motor2_pos,
                motor2_vel,
                tau_alloc[1],
                motor2_inertia,
                motor2_damping,
                dt,
            )
        else:
            if dist_mode == "dynamic" and assist_manager is not None:
                assist_status = assist_manager.update(tau_aug)
                weights = assist_status.weights
                secondary_gain = assist_status.secondary_gain
            else:
                weights = fixed_weights
                secondary_gain = fixed_secondary_gain

            tau_alloc, _ = allocator.allocate(tau_aug, weights, secondary_gain)
            prev_tau_alloc = tau_alloc.copy()

            tau_out = float(np.dot(mechanism_matrix, tau_alloc))
            output_pos, output_vel = _update_output_state(
                output_pos,
                output_vel,
                tau_out,
                inertia,
                damping,
                dt,
            )

        logger.log(
            t,
            pos_1=0.0,
            vel_1=0.0,
            tau_1=tau_alloc[0],
            iq_1=tau_alloc[0],
            tau_meas_1=tau_alloc[0],
            pos_2=motor2_pos,
            vel_2=motor2_vel,
            tau_2=tau_alloc[1],
            iq_2=tau_alloc[1],
            tau_meas_2=tau_alloc[1],
            output_pos=output_pos,
            output_vel=output_vel,
            reference_position=reference_position,
            reference_velocity=reference_velocity,
            reference_control=tau_cmd,
            tau_pid=tau_ctrl,
            tau_dob=tau_aug,
            dob_disturbance=dob_diag.get("filtered_disturbance", 0.0),
            tau_out=tau_aug,
            loop_dt=dt,
        )

    meta = {"SimMode": "plant_only", "Duration": f"{duration:.3f}s"}
    output = logger.save(metadata=meta)
    csv_path = output.get("csv")
    if csv_path is None:
        raise RuntimeError("CSV path missing after save().")
    return Path(csv_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simulate the full control pipeline using configs under simulation/config."
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("simulation/config"),
        help="Directory containing controller.yaml/reference.yaml/logger.yaml.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Simulation duration in seconds (default: inferred from reference).",
    )
    args = parser.parse_args()

    csv_path = run_simulation(args.config_dir, duration=args.duration)
    print(f"Saved simulation CSV to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
