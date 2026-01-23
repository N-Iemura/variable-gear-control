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
        repeat = bool(cfg.get("repeat", True))
        duration = float(cfg.get("step_duration", 2.0))
        if repeat:
            return wait + duration * 2.0 + 1.0
        return wait + duration + 1.0
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
    limits_vec = np.array(
        [
            abs(float(torque_limits.get("motor1", 1.0))),
            abs(float(torque_limits.get("motor2", 1.0))),
        ],
        dtype=float,
    )
    velocity_cfg = controller_cfg.get("velocity_distribution", {})
    kinematic_matrix = np.asarray(
        velocity_cfg.get("kinematic_matrix", plant_cfg.get("kinematic_matrix", mechanism_matrix)),
        dtype=float,
    )
    kinematic_vec = np.asarray(kinematic_matrix, dtype=float).reshape(2)
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
    internal_push_cfg = controller_cfg.get("internal_push", {}) or {}
    internal_push_enabled = bool(internal_push_cfg.get("enabled", False))
    internal_push_mode = str(internal_push_cfg.get("mode", "constant")).lower()
    internal_push_sign_mode = str(internal_push_cfg.get("sign_mode", "fixed")).lower()
    internal_push_sign_source = str(internal_push_cfg.get("sign_source", "tau_ref")).lower()
    internal_push_fixed_sign = float(internal_push_cfg.get("fixed_sign", 1.0))
    internal_push_beta = float(internal_push_cfg.get("beta", 0.0))
    internal_push_omega_c = float(internal_push_cfg.get("omega_c", 0.05))
    internal_push_anchor = str(internal_push_cfg.get("anchor_motor", "max")).lower()
    internal_push_tau_max = internal_push_cfg.get("tau_max")
    if internal_push_tau_max is not None:
        internal_push_tau_max = float(internal_push_tau_max)
        if internal_push_tau_max <= 0.0:
            internal_push_tau_max = None
    null_vec = np.array([mechanism_vec[1], -mechanism_vec[0]], dtype=float)
    null_scale = float(np.max(np.abs(null_vec))) if np.any(null_vec) else 0.0
    if internal_push_anchor == "motor1" and abs(null_vec[0]) > 1e-9:
        null_scale = abs(null_vec[0])
    elif internal_push_anchor == "motor2" and abs(null_vec[1]) > 1e-9:
        null_scale = abs(null_vec[1])

    m2_coupling_cfg = controller_cfg.get("motor2_coupling", {}) or {}
    m2_coupling_enabled = bool(m2_coupling_cfg.get("enabled", False))
    m2_coupling_gain = float(m2_coupling_cfg.get("gain", 0.0))
    m2_coupling_sign_mode = str(m2_coupling_cfg.get("sign_mode", "same")).lower()
    m2_coupling_tau_max = m2_coupling_cfg.get("tau_max")
    if m2_coupling_tau_max is not None:
        m2_coupling_tau_max = float(m2_coupling_tau_max)
        if m2_coupling_tau_max <= 0.0:
            m2_coupling_tau_max = None

    if duration is None:
        duration = _default_duration(reference_cfg)
    duration = float(duration)
    steps = max(int(duration / dt), 1)

    output_pos = 0.0
    output_vel = 0.0
    prev_tau_alloc = np.zeros(2, dtype=float)
    prev_reference_position = None
    motor2_pos = 0.0
    motor2_vel = 0.0
    motor2_active = False
    motor2_vel_cfg = conventional_cfg.get("motor2_velocity", {}) if sim_mode == "conventional" else {}
    motor2_mode = str(motor2_vel_cfg.get("mode", "constant")).lower()
    motor2_omega_source = str(motor2_vel_cfg.get("omega_out_source", "output")).lower()
    schedule_mode = str(motor2_vel_cfg.get("schedule_mode", "discrete")).lower()
    utilization_source = str(motor2_vel_cfg.get("utilization_source", "torque")).lower()
    if utilization_source not in {"torque", "velocity"}:
        utilization_source = "torque"
    ratio_ref_cfg = conventional_cfg.get("ratio_from_reference", {}) if sim_mode == "conventional" else {}
    ratio_ref_enabled = bool(ratio_ref_cfg.get("enabled", False))
    ratio_ref_source = str(
        ratio_ref_cfg.get("omega_out_source", motor2_omega_source)
    ).lower()
    ratio_ref_mode = str(ratio_ref_cfg.get("schedule_mode", schedule_mode)).lower()
    ratio_ref_hysteresis = abs(float(ratio_ref_cfg.get("hysteresis", 0.0)))
    ratio_ref_schedule_cfg = ratio_ref_cfg.get("speed_schedule", []) or []
    ratio_ref_schedule: list[dict[str, float]] = []
    if isinstance(ratio_ref_schedule_cfg, list):
        for entry in ratio_ref_schedule_cfg:
            if not isinstance(entry, dict):
                continue
            ratio = float(entry.get("ratio", 0.0))
            speed_min = float(entry.get("speed_min", 0.0))
            if ratio > 0.0:
                ratio_ref_schedule.append({"speed_min": speed_min, "ratio": ratio})
    ratio_ref_schedule.sort(key=lambda item: item["speed_min"])
    ratio_ref_idx = 0
    ratio_schedule_cfg = motor2_vel_cfg.get("ratio_schedule", []) or []
    ratio_schedule: list[dict[str, float]] = []
    if isinstance(ratio_schedule_cfg, list):
        for entry in ratio_schedule_cfg:
            if not isinstance(entry, dict):
                continue
            ratio = float(entry.get("ratio", 0.0))
            util_min = float(entry.get("utilization_min", 0.0))
            if ratio > 0.0:
                ratio_schedule.append({"util_min": util_min, "ratio": ratio})
    ratio_schedule.sort(key=lambda item: item["util_min"])
    ratio_idx = 0
    ratio_hysteresis = abs(float(motor2_vel_cfg.get("hysteresis", 0.0)))
    base_ratio = abs(float(mechanism_vec[0])) if abs(mechanism_vec[0]) > 1e-9 else 1.0
    a1_sign = math.copysign(1.0, mechanism_vec[0]) if abs(mechanism_vec[0]) > 1e-9 else 1.0
    k1 = float(kinematic_vec[0])
    k2 = float(kinematic_vec[1])
    velocity_limits_cfg = (controller_cfg.get("velocity_distribution", {}) or {}).get(
        "velocity_limits", {}
    )
    motor1_vel_limit = float(velocity_limits_cfg.get("motor1", 1.0))
    vel_cmd_m2_filtered = 0.0
    rate_limit = float(motor2_vel_cfg.get("rate_limit", 0.0))
    filter_alpha = motor2_vel_cfg.get("filter_alpha")
    filter_tau = motor2_vel_cfg.get("filter_tau")
    if filter_alpha is not None:
        filter_alpha = float(filter_alpha)
        if filter_alpha <= 0.0:
            filter_alpha = None
        else:
            filter_alpha = max(0.0, min(1.0, filter_alpha))
    elif filter_tau is not None:
        filter_tau = float(filter_tau)
        if filter_tau > 0.0:
            filter_alpha = dt / (filter_tau + dt)
        else:
            filter_alpha = None
    motor2_controller = None
    motor1_controller = None
    motor1_vel_enabled = False
    motor1_vel_mode = "schedule"
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
        motor1_vel_cfg = conventional_cfg.get("motor1_velocity", {}) or {}
        motor1_vel_enabled = bool(motor1_vel_cfg.get("enabled", False)) or ratio_ref_enabled
        motor1_vel_mode = str(motor1_vel_cfg.get("mode", "schedule")).lower()
        motor1_omega_source = str(motor1_vel_cfg.get("omega_out_source", "")).lower()
        if not motor1_omega_source and bool(motor1_vel_cfg.get("use_reference_derivative", False)):
            motor1_omega_source = "reference_derivative"
        if not motor1_omega_source:
            motor1_omega_source = motor2_omega_source
        motor1_pid_cfg = conventional_cfg.get("motor1_velocity_pid", {})
        m1_ctrl_gains = {
            "kp": float(motor1_pid_cfg.get("kp", 0.1)),
            "ki": float(motor1_pid_cfg.get("ki", 0.0)),
            "kd": float(motor1_pid_cfg.get("kd", 0.0)),
            "max_output": float(motor1_pid_cfg.get("max_output", torque_limits.get("motor1", 1.0))),
        }
        if motor1_vel_enabled:
            motor1_controller = VelocityController(
                gains=m1_ctrl_gains,
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

    def _interpolate_ratio(util: float) -> float:
        if not ratio_schedule:
            return base_ratio
        if util <= ratio_schedule[0]["util_min"]:
            return ratio_schedule[0]["ratio"]
        for lo, hi in zip(ratio_schedule, ratio_schedule[1:]):
            if util <= hi["util_min"]:
                u0 = float(lo["util_min"])
                u1 = float(hi["util_min"])
                r0 = float(lo["ratio"])
                r1 = float(hi["ratio"])
                if u1 <= u0 + 1e-9:
                    return r1
                t = (util - u0) / (u1 - u0)
                return r0 + (r1 - r0) * t
        return ratio_schedule[-1]["ratio"]

    def _interpolate_ratio_speed(speed: float) -> float:
        if not ratio_ref_schedule:
            return base_ratio
        if speed <= ratio_ref_schedule[0]["speed_min"]:
            return ratio_ref_schedule[0]["ratio"]
        for lo, hi in zip(ratio_ref_schedule, ratio_ref_schedule[1:]):
            if speed <= hi["speed_min"]:
                s0 = float(lo["speed_min"])
                s1 = float(hi["speed_min"])
                r0 = float(lo["ratio"])
                r1 = float(hi["ratio"])
                if s1 <= s0 + 1e-9:
                    return r1
                t = (speed - s0) / (s1 - s0)
                return r0 + (r1 - r0) * t
        return ratio_ref_schedule[-1]["ratio"]

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
        if prev_reference_position is None:
            ref_velocity_fd = 0.0
        else:
            ref_velocity_fd = (reference_position - prev_reference_position) / dt
        prev_reference_position = reference_position

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

        tau_push_vec = np.zeros(2, dtype=float)
        if internal_push_enabled and internal_push_beta > 0.0 and null_scale > 1e-9:
            if internal_push_sign_source == "tau_aug":
                push_source = tau_aug
            else:
                push_source = tau_cmd
            if internal_push_sign_mode == "output_ref":
                if abs(push_source) > 1e-9:
                    push_sign = math.copysign(1.0, push_source)
                else:
                    push_sign = 0.0
            else:
                if abs(internal_push_fixed_sign) > 1e-9:
                    push_sign = math.copysign(1.0, internal_push_fixed_sign)
                else:
                    push_sign = 0.0
            if internal_push_mode == "constant":
                speed_scale = 1.0
            elif internal_push_mode == "velocity_exp":
                if internal_push_omega_c > 1e-9:
                    speed_scale = math.exp(-abs(output_vel) / internal_push_omega_c)
                else:
                    speed_scale = 0.0
            elif internal_push_mode == "velocity_step":
                if internal_push_omega_c > 1e-9 and abs(output_vel) < internal_push_omega_c:
                    speed_scale = 1.0
                else:
                    speed_scale = 0.0
            else:
                speed_scale = 0.0
            push_mag = internal_push_beta * abs(push_source) * speed_scale
            if internal_push_tau_max is not None:
                push_mag = min(push_mag, internal_push_tau_max)
            tau_push_vec = (push_sign * push_mag / null_scale) * null_vec

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
            assist_secondary_gain = 1.0
            if assist_manager is not None:
                assist_status = assist_manager.update(tau_aug)
                assist_secondary_gain = float(assist_status.secondary_gain)
            selected_ratio = None
            a1_eff = None
            omega1_ref = None
            if ratio_ref_enabled:
                if ratio_ref_source == "reference":
                    omega_out_ref = float(reference_velocity)
                elif ratio_ref_source == "reference_derivative":
                    omega_out_ref = float(ref_velocity_fd)
                else:
                    omega_out_ref = float(output_vel)
                speed_ref = abs(omega_out_ref)
                if ratio_ref_schedule:
                    if ratio_ref_mode == "continuous":
                        selected_ratio = _interpolate_ratio_speed(speed_ref)
                    else:
                        while (
                            ratio_ref_idx + 1 < len(ratio_ref_schedule)
                            and speed_ref
                            >= ratio_ref_schedule[ratio_ref_idx + 1]["speed_min"] + ratio_ref_hysteresis
                        ):
                            ratio_ref_idx += 1
                        while (
                            ratio_ref_idx > 0
                            and speed_ref
                            < ratio_ref_schedule[ratio_ref_idx]["speed_min"] - ratio_ref_hysteresis
                        ):
                            ratio_ref_idx -= 1
                        selected_ratio = ratio_ref_schedule[ratio_ref_idx]["ratio"]
                else:
                    selected_ratio = base_ratio
                a1_eff = a1_sign * selected_ratio
                if abs(k2) > 1e-9:
                    omega1_ref = selected_ratio * a1_sign * omega_out_ref
                    vel_cmd_m2 = (omega_out_ref - k1 * omega1_ref) / k2
                else:
                    vel_cmd_m2 = 0.0
            elif motor2_mode == "schedule":
                current_ratio = (
                    ratio_schedule[ratio_idx]["ratio"] if ratio_schedule else base_ratio
                )
                a1_eff = a1_sign * current_ratio
                if utilization_source == "velocity":
                    if abs(k1) > 1e-9:
                        motor1_velocity = (output_vel - k2 * motor2_vel) / k1
                    else:
                        motor1_velocity = 0.0
                    util1 = abs(motor1_velocity) / (motor1_vel_limit + 1e-9)
                else:
                    limit1 = float(torque_limits.get("motor1", 1.0))
                    if abs(a1_eff) > 1e-9:
                        tau1_req = tau_aug / a1_eff
                        util1 = abs(tau1_req) / (limit1 + 1e-9)
                    else:
                        util1 = 0.0

                if ratio_schedule and schedule_mode == "continuous":
                    selected_ratio = _interpolate_ratio(util1)
                elif ratio_schedule:
                    while (
                        ratio_idx + 1 < len(ratio_schedule)
                        and util1 >= ratio_schedule[ratio_idx + 1]["util_min"] + ratio_hysteresis
                    ):
                        ratio_idx += 1
                    while (
                        ratio_idx > 0
                        and util1 < ratio_schedule[ratio_idx]["util_min"] - ratio_hysteresis
                    ):
                        ratio_idx -= 1
                    selected_ratio = ratio_schedule[ratio_idx]["ratio"]
                else:
                    selected_ratio = base_ratio
                a1_eff = a1_sign * selected_ratio

                if motor2_omega_source == "reference":
                    omega_out_ref = float(reference_velocity)
                elif motor2_omega_source == "reference_derivative":
                    omega_out_ref = float(ref_velocity_fd)
                else:
                    omega_out_ref = float(output_vel)
                if abs(k2) > 1e-9:
                    omega1_ref = selected_ratio * a1_sign * omega_out_ref
                    vel_cmd_m2 = (omega_out_ref - k1 * omega1_ref) / k2
                else:
                    vel_cmd_m2 = 0.0
            elif motor2_mode == "constant":
                vel_cmd_m2 = float(motor2_vel_cfg.get("value", 0.0))
            elif motor2_mode == "adaptive":
                A1 = float(mechanism_vec[0])
                if utilization_source == "velocity":
                    if abs(k1) > 1e-9:
                        motor1_velocity = (output_vel - k2 * motor2_vel) / k1
                    else:
                        motor1_velocity = 0.0
                    util1 = abs(motor1_velocity) / (motor1_vel_limit + 1e-9)
                else:
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
            vel_cmd_m2 *= assist_secondary_gain

            if rate_limit > 0.0:
                max_delta = rate_limit * dt
                delta = vel_cmd_m2 - vel_cmd_m2_filtered
                delta = max(-max_delta, min(max_delta, delta))
                vel_cmd_m2_limited = vel_cmd_m2_filtered + delta
            else:
                vel_cmd_m2_limited = vel_cmd_m2

            if filter_alpha is not None:
                vel_cmd_m2_filtered += filter_alpha * (vel_cmd_m2_limited - vel_cmd_m2_filtered)
            else:
                vel_cmd_m2_filtered = vel_cmd_m2_limited

            if motor2_controller is None:
                tau2_cmd = 0.0
            else:
                m2_feedback = PositionFeedback(position=motor2_pos, velocity=motor2_vel)
                m2_command = PositionCommand(
                    position=0.0,
                    velocity=vel_cmd_m2_filtered,
                    acceleration=0.0,
                )
                tau2_cmd, _ = motor2_controller.update(m2_command, m2_feedback)

            A1 = float(mechanism_vec[0])
            A2 = float(mechanism_vec[1])
            tau2_measured = tau2_cmd
            if motor1_controller is not None and ratio_ref_enabled and omega1_ref is not None:
                if abs(k1) > 1e-9:
                    omega1_meas = (float(output_vel) - k2 * float(motor2_vel)) / k1
                else:
                    omega1_meas = 0.0
                m1_feedback = PositionFeedback(position=0.0, velocity=omega1_meas)
                m1_command = PositionCommand(position=0.0, velocity=omega1_ref, acceleration=0.0)
                tau1_cmd, _ = motor1_controller.update(m1_command, m1_feedback)
            elif motor1_controller is not None and motor2_mode == "schedule" and omega1_ref is not None:
                if motor1_omega_source == "reference":
                    omega_out_ref_m1 = float(reference_velocity)
                elif motor1_omega_source == "reference_derivative":
                    omega_out_ref_m1 = float(ref_velocity_fd)
                else:
                    omega_out_ref_m1 = float(output_vel)
                if abs(k2) > 1e-9:
                    omega1_ref = selected_ratio * a1_sign * omega_out_ref_m1
                if abs(k1) > 1e-9:
                    omega1_meas = (float(output_vel) - k2 * float(motor2_vel)) / k1
                else:
                    omega1_meas = 0.0
                m1_feedback = PositionFeedback(position=0.0, velocity=omega1_meas)
                m1_command = PositionCommand(position=0.0, velocity=omega1_ref, acceleration=0.0)
                tau1_cmd, _ = motor1_controller.update(m1_command, m1_feedback)
            elif motor2_mode == "schedule":
                if a1_eff is None or abs(a1_eff) < 1e-9:
                    tau1_cmd = 0.0
                else:
                    tau1_cmd = tau_aug / a1_eff
            else:
                if abs(A1) < 1e-6:
                    tau1_cmd = 0.0
                else:
                    tau1_cmd = (tau_aug - A2 * tau2_measured) / A1
            if m2_coupling_enabled and m2_coupling_gain > 0.0:
                if m2_coupling_sign_mode == "opposite":
                    tau1_cmd += -m2_coupling_gain * tau2_cmd
                else:
                    tau1_cmd += m2_coupling_gain * tau2_cmd
                if m2_coupling_tau_max is not None:
                    tau1_cmd = float(
                        np.clip(tau1_cmd, -m2_coupling_tau_max, m2_coupling_tau_max)
                    )
            tau_cmds = np.array([tau1_cmd, tau2_cmd], dtype=float)
            tau_alloc = np.clip(tau_cmds + tau_push_vec, -limits_vec, limits_vec)
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
            if m2_coupling_enabled and m2_coupling_gain > 0.0:
                tau1_add = m2_coupling_gain * tau_alloc[1]
                if m2_coupling_sign_mode == "opposite":
                    tau1_add = -tau1_add
                if m2_coupling_tau_max is not None:
                    tau1_add = float(
                        np.clip(tau1_add, -m2_coupling_tau_max, m2_coupling_tau_max)
                    )
                tau_alloc[0] += tau1_add
            if internal_push_enabled:
                tau_alloc = np.clip(tau_alloc + tau_push_vec, -limits_vec, limits_vec)
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
