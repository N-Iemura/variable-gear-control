from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

try:
    import odrive
    from odrive.enums import (
        AXIS_STATE_CLOSED_LOOP_CONTROL,
        AXIS_STATE_IDLE,
        AXIS_STATE_FULL_CALIBRATION_SEQUENCE,
        CONTROL_MODE_TORQUE_CONTROL,
    )
    try:  # Optional in older firmware/tooling.
        from odrive.enums import CONTROL_MODE_VELOCITY_CONTROL, INPUT_MODE_PASSTHROUGH
    except ImportError:
        CONTROL_MODE_VELOCITY_CONTROL = 2
        INPUT_MODE_PASSTHROUGH = 1
except ImportError as exc:  # pragma: no cover - hardware dependency
    odrive = None
    AXIS_STATE_CLOSED_LOOP_CONTROL = 8
    AXIS_STATE_IDLE = 1
    AXIS_STATE_FULL_CALIBRATION_SEQUENCE = 3
    CONTROL_MODE_TORQUE_CONTROL = 1
    CONTROL_MODE_VELOCITY_CONTROL = 2
    INPUT_MODE_PASSTHROUGH = 1
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


_LOGGER = logging.getLogger(__name__)


@dataclass
class AxisSignals:
    position: float
    velocity: float
    torque_command: float
    current_iq: float
    torque_measured: float


class ODriveAxisHandle:
    """Thin wrapper around a single ODrive axis."""

    def __init__(
        self,
        device,
        axis,
        name: str,
        torque_constant: Optional[float] = None,
        is_motor: bool = True,
    ) -> None:
        self.device = device
        self.axis = axis
        self.name = name
        self.is_motor = is_motor
        self.torque_constant = torque_constant

    def prepare_for_torque_control(self) -> None:
        if not self.is_motor:
            return
        self._apply_torque_constant()
        controller_cfg = getattr(self.axis, "controller", None)
        if controller_cfg and hasattr(controller_cfg, "config"):
            controller_cfg.config.control_mode = CONTROL_MODE_TORQUE_CONTROL
            if hasattr(controller_cfg.config, "input_mode"):
                controller_cfg.config.input_mode = INPUT_MODE_PASSTHROUGH
        self.axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
        if controller_cfg and hasattr(controller_cfg, "input_torque"):
            controller_cfg.input_torque = 0.0

    def prepare_for_velocity_control(self) -> None:
        if not self.is_motor:
            return
        self._apply_torque_constant()
        controller_cfg = getattr(self.axis, "controller", None)
        if controller_cfg and hasattr(controller_cfg, "config"):
            controller_cfg.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
            if hasattr(controller_cfg.config, "input_mode"):
                controller_cfg.config.input_mode = INPUT_MODE_PASSTHROUGH
        self.axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
        if controller_cfg is not None:
            if hasattr(controller_cfg, "input_vel"):
                controller_cfg.input_vel = 0.0
            elif hasattr(controller_cfg, "vel_setpoint"):
                controller_cfg.vel_setpoint = 0.0

    def set_velocity_gains(
        self,
        vel_gain: Optional[float] = None,
        vel_integrator_gain: Optional[float] = None,
        vel_integrator_limit: Optional[float] = None,
    ) -> Dict[str, float]:
        if not self.is_motor:
            return {}
        controller_cfg = getattr(self.axis, "controller", None)
        config = getattr(controller_cfg, "config", None) if controller_cfg is not None else None
        if config is None:
            return {}
        applied: Dict[str, float] = {}
        if vel_gain is not None and hasattr(config, "vel_gain"):
            config.vel_gain = float(vel_gain)
            applied["vel_gain"] = float(getattr(config, "vel_gain"))
        if vel_integrator_gain is not None and hasattr(config, "vel_integrator_gain"):
            config.vel_integrator_gain = float(vel_integrator_gain)
            applied["vel_integrator_gain"] = float(getattr(config, "vel_integrator_gain"))
        if vel_integrator_limit is not None and hasattr(config, "vel_integrator_limit"):
            config.vel_integrator_limit = float(vel_integrator_limit)
            applied["vel_integrator_limit"] = float(getattr(config, "vel_integrator_limit"))
        return applied

    def request_state(self, state: int) -> None:
        self.axis.requested_state = state

    def wait_for_idle(self, timeout: float = 30.0) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.axis.current_state == AXIS_STATE_IDLE:
                return
            time.sleep(0.1)
        raise TimeoutError("Axis did not enter IDLE state within timeout.")

    def read_signals(self, fast: bool = False) -> AxisSignals:
        position, velocity = self._resolve_position_velocity()
        torque_cmd = 0.0
        iq_measured = 0.0
        if self.is_motor and not fast:
            motor = getattr(self.axis, "motor", None)
            if motor is not None:
                # ODrive exposes Iq in different places/casings across firmware revisions.
                iq_paths = [
                    ("current_control", "iq_measured"),
                    ("current_control", "Iq_measured"),
                    ("current_control", "Iq_setpoint"),
                    ("foc", "Iq_measured"),
                    ("foc", "Iq_setpoint"),
                    ("Iq_measured",),
                    ("Iq_setpoint",),
                ]
                for path in iq_paths:
                    obj = motor
                    for attr in path:
                        obj = getattr(obj, attr, None)
                        if obj is None:
                            break
                    else:
                        try:
                            iq_measured = float(obj)
                            break
                        except (TypeError, ValueError):
                            continue
        torque_measured = iq_measured * float(self.torque_constant or 0.0)
        return AxisSignals(
            position=position,
            velocity=velocity,
            torque_command=torque_cmd,
            current_iq=iq_measured,
            torque_measured=torque_measured,
        )

    def command_torque(self, torque: float) -> None:
        if not self.is_motor:
            return
        controller = getattr(self.axis, "controller", None)
        if controller is not None and hasattr(controller, "input_torque"):
            controller.input_torque = float(torque)

    def command_velocity(self, velocity: float) -> None:
        if not self.is_motor:
            return
        controller = getattr(self.axis, "controller", None)
        if controller is None:
            return
        if hasattr(controller, "input_vel"):
            controller.input_vel = float(velocity)
            return
        if hasattr(controller, "vel_setpoint"):
            controller.vel_setpoint = float(velocity)
            return
        raise AttributeError("ODrive controller does not expose a velocity setpoint attribute.")

    def idle(self) -> None:
        self.request_state(AXIS_STATE_IDLE)

    def _resolve_position_velocity(self) -> tuple[float, float]:
        mapper = getattr(self.axis, "pos_vel_mapper", None)
        if mapper is not None:
            position = 0.0
            for attr in ("pos_rel", "pos", "pos_estimate", "pos_abs"):
                if hasattr(mapper, attr):
                    position = float(getattr(mapper, attr))
                    break
            velocity = 0.0
            for attr in ("vel", "vel_rel", "vel_estimate"):
                if hasattr(mapper, attr):
                    velocity = float(getattr(mapper, attr))
                    break
            return position, velocity

        encoder = getattr(self.axis, "encoder", None)
        if encoder is not None:
            position = float(getattr(encoder, "pos_estimate", 0.0))
            velocity = float(getattr(encoder, "vel_estimate", 0.0))
            return position, velocity

        return 0.0, 0.0

    def _apply_torque_constant(self) -> None:
        if self.torque_constant is None:
            return
        if hasattr(self.axis, "config") and hasattr(self.axis.config, "motor"):
            self.axis.config.motor.torque_constant = float(self.torque_constant)
        elif hasattr(self.axis, "motor") and hasattr(self.axis.motor, "config"):
            self.axis.motor.config.torque_constant = float(self.torque_constant)


class ODriveInterface:
    """High-level orchestration for a pair of ODrive-controlled motors."""

    def __init__(self, hw_config: Dict[str, object]) -> None:
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "odrive Python package is required for hardware communication"
            ) from _IMPORT_ERROR

        odrive_cfg = dict(hw_config.get("odrive", {}))
        self.serial_numbers = odrive_cfg.get("serial_numbers", {})
        self.torque_constants = odrive_cfg.get("torque_constants", {})

        self.devices: Dict[str, ODriveAxisHandle] = {}
        self.output_axis: Optional[ODriveAxisHandle] = None
        self.position_offsets: Dict[str, float] = {}

    def connect(
        self,
        calibrate: bool = False,
        control_mode: str = "torque",
        axes: Sequence[str] | None = None,
    ) -> None:
        _LOGGER.info("Connecting to ODrive devices...")
        axes = list(axes) if axes is not None else ["motor1", "motor2", "output"]
        allowed = {"motor1", "motor2", "output"}
        unknown = [name for name in axes if name not in allowed]
        if unknown:
            raise ValueError(f"Unsupported axes requested: {unknown}")

        self.devices = {}
        self.output_axis = None
        self.position_offsets = {}

        if "motor1" in axes:
            self.devices["motor1"] = self._connect_axis("motor1", calibrate=calibrate, is_motor=True)
        if "motor2" in axes:
            self.devices["motor2"] = self._connect_axis("motor2", calibrate=calibrate, is_motor=True)
        if "output" in axes:
            self.output_axis = self._connect_axis("output", calibrate=False, is_motor=False)

        self.set_motor_control_mode(control_mode)
        self.zero_positions()

    def _connect_axis(
        self,
        key: str,
        calibrate: bool = False,
        is_motor: bool = True,
    ) -> ODriveAxisHandle:
        serial = self.serial_numbers.get(key)
        if serial is None:
            raise KeyError(f"Serial number for '{key}' is not defined in hardware config.")
        _LOGGER.info("Searching for ODrive (%s)...", key)
        device = odrive.find_any(serial_number=serial)
        axis = device.axis0

        torque_constant = None
        if is_motor and key in self.torque_constants:
            torque_constant = float(self.torque_constants[key])

        handle = ODriveAxisHandle(
            device=device,
            axis=axis,
            name=key,
            torque_constant=torque_constant,
            is_motor=is_motor,
        )

        if calibrate:
            _LOGGER.info("Running full calibration sequence for %s", key)
            handle.request_state(AXIS_STATE_FULL_CALIBRATION_SEQUENCE)
            handle.wait_for_idle()

        return handle

    def zero_positions(self) -> None:
        """Snapshot current positions so subsequent reads are zero-referenced."""
        for name, handle in self.devices.items():
            signals = handle.read_signals()
            self.position_offsets[name] = signals.position
        if self.output_axis is not None:
            signals = self.output_axis.read_signals()
            self.position_offsets["output"] = signals.position
        _LOGGER.info("Position offsets captured: %s", ", ".join(f"{k}={v:.6f}" for k, v in self.position_offsets.items()))

    def read_states(self, fast: bool = False) -> Dict[str, AxisSignals]:
        states: Dict[str, AxisSignals] = {}
        for name, handle in self.devices.items():
            signals = handle.read_signals(fast=fast)
            offset = self.position_offsets.get(name, 0.0)
            states[name] = AxisSignals(
                position=signals.position - offset,
                velocity=signals.velocity,
                torque_command=signals.torque_command,
                current_iq=signals.current_iq,
                torque_measured=signals.torque_measured,
            )
        if self.output_axis is not None:
            signals = self.output_axis.read_signals(fast=fast)
            offset = self.position_offsets.get("output", 0.0)
            states["output"] = AxisSignals(
                position=signals.position - offset,
                velocity=signals.velocity,
                torque_command=signals.torque_command,
                current_iq=signals.current_iq,
                torque_measured=signals.torque_measured,
            )
        return states

    def command_torques(self, tau_motor1: float, tau_motor2: float) -> None:
        if "motor1" in self.devices:
            self.devices["motor1"].command_torque(tau_motor1)
        if "motor2" in self.devices:
            self.devices["motor2"].command_torque(tau_motor2)

    def command_velocities(self, vel_motor1: float, vel_motor2: float) -> None:
        if "motor1" in self.devices:
            self.devices["motor1"].command_velocity(vel_motor1)
        if "motor2" in self.devices:
            self.devices["motor2"].command_velocity(vel_motor2)

    def set_motor_control_mode(self, mode: str) -> None:
        mode = str(mode).lower()
        for name, axis in self.devices.items():
            if not axis.is_motor:
                continue
            if mode == "velocity":
                axis.prepare_for_velocity_control()
                _LOGGER.info("Closed-loop velocity control ready on %s", name)
            elif mode == "torque":
                axis.prepare_for_torque_control()
                _LOGGER.info("Closed-loop torque control ready on %s", name)
            else:
                raise ValueError(f"Unsupported control mode: {mode}")

    def shutdown(self) -> None:
        for handle in self.devices.values():
            try:
                handle.command_torque(0.0)
                handle.idle()
            except Exception:  # pragma: no cover - best effort shutdown
                _LOGGER.exception("Failed to safely idle axis.")

    def set_velocity_gains(
        self,
        name: str,
        vel_gain: Optional[float] = None,
        vel_integrator_gain: Optional[float] = None,
        vel_integrator_limit: Optional[float] = None,
    ) -> Dict[str, float]:
        handle = self.devices.get(name)
        if handle is None:
            return {}
        return handle.set_velocity_gains(
            vel_gain=vel_gain,
            vel_integrator_gain=vel_integrator_gain,
            vel_integrator_limit=vel_integrator_limit,
        )
