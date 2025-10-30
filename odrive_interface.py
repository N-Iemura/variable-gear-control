from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

try:
    import odrive
    from odrive.enums import (
        AXIS_STATE_CLOSED_LOOP_CONTROL,
        AXIS_STATE_IDLE,
        AXIS_STATE_FULL_CALIBRATION_SEQUENCE,
        CONTROL_MODE_TORQUE_CONTROL,
    )
except ImportError as exc:  # pragma: no cover - hardware dependency
    odrive = None
    AXIS_STATE_CLOSED_LOOP_CONTROL = 8
    AXIS_STATE_IDLE = 1
    AXIS_STATE_FULL_CALIBRATION_SEQUENCE = 3
    CONTROL_MODE_TORQUE_CONTROL = 1
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


_LOGGER = logging.getLogger(__name__)


@dataclass
class AxisSignals:
    position: float
    velocity: float
    torque: float


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
        if self.torque_constant is not None:
            if hasattr(self.axis, "config") and hasattr(self.axis.config, "motor"):
                self.axis.config.motor.torque_constant = float(self.torque_constant)
            elif hasattr(self.axis, "motor") and hasattr(self.axis.motor, "config"):
                self.axis.motor.config.torque_constant = float(self.torque_constant)
        controller_cfg = getattr(self.axis, "controller", None)
        if controller_cfg and hasattr(controller_cfg, "config"):
            controller_cfg.config.control_mode = CONTROL_MODE_TORQUE_CONTROL
        self.axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
        if controller_cfg and hasattr(controller_cfg, "input_torque"):
            controller_cfg.input_torque = 0.0

    def request_state(self, state: int) -> None:
        self.axis.requested_state = state

    def wait_for_idle(self, timeout: float = 30.0) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.axis.current_state == AXIS_STATE_IDLE:
                return
            time.sleep(0.1)
        raise TimeoutError("Axis did not enter IDLE state within timeout.")

    def read_signals(self) -> AxisSignals:
        position, velocity = self._resolve_position_velocity()
        torque = 0.0
        if self.is_motor:
            controller = getattr(self.axis, "controller", None)
            if controller is not None and hasattr(controller, "input_torque"):
                torque = float(controller.input_torque)
        return AxisSignals(position=position, velocity=velocity, torque=torque)

    def command_torque(self, torque: float) -> None:
        if not self.is_motor:
            return
        controller = getattr(self.axis, "controller", None)
        if controller is not None and hasattr(controller, "input_torque"):
            controller.input_torque = float(torque)

    def idle(self) -> None:
        self.request_state(AXIS_STATE_IDLE)

    def _resolve_position_velocity(self) -> tuple[float, float]:
        mapper = getattr(self.axis, "pos_vel_mapper", None)
        if mapper is not None:
            position = float(getattr(mapper, "pos_rel", getattr(mapper, "pos_abs", 0.0)))
            velocity = float(getattr(mapper, "vel_rel", getattr(mapper, "vel_estimate", 0.0)))
            return position, velocity

        encoder = getattr(self.axis, "encoder", None)
        if encoder is not None:
            position = float(getattr(encoder, "pos_estimate", 0.0))
            velocity = float(getattr(encoder, "vel_estimate", 0.0))
            return position, velocity

        return 0.0, 0.0


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

    def connect(self, calibrate: bool = False) -> None:
        _LOGGER.info("Connecting to ODrive devices...")
        motor0 = self._connect_axis("motor0", calibrate=calibrate, is_motor=True)
        motor1 = self._connect_axis("motor1", calibrate=calibrate, is_motor=True)
        output = self._connect_axis("output", calibrate=False, is_motor=False)

        self.devices["motor0"] = motor0
        self.devices["motor1"] = motor1
        self.output_axis = output

        for name, axis in self.devices.items():
            axis.prepare_for_torque_control()
            if axis.is_motor:
                _LOGGER.info("Closed-loop torque control ready on %s", name)
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

    def read_states(self) -> Dict[str, AxisSignals]:
        states: Dict[str, AxisSignals] = {}
        for name, handle in self.devices.items():
            signals = handle.read_signals()
            offset = self.position_offsets.get(name, 0.0)
            states[name] = AxisSignals(
                position=signals.position - offset,
                velocity=signals.velocity,
                torque=signals.torque,
            )
        if self.output_axis is not None:
            signals = self.output_axis.read_signals()
            offset = self.position_offsets.get("output", 0.0)
            states["output"] = AxisSignals(
                position=signals.position - offset,
                velocity=signals.velocity,
                torque=signals.torque,
            )
        return states

    def command_torques(self, tau_motor0: float, tau_motor1: float) -> None:
        self.devices["motor0"].command_torque(tau_motor0)
        self.devices["motor1"].command_torque(tau_motor1)

    def shutdown(self) -> None:
        for handle in self.devices.values():
            try:
                handle.command_torque(0.0)
                handle.idle()
            except Exception:  # pragma: no cover - best effort shutdown
                _LOGGER.exception("Failed to safely idle axis.")
