import odrive
from odrive.enums import *
import time
import math
import matplotlib.pyplot as plt
import numpy as np

TORQUE_CONTROL= 1
MOTOR_CALIBRATION = 4
AXIS_STATE_CLOSED_LOOP_CONTROL = 8
VELOCITY_CONTROL = 2

# Find two ODrives
odrv0 = odrive.find_any(serial_number='3856345D3539')
odrv1 = odrive.find_any(serial_number='384D346F3539')
odrv2 = odrive.find_any(serial_number='3849346F3539')

# odrv0.config.brake_resistor0.resistance = 2.0
# odrv0.config.enable_brake_resistor = True
# odrv1.config.brake_resistor0.resistance = 2.0
# odrv1.config.enable_brake_resistor = True
odrv0.clear_errors()
odrv1.clear_errors()
odrv2.clear_errors()

odrv0.axis0.requested_state = AxisState.IDLE
odrv1.axis0.requested_state = AxisState.IDLE
# odrv2.axis0.requested_state = AxisState.IDLE