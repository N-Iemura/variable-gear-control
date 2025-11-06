import odrive
from odrive.enums import *
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import csv
from datetime import datetime
from collections import deque

TORQUE_CONTROL= 1
MOTOR_CALIBRATION = 4
AXIS_STATE_CLOSED_LOOP_CONTROL = 8
POSITION_CONTROL = 3

# Find two ODrives
# Tmotor
odrv0 = odrive.find_any(serial_number='3856345D3539')
# Maxon
odrv1 = odrive.find_any(serial_number='384D346F3539')
# Encoder
odrv2 = odrive.find_any(serial_number='3849346F3539')


# odrv0.axis0.requested_state = MOTOR_CALIBRATION
# time.sleep(7) 
time_data = []
time_d_data = []
current_data_0= []
vel_data_0 = []
position_data_0 = []
current_data_1 = []
vel_data_1 = []
position_data_1 = []
output_pos_data = []
output_vel_data = []

input_torque0 = []
input_torque1 = []

initial_position0 = odrv0.axis0.pos_vel_mapper.pos_rel
initial_position1 = odrv1.axis0.pos_vel_mapper.pos_rel
# initial_position2 = odrv2.axis0.pos_vel_mapper.pos_rel

odrv0.axis0.requested_state = AxisState.CLOSED_LOOP_CONTROL
odrv0.axis0.controller.config.control_mode = ControlMode.VELOCITY_CONTROL
odrv0.axis0.config.motor.torque_constant = 0.106 #(トルク定数 8.23/Kv)

odrv1.axis0.requested_state = AxisState.CLOSED_LOOP_CONTROL
odrv1.axis0.controller.config.control_mode = ControlMode.VELOCITY_CONTROL
odrv1.axis0.config.motor.torque_constant = 0.091 #(トルク定数 8.23/Kv)

odrv0.axis0.controller.config.pos_gain = 100.0
odrv0.axis0.controller.config.vel_gain = 2.5

odrv1.axis0.controller.config.pos_gain = 100.0
odrv1.axis0.controller.config.vel_gain = 2.5

start_time = time.time()  # Get the current time
time1 = 0

## motor only
Kp0 = 30 # Proportional gain
Ki0 = 0  # Integral gain
Kd0 = 1 # Derivative gain
Kp1 = 30  # Proportional gain
Ki1 = 0 # Integral gain
Kd1 = 1 # Derivative gain
prev_error0 = 0
prev_error1 = 0
prev_time = time.time()
error_integral0 = 0
error_integral1 = 0

# 指令値を格納するリスト
ref0 = []
ref1 = []
error_queue0 = deque(maxlen=5)
error_queue1 = deque(maxlen=5)

# Define impulse parameters
impulse_time = 1.0  # Time after which to apply the impulse
impulse_duration = 0.005  # Duration of the impulse in seconds
impulse_position = 0.1  # Position to set during the impulse
loop_executed = False
second_impulse_executed = False
n = 10
impulse_flags = [0] * n

### leg ###
l1=0.45
l2=0.5
fx = 2

# Function to read position commands from a CSV file
# def read_position_commands(csv_file):
#     positions = []
#     with open(csv_file, 'r') as file:
#         reader = csv.reader(file)
#         next(reader)  # Skip header
#         for row in reader:
#             positions.append((-float(row[1])/360, -float(row[2])/360))  # Read Hip-Knee Angle and Knee-Ankle Angle
#     return positions

try:
    command_index = 0
    target_velocity = 30  # 最終的な目標速度 (turn/s)
    velocity_step = 1     # 速度を増加させるステップ (turn/s)
    current_velocity = 0  # 現在の速度 (初期値は0)

    while True:
        # Calculate the elapsed time
        time_diff = time.time() - prev_time
        prev_time = time.time()
        elapsed_time = time.time() - start_time 
        
        # Current position
        current_pos0, current_pos1 = odrv0.axis0.pos_vel_mapper.pos_rel-initial_position0, odrv1.axis0.pos_vel_mapper.pos_rel-initial_position1

        # 徐々に速度を増加させる
        if current_velocity < target_velocity:
            current_velocity += velocity_step
            if current_velocity > target_velocity:
                current_velocity = target_velocity  # 目標速度を超えないようにする

        # Set velocity for odrv0 and odrv1
        odrv0.axis0.controller.input_vel = current_velocity
        odrv1.axis0.controller.input_vel = -current_velocity*2.5

        # Add the data to the list
        current_data_0.append(odrv0.axis0.motor.foc.Iq_measured)
        vel_data_0.append(odrv0.axis0.pos_vel_mapper.vel)
        position_data_0.append(current_pos0)
        current_data_1.append(odrv1.axis0.motor.foc.Iq_measured)
        vel_data_1.append(odrv1.axis0.pos_vel_mapper.vel)
        position_data_1.append(current_pos1)
        time_data.append(elapsed_time) 
        output_pos_data.append(odrv2.axis0.pos_vel_mapper.pos_rel)
        output_vel_data.append(odrv2.axis0.pos_vel_mapper.vel)

        # Wait for 0.05 seconds before the next iteration
        time.sleep(0.05)

except KeyboardInterrupt:    
    now = datetime.now()
    # Format the date and time as a string
    timestamp = now.strftime("%Y%m%d_%H%M")
    # Create the filename
    filename = f'csv/vr_{timestamp}.csv'

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'Velocity_0', 'Position_0', 'Velocity_1', 'Position_1', 'Output_pos_0', 'Output_vel_0', 'Current_0', 'Current_1'])
        # Write the data to the CSV file

        writer.writerows(zip(time_data, vel_data_0, position_data_0, vel_data_1, position_data_1, output_pos_data, output_vel_data, current_data_0, current_data_1))
    print(f"Data saved to {filename}")
    # Clear the data lists
    time_data.clear()
    time_d_data.clear()
    current_data_0.clear()
    vel_data_0.clear()
    position_data_0.clear()
    current_data_1.clear()
    vel_data_1.clear()
    position_data_1.clear()
    output_pos_data.clear()
    output_vel_data.clear()
    input_torque0.clear()
    input_torque1.clear()
    ref0.clear()
    ref1.clear()
    print("Program interrupted. Data saved and cleared.")
    # Clear errors
    odrv0.clear_errors()
    odrv1.clear_errors()
    # Reset the ODrive states
    odrv0.axis0.requested_state = AxisState.IDLE
    odrv1.axis0.requested_state = AxisState.IDLE