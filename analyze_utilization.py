import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_utilization(csv_path, label, torque_limits):
    df = pd.read_csv(csv_path, comment="#")
    
    # Calculate utilization (absolute torque / limit)
    # Avoid division by zero
    lim1 = max(torque_limits[0], 1e-6)
    lim2 = max(torque_limits[1], 1e-6)
    
    util1 = np.abs(df["tau_1"]) / lim1
    util2 = np.abs(df["tau_2"]) / lim2
    
    # Filter for times when assist is active (when tau_2 is non-zero)
    # Using a small threshold to detect activity
    active_mask = np.abs(df["tau_2"]) > 0.01
    
    if not active_mask.any():
        print(f"[{label}] No assist activity detected.")
        return

    avg_util1 = util1[active_mask].mean()
    avg_util2 = util2[active_mask].mean()
    max_util1 = util1[active_mask].max()
    max_util2 = util2[active_mask].max()
    
    print(f"--- {label} ---")
    print(f"Active samples: {active_mask.sum()}")
    print(f"Avg Utilization - Motor1: {avg_util1:.3f}, Motor2: {avg_util2:.3f}")
    print(f"Max Utilization - Motor1: {max_util1:.3f}, Motor2: {max_util2:.3f}")
    print(f"Utilization Ratio (M2/M1) - Avg: {avg_util2/avg_util1:.3f}, Max: {max_util2/max_util1:.3f}")
    print("")

# Torque limits from controller.yaml (Motor1: 2.0, Motor2: 0.5)
limits = [2.0, 0.5]

print("Analyzing Torque Utilization...\n")
analyze_utilization("csv/dob_sine_20251230_161533.csv", "Raw (Fixed Ratio)", limits)
analyze_utilization("csv/dob_sine_20251230_161456.csv", "Torque Utilization (Proposed)", limits)
