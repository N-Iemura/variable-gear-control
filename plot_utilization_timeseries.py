import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Files provided by user
files = {
    "Utilization (Proposed)": "csv/dob_sine_20251230_172543.csv",
    "Raw 1:4 (Fixed)": "csv/dob_sine_20251230_172612.csv",
    "Raw 1:16 (Fixed)": "csv/dob_sine_20251230_172628.csv"
}

# Limits used in the experiment
LIMITS = {"motor1": 1.0, "motor2": 0.5}

fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

colors = {"Utilization (Proposed)": "blue", "Raw 1:4 (Fixed)": "orange", "Raw 1:16 (Fixed)": "green"}

for label, filepath in files.items():
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        continue
        
    df = pd.read_csv(filepath, comment="#")
    t = df["time"]
    
    # 1. Position Error
    error = df["theta_ref"] - df["output_pos"]
    axes[0].plot(t, error, label=label, color=colors[label], alpha=0.8)
    
    # 2. Motor 1 Utilization
    util1 = df["tau_1"] / LIMITS["motor1"]
    axes[1].plot(t, util1, label=label, color=colors[label], alpha=0.8)
    
    # 3. Motor 2 Utilization
    util2 = df["tau_2"] / LIMITS["motor2"]
    axes[2].plot(t, util2, label=label, color=colors[label], alpha=0.8)

# Formatting
axes[0].set_title("Position Tracking Error")
axes[0].set_ylabel("Error [turn]")
axes[0].grid(True)
axes[0].legend()

axes[1].set_title(f"Motor 1 Utilization (Torque / {LIMITS['motor1']} Nm)")
axes[1].set_ylabel("Utilization Ratio")
axes[1].axhline(1.0, color='r', linestyle='--', alpha=0.5)
axes[1].axhline(-1.0, color='r', linestyle='--', alpha=0.5)
axes[1].grid(True)

axes[2].set_title(f"Motor 2 Utilization (Torque / {LIMITS['motor2']} Nm)")
axes[2].set_ylabel("Utilization Ratio")
axes[2].set_xlabel("Time [s]")
axes[2].axhline(1.0, color='r', linestyle='--', alpha=0.5)
axes[2].axhline(-1.0, color='r', linestyle='--', alpha=0.5)
axes[2].grid(True)

output_path = "analysis/utilization_timeseries_comparison.pdf"
plt.tight_layout()
plt.savefig(output_path)
print(f"Utilization time-series plot saved to {output_path}")
