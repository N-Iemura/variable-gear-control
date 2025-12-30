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

# Limits for the Limit Drop scenario
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
    
    # 2. Motor 1 Torque
    axes[1].plot(t, df["tau_1"], label=label, color=colors[label], alpha=0.8)
    
    # 3. Motor 2 Torque
    axes[2].plot(t, df["tau_2"], label=label, color=colors[label], alpha=0.8)

# Formatting
axes[0].set_title("Position Tracking Error")
axes[0].set_ylabel("Error [turn]")
axes[0].grid(True)
axes[0].legend()

axes[1].set_title(f"Motor 1 Torque (Limit: {LIMITS['motor1']} Nm)")
axes[1].set_ylabel("Torque [Nm]")
axes[1].axhline(LIMITS['motor1'], color='r', linestyle='--', alpha=0.5)
axes[1].axhline(-LIMITS['motor1'], color='r', linestyle='--', alpha=0.5)
axes[1].grid(True)

axes[2].set_title(f"Motor 2 Torque (Limit: {LIMITS['motor2']} Nm)")
axes[2].set_ylabel("Torque [Nm]")
axes[2].set_xlabel("Time [s]")
axes[2].axhline(LIMITS['motor2'], color='r', linestyle='--', alpha=0.5)
axes[2].axhline(-LIMITS['motor2'], color='r', linestyle='--', alpha=0.5)
axes[2].grid(True)

output_path = "analysis/comparison_limit_drop_3cases.pdf"
plt.tight_layout()
plt.savefig(output_path)
print(f"Comparison plot saved to {output_path}")
