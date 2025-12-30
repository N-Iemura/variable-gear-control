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

# Data storage
labels = []
avg_util_m1 = []
avg_util_m2 = []
max_util_m1 = []
max_util_m2 = []

for label, filepath in files.items():
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        continue
        
    df = pd.read_csv(filepath, comment="#")
    
    # Calculate Utilization
    u1 = np.abs(df["tau_1"]) / LIMITS["motor1"]
    u2 = np.abs(df["tau_2"]) / LIMITS["motor2"]
    
    labels.append(label)
    avg_util_m1.append(u1.mean())
    avg_util_m2.append(u2.mean())
    max_util_m1.append(u1.max())
    max_util_m2.append(u2.max())

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1. Average Utilization Comparison
x = np.arange(len(labels))
width = 0.35

axes[0].bar(x - width/2, avg_util_m1, width, label='Motor 1', color='skyblue')
axes[0].bar(x + width/2, avg_util_m2, width, label='Motor 2', color='salmon')

axes[0].set_ylabel('Average Utilization (Torque / Limit)')
axes[0].set_title('Average Motor Utilization Comparison')
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, rotation=15)
axes[0].legend()
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for i, v in enumerate(avg_util_m1):
    axes[0].text(i - width/2, v + 0.005, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
for i, v in enumerate(avg_util_m2):
    axes[0].text(i + width/2, v + 0.005, f"{v:.3f}", ha='center', va='bottom', fontsize=9)


# 2. Max Utilization Comparison (Peak Load)
axes[1].bar(x - width/2, max_util_m1, width, label='Motor 1', color='skyblue')
axes[1].bar(x + width/2, max_util_m2, width, label='Motor 2', color='salmon')

axes[1].set_ylabel('Max Utilization (Peak Torque / Limit)')
axes[1].set_title('Peak Motor Utilization Comparison')
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, rotation=15)
axes[1].legend()
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for i, v in enumerate(max_util_m1):
    axes[1].text(i - width/2, v + 0.005, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
for i, v in enumerate(max_util_m2):
    axes[1].text(i + width/2, v + 0.005, f"{v:.3f}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
output_path = "analysis/utilization_comparison_bar.pdf"
plt.savefig(output_path)
print(f"Utilization comparison plot saved to {output_path}")
