import pandas as pd
import numpy as np

def analyze_saturation_margin(csv_path, label, limits):
    df = pd.read_csv(csv_path, comment="#")
    
    # Calculate utilization (absolute torque / limit)
    lim1, lim2 = limits
    util1 = np.abs(df["tau_1"]) / lim1
    util2 = np.abs(df["tau_2"]) / lim2
    
    # Filter for active assist periods
    active_mask = np.abs(df["tau_2"]) > 0.01
    
    if not active_mask.any():
        print(f"[{label}] No assist activity detected.")
        return

    u1 = util1[active_mask]
    u2 = util2[active_mask]
    
    # "Imbalance" metric: difference in utilization
    imbalance = np.abs(u1 - u2)
    
    print(f"--- {label} ---")
    print(f"Max Utilization: M1={u1.max():.3f}, M2={u2.max():.3f}")
    print(f"Avg Utilization: M1={u1.mean():.3f}, M2={u2.mean():.3f}")
    print(f"Utilization Imbalance (smaller is better): Avg={imbalance.mean():.3f}, Max={imbalance.max():.3f}")
    print("")

limits = [0.4, 0.1] # Pseudo limits
print("Analyzing Saturation Margin...\n")
analyze_saturation_margin("csv/dob_sine_20251230_163722.csv", "Raw (Fixed)", limits)
analyze_saturation_margin("csv/dob_sine_20251230_163751.csv", "Utilization (Proposed)", limits)
