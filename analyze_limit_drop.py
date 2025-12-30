import pandas as pd
import numpy as np

def analyze_limit_drop_scenario(csv_path, label, limits):
    df = pd.read_csv(csv_path, comment="#")
    
    lim1, lim2 = limits
    util1 = np.abs(df["tau_1"]) / lim1
    util2 = np.abs(df["tau_2"]) / lim2
    
    active_mask = np.abs(df["tau_2"]) > 0.01
    
    if not active_mask.any():
        print(f"[{label}] No assist activity detected.")
        return

    u1 = util1[active_mask]
    u2 = util2[active_mask]
    
    # Count saturation events (utilization > 99%)
    sat1 = (u1 > 0.99).sum()
    sat2 = (u2 > 0.99).sum()
    total = len(u1)
    
    print(f"--- {label} ---")
    print(f"Avg Utilization: M1={u1.mean():.3f}, M2={u2.mean():.3f}")
    print(f"Max Utilization: M1={u1.max():.3f}, M2={u2.max():.3f}")
    print(f"Saturation Count: M1={sat1}/{total} ({sat1/total*100:.1f}%), M2={sat2}/{total} ({sat2/total*100:.1f}%)")
    print("")

limits = [2.0, 0.2] # Limit drop scenario
print("Analyzing Limit Drop Scenario...\n")
analyze_limit_drop_scenario("csv/dob_sine_20251230_164753.csv", "Raw (Fixed 4:1)", limits)
analyze_limit_drop_scenario("csv/dob_sine_20251230_164726.csv", "Utilization (Proposed)", limits)
