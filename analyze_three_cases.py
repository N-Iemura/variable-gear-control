import pandas as pd
import numpy as np

def analyze_limit_drop_scenario(csv_path, label, limits):
    try:
        df = pd.read_csv(csv_path, comment="#")
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        return

    lim1, lim2 = limits
    util1 = np.abs(df["tau_1"]) / lim1
    util2 = np.abs(df["tau_2"]) / lim2
    
    # Consider "active" when total torque is significant, or just analyze all data
    # Here we analyze the whole dataset to see overall behavior
    
    # Count saturation events (utilization > 99%)
    sat1 = (util1 > 0.99).sum()
    sat2 = (util2 > 0.99).sum()
    total = len(df)
    
    # Calculate RMS Error
    error = df["theta_ref"] - df["output_pos"]
    rms_error = np.sqrt((error**2).mean())
    max_error = np.abs(error).max()

    print(f"--- {label} ---")
    print(f"RMS Error: {rms_error:.5f} turn")
    print(f"Max Error: {max_error:.5f} turn")
    print(f"Avg Utilization: M1={util1.mean():.3f}, M2={util2.mean():.3f}")
    print(f"Max Utilization: M1={util1.max():.3f}, M2={util2.max():.3f}")
    print(f"Saturation Count: M1={sat1}/{total} ({sat1/total*100:.1f}%), M2={sat2}/{total} ({sat2/total*100:.1f}%)")
    print("")

limits = [1.0, 0.5] # High load scenario
print("Analyzing High Load Scenario (Limits: M1=1.0, M2=0.5)...\n")

files = {
    "Utilization (Proposed)": "csv/dob_sine_20251230_172543.csv",
    "Raw 1:4 (Fixed)": "csv/dob_sine_20251230_172612.csv",
    "Raw 1:16 (Fixed)": "csv/dob_sine_20251230_172628.csv"
}

for label, path in files.items():
    analyze_limit_drop_scenario(path, label, limits)
