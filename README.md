# ODrive-based Variable Gear Assist Control System — Specification
**Version:** 1.0 (Draft)

---

## 1. System Overview
This repository implements a dual-motor differential mechanism with a variable gear ratio using **ODrive** motor drivers. The control stack provides **position tracking**, **torque control**, **Disturbance Observer (DOB)** compensation, and **Assist-as-Needed (A/N)** logic that engages the second motor only when the primary motor load ratio exceeds a threshold.

**Signal flow**
```
θ_out_ref → [Position PID + FF] → τ_out_cmd
          → [DOB]               → τ_out_aug
          → [A/N + Torque Split]→ [τ0, τ1]
          → ODrive (Torque Mode)→ Plant
```

---

## 2. Hardware Configuration

### 2.1 Bill of Materials (example)
| Item | Qty | Notes |
|---|---:|---|
| ODrive S1 | 2 | Dual-axis BLDC driver, USB/CAN |
| BLDC Motor (Motor0) | 1 | Main torque source (higher torque), with encoder |
| BLDC Motor (Motor1) | 1 | Assist motor (speed/assist), with encoder |
| Incremental Encoder(s) | 2 | ≥ 8192 CPR (or equivalent), differential recommended |
| Power Supply | 1 | e.g., 24–48 V DC, sized per peak current |
| E-stop / Power Switch | 1 | Latching emergency stop with main relay |
| Main Fuse | 1 | Sized for PSU; fast-acting recommended |
| Per-axis Fuse / PTC | 2 | Optional, protects each motor branch |
| Wiring / Connectors | — | Shielded motor phases, twisted pairs for encoders |
| Chassis / Ground | — | Single-point star ground to PSU return |

> Replace parts with your actual hardware; values below are safe defaults to start commissioning.

### 2.2 Power & Limits (typical starting points)
- **Bus voltage (Vbus):** 24–48 V (set by your PSU).  
- **ODrive current limits (per axis):**  
  - `axis.motor.config.current_lim`: 20–40 A (match motor/driver capability)  
  - `axis.controller.config.vel_limit`: motor-safe speed limit  
  - `odrv.config.dc_max_negative_current`: 0 (disable regen into PSU unless you have a dump resistor)  
- **Thermal & wiring:** Use appropriate gauge, keep motor phase leads short; ensure airflow for ODrive.

### 2.3 Encoder & Motor Wiring (textual)
- **Motors:** U/V/W → ODrive M0/M1 phases. Keep cable lengths short and routed away from encoders.  
- **Encoders:** A+/A−, B+/B−, Z+/Z− → ODrive encoder inputs; use **twisted, shielded pairs**. Shield bonded at **one end** only (controller side).  
- **Grounding:** Single-point ground at PSU negative; avoid ground loops.  
- **E-stop:** Hard-cuts PSU via relay; optional soft-stop signals to host PC/ODrive for graceful decel.  
- **EMI:** Separate power and signal harnesses; add ferrites if necessary.

### 2.4 Initial ODrive Setup (illustrative values)
> Use `odrivetool` or API. Adjust to your motors’ datasheets.

```
# Bus/regeneration
odrv.config.dc_bus_undervoltage_trip_level = 20.0
odrv.config.dc_bus_overvoltage_trip_level  = 56.0
odrv.config.dc_max_negative_current        = 0.0   # disable regen into PSU

# Axis 0 (repeat for Axis 1 with appropriate values)
axis.motor.config.pole_pairs            = <pp>
axis.motor.config.torque_constant       = <Kt_Nm_per_A>
axis.motor.config.current_lim           = 30.0
axis.motor.config.requested_current_range = 40.0

axis.encoder.config.cpr                 = 8192
axis.encoder.config.mode                = ENCODER_MODE_INCREMENTAL
axis.encoder.config.use_index           = True
axis.encoder.config.bandwidth           = 1000.0

axis.controller.config.control_mode     = CONTROL_MODE_TORQUE_CONTROL
axis.controller.config.input_mode       = INPUT_MODE_PASSTHROUGH
axis.controller.config.vel_limit        = <rad_s_limit>

# Calibration and saving
axis.requested_state                    = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
odrv.save_configuration()
```

### 2.5 Safety Checklist
- ✅ **E-stop** cuts power; verify latch/unlatch behavior.  
- ✅ **Fusing** present (main + per-axis recommended).  
- ✅ **No regen** to PSU (or use dump resistor).  
- ✅ **Thermal monitoring** enabled; driver and motors have airflow.  
- ✅ **Torque & rate limits** enforced in software (see §5.3).  

---

## 3. Kinematics & Dynamics (recap)
- Output speed: `ω_out = a0·ω0 + a1·ω1`  
- Output torque: `τ_out = a0·τ0 + a1·τ1`  
- Weighted minimum-norm split (assist weight `w1`):  
  ```
  [τ0, τ1]^T = W·A^T·(A·W·A^T)^{-1} · τ_out_aug,
  W = diag(1, w1)
  ```

---

## 4. Control Overview

### 4.1 Position Controller (outer loop)
PID + feedforward:
```
τ_out_cmd = Kp·e_θ + Kd·e_ω + Ki·∫e_θ + J_nom·θ̈_ref + b_nom·θ̇_ref
```
- Target bandwidth: **3–6 Hz** (host @ 200 Hz loop).

### 4.2 Disturbance Observer (DOB)
```
d̂ = Q(s)·[J_nom·ω̇ + b_nom·ω − τ_out_cmd]
τ_out_aug = τ_out_cmd − d̂
```
- Low-pass cutoff **10–20 Hz**; discrete α = `1 − exp(−2π f_c Ts)`.

### 4.3 Assist-as-Needed (A/N)
Load ratio on Motor0:
```
r_τ = |τ_out_aug / a0| / τ0,max
ON if r_τ ≥ 0.65; OFF if r_τ ≤ 0.55
ṡ = (1/τ_s)(s* − s), τ_s ≈ 0.15 s
w1(s) = w_off^(1−s) · w_on^s
```

### 4.4 Torque Limits and Projection
Before sending `[τ0, τ1]`, project onto safety set `𝔅`:
```
|τ_i| ≤ τ_i,max,   |dτ_i/dt| ≤ ρ_i,   sign consistency (same direction assist)
```

---

## 5. Execution Layout (files)

| File | Role |
|---|---|
| `main_control_odrive.py` | Main control loop (host @ 200 Hz) |
| `odrive_interface.py` | ODrive comms (USB/CAN), torque set/readback |
| `dob_estimator.py` | Disturbance Observer |
| `position_controller.py` | PID + feedforward |
| `assist_manager.py` | A/N logic (thresholds, smoothing) |
| `torque_distribution.py` | Weighted min-norm torque split |
| `identification.py` | J_nom, b_nom identification |
| `logger.py` | CSV logging & health |
| `config/*.yaml` | Parameters (A, limits, gains, DOB, A/N) |

---

## 6. Identification (J_nom, b_nom)
1) Disable A/N, hold Motor1 off.  
2) Apply a ramp torque (e.g., 0 → 0.3 N·m in ~80 ms; hold ~200 ms).  
3) Sample `ω`, estimate `ω̇` (LPF + Savitzky–Golay).  
4) Solve least-squares for `J, b` with `τ ≈ J·ω̇ + b·ω`.  
5) Validate by comparing predicted `τ_pred` vs actual τ (±10% target).

---

## 7. Parameters (example)
```
A = [-0.05, 0.0815]
J_nom = 0.015  # kg·m²
b_nom = 0.002  # N·m·s/rad
Ts = 0.005     # s (200 Hz)
DOB fc = 20    # Hz
τ_max = [0.8, 0.4]  # N·m
A/N thresholds: on=0.65, off=0.55, τ_s=0.15 s
```

---

## 8. Commissioning Checklist
- Position loop stable (≤ 6 Hz), DOB enabled (≤ 20 Hz).  
- Smooth A/N engagement/disengagement, no torque steps.  
- `[τ0, τ1]` within limits and project properly.  
- No encoder faults; index alignment verified.  
- E-stop and power path verified under load.

---

## 9. Notes
- For USB jitter, consider moving to **CAN** for lower latency.  
- If regen is required, add a dump resistor and configure ODrive accordingly.  
- Keep encoder and power wiring separated; use star ground.

