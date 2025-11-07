# ODrive対応 可変減速機構 トルク制御システム
**Ver. 1.0 / Draft**

---

## 1. システム概要
本システムは、2つのブラシレスモータ（Motor1, Motor2）を差動的に組み合わせて出力軸の減速比を可変とする  
「可変減速機構（Variable Gear Ratio Mechanism）」を対象とした制御系である。  
低出力の主モータ（Motor1）を基幹トルク源、補助モータ（Motor2）を高速度域でのアシスト源として用い、  
負担率に基づくAssist-as-Needed（A/N）制御を実現する。

**信号の流れ**
```
θ_out_ref → [位置制御(PID + FF)] → τ_out_cmd
          → [外乱オブザーバ(DOB)] → τ_out_aug
          → [A/N制御 + トルク配分] → [τ1, τ2]
          → ODrive（トルク制御モード）→ 機構出力
```

---

## 2. ハードウェア構成

### 2.1 使用機器例
| 機器 | 数量 | 備考 |
|------|------:|------|
| ODrive S1 | 2 | 2軸BLDCドライバ、USBまたはCAN接続 |
| BLDCモータ（Motor1） | 1 | 主トルク源、高トルク型、エンコーダ付き |
| BLDCモータ（Motor2） | 1 | 補助モータ（スピード用）、エンコーダ付き |
| 絶対値エンコーダ | 2 | 16384 CPR |

---

### 2.2 推奨電源設定
- **バス電圧（Vbus）：** 48V  
- **ODrive設定例：**
  - `axis.motor.config.current_lim = 30 A`
  - `axis.motor.config.requested_current_range = 40 A`
  - `axis.controller.config.vel_limit = <モータ許容速度>`
  - `odrv.config.dc_max_negative_current = 0`（回生無効）  

---

### 2.3 配線指針
- **モータ配線：** U/V/W → ODrive M0/M1 出力端子  
- **エンコーダ配線：** A+/A−, B+/B−, Z+/Z− → ODrive入力  
　ツイストペア＆シールド使用（片側のみ接地）  
- **接地：** 電源負極を基準としたスタ―接地推奨  
- **E-Stop:** 電源リレーを直接遮断する構成  
- **ノイズ対策:** 電力線と信号線を分離し、必要に応じてフェライト追加  

---

### 2.4 初期設定例（odrivetool）
```
odrv.config.dc_bus_undervoltage_trip_level = 20.0
odrv.config.dc_bus_overvoltage_trip_level  = 56.0
odrv.config.dc_max_negative_current        = 0.0

axis.motor.config.pole_pairs               = <pp>
axis.motor.config.torque_constant          = <Kt [N·m/A]>
axis.motor.config.current_lim              = 30.0
axis.motor.config.requested_current_range  = 40.0

axis.encoder.config.cpr                    = 16384
axis.encoder.config.mode                   = ENCODER_MODE_INCREMENTAL
axis.encoder.config.use_index              = True
axis.encoder.config.bandwidth              = 1000.0

axis.controller.config.control_mode        = CONTROL_MODE_TORQUE_CONTROL
axis.controller.config.input_mode          = INPUT_MODE_PASSTHROUGH
axis.controller.config.vel_limit           = <rad_s_limit>

axis.requested_state                       = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
odrv.save_configuration()

※ 実運用では Web GUI で詳細パラメータを設定し、スクリプト側ではトルク制御モード移行とトルク定数の設定のみを行う想定。
```

---

### 2.5 安全確認項目
- ✅ 非常停止スイッチで電源遮断が可能であること  
- ✅ 各軸に過電流保護が入っていること  
- ✅ 回生電流が電源に戻らない構成（必要ならダンプ抵抗）  
- ✅ 冷却と熱監視が十分であること  
- ✅ トルクおよび速度の上限がソフトウェアで設定されていること  

---

## 3. 制御系概要
- 出力角速度： `ω_out = a0·ω0 + a1·ω1`（機構構造に依存する運動学的な関係例）  
- 出力トルク： `τ_out = A0·τ0 + A1·τ1` （config の `mechanism_matrix = [A0, A1]` を使用）  
  - `A_i` は「モータ i のトルクが出力に与える寄与係数」であり、速度係数 `a_i` の逆数とは限らない  
  - エネルギ整合をとる場合は、実測した速度比・効率から `A_i = (η_i·ω_i/ω_out)` などの形で設定する
- 最小ノルムトルク分配（重み付き）：  
  ```
  [τ0, τ1]^T = W·A^T·(A·W·A^T)^(-1) · τ_out_aug
  W = diag(1, w1)
  ```

---

## 4. 制御アルゴリズム

### 4.1 位置制御（外側ループ）
```
τ_out_cmd = Kp·e_θ + Kd·e_ω + Ki·∫e_θ + J_nom·θ̈_ref + b_nom·θ̇_ref
```
- 帯域：3〜6 Hz（200 Hz制御周期を想定）

### 4.2 外乱オブザーバ（DOB）
```
d̂ = Q(s)·[J_nom·ω̇ + b_nom·ω − τ_out_cmd]
τ_out_aug = τ_out_cmd − d̂
```
- カットオフ周波数：10〜20 Hz  
- 離散化： `α = 1 − exp(−2π f_c Ts)`  

### 4.3 Assist-as-Needed制御（A/N）
```
r_τ = |τ_out_aug / A0| / τ0,max
ON if r_τ ≥ 0.65; OFF if r_τ ≤ 0.55
ṡ = (1/τ_s)(s* − s), τ_s ≈ 0.15 s
w1(s) = w_off^(1−s) · w_on^s
```

### 4.4 トルク制限および再投影
```
|τ_i| ≤ τ_i,max,   |dτ_i/dt| ≤ ρ_i,   sign(τ0) = sign(τ1)
```

---

## 5. ファイル構成
| ファイル名 | 役割 |
|-------------|------|
| main_control_odrive.py | メイン制御ループ（200 Hz） |
| odrive_interface.py | ODrive通信制御（USB/CAN） |
| dob_estimator.py | 外乱推定器 |
| position_controller.py | PID + フィードフォワード |
| assist_manager.py | A/N制御ロジック |
| torque_distribution.py | トルク分配計算 |
| identification.py | 慣性・粘性係数同定 |
| logger.py | ログ保存処理 |
| plot_csv.py | 既存CSVからθ/τを再プロット |
| config/*.yaml | パラメータ設定ファイル群 |

---

## 6. 同定手順（J_nom, b_nom）
1. A/N制御を無効化（Motor2を固定）  
2. τ_out にランプ入力（例：0→0.3 N·m）  
3. ω, ω̇ をLPF処理して取得  
4. 最小二乗法で推定：  
   ```
   [J_eq, b_eq]^T = (Φ^TΦ)^(-1)Φ^T τ
   Φ = [ω̇, ω]
   ```
5. 推定精度を±10%以内で確認

---

## 7. パラメータ例
```
A = [-20.0, 12.26993865]
J_nom = 0.015  # kg·m²
b_nom = 0.002  # N·m·s/rad
Ts = 0.005     # s (200 Hz)
DOB fc = 20    # Hz
τ_max = [0.8, 0.4]  # N·m
A/N thresholds: on=0.65, off=0.55, τ_s=0.15 s
```

---

## 8. 確認項目
- 位置ループ安定（帯域6Hz以下）  
- DOB応答安定（カットオフ20Hz以下）  
- A/N制御ON/OFF時にトルク段差なし  
- トルク飽和・符号不整合がないこと  
- エンコーダインデックス整合確認済み  
- 非常停止動作確認済み

---

## 9. 備考
- USB通信のジッタが問題となる場合、CAN通信に移行推奨。  
- 回生を有効にする場合、ダンプ抵抗を実装。  
- 配線は電力系と信号系を分離し、接地は一点アース。

---

## 10. ログ保存と可視化
- `main_control_odrive.py` を実行すると、制御中のデータを `csv/` に時系列CSVとして保存。
- Matplotlib が使用可能な場合は、同じファイル名で `fig/` にPDFプロットを自動生成。
- 保存先やファイル名接頭辞は `config/logger.yaml` で変更可能。
- 追加で `python3 plot_csv.py csv/<file>.csv --show` を実行すると、既存CSVから位置・速度・`τ_1/2` の再プロットや別ファイル保存が可能。
- CSVには `pos_1/vel_1/tau_1` と `pos_2/vel_2/tau_2` が記録され、Motor1/2にそれぞれ対応する。
