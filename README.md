# ODrive対応 可変減速機構 制御ソフトウェア
**Ver. 1.1**

---

## 1. 概要
2台のBLDCモータ（Motor1/2）で可変減速機構を駆動し、出力軸の位置/速度を制御するためのPC側制御ソフトウェアです。  
ODriveはトルク制御または速度制御で動作させ、PC側で出力軸の外側ループとトルク/速度分配を行います。

---

## 2. 現在の制御構造（`config/controller.yaml` デフォルト）
現在の設定は **トルク制御モード** です。

- `command_type: position`
- `motor_control_mode: torque`
- 外側ループ: 位置PID（FFは無効）
- DOB: 無効
- 摩擦FF: 無効
- Assist-as-Needed: 有効（balanced）

**信号の流れ（トルク制御）**
```
θ_out_ref → [位置PID] → τ_out_cmd
         → [摩擦FF (無効)] → τ_out_cmd
         → [DOB (無効)] → τ_out_aug
         → [Assist + トルク分配] → [τ1, τ2]
         → ODrive（トルク制御）→ 機構出力
```

---

## 3. 実装されている制御機能（有効/無効は設定次第）
### 3.1 外側ループ
- **位置PID / 速度PID**: `position_controller.py`
- FF（慣性・粘性）: `use_feedforward` で有効/無効

### 3.2 外乱オブザーバ（DOB）
- `dob_estimator.py`
- `dob.enabled` で切替
- `torque_input_mode: command/applied` 対応

### 3.3 Assist-as-Needed（A/N）
- `assist_manager.py`
- 荷重比に応じた重みの切替
- `assist_manager.enabled` で切替

### 3.4 トルク分配
- `torque_distribution.py`
- 最小ノルム解 + 制限再投影
- トルクリミット/レートリミット/符号整合

### 3.5 速度分配（オプション）
- `velocity_distribution.py`
- `motor_control_mode: velocity` のとき使用
- `direct_velocity: true` なら **速度指令直結**
- `direct_velocity: false` なら `τ_out→ω_out` 変換を使用

### 3.6 ODrive制御
- `odrive_interface.py`
- トルク/速度制御モード切替
- 速度ゲイン設定（`odrive_velocity_gains`）

---

## 4. 制御構造の切替
### 4.1 トルク制御（現行）
`config/controller.yaml`
```
command_type: position
motor_control_mode: torque
outer_pid:
  kp: 10.0
  ki: 10.0
  kd: 2.0
  use_feedforward: false
```
`config/reference.yaml`
```
command_type: position
step:
  output_amplitude: 0.1
```

### 4.2 速度制御（ODrive内速度PID）
`config/controller.yaml`
```
command_type: velocity
motor_control_mode: velocity
velocity_distribution:
  direct_velocity: true
```
`config/reference.yaml`
```
command_type: velocity
step:
  output_amplitude: 0.1
```

**補足**  
`direct_velocity: false` にすると `τ_out→ω_out` 変換を使う構造になります。  
その場合は `plant.inertia` が必須です。

---

## 5. パラメータの要点
- `motor_control_mode`: ODriveの制御モード（`torque` / `velocity`）
- `command_type`: 外側ループの入力種別（`position` / `velocity`）
- `outer_pid` / `velocity_pid`: 外側ループのゲイン
- `use_feedforward`: FFの有効/無効
- `assist_manager`: A/N制御の有効/無効
- `dob`: 外乱オブザーバの有効/無効
- `torque_limits` / `torque_rate_limits`: トルクの制限
- `velocity_distribution`: 速度分配・直結モード設定

---

## 6. ファイル構成
| ファイル名 | 役割 |
|-------------|------|
| main_control_odrive.py | メイン制御ループ |
| odrive_interface.py | ODrive通信制御 |
| position_controller.py | 位置/速度PID（FF切替可） |
| dob_estimator.py | 外乱オブザーバ |
| assist_manager.py | Assist-as-Needed制御 |
| torque_distribution.py | トルク分配 |
| velocity_distribution.py | 速度分配 |
| logger.py | ログ保存 |
| plot_csv.py | CSVプロット |
| config/*.yaml | 各種設定 |

---

## 7. ログ保存と可視化
- `main_control_odrive.py` 実行で `csv/` にログ保存
- Matplotlib があれば `fig/` にPDF出力
- 既存CSVは `python3 plot_csv.py csv/<file>.csv --show`

---

## 8. 注意事項
- 速度制御は ODrive 内部PIDが主役です。ゲインが強すぎると振動しやすいので注意してください。
- トルク制御では `outer_pid` を 0 にすると動きません。
- `plant.inertia` を 0 にすると `τ_out→ω_out` 変換が停止します。
