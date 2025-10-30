# ODrive対応 可変減速機構 トルク制御システム 仕様書
**Ver. 1.0 / Draft**

---

## 1. システム概要
本システムは、2つのブラシレスモータ（Motor0, Motor1）を差動的に組み合わせて出力軸の減速比を可変とする  
「可変減速機構（Variable Gear Ratio Mechanism）」を対象とした制御系である。  
低出力の主モータ（Motor0）をベーストルク源、補助モータ（Motor1）を高速度域でのアシスト源として利用し、  
負担率に基づくAssist-as-Needed（A/N）制御を実現する。

**制御信号の経路：**
```
θ_out_ref → [位置制御(PID)] → τ_out_cmd
          → [外乱オブザーバ(DOB)] → τ_out_aug
          → [A/N制御] + [トルク配分] → [τ0, τ1]
          → ODrive (トルク制御モード)
```

---

## 2. 制御階層構成
- 第1層: 電流制御 (ODrive内部)  
- 第2層: トルク制御・外乱補償 (PC側)  
- 第3層: 位置制御＋Assist-as-Needed (PC側)

**角速度関係**
```
ω_out = a0·ω0 + a1·ω1
```

**トルク関係**
```
τ = A^T τ_out + τ_null
τ = A^T (A A^T)^(-1) τ_out    (最小ノルム解)
```

---

## 3. 制御アルゴリズム概要

### 3.1 位置制御器（最外層）
PID + フィードフォワード：
```
τ_out_cmd = Kp·e_θ + Kd·e_ω + Ki∫e_θ + J_eq·θ̈_ref + b_eq·θ̇_ref
```
- 出力: 希望出力トルク τ_out_cmd  
- 帯域: 3–6 Hz  

---

### 3.2 外乱オブザーバ（DOB）
- 推定式：  
  `d̂ = Q(s)[J_nom ω̇ + b_nom ω - τ_out]`
- 補償後トルク：  
  `τ_out_aug = τ_out_cmd - d̂`
- カットオフ周波数: 10–20 Hz  
- 慣性パラメータ J_nom, b_nom は実験同定により決定。

---

### 3.3 トルク配分ロジック
モータトルク分配：  
```
[τ0, τ1]^T = A^T (A A^T)^(-1) W^(-1) τ_out_aug
```
**制約条件**
- 同方向動作（sign(τ0) = sign(τ1)）  
- 飽和: |τ_i| ≤ τ_max,i  

---

### 3.4 Assist-as-Needed制御（A/N）
主モータの負担率：  
```
r_τ = |τ_out_aug / a0| / τ_0,max
```
アシストゲイン更新：  
```
ṡ = (1/τ_s)(s* - s)
```
**条件**
- on閾値: r_τ ≥ 0.65  
- off閾値: r_τ ≤ 0.55  
- 時定数 τ_s = 0.15 s  

**トルク配分重み**
```
W = diag(1, w1(s))
w1(s) = w_off^(1-s) w_on^s
```

---

### 3.5 トルク制御
- ODrive 内部のトルク制御（約8kHz）を利用。  
- Python側から `axis.controller.input_torque` で指令。  
- 単位: N·m（内部でKt換算済み）

---

## 4. 同定プロセス（J_eq, b_eq）
1. A/Nを無効化（s固定）  
2. τ_outにランプ入力（例: 0→0.3 N·m）  
3. ω, ω̇ をローパス後サンプリング  
4. 最小二乗法で同定：  
   ```
   [J_eq, b_eq]^T = (Φ^TΦ)^(-1)Φ^T τ
   Φ = [ω̇, ω]
   ```

---

## 5. 実行構成
| ファイル名 | 役割 |
|-------------|------|
| main_control_odrive.py | 制御ループ本体 |
| odrive_interface.py | ODrive通信 |
| dob_estimator.py | 外乱推定 |
| position_controller.py | PID + FF |
| assist_manager.py | A/N制御 |
| torque_distribution.py | トルク配分 |
| identification.py | 同定実験 |
| logger.py | ログ保存 |
| config/*.yaml | パラメータ管理 |

---

## 6. パラメータ例
```
a0 = -0.05, a1 = 0.0815
J_nom = 0.015 [kg·m²]
b_nom = 0.002 [N·m·s/rad]
fc = 20 [Hz]
τ_max = [0.8, 0.4] [N·m]
Ts = 0.005 [s]
th_on/off = 0.65 / 0.55
τ_s = 0.15 [s]
```

---

## 7. 想定動作
- 通常時: Motor0単独で駆動（高減速・高トルク）  
- 高速要求時: Motor1が同方向に回転し減速比を低下 → 速度増加  
- 逆方向回転は禁止（トルク増強には非効率）  
- 外乱補償はDOBが吸収。

---

## 8. 評価項目
| 項目 | 判定基準 |
|------|------------|
| 応答帯域 | 3–6 Hz |
| 定常誤差 | < 0.5° |
| 外乱抑制 | 50%以上減衰 |
| トルク飽和 | 無 |
| A/N発動 | スムーズなON/OFF |

---

## 9. 運用方法
1. ODriveキャリブレーション  
2. config設定  
3. identification.py 実行  
4. main_control_odrive.py 実行  
5. ログ解析とゲイン調整  
