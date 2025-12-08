# 出力軸摩擦（クーロン＋粘性）の同定実験手順

目的：出力軸のクーロン摩擦 τ_c^out と粘性摩擦 B_out を、モータ電流と機構モデルから同定する。

この手順は、明日そのまま現場で試せる具体性を目標にしています。A/B/C の3フェーズを含め、ログのフォーマット、しきい値、解析コマンドまで記載します。

---

## 前提・記号
- モータ2台：M1, M2
- 出力軸エンコーダあり → ω_out が取得可能
- モータ電流 i₁, i₂ が読める（トルク定数 Kt₁, Kt₂ は既知）
- 可変減速状態 φ が分かっている（N₁(φ), N₂(φ) が算出可能）
- 設計から符号・ゲイン（N₁, N₂ の符号方向、トルクの正負）が既知

摩擦モデル（まずは素直な2項）
- τ_f_out(ω_out) = τ_c_out · sgn(ω_out) + B_out · ω_out

---

## 0. 共通の設定（全フェーズ共通）

制御モード（推奨）
- 内側：電流（トルク）制御ループを有効
- 外側：ゆっくりした速度指令 または 位置スキャン
- もし電流ループが使えない場合：電流指令値からトルクを推定（Kt を掛ける）

ログ信号（必須列名の例）
- 時間: `time` [s]
- モータ電流: `i1`, `i2` [A]
- モータ速度: `omega1`, `omega2` [rad/s]
- 出力軸速度: `omega_out` [rad/s]
- 可変減速状態: `phi` [-]（N₁(φ), N₂(φ) を計算可能な表現）
- 可能なら補助: 電流指令 `i1_cmd`, `i2_cmd`／速度指令 `omega_out_cmd`

制御条件
- 低速・小加速度で動かす（準定常）
  - 目安：|ω̇_out| を小さく。解析時は |ω̇_out| < しきい値 を抽出。
- 往復で ±方向を必ず通る波形（sgn(ω_out) が両符号になる）

おすすめ指令波形
- 出力軸の位置に低周波の三角波 or サイン波（周期 1〜5 s）
- 速度上限・ストローク安全を最優先（可動域に注意）

---

## フェーズA：M2 を固定して M1 だけ駆動

A-1. 条件
- M2：位置サーボで角度固定（高ゲイン）。可能なら外乱オブザーバも有効。
- M1：出力軸が低速で往復するような位置／速度指令。±方向とも必ず通す。
- 実質自由度はほぼ1。機構学より ω_out ≈ k₁(φ) · ω₁ を想定。

A-2. 測定
- T 秒（目安 60–180 s）動作させ、上記の全信号を CSV ログ。

---

## フェーズB：M1 を固定して M2 だけ駆動（A の逆）

B-1. 条件
- M1：位置サーボで固定
- M2：低速で往復（A と似た軌道）

B-2. 測定
- A と同様に全信号を CSV ログ。

---

## フェーズC：両モータを動かす（任意だが精度向上）

C-1. 条件の例
- M1 と M2 に位相のずれたサイン波の速度指令を与え、ω₁(t), ω₂(t) を“独立っぽく”励起。
- 出力軸の速度・ストロークは必ず安全範囲に制限。

狙い
- A/B は自由度 1。C で ω₁, ω₂ が別方向に励起され、回帰行列の rank が上がり、摩擦パラメータ分離が容易に。

C-2. 測定
- 同様に全信号を CSV ログ。

---

## ログファイルの形式（例）
- 1つのフェーズにつき 1 CSV（例：`csv/phaseA_YYYYMMDD_hhmmss.csv`）
- ヘッダ行あり。推奨列：
  - `time,i1,i2,omega1,omega2,omega_out,phi`
- コメント行先頭は `#`（既存の解析スクリプトでスキップされます）

---

## 解析ステップ（Python）
既存の `identification.py` を使って実施します（出力軸の摩擦モデルに特化）。

1. 出力トルク y_k = τ_out,k の再構成
   - 各時刻 k で
     - τ₁ₖ = Kt₁ · i₁ₖ
     - τ₂ₖ = Kt₂ · i₂ₖ
     - τ_out,k = N₁(φₖ)·τ₁ₖ + N₂(φₖ)·τ₂ₖ
   - メモ：`identification.py` は既定では `tau_out` 列を読む想定ですが、計算列を追加して保存するか、前処理で作成してください。

2. 準定常区間だけ抽出
   - |ω̇_out| が小さいサンプルのみ使用（例：|ω̇_out| < しきい値）
   - 準定常なら J_out·ω̇_out ≈ 0 ⇒ τ_out ≈ τ_f_out とみなせる

3. 回帰行列 Φ の構成
   - 各サンプル k で
     - φ₁ₖ = sgn(ω_out,k)
     - φ₂ₖ = ω_out,k
   - Φ = [φ₁ₖ, φ₂ₖ] の N×2 行列、y = τ_out の N×1 ベクトル
   - A/B/C 全フェーズのデータをまとめて使う

4. 最小二乗推定
   - θ = [τ_c_out, B_out]
   - θ^ = (ΦᵀΦ)⁻¹ Φᵀ y を計算

5. 品質チェック
   - 予測 τ^_f,k = τ^_c_out·sgn(ω_out,k) + B^_out·ω_out,k
   - 実測 τ_out,k と重ね描き。誤差のランダム性、±方向のバイアス有無を確認。
   - 問題があれば：
     - 低速区間にもっと絞る（|ω̇_out| 閾値を小さく）
     - モデルを Stribeck（|ω| の非線形項）へ拡張
     - A/B/C の励起量の再検討（rank 不足対策）

---

## すぐ試せるミニスクリプト（前処理→同定）
以下は PowerShell での実行例です。必要に応じて列名を合わせてください。

```powershell
# 1) 複数 CSV を一つに統合（A/B/C を結合）
python - <<'PY'
import pandas as pd
from pathlib import Path
import yaml
import numpy as np

# 入力 CSV のリスト（必要に応じて編集）
inputs = [
    'csv/phaseA_*.csv',
    'csv/phaseB_*.csv',
    'csv/phaseC_*.csv',
]
files = []
for pat in inputs:
    files += [str(p) for p in Path('.').glob(pat)]
if not files:
    raise SystemExit('CSV が見つかりません。patterns を調整してください。')

# 読み込み＆結合
frames = [pd.read_csv(f, comment='#') for f in files]
df = pd.concat(frames, ignore_index=True)

# パラメータ（編集）
Kt1 = 0.1  # [Nm/A] 例
Kt2 = 0.1  # [Nm/A] 例

def N1(phi):
    # 例：線形近似やテーブル参照に差し替えてください
    return 1.0

def N2(phi):
    return 1.0

# 出力トルク再構成
if not {'time','i1','i2','omega_out','phi'} <= set(df.columns):
    raise SystemExit('必要列が不足：time,i1,i2,omega_out,phi')

df['tau1'] = Kt1 * df['i1']
df['tau2'] = Kt2 * df['i2']
df['tau_out'] = df.apply(lambda r: N1(r['phi'])*r['tau1'] + N2(r['phi'])*r['tau2'], axis=1)

# 準定常抽出：|ω̇_out| < thresh
# 時間が等間隔でない場合は近似で平均 dt を使用
dt = np.diff(df['time'].to_numpy())
if (dt <= 0).any():
    raise SystemExit('time が単調増加でありません')
dt_mean = float(np.mean(dt))
omega = df['omega_out'].to_numpy()
omega_dot = np.gradient(omega, dt_mean)
thresh = 0.2  # [rad/s^2] 例・要調整（低速運転ならもっと小さく）
mask = np.abs(omega_dot) < thresh
sel = df.loc[mask].copy()
if len(sel) < 100:
    print(f'注意：抽出サンプルが少ない ({len(sel)})。閾値や波形を再検討。')

# 同定用 CSV を書き出し（identification.py が読む列名に合わせる）
out = sel[['time','tau_out','omega_out']].copy()
out.to_csv('csv/ident_for_friction.csv', index=False)
print('csv/ident_for_friction.csv を生成しました')
PY

# 2) 既存スクリプトで最小二乗（粘性＋クーロン）
python .\identification.py csv\ident_for_friction.csv --velocity-column omega_out --torque-column tau_out --filter-alpha 0.0 --output-file csv\ident_friction_result.yaml
```

- `--filter-alpha 0.0` は極力そのままの ω_out を使う設定（ノイズが大きければ 0.05–0.2 を試してください）
- 出力 `csv/ident_friction_result.yaml` には推定結果（`inertia` は本スクリプトの一般機能上の項目です。摩擦同定では `damping` が B_out に対応、クーロン成分は前処理の sgn 項を Φ に入れた場合に別途計算します）。

注：本リポジトリの `identification.py` は一般の慣性＋粘性推定向けで、クーロン項は内蔵していません。上記の前処理で Φ=[sgn(ω_out), ω_out] を組み、`np.linalg.lstsq` を直接使う形に拡張したい場合は、別途 `scripts/identify_friction.py` を用意するのが安全です（次の「拡張案」参照）。

---

## 拡張案：クーロン＋粘性に特化した簡易スクリプト
将来、`scripts/identify_friction.py` を追加して次の処理を自動化できます：
- CSV 読み込み（A/B/C をまとめて）
- τ_out の再構成（Kt, N₁(φ), N₂(φ) を関数／テーブルで注入）
- 準定常抽出（|ω̇_out| 閾値）
- Φ=[sgn(ω_out), ω_out] と y=τ_out を作成
- θ=[τ_c_out, B_out] を最小二乗で推定
- 品質レポート・プロット（誤差分布、±方向バイアス検出）

---

## すぐ使えるチェックリスト（印刷推奨）
- [ ] 出力軸速度が安全範囲（max rad/s）に制限されている
- [ ] 位置／速度波形の周期（1–5 s）設定済み
- [ ] フェーズA：M2 固定／M1 往復のコマンド準備
- [ ] フェーズB：M1 固定／M2 往復のコマンド準備
- [ ] フェーズC：M1/M2 位相ずれサインの速度指令準備（任意）
- [ ] ログ列：time,i1,i2,omega1,omega2,omega_out,phi を記録
- [ ] 実験 T（60–180 s/フェーズ）を確保
- [ ] CSV が保存されている（ヘッダあり、コメントは #）
- [ ] 解析の Kt₁,Kt₂ と N₁(φ),N₂(φ) の式（またはテーブル）が準備済み

---

## トラブルシュート
- rank が低い（推定が不安定）
  - C フェーズで ω₁, ω₂ を独立に励起する
  - ±方向のデータバランスを確保（sgn が両符号）
- |ω̇_out| の閾値が高すぎる／低すぎる
  - サンプル数と残差を見ながら調整（0.05–0.5 rad/s^2 など運転に応じて）
- ノイズが大きい
  - 低い周波数の波形、フィルタの α を小さく（0.05–0.2）
  - データ量を増やす（各フェーズ T を長めに）

---

## 出力物（例）
- `csv/ident_for_friction.csv`：準定常抽出後の同定用データ
- `csv/ident_friction_result.yaml`：推定結果（B_out, 参考値など）
- プロット（任意）：τ_out と τ^_f の比較、誤差ヒストグラム

---

この文書の意図は「トルクセンサなしでも、モータ電流＋機構モデルから τ_out を理論的に再構成し、A/B/C のモードで rank を稼いでパラメータ可同定性を確保する」ことです。低速域の摩擦補償（クーロン＋粘性）を、実験と回帰の基本式に基づいて設計できます。