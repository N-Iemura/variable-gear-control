"""
トルク制御PID環境 - 最小ノルム配分 + モータ優先バイアス版（コピペ即実行）

目的:
- 出力角θのみを外側PIDで追従
- モータ重み付き最小ノルムで出力トルク τ_out* を各モータへ配分
- motor0 を優先しつつ、負荷が高まった際に motor1 を自動解放して可変減速比を実現
- 既存のODriveトルクモードにそのまま繋ぐ

使い方:
1) 本ファイルを保存して `python norm_v2_preferred.py` を実行。
2) 終了時に CSV とグラフの保存/破棄を選べます。

注意:
- ODriveのシリアル番号やトルク定数は環境に合わせて変更してください。
- 出力角θは独立エンコーダ (odrv2) で計測する想定です。
"""

# ===================== 標準ライブラリ =====================
import csv
import json
import math
import os
import threading
import time
from datetime import datetime

# ===================== サードパーティ =====================
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ===================== ODrive =====================
import odrive
from odrive.enums import *

# ===================== 日本語フォント設定 =====================
plt.rcParams['font.family'] = 'DejaVu Sans'
japanese_fonts = ['Noto Sans CJK JP', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic']
for font in japanese_fonts:
    if font in [f.name for f in fm.fontManager.ttflist]:
        plt.rcParams['font.family'] = font
        break
else:
    print("警告: 日本語フォントが見つかりません。グラフは英語表記になります。")

# ==================================================================================
# 設定
# ==================================================================================
REFERENCE_PROFILE = {
    'active_profile': 'ramp',  # 'step', 'sine', 'chirp'
    'file_label': None,        # CSV/グラフの識別子に使う任意文字列
    'step': {
        'initial_wait': 1.0,        # 最初のステップまでの待機[秒]
        'step_duration': 20.0,      # 各ステップの持続[秒]
        'output_amplitude': 0.2,    # 出力θ振幅[turn]
        'offset': 0.0,
    },
    'sine': {
        'initial_wait': 1.0,        # 開始前の待機[秒]
        'output_amplitude': 0.1,   # 正弦波振幅[turn]
        'frequency_hz': 0.1,       # 周波数[Hz]
        'offset': 0.0,              # バイアス
    },
    'chirp': {
        'initial_wait': 1.0,
        'output_amplitude': 0.2,
        'start_frequency_hz': 0.02,
        'end_frequency_hz': 0.2,
        'duration': 60.0,           # 開始からこの秒数で終端周波数へ
        'offset': 0.0,
    },
    'ramp': {
        'initial_wait': 1.0,
        'start_value': 0.0,
        'end_value': 10.0,
        'ramp_duration': 10.0,      # [s] start -> end の時間
        'hold_duration': 5.0,       # [s] end_value で保持
        'return_duration': 0.0,     # [s] >0 のとき start_value へ線形で戻す
        'repeat': True,             # True なら周期的に繰り返す
    },
    'ramp_b': {
        'initial_wait': 1.0,
        'start_value': 0.0,
        'end_value': 1.0,
        'ramp_duration': 1.0,
        'hold_duration': 0.0,
        'return_duration': 1.0,
        'start_hold_duration': 0.0,  # start_value で保持する時間
        'repeat': True,
    },
    'ramp_step': {
        'initial_wait': 1.0,
        'start_value': 0.0,
        'end_value': 1.5,
        'ramp_duration': 3.0,
        'hold_duration': 5.0,
        'return_duration': 0.0,
        'repeat': True,
        'step': {
            'amplitude': 0.2,       # 追加ステップ振幅[turn]
            'duration': 10.0,        # ステップ保持時間[秒]
            'start_after': -0.01,     # 初期待機後、この秒数経過で最初のステップ
            'period': 3.0,          # 周期的に繰り返す場合の周期[秒]
            'repeat': False,        # Falseで一度だけステップ
            'offset_in_cycle': 0.0, # 周期内でステップを入れる位置[秒]
            'align_to_ramp_end': True,  # Trueならランプ完了直後に挿入
        },
    },
    'nstep': {
        'initial_wait': 1.0,
        'values': [0.01, 0.02, 0.01, 0],  # 順番に保持したいθ[turn]
        'hold_duration': 3.0,             # 各値の保持時間[秒]
        'repeat': True,                   # Trueで最後まで行ったら先頭に戻る
        'wait_value': 0.0,                # initial_wait中に保持する値
    },
}

# 後方互換性用のエイリアス
STEP_CONFIG = REFERENCE_PROFILE['step']

# 制御モード: 'output_pid' は既存の単一PID, 'per_motor_pid' はモータ毎PID
CONTROL_MODE = 'per_motor_pid'

# 出力θのPIDゲイン（外側）※CONTROL_MODE='output_pid' のとき使用
OUTPUT_PID = {'kp': 1.00, 'ki': 0.8, 'kd': 0.001, 'max_output': 200.0}

# per_motor_pid モードでも外側PIDを併用する場合は True
ENABLE_OUTER_PID_IN_PER_MOTOR = True

# PID微分項のモード ('measurement' または 'error') とフィルタ係数
PID_DERIVATIVE_MODE = 'error'
PID_DERIVATIVE_FILTER_ALPHA = 0.2

# 各モータ用PIDゲイン（CONTROL_MODE='per_motor_pid' のとき使用）
MOTOR_PID = {
    'motor0': {'kp': 2.81, 'ki': 0.02, 'kd': 0.05, 'max_output': 5.0},   # T-motor
    'motor1': {'kp': 0.05, 'ki': 0.03, 'kd': 0.01, 'max_output': 0.2}     # Maxon
}

# 関節基準姿勢（必要に応じて調整）
JOINT_REFERENCE = np.array([0.0, 0.0], dtype=float)

# 可変減速制御設定（motor0 の負荷比で motor1 の寄与を補間）
VARIABLE_RATIO_CONFIG = {
    'enabled': True,
    'primary_motor': 'motor0',        # 通常トルクを担う側
    'release_start_ratio': 0.5,      # この比率を超えると開放を開始
    'release_full_ratio': 0.9,        # この比率で完全開放
    'secondary_gain_hold': 0.0,       # ホールド時の secondary_gain
    'secondary_gain_release': 1.0,    # 開放時の secondary_gain
}

# 片側モータを固定する場合の設定（'motor0' / 'motor1' / None）
FREEZE_CONFIG = {
    'motor_to_freeze': 'None',
    'kp': 0.1,
    'kd': 0.002,
}

# モータ使用バイアス設定（preferred_motor を優先し、もう一方を抑える）
TORQUE_PREFERENCE = {
    'preferred_motor': 'motor0',  # 'motor0', 'motor1', または None
    'secondary_gain': 0.0,        # 0.0 で完全抑制, 1.0 でバイアス無し
}

# モータごとの重み（重みが大きいほどそのモータを使いづらくする）
# hold/release を切り替えて可変減速の状態に応じた重みを指定可能
TORQUE_WEIGHTING = {
    'hold': {
        'motor0': 1.0,
        'motor1': 100000.0,
    },
    'release': {
        'motor0': 1.0/6.0,
        'motor1': 1.0,
    },
}

# 安全制限
SAFETY_CONFIG = {
    'max_torque0': 6.0,      # T-motor 最大トルク[Nm]
    'max_torque1': 1.00,      # Maxon 最大トルク[Nm]
}

# ODrive接続設定（必要に応じて変更）
ODRIVE_SERIAL = {
    'motor0': '3856345D3539',  # T-motor 側
    'motor1': '384D346F3539',  # Maxon 側
    'output': '3849346F3539',  # 出力エンコーダ
}
ODRIVE_TORQUE_CONSTANT = {
    'motor0': 0.106,  # Nm/A
    'motor1': 0.091,  # Nm/A
}

# 制御周期 [Hz]
CONTROL_FREQUENCY = 200

# 出力ファイル関連
CSV_DIR = 'csv'
FIG_DIR = 'fig'
DATA_FILENAME_PREFIX = 'norm2'
PLOT_FILENAME_SUFFIX = '_plot.pdf'
# グラフファイル名に含めるコマンド値を固定したい場合に設定 (例: [0.0, 2.0])
# None の場合は記録されたコマンドから自動抽出
FILENAME_COMMAND_VALUES = None
# コマンド値の小数点以下桁数 (自動抽出/固定いずれも適用)
FILENAME_DECIMALS = 4

# ==================================================================================
# ユーティリティ: 機構行列 / 最小ノルム配分 / ヌル空間射影
# ==================================================================================

def _get_weight_vector(mode):
    cfg = TORQUE_WEIGHTING.get(mode, TORQUE_WEIGHTING.get('release', {}))
    w = np.array([
        float(cfg.get('motor0', 1.0)),
        float(cfg.get('motor1', 1.0)),
    ], dtype=float)
    if np.any(w <= 0.0):
        raise ValueError("TORQUE_WEIGHTING の値は正の実数である必要があります。")
    return w


def get_A(q=None):
    """機構の出力写像 A = [a1 a2] (1x2)
    姿勢依存の場合は q から計算する。まずは定数でOK。
    例: a1 = -1/20, a2 = 163/2000
    """
    return np.array([[-1/20, 163/2000]])


def min_norm_torque_split(A, tau_out, weights=None):
    """重み付き最小ノルムのトルク配分: τ* = W^{-1} A^T (A W^{-1} A^T)^(-1) τ_out"""
    if weights is None:
        weights = _get_weight_vector('release')
    weights = np.asarray(weights, dtype=float).reshape(2)
    if np.any(weights <= 0.0):
        raise ValueError("weights must be positive.")
    At = A.T  # shape (2,1)
    W_inv = np.diag(1.0 / weights)
    W_inv_At = W_inv @ At
    s = float(A @ W_inv_At)  # = A W^-1 A^T
    if s < 1e-8:
        raise ValueError("Mechanism matrix A is near-singular.")
    scale = float(tau_out) / s
    return (W_inv_At * scale).reshape(2)


MOTOR_OUTPUT_GAINS = np.array([20.0, 2000.0 / 163.0], dtype=float)


def motor_torque_to_output(tau_vec):
    """モータトルクから出力トルクへ変換: τ_out = τ0*20 + τ1*(2000/163)"""
    tau_vec = np.asarray(tau_vec, dtype=float).reshape(2)
    return float(np.dot(MOTOR_OUTPUT_GAINS, tau_vec))


def _solve_torque_with_limits(
    A, tau_desired, torque_limits, tau_preferred=None, weights=None, tol=1e-9
):
    """
    A:              1x2 行列
    tau_desired:    望ましい出力トルク (スカラー)
    torque_limits:  [limit0, limit1]
    tau_preferred:  ヌル空間成分など、可能なら近づけたい候補
    戻り値: (tau_solution[2], 実現された出力トルク)
    """
    A = np.asarray(A, dtype=float).reshape(1, 2)
    if weights is None:
        weights = _get_weight_vector('release')
    weights = np.asarray(weights, dtype=float).reshape(2)
    if np.any(weights <= 0.0):
        raise ValueError("weights must be positive.")
    limits = np.asarray(torque_limits, dtype=float)
    a1, a2 = A[0]

    # フィージビリティチェック: 角の値から達成可能な出力トルク範囲を把握
    corners = np.array([
        [ limits[0],  limits[1]],
        [ limits[0], -limits[1]],
        [-limits[0],  limits[1]],
        [-limits[0], -limits[1]],
    ], dtype=float)
    tau_corner_vals = corners @ A.T  # shape (4,1)
    tau_min = float(np.min(tau_corner_vals))
    tau_max = float(np.max(tau_corner_vals))
    tau_target = float(np.clip(tau_desired, tau_min, tau_max))

    # 最小ノルム解（等式を満たす）
    tau_base = min_norm_torque_split(A, tau_target, weights=weights)
    if tau_preferred is None:
        tau_preferred = tau_base.copy()
    tau_preferred = np.asarray(tau_preferred, dtype=float).reshape(2)

    # 既に制限内なら終了
    if np.all(np.abs(tau_base) <= limits + 1e-9):
        return tau_base, tau_target

    # 等式を維持したままボックスへ射影（ヌル空間方向を利用）
    n = np.array([a2, -a1], dtype=float)  # ヌル空間基底（A の幾何学的ヌル空間）
    n_norm_sq = float(np.dot(n, n))

    def project_with_preference(pref):
        if n_norm_sq < tol:
            return None
        alpha_opt = float(np.dot(n, pref - tau_base) / n_norm_sq)
        alpha_low, alpha_high = -np.inf, np.inf
        for i in range(2):
            n_i = n[i]
            if abs(n_i) < tol:
                # この軸では調整できない -> ベースが制限を超えるなら不可
                if abs(tau_base[i]) <= limits[i] + 1e-9:
                    continue
                return None
            low = (-limits[i] - tau_base[i]) / n_i
            high = (limits[i] - tau_base[i]) / n_i
            if low > high:
                low, high = high, low
            alpha_low = max(alpha_low, low)
            alpha_high = min(alpha_high, high)
            if alpha_low > alpha_high:
                return None
        alpha = float(min(max(alpha_opt, alpha_low), alpha_high))
        tau_candidate = tau_base + alpha * n
        if np.all(np.abs(tau_candidate) <= limits + 1e-8):
            return tau_candidate
        return None

    candidate = project_with_preference(tau_preferred)
    if candidate is not None:
        return candidate, tau_target

    candidate = project_with_preference(tau_base)
    if candidate is not None:
        return candidate, tau_target

    # それでも見つからない場合は、片方のモータを限界に固定して求める
    best = None
    eps = tol

    def evaluate_candidate(tau_vec):
        nonlocal best
        if tau_vec is None:
            return
        if not np.all(np.abs(tau_vec) <= limits + 1e-6):
            return
        tau_out = float(A @ tau_vec.reshape(2, 1))
        err_out = abs(tau_out - tau_target)
        pref_err = np.linalg.norm(tau_vec - tau_preferred)
        score = (err_out, pref_err)
        if best is None or score < best[0]:
            best = (score, tau_vec, tau_out)

    # motor0 を限界に貼り付け
    if abs(a2) > eps:
        for s0 in (-1, 1):
            t0 = s0 * limits[0]
            t1 = (tau_target - a1 * t0) / a2
            if abs(t1) <= limits[1] + 1e-6:
                evaluate_candidate(np.array([t0, t1], dtype=float))

    # motor1 を限界に貼り付け
    if abs(a1) > eps:
        for s1 in (-1, 1):
            t1 = s1 * limits[1]
            t0 = (tau_target - a2 * t1) / a1
            if abs(t0) <= limits[0] + 1e-6:
                evaluate_candidate(np.array([t0, t1], dtype=float))

    # それでも不可なら角の中で最も出力が近いものを採用
    if best is None:
        for vec in corners:
            evaluate_candidate(vec)

    if best is None:
        # 理論上ここには来ないはずだが、最悪は単純クリップ
        tau_fallback = np.clip(tau_base, -limits, limits)
        tau_out = float(A @ tau_fallback.reshape(2, 1))
        return tau_fallback, tau_out

    _, tau_vec, tau_out = best
    return tau_vec, tau_out


def _apply_torque_preference(tau_candidate):
    """preferred_motor を優先しつつ、もう一方のトルクを抑制した理想ベクトルを生成"""
    motor = TORQUE_PREFERENCE.get('preferred_motor')
    gain = float(TORQUE_PREFERENCE.get('secondary_gain', 1.0))
    gain = float(np.clip(gain, 0.0, 1.0))
    tau_pref = np.asarray(tau_candidate, dtype=float).reshape(2)
    if motor == 'motor0':
        tau_pref = np.array([tau_pref[0], tau_pref[1] * gain], dtype=float)
    elif motor == 'motor1':
        tau_pref = np.array([tau_pref[0] * gain, tau_pref[1]], dtype=float)
    else:
        tau_pref = tau_pref.copy()
    return tau_pref


def _compute_release_alpha(primary_tau, limit, start_ratio, full_ratio):
    """一次モータの負荷比に応じて 0.0 (完全ホールド) ～ 1.0 (完全開放) を返す"""
    limit = max(float(limit), 1e-6)
    load_ratio = abs(float(primary_tau)) / limit
    if full_ratio <= start_ratio:
        return float(load_ratio >= full_ratio)
    if load_ratio <= start_ratio:
        return 0.0
    if load_ratio >= full_ratio:
        return 1.0
    span = full_ratio - start_ratio
    return float((load_ratio - start_ratio) / span)


def _lerp(a, b, alpha):
    """スカラー線形補間"""
    return float((1.0 - alpha) * float(a) + alpha * float(b))


def _lerp_vec(vec_a, vec_b, alpha):
    """ベクトル線形補間"""
    vec_a = np.asarray(vec_a, dtype=float)
    vec_b = np.asarray(vec_b, dtype=float)
    return (1.0 - alpha) * vec_a + alpha * vec_b


def project_torque_to_limits(
    A, tau_candidate, torque_limits, tau_preferred=None, weights=None
):
    """候補トルクを、重み付き最小ノルムとバイアスを考慮しつつ安全範囲へ投影"""
    tau_candidate = np.asarray(tau_candidate, dtype=float).reshape(2)
    desired_output = float((A @ tau_candidate.reshape(2, 1)).item())
    if tau_preferred is None:
        tau_preferred = _apply_torque_preference(tau_candidate)
    tau_res, _ = _solve_torque_with_limits(
        A, desired_output, torque_limits, tau_preferred=tau_preferred, weights=weights
    )
    return tau_res


def solve_torque_with_fixed_motor(A, tau_out_desired, freeze_idx, tau_freeze, eps=1e-8):
    """固定モータのトルクを指定した上で出力トルクを達成する解を返す"""
    A = np.asarray(A, dtype=float).reshape(1, 2)
    tau = np.zeros(2, dtype=float)
    tau[freeze_idx] = float(tau_freeze)
    active_idx = 1 - freeze_idx
    a_active = float(A[0, active_idx])
    a_freeze = float(A[0, freeze_idx])
    if abs(a_active) < eps:
        raise ValueError("Active motor gain is too small to solve for output torque.")
    tau_active = (float(tau_out_desired) - a_freeze * tau[freeze_idx]) / a_active
    tau[active_idx] = tau_active
    return tau

# ==================================================================================
# 目標生成
# ==================================================================================

def get_active_profile_name():
    profile = REFERENCE_PROFILE.get('active_profile', 'step')
    if profile not in REFERENCE_PROFILE or not isinstance(REFERENCE_PROFILE[profile], dict):
        raise ValueError(f"未定義の目標プロファイル: {profile}")
    return profile


def get_profile_label():
    label = REFERENCE_PROFILE.get('file_label')
    if label:
        return str(label)
    return get_active_profile_name()


def sanitize_label_for_filename(label):
    safe = ''.join(ch if (ch.isalnum() or ch in ('-', '_')) else '_' for ch in label)
    return safe or 'profile'


def _normalize_filename_values(values):
    if values is None:
        return None
    arr = np.asarray(values, dtype=float).ravel()
    return arr if arr.size > 0 else None


def _infer_profile_filename_values(profile, cfg):
    if profile == 'step':
        offset = float(cfg.get('offset', 0.0))
        amp = float(cfg.get('output_amplitude', 0.0))
        if abs(amp) < 1e-12:
            return _normalize_filename_values([offset])
        return _normalize_filename_values([offset, offset + amp])
    if profile in ('sine', 'chirp'):
        offset = float(cfg.get('offset', 0.0))
        amp = float(cfg.get('output_amplitude', 0.0))
        if abs(amp) < 1e-12:
            return _normalize_filename_values([offset])
        return _normalize_filename_values([offset - amp, offset + amp])
    if profile in ('ramp', 'ramp_b'):
        start_value = float(cfg.get('start_value', 0.0))
        end_value = float(cfg.get('end_value', start_value))
        return _normalize_filename_values([start_value, end_value])
    if profile == 'ramp_step':
        start_value = float(cfg.get('start_value', 0.0))
        end_value = float(cfg.get('end_value', start_value))
        step_cfg = cfg.get('step', {}) if isinstance(cfg.get('step'), dict) else {}
        amplitude = float(step_cfg.get('amplitude', 0.0))
        candidates = [start_value, end_value]
        if abs(amplitude) >= 1e-12:
            candidates.append(end_value + amplitude)
            if not bool(step_cfg.get('align_to_ramp_end', True)):
                candidates.append(start_value + amplitude)
        return _normalize_filename_values(candidates)
    if profile == 'nstep':
        values = list(cfg.get('values', []))
        wait_value = cfg.get('wait_value')
        if wait_value is not None:
            values = [wait_value] + values
        return _normalize_filename_values(values)
    return None


def get_filename_command_values_override():
    profile = get_active_profile_name()
    cfg = REFERENCE_PROFILE.get(profile, {})
    explicit = _normalize_filename_values(cfg.get('filename_values'))
    if explicit is not None:
        return explicit
    global_override = _normalize_filename_values(FILENAME_COMMAND_VALUES)
    if global_override is not None:
        return global_override
    inferred = _infer_profile_filename_values(profile, cfg)
    if inferred is not None:
        return inferred
    return None


def resolve_command_values(base_command, theta_ref_turn=None):
    override_values = get_filename_command_values_override()
    if override_values is not None:
        values = np.asarray(override_values, dtype=float).ravel()
    else:
        base_arr = np.asarray(base_command, dtype=float).ravel()
        if base_arr.size == 0:
            base_arr = np.array([0.0], dtype=float)
        values = np.unique(np.round(base_arr, decimals=FILENAME_DECIMALS))
        if theta_ref_turn is None or len(theta_ref_turn) == 0:
            theta_ref_turn = base_arr
        else:
            theta_ref_turn = np.asarray(theta_ref_turn, dtype=float).ravel()
        if values.size > 20 and theta_ref_turn.size > 0:
            values = np.unique(
                np.round(
                    [np.min(theta_ref_turn), np.max(theta_ref_turn)],
                    decimals=FILENAME_DECIMALS,
                )
            )
    return np.round(values, decimals=FILENAME_DECIMALS)


def format_command_value(value):
    txt = f"{float(value):.{FILENAME_DECIMALS}f}".rstrip('0').rstrip('.')
    return txt if txt else "0"


def _format_vector(values):
    arr = np.asarray(values, dtype=float).ravel()
    return "[" + ", ".join(f"{val:g}" for val in arr) + "]"


def _format_profile_settings(profile, cfg):
    def fmt(value):
        if isinstance(value, float):
            return f"{value:g}"
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value, dtype=float).ravel()
            return "[" + ", ".join(f"{v:g}" for v in arr) + "]"
        return str(value)

    settings = []
    if profile == 'step':
        settings = [
            f"wait={fmt(cfg.get('initial_wait', 0.0))}s",
            f"step={fmt(cfg.get('step_duration', 0.0))}s",
            f"amp={fmt(cfg.get('output_amplitude', 0.0))}turn",
            f"offset={fmt(cfg.get('offset', 0.0))}",
        ]
    elif profile == 'sine':
        settings = [
            f"wait={fmt(cfg.get('initial_wait', 0.0))}s",
            f"amp={fmt(cfg.get('output_amplitude', 0.0))}turn",
            f"freq={fmt(cfg.get('frequency_hz', 0.0))}Hz",
            f"offset={fmt(cfg.get('offset', 0.0))}",
        ]
    elif profile == 'chirp':
        settings = [
            f"wait={fmt(cfg.get('initial_wait', 0.0))}s",
            f"amp={fmt(cfg.get('output_amplitude', 0.0))}turn",
            f"f0={fmt(cfg.get('start_frequency_hz', 0.0))}Hz",
            f"f1={fmt(cfg.get('end_frequency_hz', 0.0))}Hz",
            f"dur={fmt(cfg.get('duration', 0.0))}s",
            f"offset={fmt(cfg.get('offset', 0.0))}",
        ]
    elif profile == 'ramp':
        settings = [
            f"wait={fmt(cfg.get('initial_wait', 0.0))}s",
            f"start={fmt(cfg.get('start_value', 0.0))}",
            f"end={fmt(cfg.get('end_value', 0.0))}",
            f"ramp={fmt(cfg.get('ramp_duration', 0.0))}s",
            f"hold={fmt(cfg.get('hold_duration', 0.0))}s",
            f"return={fmt(cfg.get('return_duration', 0.0))}s",
        ]
    elif profile == 'ramp_b':
        settings = [
            f"wait={fmt(cfg.get('initial_wait', 0.0))}s",
            f"start={fmt(cfg.get('start_value', 0.0))}",
            f"end={fmt(cfg.get('end_value', 0.0))}",
            f"ramp={fmt(cfg.get('ramp_duration', 0.0))}s",
            f"hold={fmt(cfg.get('hold_duration', 0.0))}s",
            f"return={fmt(cfg.get('return_duration', 0.0))}s",
            f"start_hold={fmt(cfg.get('start_hold_duration', 0.0))}s",
        ]
    elif profile == 'ramp_step':
        step_cfg = cfg.get('step', {}) if isinstance(cfg.get('step'), dict) else {}
        settings = [
            f"wait={fmt(cfg.get('initial_wait', 0.0))}s",
            f"start={fmt(cfg.get('start_value', 0.0))}",
            f"end={fmt(cfg.get('end_value', 0.0))}",
            f"ramp={fmt(cfg.get('ramp_duration', 0.0))}s",
            f"hold={fmt(cfg.get('hold_duration', 0.0))}s",
            f"return={fmt(cfg.get('return_duration', 0.0))}s",
            f"step_amp={fmt(step_cfg.get('amplitude', 0.0))}",
            f"step_dur={fmt(step_cfg.get('duration', 0.0))}s",
        ]
        if step_cfg:
            settings.append(f"step_period={fmt(step_cfg.get('period', 0.0))}s")
            settings.append(f"step_repeat={step_cfg.get('repeat', True)}")
    elif profile == 'nstep':
        settings = [
            f"wait={fmt(cfg.get('initial_wait', 0.0))}s",
            f"values={fmt(cfg.get('values', []))}",
            f"hold={fmt(cfg.get('hold_duration', 0.0))}s",
            f"repeat={cfg.get('repeat', True)}",
        ]
        if 'wait_value' in cfg:
            settings.append(f"wait_val={fmt(cfg.get('wait_value', 0.0))}")
    else:
        settings = [f"{k}={fmt(v)}" for k, v in cfg.items()]
    return ", ".join(settings)


def _compute_ramp_profile(cfg, elapsed_time):
    initial_wait = float(cfg.get('initial_wait', 0.0))
    start_value = float(cfg.get('start_value', 0.0))
    end_value = float(cfg.get('end_value', start_value))
    ramp_duration = max(float(cfg.get('ramp_duration', 0.0)), 1e-6)
    hold_duration = max(float(cfg.get('hold_duration', 0.0)), 0.0)
    return_duration = max(float(cfg.get('return_duration', 0.0)), 0.0)
    repeat = bool(cfg.get('repeat', True))

    cycle = ramp_duration + hold_duration + return_duration
    if elapsed_time < initial_wait:
        return start_value, 0.0, cycle, repeat, ramp_duration

    def ramp_value(phase, repeat_mode):
        if phase < ramp_duration:
            ratio = phase / ramp_duration
            return start_value + (end_value - start_value) * ratio
        phase -= ramp_duration
        if phase < hold_duration:
            return end_value
        phase -= hold_duration
        if return_duration > 0.0:
            ratio = min(max(phase / return_duration, 0.0), 1.0)
            return end_value + (start_value - end_value) * ratio
        return start_value if repeat_mode else end_value

    t_after_wait = elapsed_time - initial_wait
    if cycle <= 0.0:
        return end_value, max(t_after_wait, 0.0), cycle, repeat, ramp_duration

    if repeat:
        phase = t_after_wait % cycle
        value = ramp_value(phase, True)
    else:
        if t_after_wait >= cycle:
            final_value = start_value if return_duration > 0.0 else end_value
            value = final_value
        else:
            value = ramp_value(t_after_wait, False)

    return value, max(t_after_wait, 0.0), cycle, repeat, ramp_duration


def generate_output_reference(elapsed_time):
    """出力θの目標値をプロファイルに応じて生成"""
    profile = get_active_profile_name()
    cfg = REFERENCE_PROFILE[profile]

    if profile == 'step':
        initial_wait = cfg['initial_wait']
        step_duration = cfg['step_duration']
        amp = cfg['output_amplitude']
        offset = cfg.get('offset', 0.0)

        if elapsed_time < initial_wait:
            return offset

        period = step_duration * 4.0
        if period <= 0.0:
            return offset
        cyc = (elapsed_time - initial_wait) % period
        if cyc < step_duration:
            return offset + amp
        else:
            return offset

    if profile == 'sine':
        initial_wait = cfg['initial_wait']
        amp = cfg['output_amplitude']
        freq = cfg['frequency_hz']
        offset = cfg.get('offset', 0.0)
        if elapsed_time < initial_wait:
            return offset
        t = elapsed_time - initial_wait
        return offset + amp * math.sin(2.0 * math.pi * freq * t)

    if profile == 'chirp':
        initial_wait = cfg['initial_wait']
        amp = cfg['output_amplitude']
        f0 = cfg['start_frequency_hz']
        f1 = cfg['end_frequency_hz']
        duration = max(cfg['duration'], 1e-6)
        offset = cfg.get('offset', 0.0)
        if elapsed_time < initial_wait:
            return offset
        t = elapsed_time - initial_wait
        k = (f1 - f0) / duration
        if t > duration:
            phase = 2.0 * math.pi * (f0 * duration + 0.5 * k * duration ** 2) + 2.0 * math.pi * f1 * (t - duration)
        else:
            phase = 2.0 * math.pi * (f0 * t + 0.5 * k * t ** 2)
        return offset + amp * math.sin(phase)

    if profile == 'ramp':
        value, _, _, _, _ = _compute_ramp_profile(cfg, elapsed_time)
        return value

    if profile == 'ramp_b':
        initial_wait = float(cfg.get('initial_wait', 0.0))
        if elapsed_time < initial_wait:
            return float(cfg.get('start_value', 0.0))

        start_value = float(cfg.get('start_value', 0.0))
        end_value = float(cfg.get('end_value', start_value))
        ramp_duration = max(float(cfg.get('ramp_duration', 0.0)), 1e-6)
        hold_duration = max(float(cfg.get('hold_duration', 0.0)), 0.0)
        return_duration = max(float(cfg.get('return_duration', 0.0)), 1e-6)
        start_hold = max(float(cfg.get('start_hold_duration', 0.0)), 0.0)
        repeat = bool(cfg.get('repeat', True))

        t_after_wait = elapsed_time - initial_wait
        cycle = start_hold + ramp_duration + hold_duration + return_duration
        if cycle <= 0.0:
            return start_value

        if repeat:
            phase = t_after_wait % cycle
        else:
            phase = min(t_after_wait, cycle)

        if phase < start_hold:
            return start_value
        phase -= start_hold

        if phase < ramp_duration:
            ratio = phase / ramp_duration
            return start_value + (end_value - start_value) * ratio
        phase -= ramp_duration

        if phase < hold_duration:
            return end_value
        phase -= hold_duration

        if phase < return_duration:
            ratio = phase / return_duration
            return end_value + (start_value - end_value) * ratio

        return start_value if repeat or t_after_wait >= cycle else end_value

    if profile == 'ramp_step':
        base_value, t_after_wait, cycle, repeat_ramp, ramp_duration = _compute_ramp_profile(cfg, elapsed_time)
        step_cfg = cfg.get('step', {})
        amplitude = float(step_cfg.get('amplitude', 0.0))
        if amplitude == 0.0:
            return base_value

        align_to_ramp_end = bool(step_cfg.get('align_to_ramp_end', False))
        start_after_base = max(float(step_cfg.get('start_after', 0.0)), 0.0)
        start_after = start_after_base + (ramp_duration if align_to_ramp_end else 0.0)
        duration = max(float(step_cfg.get('duration', 0.0)), 0.0)
        repeat_step = bool(step_cfg.get('repeat', True))

        if t_after_wait < start_after:
            return base_value

        elapsed_since_start = t_after_wait - start_after
        step_active = False

        if repeat_step:
            period_default = cycle if (cycle > 0.0 and repeat_ramp) else duration if duration > 0.0 else 1.0
            period = float(step_cfg.get('period', period_default))
            if period <= 0.0:
                period = max(period_default, 1e-6)
            offset = float(step_cfg.get('offset_in_cycle', 0.0))
            if align_to_ramp_end:
                offset = (offset + ramp_duration) % period
            phase = (elapsed_since_start - offset) % period
            if duration <= 0.0:
                step_active = phase < 1e-6
            else:
                step_active = phase < duration
        else:
            if duration <= 0.0:
                step_active = elapsed_since_start >= 0.0 and elapsed_since_start < 1e-6
            else:
                step_active = 0.0 <= elapsed_since_start < duration

        if step_active:
            return base_value + amplitude
        return base_value

    if profile == 'nstep':
        initial_wait = float(cfg.get('initial_wait', 0.0))
        values = list(cfg.get('values', []))
        hold_duration = max(float(cfg.get('hold_duration', 0.0)), 1e-6)
        repeat = bool(cfg.get('repeat', True))
        if not values:
            return 0.0
        wait_value = float(cfg.get('wait_value', values[0]))
        if elapsed_time < initial_wait:
            return wait_value
        t = elapsed_time - initial_wait
        idx = int(t // hold_duration)
        if repeat:
            idx = idx % len(values)
        else:
            idx = min(idx, len(values) - 1)
        return float(values[idx])

    raise ValueError(f"未対応の目標プロファイル: {profile}")

# ==================================================================================
# PID コントローラ（スレッドセーフ）
# ==================================================================================

class PIDController:
    def __init__(
        self,
        kp=0.0,
        ki=0.0,
        kd=0.0,
        max_output=10.0,
        min_output=-10.0,
        derivative_filter_alpha=1.0,
        derivative_mode='measurement',
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.min_output = min_output
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = time.time()
        self.prev_feedback = 0.0
        self.prev_feedback_derivative = 0.0
        self.derivative_filter_alpha = float(
            min(max(derivative_filter_alpha, 0.0), 1.0)
        )
        self.derivative_mode = derivative_mode
        self.lock = threading.Lock()

    def update(self, setpoint, feedback):
        with self.lock:
            t = time.time()
            dt = t - self.prev_time
            if dt <= 0.0:
                dt = 1e-6
            e = setpoint - feedback
            P = self.kp * e
            self.integral += e * dt
            if self.ki > 0:
                i_lim = self.max_output / self.ki
                self.integral = max(min(self.integral, i_lim), -i_lim)
            I = self.ki * self.integral
            if self.derivative_mode == 'measurement':
                # derivative on measurement with optional low-pass filtering
                raw_feedback_derivative = (feedback - self.prev_feedback) / dt
                alpha = self.derivative_filter_alpha
                feedback_derivative = (
                    alpha * raw_feedback_derivative
                    + (1.0 - alpha) * self.prev_feedback_derivative
                )
                D = -self.kd * feedback_derivative
                self.prev_feedback_derivative = feedback_derivative
            else:
                D = self.kd * (e - self.prev_error) / dt
                self.prev_feedback_derivative = 0.0
            u = P + I + D
            u = max(min(u, self.max_output), self.min_output)
            self.prev_error = e
            self.prev_feedback = feedback
            self.prev_time = t
            return u, e, P, I, D

# ==================================================================================
# 可視化/解析
# ==================================================================================

def _estimate_linear_map(x0, x1, y, include_bias=False):
    arr0 = np.asarray(x0, dtype=float)
    arr1 = np.asarray(x1, dtype=float)
    arr_y = np.asarray(y, dtype=float)
    mask = np.isfinite(arr0) & np.isfinite(arr1) & np.isfinite(arr_y)
    n = int(mask.sum())
    required = 3 if include_bias else 2
    if n < required:
        return None
    X = np.column_stack([arr0[mask], arr1[mask]])
    if include_bias:
        X = np.column_stack([X, np.ones(n)])
    coef, _, _, _ = np.linalg.lstsq(X, arr_y[mask], rcond=None)
    gains = coef[:2]
    bias = float(coef[2]) if include_bias else 0.0
    residual = arr_y[mask] - X @ coef
    rms = float(np.sqrt(np.mean(residual ** 2))) if n > 0 else float("nan")
    return gains, bias, rms, n


def analyze_and_plot_response(csv_filename):
    profile_label = get_profile_label()
    print(f"応答解析を開始 ({profile_label}) : {csv_filename}")
    df = pd.read_csv(csv_filename, comment='#')

    has_backlash_cols = {'theta_ref_raw', 'backlash_state'}.issubset(df.columns)
    n_rows = 2 + (2 if has_backlash_cols else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    t = df['time'].to_numpy(dtype=float)
    if 'theta_ref_raw' in df.columns:
        base_command = df['theta_ref_raw'].to_numpy(dtype=float)
    else:
        base_command = df['theta_ref'].to_numpy(dtype=float)
    theta_ref_turn = df['theta_ref'].to_numpy(dtype=float)
    theta_turn = df['output_pos'].to_numpy(dtype=float)
    torque1 = df['motor0_torque'].to_numpy(dtype=float)
    torque2 = df['motor1_torque'].to_numpy(dtype=float)
    theta_ref_raw_turn = df['theta_ref_raw'].to_numpy(dtype=float) if 'theta_ref_raw' in df.columns else None
    backlash_state_turn = df['backlash_state'].to_numpy(dtype=float) if 'backlash_state' in df.columns else None

    turn_to_deg = lambda arr: np.asarray(arr, dtype=float) * 360.0
    theta_ref_deg = turn_to_deg(theta_ref_turn)
    theta_deg = turn_to_deg(theta_turn)
    theta_ref_raw_deg = turn_to_deg(theta_ref_raw_turn) if theta_ref_raw_turn is not None else None
    backlash_state_deg = turn_to_deg(backlash_state_turn) if backlash_state_turn is not None else None

    est_pos = _estimate_linear_map(
        df['motor0_pos'].to_numpy(dtype=float),
        df['motor1_pos'].to_numpy(dtype=float),
        df['output_pos'].to_numpy(dtype=float),
        include_bias=True,
    )
    est_vel = _estimate_linear_map(
        df['motor0_vel'].to_numpy(dtype=float),
        df['motor1_vel'].to_numpy(dtype=float),
        df['output_vel'].to_numpy(dtype=float),
        include_bias=False,
    )

    if est_pos:
        (g_pos, bias_pos, rms_pos, n_pos) = est_pos
        print("推定A (位置ベース): "
              f"a1={g_pos[0]:.6f}, a2={g_pos[1]:.6f}, bias={bias_pos:.6f}, "
              f"RMS={rms_pos:.6e} [turn], samples={n_pos}")
    else:
        print("推定A (位置ベース): 十分なサンプルがありません。")

    if est_vel:
        (g_vel, bias_vel, rms_vel, n_vel) = est_vel
        print("推定A (速度ベース): "
              f"a1={g_vel[0]:.6f}, a2={g_vel[1]:.6f}, bias={bias_vel:.6f} (固定0), "
              f"RMS={rms_vel:.6e} [turn/s], samples={n_vel}")
    else:
        print("推定A (速度ベース): 十分なサンプルがありません。")

    theta_ax = axes[0]
    theta_ax.plot(t, theta_ref_deg, '--', label='θ_ref')
    theta_ax.plot(t, theta_deg, '-', label='θ')
    if theta_ref_raw_deg is not None:
        theta_ax.plot(t, theta_ref_raw_deg, ':', label='θ_ref_raw')
    theta_ax.set_ylabel('θ [deg]')
    theta_ax.legend()

    next_axis_idx = 1

    if has_backlash_cols and theta_ref_raw_deg is not None:
        delta_ref_deg = theta_ref_deg - theta_ref_raw_deg
        delta_ax = axes[next_axis_idx]
        delta_ax.plot(t, delta_ref_deg, '-', label='補償量 Δθ_ref')
        if backlash_state_deg is not None:
            delta_ax.plot(t, backlash_state_deg, '--', label='推定プレイ状態')
        delta_ax.axhline(0.0, color='black', linewidth=0.8, linestyle=':')
        delta_ax.set_ylabel('補償 [deg]')
        delta_ax.legend()
        next_axis_idx += 1

        error_ax = axes[next_axis_idx]
        error_comp_deg = theta_deg - theta_ref_deg
        error_ax.plot(t, error_comp_deg, '-', label='θ - θ_ref (補償後)')
        if theta_ref_raw_deg is not None:
            error_raw_deg = theta_deg - theta_ref_raw_deg
            error_ax.plot(t, error_raw_deg, '--', label='θ - θ_ref_raw (補償前)')
        error_ax.axhline(0.0, color='black', linewidth=0.8, linestyle=':')
        error_ax.set_ylabel('追従誤差 [deg]')
        error_ax.legend()
        next_axis_idx += 1

    torque_ax = axes[-1]
    torque_ax.plot(t, torque1, color='red', label='Input torque1')
    torque_ax.plot(t, torque2, color='green', label='Input torque2')
    torque_ax.set_xlabel('Time [s]')
    torque_ax.set_ylabel('Torque [Nm]')
    torque_ax.legend()

    # グリッド線なし、目盛り内向き
    for ax in axes:
        ax.grid(False)
        ax.tick_params(axis='both', direction='in', length=6, width=0.8)

    os.makedirs(FIG_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(csv_filename))[0]
    override_values = get_filename_command_values_override()
    if override_values is not None:
        command_values = np.asarray(override_values, dtype=float).ravel()
    else:
        command_values = resolve_command_values(base_command, theta_ref_turn)

    command_slug = "_".join(format_command_value(v) for v in command_values)
    command_slug = command_slug.strip()
    if command_slug:
        command_slug = command_slug.replace(' ', '')
        fig_filename = f"{command_slug}_{base_name}{PLOT_FILENAME_SUFFIX}"
    else:
        fig_filename = f"{base_name}{PLOT_FILENAME_SUFFIX}"
    fig_path = os.path.join(FIG_DIR, fig_filename)
    command_values_text = ", ".join(format_command_value(v) for v in command_values) or "-"

    fig.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.show()

    print("\n" + "="*60)
    print("データファイルの処理を選択してください:")
    print("  [1] CSVとグラフの両方を保存")
    print("  [2] グラフのみ保存（CSVは破棄）")
    print("  [3] CSVのみ保存（グラフは破棄）")
    print("  [4] 両方とも破棄")
    print("="*60)

    while True:
        try:
            choice = input("選択 (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                break
            else:
                print("1,2,3,4 のいずれかを入力してください。")
        except KeyboardInterrupt:
            print("\n処理をキャンセルします。")
            choice = '4'
            break

    final_csv_path = csv_filename
    final_graph_path = fig_path

    try:
        if choice == '1':
            print(f"✅ CSV保存: {csv_filename}")
            print(f"✅ グラフ保存: {fig_path}")
        elif choice == '2':
            os.remove(csv_filename)
            final_csv_path = None
            print(f"🗑️ CSV削除: {csv_filename}")
            print(f"✅ グラフ保存: {fig_path}")
        elif choice == '3':
            os.remove(fig_path)
            final_graph_path = None
            print(f"✅ CSV保存: {csv_filename}")
            print(f"🗑️ グラフ削除: {fig_path}")
        elif choice == '4':
            os.remove(csv_filename)
            os.remove(fig_path)
            final_csv_path = None
            final_graph_path = None
            print("🗑️ 両方削除")
    except Exception as e:
        print(f"⚠️ ファイル削除エラー: {e}")

    plt.close('all')
    return final_graph_path, final_csv_path

# ==================================================================================
# メイン
# ==================================================================================

def main():
    print("=== トルク制御PID環境 - 最小ノルム配分 版 ===")
    print(f"使用する応答プロファイル: {get_profile_label()} (type={get_active_profile_name()})")

    # ---- ODrive 接続 ----
    print("ODriveを検索中...")
    try:
        odrv0 = odrive.find_any(serial_number=ODRIVE_SERIAL['motor0'])
        odrv1 = odrive.find_any(serial_number=ODRIVE_SERIAL['motor1'])
        odrv2 = odrive.find_any(serial_number=ODRIVE_SERIAL['output'])
        print("ODrive接続完了")
    except Exception as e:
        print(f"ODrive接続エラー: {e}")
        return

    # 初期位置
    initial_position0 = odrv0.axis0.pos_vel_mapper.pos_rel
    initial_position1 = odrv1.axis0.pos_vel_mapper.pos_rel
    initial_position2 = odrv2.axis0.pos_vel_mapper.pos_rel

    # ---- モータをトルク制御モードへ ----
    print("モータをトルク制御モードに設定中...")
    odrv0.axis0.requested_state = AxisState.CLOSED_LOOP_CONTROL
    odrv0.axis0.controller.config.control_mode = ControlMode.TORQUE_CONTROL
    odrv0.axis0.config.motor.torque_constant = ODRIVE_TORQUE_CONSTANT['motor0']

    odrv1.axis0.requested_state = AxisState.CLOSED_LOOP_CONTROL
    odrv1.axis0.controller.config.control_mode = ControlMode.TORQUE_CONTROL
    odrv1.axis0.config.motor.torque_constant = ODRIVE_TORQUE_CONSTANT['motor1']
    print("モータ設定完了")

    # ---- コントローラ ----
    output_pid = None
    motor_pid = {}
    if CONTROL_MODE == 'output_pid':
        output_pid = PIDController(
            kp=OUTPUT_PID['kp'],
            ki=OUTPUT_PID['ki'],
            kd=OUTPUT_PID['kd'],
            max_output=OUTPUT_PID['max_output'],
            min_output=-OUTPUT_PID['max_output'],
            derivative_filter_alpha=PID_DERIVATIVE_FILTER_ALPHA,
            derivative_mode=PID_DERIVATIVE_MODE,
        )
    elif CONTROL_MODE == 'per_motor_pid':
        if ENABLE_OUTER_PID_IN_PER_MOTOR:
            output_pid = PIDController(
                kp=OUTPUT_PID['kp'],
                ki=OUTPUT_PID['ki'],
                kd=OUTPUT_PID['kd'],
                max_output=OUTPUT_PID['max_output'],
                min_output=-OUTPUT_PID['max_output'],
                derivative_filter_alpha=PID_DERIVATIVE_FILTER_ALPHA,
                derivative_mode=PID_DERIVATIVE_MODE,
            )
        for key, cfg in MOTOR_PID.items():
            motor_pid[key] = PIDController(
                kp=cfg['kp'],
                ki=cfg['ki'],
                kd=cfg['kd'],
                max_output=cfg['max_output'],
                min_output=-cfg['max_output'],
                derivative_filter_alpha=PID_DERIVATIVE_FILTER_ALPHA,
                derivative_mode=PID_DERIVATIVE_MODE,
            )
    else:
        raise ValueError(f"未知のCONTROL_MODE: {CONTROL_MODE}")

    # ---- ログ ----
    data_lock = threading.Lock()
    data_log = {
        'time': [],
        'motor0': {'pos': [], 'vel': [], 'torque': []},
        'motor1': {'pos': [], 'vel': [], 'torque': []},
        'output': {'pos': [], 'vel': []},
        'theta_ref': [],
        'theta_ctrl': [],
        'tau_out': [],
    }

    # ---- 可変減速設定 ----
    weights_release_vec = _get_weight_vector('release')
    weights_hold_vec = _get_weight_vector('hold')
    q_ref = JOINT_REFERENCE.copy()
    adaptive_enabled = bool(VARIABLE_RATIO_CONFIG.get('enabled', False))
    primary_motor = VARIABLE_RATIO_CONFIG.get('primary_motor', 'motor0')
    primary_idx = {'motor0': 0, 'motor1': 1}.get(primary_motor, 0)
    secondary_idx = 1 - primary_idx
    release_start_ratio = float(VARIABLE_RATIO_CONFIG.get('release_start_ratio', 0.7))
    release_full_ratio = float(VARIABLE_RATIO_CONFIG.get('release_full_ratio', 0.95))
    secondary_gain_hold = float(VARIABLE_RATIO_CONFIG.get('secondary_gain_hold', 0.0))
    secondary_gain_release = float(VARIABLE_RATIO_CONFIG.get('secondary_gain_release', 1.0))
    TORQUE_PREFERENCE['preferred_motor'] = primary_motor
    freeze_idx_config = {'motor0': 0, 'motor1': 1}.get(FREEZE_CONFIG['motor_to_freeze'])
    freeze_kp = FREEZE_CONFIG['kp']
    freeze_kd = FREEZE_CONFIG['kd']
    torque_limits = np.array([
        SAFETY_CONFIG['max_torque0'],
        SAFETY_CONFIG['max_torque1'],
    ], dtype=float)

    start_time = time.time()
    dt_target = 1.0 / CONTROL_FREQUENCY
    
    print("=== 制御開始 (Ctrl+Cで停止) ===")
    
    try:
        while True:
            t0 = time.time()
            elapsed = t0 - start_time

            # ---- 目標 ----
            theta_ref = generate_output_reference(elapsed)

            tau_out_target = None
            current_freeze_idx = freeze_idx_config
            release_alpha = 1.0

            # ---- 計測 ----
            q0 = odrv0.axis0.pos_vel_mapper.pos_rel - initial_position0
            q1 = odrv1.axis0.pos_vel_mapper.pos_rel - initial_position1
            qout = odrv2.axis0.pos_vel_mapper.pos_rel - initial_position2
            dq0 = odrv0.axis0.pos_vel_mapper.vel
            dq1 = odrv1.axis0.pos_vel_mapper.vel
            dqout = odrv2.axis0.pos_vel_mapper.vel  # 参考

            q = np.array([q0, q1])
            qdot = np.array([dq0, dq1])

            # ---- 機構行列 ----
            A = get_A(q)
            tau_cmd_prelimit = np.zeros(2)
            tau_cmd = np.zeros(2)
            theta_ctrl_cmd = theta_ref - qout
            weights_current = np.array(weights_release_vec, dtype=float)
    
            if CONTROL_MODE == 'output_pid':
                tau_out_desired, _, _, _, _ = output_pid.update(theta_ref, qout)
                tau_out_target = tau_out_desired

                if adaptive_enabled:
                    tau_hold = min_norm_torque_split(
                        A, tau_out_desired, weights=np.array(weights_hold_vec, dtype=float)
                    )
                    primary_tau = tau_hold[primary_idx]
                    release_alpha = _compute_release_alpha(
                        primary_tau,
                        torque_limits[primary_idx],
                        release_start_ratio,
                        release_full_ratio,
                    )
                    weights_current = np.array(
                        _lerp_vec(weights_hold_vec, weights_release_vec, release_alpha),
                        dtype=float,
                    )
                else:
                    release_alpha = 1.0
                    weights_current = np.array(weights_release_vec, dtype=float)

                tau_cmd_prelimit = min_norm_torque_split(
                    A, tau_out_desired, weights=weights_current
                )
                TORQUE_PREFERENCE['secondary_gain'] = _lerp(
                    secondary_gain_hold, secondary_gain_release, release_alpha
                )

            elif CONTROL_MODE == 'per_motor_pid':
                theta_err_raw = theta_ref - qout
                theta_ctrl = theta_err_raw
                if output_pid is not None:
                    theta_ctrl, _, _, _, _ = output_pid.update(theta_ref, qout)
                At = A.T
                s = float(A @ At)
                if s < 1e-8:
                    raise ValueError("Mechanism matrix A is near-singular.")

                delta_q = np.zeros(2)
                if current_freeze_idx is None:
                    delta_q = (At / s).flatten() * theta_ctrl
                else:
                    active_idx = 1 - current_freeze_idx
                    a_active = float(A[0, active_idx])
                    if abs(a_active) > 1e-8:
                        delta_q[active_idx] = theta_ctrl / a_active
                    else:
                        delta_q[active_idx] = 0.0

                q_des = q_ref + delta_q

                keys = ['motor0', 'motor1']
                for idx, key in enumerate(keys):
                    if current_freeze_idx is not None and idx == current_freeze_idx:
                        tau_cmd_prelimit[idx] = float(-freeze_kp * q[idx] - freeze_kd * qdot[idx])
                    else:
                        pid = motor_pid[key]
                        u, _, _, _, _ = pid.update(q_des[idx], q[idx])
                        tau_cmd_prelimit[idx] = float(u)

                if adaptive_enabled and current_freeze_idx is None:
                    primary_tau = tau_cmd_prelimit[primary_idx]
                    release_alpha = _compute_release_alpha(
                        primary_tau,
                        torque_limits[primary_idx],
                        release_start_ratio,
                        release_full_ratio,
                    )
                    weights_current = np.array(
                        _lerp_vec(weights_hold_vec, weights_release_vec, release_alpha),
                        dtype=float,
                    )
                else:
                    release_alpha = 1.0
                    weights_current = np.array(weights_release_vec, dtype=float)

                secondary_gain = _lerp(
                    secondary_gain_hold, secondary_gain_release, release_alpha
                )
                TORQUE_PREFERENCE['secondary_gain'] = secondary_gain

                if current_freeze_idx is not None and current_freeze_idx != secondary_idx:
                    tau_cmd_prelimit[current_freeze_idx] = float(
                        -freeze_kp * q[current_freeze_idx] - freeze_kd * qdot[current_freeze_idx]
                    )
                elif current_freeze_idx is None:
                    # scale secondary motor torque; keep primary as-is
                    tau_cmd_prelimit[secondary_idx] *= secondary_gain

                theta_ctrl_cmd = float(theta_ctrl)
    
            else:
                raise RuntimeError(f"未対応のCONTROL_MODE: {CONTROL_MODE}")
    
            # ---- 飽和を考慮した再割り当て ----
            if current_freeze_idx is not None:
                tau_cmd = np.zeros(2, dtype=float)
                tau_freeze = float(
                    np.clip(
                        tau_cmd_prelimit[current_freeze_idx],
                        -torque_limits[current_freeze_idx],
                        torque_limits[current_freeze_idx],
                    )
                )
                if tau_out_target is not None:
                    try:
                        tau_cmd = solve_torque_with_fixed_motor(
                            A, tau_out_target, current_freeze_idx, tau_freeze
                        )
                    except ValueError:
                        tau_cmd = np.asarray(tau_cmd_prelimit, dtype=float).copy()
                        tau_cmd[current_freeze_idx] = tau_freeze
                else:
                    tau_cmd = np.asarray(tau_cmd_prelimit, dtype=float).copy()
                    tau_cmd[current_freeze_idx] = tau_freeze
                tau_cmd[current_freeze_idx] = tau_freeze
                active_idx = 1 - current_freeze_idx
                tau_cmd[active_idx] = float(
                    np.clip(
                        tau_cmd[active_idx],
                        -torque_limits[active_idx],
                        torque_limits[active_idx],
                    )
                )
            else:
                tau_cmd = project_torque_to_limits(
                    A, tau_cmd_prelimit, torque_limits, weights=weights_current
                )
            tau_out_disp = motor_torque_to_output(tau_cmd)
    
            # ---- 出力 ----
            odrv0.axis0.controller.input_torque = tau_cmd[0]
            odrv1.axis0.controller.input_torque = tau_cmd[1]
    
            # ---- ログ ----
            with data_lock:
                data_log['time'].append(elapsed)
                data_log['motor0']['pos'].append(q0)
                data_log['motor0']['vel'].append(dq0)
                data_log['motor0']['torque'].append(float(tau_cmd[0]))
                data_log['motor1']['pos'].append(q1)
                data_log['motor1']['vel'].append(dq1)
                data_log['motor1']['torque'].append(float(tau_cmd[1]))
                data_log['output']['pos'].append(qout)
                data_log['output']['vel'].append(dqout)
                data_log['theta_ref'].append(theta_ref)
                data_log['theta_ctrl'].append(float(theta_ctrl_cmd))
                data_log['tau_out'].append(float(tau_out_disp))
    
            # ---- タイミング調整 ----
            dt = time.time() - t0
            sleep = dt_target - dt
            if sleep > 0:
                time.sleep(sleep)
    
    except KeyboardInterrupt:
        print("\n制御を停止中...")
    finally:
        try:
            odrv0.axis0.controller.input_torque = 0.0
            odrv1.axis0.controller.input_torque = 0.0
            odrv0.axis0.requested_state = AxisState.IDLE
            odrv1.axis0.requested_state = AxisState.IDLE
        except Exception:
            pass
    
        # ---- CSV保存 ----
        os.makedirs(CSV_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        profile_slug = sanitize_label_for_filename(get_profile_label())
        csv_filename = os.path.join(CSV_DIR, f"{DATA_FILENAME_PREFIX}_{profile_slug}_{timestamp}.csv")
        profile_name = get_active_profile_name()
        profile_settings_text = _format_profile_settings(profile_name, REFERENCE_PROFILE[profile_name])
        theta_ref_values = np.asarray(data_log['theta_ref'], dtype=float)
        command_values_for_meta = resolve_command_values(theta_ref_values, theta_ref_values)
        command_values_text = ", ".join(format_command_value(v) for v in command_values_for_meta) or "-"
        control_mode_text = CONTROL_MODE
        if CONTROL_MODE == 'per_motor_pid':
            outer_state = 'on' if ENABLE_OUTER_PID_IN_PER_MOTOR else 'off'
            control_mode_text = f"{CONTROL_MODE}(outer={outer_state})"
    
        def _fmt_pid(cfg):
            return f"kp={cfg['kp']},ki={cfg['ki']},kd={cfg['kd']},max={cfg.get('max_output', 0.0)}"
    
        nullspace_text = "disabled"
        freeze_text = (
            f"motor={FREEZE_CONFIG.get('motor_to_freeze', 'None')},"
            f"kp={FREEZE_CONFIG.get('kp', 0.0)},"
            f"kd={FREEZE_CONFIG.get('kd', 0.0)}"
        )
        torque_limits_text = _format_vector([
            SAFETY_CONFIG.get('max_torque0', 0.0),
            SAFETY_CONFIG.get('max_torque1', 0.0),
        ])
        metadata_lines = [
            f"Profile={profile_name}",
            f"ProfileLabel={get_profile_label()}",
            f"CommandValues={command_values_text}",
            f"Settings={profile_settings_text}",
            f"ControlMode={control_mode_text}",
            f"OutputPID={_fmt_pid(OUTPUT_PID)}",
            f"MotorPID_motor0={_fmt_pid(MOTOR_PID['motor0'])}",
            f"MotorPID_motor1={_fmt_pid(MOTOR_PID['motor1'])}",
            f"PIDDerivativeMode={PID_DERIVATIVE_MODE}",
            f"PIDDerivativeFilterAlpha={PID_DERIVATIVE_FILTER_ALPHA}",
            f"Nullspace={nullspace_text}",
            f"Freeze={freeze_text}",
            f"TorqueLimits={torque_limits_text}",
        ]
    
        with open(csv_filename, 'w', newline='') as f:
            for line in metadata_lines:
                f.write(f"# {line}\n")
            w = csv.writer(f)
            w.writerow([
                'time',
                'motor0_pos','motor0_vel','motor0_torque',
                'motor1_pos','motor1_vel','motor1_torque',
                'output_pos','output_vel',
                'theta_ref','theta_ctrl','tau_out'
            ])
            for i in range(len(data_log['time'])):
                w.writerow([
                    data_log['time'][i],
                    data_log['motor0']['pos'][i], data_log['motor0']['vel'][i], data_log['motor0']['torque'][i],
                    data_log['motor1']['pos'][i], data_log['motor1']['vel'][i], data_log['motor1']['torque'][i],
                    data_log['output']['pos'][i], data_log['output']['vel'][i],
                    data_log['theta_ref'][i], data_log['theta_ctrl'][i], data_log['tau_out'][i]
                ])
        print(f"データ保存完了: {csv_filename}")
    
        # ---- 可視化 ----
        try:
            print("\n=== 応答解析とグラフ表示 ===")
            final_graph_path, final_csv_path = analyze_and_plot_response(csv_filename)
            print("\n=== 最終的なファイル状況 ===")
            print(f"CSV: {final_csv_path if final_csv_path else '削除済み'}")
            print(f"FIG: {final_graph_path if final_graph_path else '削除済み'}")
        except Exception as e:
            print(f"応答解析エラー: {e}")
            print("手動でCSVファイルを確認してください。")
        print("制御終了")
    
    
if __name__ == '__main__':
    main()
