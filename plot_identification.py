#!/usr/bin/env python3
"""同定実験ログを解析して J/B を推定し、結果を可視化するスクリプト。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np

TURN_TO_RAD = 2.0 * np.pi
TURN_TO_DEG = 360.0
RAD_TO_DEG = 180.0 / np.pi

plt.rcParams.update({"font.family": "Times New Roman"})


def _read_ident_log(path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """コメント付きCSVを読み込み、列データとメタデータを返す。"""
    metadata: Dict[str, str] = {}
    data_lines: list[str] = []
    with path.open("r", newline="") as f:
        for line in f:
            if line.startswith("#"):
                line = line[1:].strip()
                if "=" in line:
                    key, value = line.split("=", 1)
                    metadata[key.strip()] = value.strip()
                continue
            if line.strip():
                data_lines.append(line)
    if not data_lines:
        raise ValueError(f"{path} にデータ行が見つかりません。")

    reader = csv.DictReader(data_lines)
    columns: Dict[str, list[float]] = {name: [] for name in (reader.fieldnames or [])}
    for row in reader:
        for key, value in row.items():
            columns[key].append(float(value))
    series = {name: np.asarray(values, dtype=float) for name, values in columns.items()}
    return series, metadata


def _resolve(series: Dict[str, np.ndarray], candidates: Iterable[str]) -> np.ndarray:
    for name in candidates:
        if name in series:
            return series[name]
    raise KeyError(f"列 {list(candidates)} が見つかりません。候補: {list(series.keys())}")


def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr
    kernel = np.ones(int(window), dtype=float)
    kernel /= kernel.sum()
    return np.convolve(arr, kernel, mode="same")


def _fit_jb(time: np.ndarray, omega_turns: np.ndarray, tau_out: np.ndarray, smooth: int) -> Dict[str, np.ndarray]:
    omega_rad = omega_turns * TURN_TO_RAD
    if smooth > 1:
        omega_rad = _moving_average(omega_rad, smooth)
    omega_dot = np.gradient(omega_rad, time)

    mask = np.isfinite(time) & np.isfinite(omega_rad) & np.isfinite(omega_dot) & np.isfinite(tau_out)
    if mask.sum() < 2:
        raise ValueError("有効サンプル数が不足しています。")

    design = np.column_stack((omega_dot[mask], omega_rad[mask]))
    y = tau_out[mask]
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    j_hat, b_hat = coeffs

    tau_model = j_hat * omega_dot + b_hat * omega_rad
    residual = tau_out - tau_model
    rmse = float(np.sqrt(np.mean((y - (design @ coeffs)) ** 2)))

    return {
        "J_hat": float(j_hat),
        "B_hat": float(b_hat),
        "omega_rad": omega_rad,
        "omega_dot": omega_dot,
        "tau_model": tau_model,
        "residual": residual,
        "rmse": rmse,
    }


def _format_fit_summary(fit: Dict[str, np.ndarray]) -> str:
    return (
        f"J = {fit['J_hat']:.6f} kg·m^2, "
        f"B = {fit['B_hat']:.6f} N·m·s/rad, "
        f"RMSE = {fit['rmse']:.5f} N·m"
    )


def plot_identification(
    csv_path: Path, save_path: Path | None, show: bool, smooth: int
) -> Tuple[Path, Path]:
    series, metadata = _read_ident_log(csv_path)
    time = _resolve(series, ("time",))
    tau_out = _resolve(series, ("tau_out", "theta_ctrl"))
    omega_turns = _resolve(series, ("output_vel", "theta_dot"))

    fit = _fit_jb(time, omega_turns, tau_out, smooth)
    summary = _format_fit_summary(fit)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(time, tau_out, label=r"$\tau_{\mathrm{out}}$")
    axes[0].plot(time, fit["tau_model"], "--", label=r"$J\dot{\omega} + B\omega$")
    axes[0].set_ylabel(r"$\tau$ [Nm]")
    axes[0].legend(loc="upper right")

    omega_deg = omega_turns * TURN_TO_DEG
    omega_dot_deg = fit["omega_dot"] * RAD_TO_DEG
    axes[1].plot(time, omega_deg, label=r"$\omega_{\mathrm{out}}$ [deg/s]")
    ax_acc = axes[1].twinx()
    ax_acc.plot(time, omega_dot_deg, color="tab:orange", alpha=0.7, label=r"$\dot{\omega}$ [deg/s$^2$]")
    axes[1].set_ylabel(r"$\omega$ [deg/s]")
    ax_acc.set_ylabel(r"$\dot{\omega}$ [deg/s$^2$]")
    axes[1].legend(loc="upper left")
    ax_acc.legend(loc="upper right")

    axes[2].plot(time, fit["residual"], label=r"$\tau_{\mathrm{out}} - (J\dot{\omega}+B\omega)$")
    axes[2].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[2].set_ylabel(r"Residual [Nm]")
    axes[2].set_xlabel(r"Time [s]")
    axes[2].legend(loc="upper right")

    for ax in axes:
        ax.grid(False)
        ax.tick_params(axis="both", direction="in", length=6, width=0.9)

    scatter_fig, scatter_ax = plt.subplots(figsize=(5.5, 4.2))
    scatter_ax.scatter(
        fit["omega_dot"],
        tau_out,
        s=24,
        alpha=0.5,
        label="samples",
        edgecolor="none",
    )
    scatter_ax.plot(
        fit["omega_dot"],
        fit["tau_model"],
        color="black",
        linewidth=1.3,
        label="linear fit",
    )
    scatter_ax.set_xlabel(r"$\dot{\omega}$ [rad/s$^2$]")
    scatter_ax.set_ylabel(r"$\tau_{\mathrm{out}}$ [Nm]")
    scatter_ax.legend(loc="best")
    scatter_ax.tick_params(axis="both", direction="in", length=6, width=0.9)
    scatter_fig.tight_layout()

    if save_path is None:
        parent = csv_path.parent
        if parent.name.lower() == "csv":
            base_dir = parent.parent / "identification"
        else:
            base_dir = parent / "identification"
        base_dir.mkdir(parents=True, exist_ok=True)
        main_path = base_dir / f"{csv_path.stem}_ident.pdf"
    else:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        main_path = save_path

    scatter_path = main_path.with_name(f"{main_path.stem}_fit{main_path.suffix}")

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(main_path, dpi=300, bbox_inches="tight")
    scatter_fig.savefig(scatter_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    plt.close(scatter_fig)

    print(summary)
    print("Scatter plot shows tau_out vs omega_dot; linearity indicates how well the single-inertia model fits.")

    return main_path, scatter_path


def main() -> None:
    parser = argparse.ArgumentParser(description="同定ログを可視化して J/B を推定するツール")
    parser.add_argument("csv_path", type=Path, help="同定ログCSVへのパス")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="保存先PDF（省略時はidentificationフォルダ）",
    )
    parser.add_argument("--smooth", type=int, default=5, help="速度の移動平均窓長（サンプル数）")
    parser.add_argument("--show", action="store_true", help="プロットウィンドウを表示する場合に指定")
    args = parser.parse_args()

    main_path, scatter_path = plot_identification(args.csv_path, args.save, args.show, args.smooth)
    print(f"Saved plot to {main_path}")
    print(f"Saved fit plot to {scatter_path}")


if __name__ == "__main__":
    main()
