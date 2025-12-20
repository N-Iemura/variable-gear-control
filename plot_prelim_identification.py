#!/usr/bin/env python3
"""Plot preliminary identification experiment logs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
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
        raise ValueError(f"No data rows found in {path}")
    reader = csv.DictReader(data_lines)
    if not reader.fieldnames:
        raise ValueError("CSV header is missing.")
    columns: Dict[str, list[float]] = {name: [] for name in reader.fieldnames}
    for row in reader:
        for key, value in row.items():
            try:
                columns[key].append(float(value))
            except (TypeError, ValueError):
                columns[key].append(float("nan"))
    series = {name: np.asarray(values, dtype=float) for name, values in columns.items()}
    return series, metadata


def _resolve(series: Dict[str, np.ndarray], candidates: Iterable[str]) -> np.ndarray:
    for name in candidates:
        if name in series:
            return series[name]
    raise KeyError(f"Missing required columns: {list(candidates)}")


def _resolve_optional(series: Dict[str, np.ndarray], candidates: Iterable[str]) -> np.ndarray | None:
    for name in candidates:
        if name in series:
            return series[name]
    return None


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_experiment(exp_arg: str | None, metadata: Dict[str, str], path: Path) -> str:
    if exp_arg:
        return "exp5" if exp_arg == "motor2_only" else exp_arg
    meta = metadata.get("Experiment")
    if meta == "motor2_only":
        return "exp5"
    if meta in {"exp1", "exp2", "exp3", "exp4", "exp5"}:
        return meta
    stem = path.stem.lower()
    for candidate in ("exp1", "exp2", "exp3", "exp4", "exp5"):
        if candidate in stem:
            return candidate
    raise ValueError("Experiment type could not be inferred. Use --experiment.")


def _steady_mask(series: Dict[str, np.ndarray]) -> np.ndarray:
    steady = _resolve_optional(series, ("steady",))
    if steady is None:
        return np.ones(len(next(iter(series.values()))), dtype=bool)
    return steady > 0.5


def _default_save_path(csv_path: Path, exp_name: str) -> Path:
    parent = csv_path.parent
    if parent.name.lower() == "csv":
        base_dir = parent.parent / "identification"
    else:
        base_dir = parent / "identification"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{csv_path.stem}_{exp_name}.pdf"


def _resolve_kinematic(args: argparse.Namespace, metadata: Dict[str, str]) -> Tuple[float | None, float | None]:
    a1 = args.a1 if args.a1 is not None else _parse_float(metadata.get("KinematicA1"))
    a2 = args.a2 if args.a2 is not None else _parse_float(metadata.get("KinematicA2"))
    if a1 is None or a2 is None:
        return None, None
    return float(a1), float(a2)


def _plot_exp1(
    time_s: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    w_out: np.ndarray,
    mask: np.ndarray,
    a1_theory: float | None,
    a2_theory: float | None,
) -> plt.Figure:
    X = np.column_stack((w1[mask], w2[mask]))
    y = w_out[mask]
    theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a1_hat, a2_hat = theta
    y_hat = X @ theta
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2))) if y.size else float("nan")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(time_s, w1, label="w1")
    axes[0].plot(time_s, w2, label="w2")
    axes[0].plot(time_s, w_out, label="w_out")
    axes[0].set_ylabel("rad/s")
    axes[0].legend(loc="upper right")

    axes[1].scatter(y, y_hat, s=20, alpha=0.6, label="steady samples")
    min_val = float(np.nanmin([np.min(y), np.min(y_hat)]))
    max_val = float(np.nanmax([np.max(y), np.max(y_hat)]))
    axes[1].plot([min_val, max_val], [min_val, max_val], "--", color="black", label="y=x")
    axes[1].set_xlabel("measured w_out [rad/s]")
    axes[1].set_ylabel("predicted w_out [rad/s]")
    axes[1].legend(loc="upper left")

    residual = y - y_hat
    axes[2].plot(time_s[mask], residual, label="residual")
    axes[2].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[2].set_ylabel("rad/s")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(loc="upper right")

    title = f"exp1: a1={a1_hat:.6g}, a2={a2_hat:.6g}, rmse={rmse:.4g}"
    if a1_theory is not None and a2_theory is not None:
        err_a1 = 100.0 * (a1_hat - a1_theory) / a1_theory if a1_theory != 0.0 else float("nan")
        err_a2 = 100.0 * (a2_hat - a2_theory) / a2_theory if a2_theory != 0.0 else float("nan")
        title += f" (theory a1={a1_theory:.6g}, a2={a2_theory:.6g}, err={err_a1:.2f}%,{err_a2:.2f}%)"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    print(title)
    return fig


def _plot_exp2(
    time_s: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    w_out: np.ndarray,
    step_id: np.ndarray,
    cmd_rho: np.ndarray,
    mask: np.ndarray,
    a1_theory: float | None,
    a2_theory: float | None,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(time_s, w1, label="w1")
    axes[0].plot(time_s, w2, label="w2")
    axes[0].plot(time_s, w_out, label="w_out")
    axes[0].set_ylabel("rad/s")
    axes[0].legend(loc="upper right")

    step_ids = np.unique(step_id[mask].astype(int)) if mask.any() else np.unique(step_id.astype(int))
    rho_vals = []
    i_eff_vals = []
    for sid in step_ids:
        step_mask = mask & (step_id == sid)
        if not step_mask.any():
            continue
        rho_mean = float(np.nanmean(cmd_rho[step_mask]))
        w1_mean = float(np.nanmean(w1[step_mask]))
        w_out_mean = float(np.nanmean(w_out[step_mask]))
        if abs(w_out_mean) < 1e-6:
            continue
        rho_vals.append(rho_mean)
        i_eff_vals.append(w1_mean / w_out_mean)
    rho_vals = np.asarray(rho_vals, dtype=float)
    i_eff_vals = np.asarray(i_eff_vals, dtype=float)

    axes[1].scatter(rho_vals, i_eff_vals, s=30, label="measured")
    if rho_vals.size > 0:
        rho_min, rho_max = float(np.min(rho_vals)), float(np.max(rho_vals))
    else:
        rho_min, rho_max = -1.0, 1.0
    if a1_theory is not None and a2_theory is not None:
        rho_line = np.linspace(rho_min - 0.2, rho_max + 0.2, 200)
        i_eff_theory = 1.0 / (a1_theory + a2_theory * rho_line)
        axes[1].plot(rho_line, i_eff_theory, "--", color="black", label="theory")
    axes[1].set_xlabel("rho (=w2/w1)")
    axes[1].set_ylabel("i_eff (=w1/w_out)")
    axes[1].legend(loc="best")

    axes[2].plot(rho_vals, i_eff_vals, "o-", label="i_eff")
    axes[2].set_xlabel("rho (=w2/w1)")
    axes[2].set_ylabel("i_eff")
    axes[2].legend(loc="best")

    fig.suptitle("exp2: effective reduction vs rho", fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if rho_vals.size:
        for rho_val, ieff in zip(rho_vals, i_eff_vals):
            print(f"rho={rho_val:.3g}, i_eff={ieff:.4g}")
    return fig


def _plot_exp3(
    time_s: np.ndarray,
    w1: np.ndarray,
    tau1: np.ndarray,
    mask: np.ndarray,
    axis_label: str,
    tau_label: str,
    title_prefix: str,
) -> plt.Figure:
    omega = w1[mask]
    tau = tau1[mask]
    X = np.column_stack((np.sign(omega), omega))
    theta, _, _, _ = np.linalg.lstsq(X, tau, rcond=None)
    tau_c, damping = theta
    tau_fit = tau_c * np.sign(w1) + damping * w1
    rmse = float(np.sqrt(np.mean((tau - (X @ theta)) ** 2))) if tau.size else float("nan")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    axes[0].plot(time_s, w1, label=axis_label)
    axes[0].set_ylabel("rad/s")
    axes[0].legend(loc="upper right")
    if time_s.size:
        axes[0].set_xlim(time_s.min(), time_s.max())

    axes[1].scatter(omega, tau, s=24, alpha=0.6, label="steady samples")
    axes[1].plot(w1, tau_fit, color="black", linewidth=1.2, label="fit")
    axes[1].set_xlabel(f"{axis_label} [rad/s]")
    axes[1].set_ylabel(f"{tau_label} [Nm]")
    axes[1].legend(loc="best")

    residual = tau1 - tau_fit
    axes[2].plot(time_s, residual, label="residual")
    axes[2].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[2].set_ylabel("Nm")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(loc="upper right")
    if time_s.size:
        axes[2].set_xlim(time_s.min(), time_s.max())

    title = f"{title_prefix}: tau_c={tau_c:.6g}, b={damping:.6g}, rmse={rmse:.4g}"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    print(title)
    return fig


def _plot_exp4(
    time_s: np.ndarray,
    pos_1: np.ndarray,
    pos_out: np.ndarray,
    w1: np.ndarray,
    w_out: np.ndarray,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(time_s, pos_1, label="pos_1")
    axes[0].plot(time_s, pos_out, label="pos_out")
    axes[0].set_ylabel("rad")
    axes[0].legend(loc="upper right")

    axes[1].scatter(pos_1, pos_out, s=16, alpha=0.6)
    axes[1].set_xlabel("pos_1 [rad]")
    axes[1].set_ylabel("pos_out [rad]")

    axes[2].plot(time_s, w1, label="w1")
    axes[2].plot(time_s, w_out, label="w_out")
    axes[2].set_ylabel("rad/s")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(loc="upper right")

    fig.suptitle("exp4: backlash check", fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot preliminary identification results.")
    parser.add_argument("csv_path", type=Path, help="CSV log from prelim identification.")
    parser.add_argument(
        "--experiment",
        choices=["exp1", "exp2", "exp3", "exp4", "exp5", "motor2_only"],
        default=None,
    )
    parser.add_argument("--save", type=Path, default=None, help="Output PDF path.")
    show_group = parser.add_mutually_exclusive_group()
    show_group.add_argument("--show", action="store_true", help="Show the plot window (default).")
    show_group.add_argument("--no-show", action="store_true", help="Do not show the plot window.")
    parser.add_argument("--a1", type=float, default=None, help="Theory a1 for exp2/exp1.")
    parser.add_argument("--a2", type=float, default=None, help="Theory a2 for exp2/exp1.")
    args = parser.parse_args(argv)

    series, metadata = _read_csv(args.csv_path)
    exp_name = _resolve_experiment(args.experiment, metadata, args.csv_path)
    time_s = _resolve(series, ("time_s", "time"))
    mask = _steady_mask(series)
    a1_theory, a2_theory = _resolve_kinematic(args, metadata)

    if exp_name == "exp1":
        w1 = _resolve(series, ("meas_w1_rad_s", "vel_1", "w1"))
        w2 = _resolve(series, ("meas_w2_rad_s", "vel_2", "w2"))
        w_out = _resolve(series, ("meas_w_out_rad_s", "output_vel", "w_out"))
        fig = _plot_exp1(time_s, w1, w2, w_out, mask, a1_theory, a2_theory)
    elif exp_name == "exp2":
        w1 = _resolve(series, ("meas_w1_rad_s", "vel_1", "w1"))
        w2 = _resolve(series, ("meas_w2_rad_s", "vel_2", "w2"))
        w_out = _resolve(series, ("meas_w_out_rad_s", "output_vel", "w_out"))
        step_id = _resolve(series, ("step_id",))
        cmd_rho = _resolve_optional(series, ("cmd_rho",))
        if cmd_rho is None:
            cmd_w1 = _resolve(series, ("cmd_w1_rad_s",))
            cmd_w2 = _resolve(series, ("cmd_w2_rad_s",))
            cmd_rho = np.where(np.abs(cmd_w1) > 1e-9, cmd_w2 / cmd_w1, np.nan)
        fig = _plot_exp2(
            time_s,
            w1,
            w2,
            w_out,
            step_id,
            cmd_rho,
            mask,
            a1_theory,
            a2_theory,
        )
    elif exp_name == "exp3":
        w1 = _resolve(series, ("meas_w1_rad_s", "vel_1", "w1"))
        tau1 = _resolve(
            series,
            ("tau_meas_1_Nm", "cmd_tau_1_Nm", "tau_1", "tau_meas_1"),
        )
        fig = _plot_exp3(time_s, w1, tau1, mask, "w1", "tau1", "exp3")
    elif exp_name == "exp5":
        w2 = _resolve(series, ("meas_w2_rad_s", "vel_2", "w2"))
        tau2 = _resolve(
            series,
            ("tau_meas_2_Nm", "cmd_tau_2_Nm", "tau_2", "tau_meas_2"),
        )
        fig = _plot_exp3(time_s, w2, tau2, mask, "w2", "tau2", "exp5")
    elif exp_name == "exp4":
        w1 = _resolve(series, ("meas_w1_rad_s", "vel_1", "w1"))
        w_out = _resolve(series, ("meas_w_out_rad_s", "output_vel", "w_out"))
        pos_1 = _resolve(series, ("pos_1_rad", "pos_1", "pos_1_turn"))
        pos_out = _resolve(series, ("pos_out_rad", "output_pos", "pos_out"))
        fig = _plot_exp4(time_s, pos_1, pos_out, w1, w_out)
    else:
        raise ValueError(f"Unsupported experiment: {exp_name}")

    save_path = args.save or _default_save_path(args.csv_path, exp_name)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    show = not args.no_show
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved plot: {save_path}")


if __name__ == "__main__":
    main()
