#!/usr/bin/env python3
"""Compare inertia identification between single-motor and dual-motor drive."""

from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.family": "Times New Roman"})


@dataclass
class FitResult:
    inertia: float
    damping: float
    rmse: float
    residual_norm: float


@dataclass
class FitData:
    time: np.ndarray
    omega: np.ndarray
    omega_dot: np.ndarray
    tau_out: np.ndarray
    tau_model: np.ndarray


@dataclass
class ModeOutcome:
    label: str
    result: FitResult
    data: FitData
    source_files: list[Path]


def lowpass_filter(data: np.ndarray, alpha: float) -> np.ndarray:
    """Simple first-order low-pass filter."""
    if alpha <= 0:
        return np.asarray(data, dtype=float)
    filtered = np.empty_like(data, dtype=float)
    acc = float(data[0])
    for idx, value in enumerate(data):
        acc += alpha * (float(value) - acc)
        filtered[idx] = acc
    return filtered


class GearModel:
    """Evaluates N1(phi) and N2(phi) either from polynomials or a lookup table."""

    def __init__(
        self,
        n1_coeffs: Sequence[float] | None = None,
        n2_coeffs: Sequence[float] | None = None,
        table: pd.DataFrame | None = None,
    ) -> None:
        self.n1_coeffs = np.asarray(n1_coeffs if n1_coeffs is not None else [1.0], dtype=float)
        self.n2_coeffs = np.asarray(n2_coeffs if n2_coeffs is not None else [1.0], dtype=float)
        if table is not None and not {"phi", "n1", "n2"} <= set(table.columns):
            raise ValueError("Gear table must contain phi, n1, n2 columns.")
        self.table = table.sort_values("phi") if table is not None else None

    def _interp(self, phi: np.ndarray, column: str, coeffs: np.ndarray) -> np.ndarray:
        if self.table is not None:
            phi_table = self.table["phi"].to_numpy(dtype=float)
            values = self.table[column].to_numpy(dtype=float)
            return np.interp(phi, phi_table, values)
        return np.polyval(coeffs, phi)

    def n1(self, phi: np.ndarray) -> np.ndarray:
        return self._interp(phi, "n1", self.n1_coeffs)

    def n2(self, phi: np.ndarray) -> np.ndarray:
        return self._interp(phi, "n2", self.n2_coeffs)


def _expand_inputs(patterns: Sequence[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        matched = [Path(p) for p in glob.glob(pattern)]
        if matched:
            files.extend(matched)
        else:
            candidate = Path(pattern)
            if candidate.exists():
                files.append(candidate)
    unique = list(dict.fromkeys(files))  # preserve order
    if not unique:
        raise FileNotFoundError("No CSV files matched the given patterns.")
    return unique


def _load_csvs(
    paths: Sequence[Path],
    time_column: str,
    required_columns: Iterable[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    time_offset = 0.0
    for path in paths:
        df = pd.read_csv(path, comment="#")
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise KeyError(f"{path} is missing required columns: {missing}")
        df = df.copy()
        time = df[time_column].to_numpy(dtype=float)
        if len(time) < 2:
            raise ValueError(f"{path} does not contain enough samples.")
        dt = np.diff(time)
        if np.any(dt <= 0):
            raise ValueError(f"Time column is not strictly increasing in {path}.")
        df[time_column] = time + time_offset
        time_offset = float(df[time_column].iloc[-1] + np.mean(dt))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _compute_tau_out(
    df: pd.DataFrame,
    gear: GearModel,
    kt1: float,
    kt2: float,
    i1_column: str,
    i2_column: str,
    phi_column: str,
) -> np.ndarray:
    tau1 = kt1 * df[i1_column].to_numpy(dtype=float)
    tau2 = kt2 * df[i2_column].to_numpy(dtype=float)
    phi = df[phi_column].to_numpy(dtype=float)
    n1 = gear.n1(phi)
    n2 = gear.n2(phi)
    return n1 * tau1 + n2 * tau2


def _fit_inertia(
    time: np.ndarray,
    omega_raw: np.ndarray,
    tau_out: np.ndarray,
    filter_alpha: float,
) -> ModeOutcome:
    dt = np.diff(time)
    if np.any(dt <= 0):
        raise ValueError("Time must be strictly increasing after concatenation.")
    dt_mean = float(np.mean(dt))

    omega = lowpass_filter(omega_raw, filter_alpha)
    omega_dot = np.gradient(omega, dt_mean)

    design = np.column_stack((omega_dot, omega))
    theta, residuals, _, _ = np.linalg.lstsq(design, tau_out, rcond=None)
    inertia, damping = theta
    if residuals.size:
        residual_norm = float(residuals[0])
    else:
        residual_norm = float(np.linalg.norm(design @ theta - tau_out))
    tau_model = design @ theta
    rmse = float(np.sqrt(np.mean((tau_model - tau_out) ** 2)))

    result = FitResult(float(inertia), float(damping), rmse, residual_norm)
    data = FitData(time, omega, omega_dot, tau_out, tau_model)
    return result, data


def _default_save_prefix(first_input: Path) -> Path:
    parent = first_input.parent
    if parent.name.lower() == "csv":
        base_dir = parent.parent / "identification"
    else:
        base_dir = parent / "identification"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / "compare_inertia"


def _plot_mode(outcome: ModeOutcome, save_path: Path, show: bool) -> Path:
    time = outcome.data.time
    tau_out = outcome.data.tau_out
    tau_model = outcome.data.tau_model
    omega = outcome.data.omega
    omega_dot = outcome.data.omega_dot

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(time, tau_out, label=r"$\tau_{\mathrm{out}}$")
    axes[0].plot(time, tau_model, "--", label=r"$\hat{J}\dot{\omega} + \hat{B}\omega$")
    axes[0].set_ylabel(r"$\tau$ [Nm]")
    axes[0].legend(loc="upper right")

    omega_deg = omega * (180.0 / np.pi)
    omega_dot_deg = omega_dot * (180.0 / np.pi)
    axes[1].plot(time, omega_deg, label=r"$\omega_{\mathrm{out}}$ [deg/s]")
    ax_acc = axes[1].twinx()
    ax_acc.plot(
        time,
        omega_dot_deg,
        color="tab:orange",
        alpha=0.7,
        label=r"$\dot{\omega}$ [deg/s$^2$]",
    )
    axes[1].set_ylabel(r"$\omega$ [deg/s]")
    ax_acc.set_ylabel(r"$\dot{\omega}$ [deg/s$^2$]")
    axes[1].legend(loc="upper left")
    ax_acc.legend(loc="upper right")

    axes[2].plot(time, tau_out - tau_model, label="residual")
    axes[2].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[2].set_ylabel("Residual [Nm]")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend(loc="upper right")

    for ax in axes:
        ax.grid(False)
        ax.tick_params(axis="both", direction="in", length=6, width=0.9)

    fig.suptitle(
        f"{outcome.label}: J={outcome.result.inertia:.5f}, B={outcome.result.damping:.5f}, RMSE={outcome.result.rmse:.5f}"
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path


def _plot_scatter(outcomes: Sequence[ModeOutcome], save_path: Path, show: bool) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue", "tab:orange"])

    for idx, outcome in enumerate(outcomes):
        color = colors[idx % len(colors)]
        tau_eff = outcome.data.tau_out - outcome.result.damping * outcome.data.omega
        ax.scatter(
            outcome.data.omega_dot,
            tau_eff,
            s=18,
            alpha=0.45,
            color=color,
            edgecolor="none",
            label=f"{outcome.label} (tau - B*omega)",
        )
        omega_dot_line = np.linspace(
            float(np.min(outcome.data.omega_dot)),
            float(np.max(outcome.data.omega_dot)),
            300,
        )
        tau_line = outcome.result.inertia * omega_dot_line
        ax.plot(
            omega_dot_line,
            tau_line,
            color=color,
            linewidth=1.4,
            label=f"{outcome.label} fit slope (J)",
        )

    ax.set_xlabel(r"$\dot{\omega}_{\mathrm{out}}$ [rad/s$^2$]")
    ax.set_ylabel(r"$\tau_{\mathrm{out}} - B\omega$ [Nm]")
    ax.legend(loc="best")
    ax.tick_params(axis="both", direction="in", length=6, width=0.9)
    ax.grid(False)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path


def _run_mode(
    label: str,
    inputs: Sequence[str],
    gear: GearModel,
    time_column: str,
    omega_column: str,
    phi_column: str,
    i1_column: str,
    i2_column: str,
    kt1: float,
    kt2: float,
    filter_alpha: float,
) -> ModeOutcome:
    paths = _expand_inputs(inputs)
    required = {time_column, omega_column, phi_column, i1_column, i2_column}
    df = _load_csvs(paths, time_column, required)
    tau_out = _compute_tau_out(df, gear, kt1, kt2, i1_column, i2_column, phi_column)
    time = df[time_column].to_numpy(dtype=float)
    omega = df[omega_column].to_numpy(dtype=float)

    result, data = _fit_inertia(time, omega, tau_out, filter_alpha)
    return ModeOutcome(label, result, data, paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify inertia/damping for single-motor vs dual-motor drive to confirm differences."
    )
    parser.add_argument(
        "--single",
        nargs="+",
        required=True,
        dest="single_inputs",
        help="CSV files or globs for single-motor operation (e.g., csv/phaseA_single*.csv).",
    )
    parser.add_argument(
        "--dual",
        nargs="+",
        required=True,
        dest="dual_inputs",
        help="CSV files or globs for dual-motor operation (e.g., csv/phaseA_dual*.csv).",
    )
    parser.add_argument("--time-column", default="time", help="Time column name (default: time).")
    parser.add_argument("--omega-column", default="omega_out", help="Output velocity column (rad/s).")
    parser.add_argument("--phi-column", default="phi", help="Variable gear state column (default: phi).")
    parser.add_argument("--i1-column", default="i1", help="Motor 1 current column (A).")
    parser.add_argument("--i2-column", default="i2", help="Motor 2 current column (A).")
    parser.add_argument("--kt1", type=float, default=1.0, help="Torque constant for motor 1 [Nm/A].")
    parser.add_argument("--kt2", type=float, default=1.0, help="Torque constant for motor 2 [Nm/A].")
    parser.add_argument(
        "--n1-coeffs",
        type=float,
        nargs="+",
        default=[1.0],
        help="Polynomial coefficients for N1(phi) (highest order first, default: 1.0).",
    )
    parser.add_argument(
        "--n2-coeffs",
        type=float,
        nargs="+",
        default=[1.0],
        help="Polynomial coefficients for N2(phi) (highest order first, default: 1.0).",
    )
    parser.add_argument(
        "--gear-table",
        type=Path,
        help="Optional CSV table with phi,n1,n2 columns for interpolation instead of polynomials.",
    )
    parser.add_argument(
        "--filter-alpha",
        type=float,
        default=0.05,
        help="Low-pass alpha for omega_out before differentiation (default: 0.05; 0 disables).",
    )
    parser.add_argument(
        "--save-prefix",
        type=Path,
        help="Prefix for output PDFs (default: identification/compare_inertia).",
    )
    parser.add_argument("--show", action="store_true", help="Show plots interactively.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    table = None
    if args.gear_table is not None:
        table = pd.read_csv(args.gear_table)
    gear = GearModel(args.n1_coeffs, args.n2_coeffs, table)

    single_outcome = _run_mode(
        "single-motor",
        args.single_inputs,
        gear,
        args.time_column,
        args.omega_column,
        args.phi_column,
        args.i1_column,
        args.i2_column,
        args.kt1,
        args.kt2,
        float(args.filter_alpha),
    )
    dual_outcome = _run_mode(
        "dual-motor",
        args.dual_inputs,
        gear,
        args.time_column,
        args.omega_column,
        args.phi_column,
        args.i1_column,
        args.i2_column,
        args.kt1,
        args.kt2,
        float(args.filter_alpha),
    )

    save_prefix = args.save_prefix or _default_save_prefix(single_outcome.source_files[0])
    single_path = save_prefix.with_name(f"{save_prefix.stem}_single.pdf")
    dual_path = save_prefix.with_name(f"{save_prefix.stem}_dual.pdf")
    scatter_path = save_prefix.with_name(f"{save_prefix.stem}_scatter.pdf")

    _plot_mode(single_outcome, single_path, args.show)
    _plot_mode(dual_outcome, dual_path, args.show)
    _plot_scatter([single_outcome, dual_outcome], scatter_path, args.show)

    print(f"[single-motor] J={single_outcome.result.inertia:.6g}, B={single_outcome.result.damping:.6g}, RMSE={single_outcome.result.rmse:.4g}")
    print(f"[dual-motor]   J={dual_outcome.result.inertia:.6g}, B={dual_outcome.result.damping:.6g}, RMSE={dual_outcome.result.rmse:.4g}")
    print(f"Saved plots: {single_path}, {dual_path}, {scatter_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
