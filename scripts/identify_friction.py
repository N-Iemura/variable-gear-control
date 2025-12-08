#!/usr/bin/env python3
"""Friction identification for the output shaft using phase A/B/C logs."""

from __future__ import annotations

import argparse
import glob
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

plt.rcParams.update({"font.family": "Times New Roman"})


@dataclass
class FrictionResult:
    tau_c: float
    damping: float
    rmse: float
    used_samples: int
    total_samples: int


@dataclass
class IdentificationData:
    time: np.ndarray
    omega: np.ndarray
    omega_dot: np.ndarray
    tau_out: np.ndarray
    tau_model: np.ndarray
    mask: np.ndarray


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


def identify_friction(
    df: pd.DataFrame,
    time_column: str,
    omega_column: str,
    tau_out: np.ndarray,
    omega_dot_thresh: float,
    filter_alpha: float,
) -> tuple[FrictionResult, IdentificationData]:
    time = df[time_column].to_numpy(dtype=float)
    dt = np.diff(time)
    if np.any(dt <= 0):
        raise ValueError("Time must be strictly increasing after concatenation.")
    dt_mean = float(np.mean(dt))

    omega_raw = df[omega_column].to_numpy(dtype=float)
    omega = lowpass_filter(omega_raw, filter_alpha)
    omega_dot = np.gradient(omega, dt_mean)

    mask = np.abs(omega_dot) < omega_dot_thresh
    if mask.sum() < 3:
        raise ValueError("Too few samples remain after steady-state filtering.")

    design = np.column_stack((np.sign(omega[mask]), omega[mask]))
    y = tau_out[mask]
    theta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    tau_c, damping = theta
    rmse = float(np.sqrt(np.mean((design @ theta - y) ** 2)))

    tau_model = tau_c * np.sign(omega) + damping * omega
    result = FrictionResult(
        tau_c=float(tau_c),
        damping=float(damping),
        rmse=rmse,
        used_samples=int(mask.sum()),
        total_samples=int(len(df)),
    )
    ident_data = IdentificationData(
        time=time,
        omega=omega,
        omega_dot=omega_dot,
        tau_out=tau_out,
        tau_model=tau_model,
        mask=mask,
    )
    return result, ident_data


def _default_plot_path(first_input: Path, stem_suffix: str) -> Path:
    parent = first_input.parent
    if parent.name.lower() == "csv":
        base_dir = parent.parent / "identification"
    else:
        base_dir = parent / "identification"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{first_input.stem}_{stem_suffix}.pdf"


def plot_results(
    result: FrictionResult,
    data: IdentificationData,
    save_path: Path | None,
    show: bool,
) -> tuple[Path, Path]:
    main_path = save_path or _default_plot_path(Path("csv"), "friction")
    main_path.parent.mkdir(parents=True, exist_ok=True)
    scatter_path = main_path.with_name(f"{main_path.stem}_fit{main_path.suffix}")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(data.time, data.tau_out, label=r"$\tau_{\mathrm{out}}$")
    axes[0].plot(
        data.time,
        data.tau_model,
        "--",
        label=r"$\hat{\tau}_c \mathrm{sgn}(\omega) + \hat{B}\omega$",
    )
    axes[0].set_ylabel(r"$\tau$ [Nm]")
    axes[0].legend(loc="upper right")

    omega_deg = data.omega * (180.0 / np.pi)
    omega_dot_deg = data.omega_dot * (180.0 / np.pi)
    axes[1].plot(data.time, omega_deg, label=r"$\omega_{\mathrm{out}}$ [deg/s]")
    ax_acc = axes[1].twinx()
    ax_acc.plot(
        data.time,
        omega_dot_deg,
        color="tab:orange",
        alpha=0.7,
        label=r"$\dot{\omega}$ [deg/s$^2$]",
    )
    axes[1].set_ylabel(r"$\omega$ [deg/s]")
    ax_acc.set_ylabel(r"$\dot{\omega}$ [deg/s$^2$]")
    axes[1].legend(loc="upper left")
    ax_acc.legend(loc="upper right")

    axes[2].plot(data.time, data.tau_out - data.tau_model, label="residual")
    axes[2].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[2].set_ylabel("Residual [Nm]")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend(loc="upper right")

    for ax in axes:
        ax.grid(False)
        ax.tick_params(axis="both", direction="in", length=6, width=0.9)

    scatter_fig, scatter_ax = plt.subplots(figsize=(5.5, 4.2))
    scatter_ax.scatter(
        data.omega,
        data.tau_out,
        s=18,
        alpha=0.35,
        label="all samples",
        edgecolor="none",
    )
    scatter_ax.scatter(
        data.omega[data.mask],
        data.tau_out[data.mask],
        s=24,
        alpha=0.7,
        label="steady samples",
        edgecolor="none",
    )
    omega_line = np.linspace(
        float(np.min(data.omega)),
        float(np.max(data.omega)),
        200,
    )
    tau_line = result.tau_c * np.sign(omega_line) + result.damping * omega_line
    scatter_ax.plot(
        omega_line,
        tau_line,
        color="black",
        linewidth=1.3,
        label="fit",
    )
    scatter_ax.set_xlabel(r"$\omega_{\mathrm{out}}$ [rad/s]")
    scatter_ax.set_ylabel(r"$\tau_{\mathrm{out}}$ [Nm]")
    scatter_ax.legend(loc="best")
    scatter_ax.tick_params(axis="both", direction="in", length=6, width=0.9)
    scatter_fig.tight_layout()

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(main_path, dpi=300, bbox_inches="tight")
    scatter_fig.savefig(scatter_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    plt.close(scatter_fig)
    return main_path, scatter_path


def _write_outputs(
    result: FrictionResult,
    df: pd.DataFrame,
    ident_data: IdentificationData,
    output_csv: Path | None,
    output_yaml: Path | None,
    time_column: str,
    omega_column: str,
    omega_dot_thresh: float,
) -> None:
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        steady = pd.DataFrame(
            {
                "time": ident_data.time[ident_data.mask],
                "omega_out": ident_data.omega[ident_data.mask],
                "tau_out": ident_data.tau_out[ident_data.mask],
            }
        )
        steady.to_csv(output_csv, index=False)
        print(f"Exported steady-state samples to {output_csv}")

    if output_yaml is not None:
        if yaml is None:
            print(
                f"PyYAML is not installed; skipping YAML output ({output_yaml}). "
                "Install with `pip install pyyaml` if YAML export is needed.",
                file=sys.stderr,
            )
        else:
            output_yaml.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "tau_c_out": result.tau_c,
                "B_out": result.damping,
                "rmse": result.rmse,
                "used_samples": result.used_samples,
                "total_samples": result.total_samples,
                "omega_dot_threshold": omega_dot_thresh,
                "time_column": time_column,
                "omega_column": omega_column,
            }
            with output_yaml.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False)
            print(f"Saved identification summary to {output_yaml}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate Coulomb and viscous friction on the output shaft from phase A/B/C logs. "
            "CSV files can contain comment lines starting with '#'."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="CSV files or glob patterns (e.g., csv/phaseA_*.csv csv/phaseB_*.csv).",
    )
    parser.add_argument("--time-column", default="time", help="Time column name (default: time).")
    parser.add_argument("--omega-column", default="omega_out", help="Output velocity column (rad/s).")
    parser.add_argument("--phi-column", default="phi", help="Variable gear state column name (default: phi).")
    parser.add_argument("--i1-column", default="i1", help="Motor 1 current column name (A).")
    parser.add_argument("--i2-column", default="i2", help="Motor 2 current column name (A).")
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
        default=0.0,
        help="Low-pass filter alpha for omega_out before differentiation (default: 0.0, disabled).",
    )
    parser.add_argument(
        "--omega-dot-thresh",
        type=float,
        default=0.2,
        help="Threshold on |omega_dot| for steady-state selection [rad/s^2].",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("csv/ident_for_friction.csv"),
        help="Path to export steady-state samples (default: csv/ident_for_friction.csv).",
    )
    parser.add_argument(
        "--output-yaml",
        type=Path,
        default=Path("csv/ident_friction_result.yaml"),
        help="Path to export identification summary (default: csv/ident_friction_result.yaml).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Path to save the main PDF plot (default: identification/<stem>_friction.pdf).",
    )
    parser.add_argument("--show", action="store_true", help="Show plots interactively.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_paths = _expand_inputs(args.inputs)
    required = {args.time_column, args.omega_column, args.i1_column, args.i2_column, args.phi_column}
    df = _load_csvs(input_paths, args.time_column, required)

    table = None
    if args.gear_table is not None:
        table = pd.read_csv(args.gear_table)
    gear = GearModel(args.n1_coeffs, args.n2_coeffs, table)

    tau_out = _compute_tau_out(df, gear, args.kt1, args.kt2, args.i1_column, args.i2_column, args.phi_column)
    result, ident_data = identify_friction(
        df,
        args.time_column,
        args.omega_column,
        tau_out,
        omega_dot_thresh=float(args.omega_dot_thresh),
        filter_alpha=float(args.filter_alpha),
    )

    first_input = input_paths[0]
    save_path = args.save or _default_plot_path(first_input, "friction")
    main_path, scatter_path = plot_results(result, ident_data, save_path, args.show)

    _write_outputs(
        result,
        df,
        ident_data,
        args.output_csv,
        args.output_yaml,
        args.time_column,
        args.omega_column,
        float(args.omega_dot_thresh),
    )

    print(
        f"tau_c_out = {result.tau_c:.6g} Nm, "
        f"B_out = {result.damping:.6g} Nm*s/rad, "
        f"RMSE = {result.rmse:.6g} Nm "
        f"({result.used_samples}/{result.total_samples} samples used)",
    )
    print(f"Saved plot to {main_path}")
    print(f"Saved fit plot to {scatter_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
