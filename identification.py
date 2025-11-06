from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import yaml


@dataclass
class IdentificationResult:
    inertia: float
    damping: float
    residual_norm: float


def lowpass_filter(data: np.ndarray, alpha: float) -> np.ndarray:
    """Simple first-order low-pass filter."""
    filtered = np.empty_like(data, dtype=float)
    acc = float(data[0])
    for idx, value in enumerate(data):
        acc += alpha * (float(value) - acc)
        filtered[idx] = acc
    return filtered


def compute_velocity_from_position(time: np.ndarray, position: np.ndarray) -> np.ndarray:
    time = np.asarray(time, dtype=float)
    position = np.asarray(position, dtype=float)
    dt = np.diff(time)
    if np.any(dt <= 0):
        raise ValueError("Time vector must be strictly increasing.")
    dt_mean = float(np.mean(dt))
    velocity = np.gradient(position, dt_mean)
    return velocity


def estimate_inertia_damping(
    time: np.ndarray,
    velocity: np.ndarray,
    torque: np.ndarray,
    filter_alpha: float = 0.1,
) -> IdentificationResult:
    time = np.asarray(time, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    torque = np.asarray(torque, dtype=float)

    dt = np.diff(time)
    if np.any(dt <= 0):
        raise ValueError("Time vector must be strictly increasing.")
    dt_mean = float(np.mean(dt))

    omega = lowpass_filter(velocity, filter_alpha)
    omega_dot = np.gradient(omega, dt_mean)

    Phi = np.column_stack((omega_dot, omega))
    theta, residuals, _, _ = np.linalg.lstsq(Phi, torque, rcond=None)
    inertia, damping = theta
    residual_norm = float(residuals[0]) if residuals.size else float(
        np.linalg.norm(Phi @ theta - torque)
    )
    return IdentificationResult(float(inertia), float(damping), residual_norm)


def identify_from_csv(
    path: Path,
    velocity_column: str = "output_vel",
    torque_column: str = "tau_out",
    filter_alpha: float = 0.1,
    position_column: str | None = None,
    velocity_min_std: float = 1e-6,
) -> IdentificationResult:
    df = pd.read_csv(path, comment="#")
    if "time" not in df.columns:
        raise KeyError("CSV file must contain 'time' column.")
    if velocity_column not in df.columns or torque_column not in df.columns:
        raise KeyError("CSV file missing required columns for identification.")
    time = df["time"].to_numpy()
    torque = df[torque_column].to_numpy()
    velocity = df[velocity_column].to_numpy()
    if np.std(velocity) <= float(velocity_min_std):
        if position_column is None or position_column not in df.columns:
            raise ValueError(
                "Velocity column is nearly constant; provide position_column to derive velocity."
            )
        position = df[position_column].to_numpy()
        velocity = compute_velocity_from_position(time, position)
    return estimate_inertia_damping(
        time,
        velocity,
        torque,
        filter_alpha=filter_alpha,
    )


def _format_result(result: IdentificationResult) -> str:
    return (
        f"inertia: {result.inertia:.6g}\n"
        f"damping: {result.damping:.6g}\n"
        f"residual_norm: {result.residual_norm:.6g}"
    )


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _write_result_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _merge_with_defaults(defaults: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(defaults)
    merged.update(override)
    return merged


def _run_from_config(config_path: Path) -> int:
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return 1
    config = _load_yaml(config_path)
    tasks_cfg: Iterable[Dict[str, Any]]
    if "tasks" in config and isinstance(config["tasks"], Iterable):
        base_defaults = {k: v for k, v in config.items() if k != "tasks"}
        tasks_cfg = (
            _merge_with_defaults(base_defaults, task_cfg if isinstance(task_cfg, dict) else {})
            for task_cfg in config["tasks"]
        )
    else:
        tasks_cfg = [config]

    base_dir = config_path.parent
    overall_success = True

    for idx, task_cfg in enumerate(tasks_cfg, start=1):
        try:
            csv_value = task_cfg.get("csv_path")
            if not csv_value:
                raise KeyError("csv_path is required in task configuration.")
            csv_path = _resolve_path(base_dir, csv_value)
            velocity_column = task_cfg.get("velocity_column", "output_vel")
            torque_column = task_cfg.get("torque_column", "tau_out")
            position_column = task_cfg.get("position_column")
            filter_alpha = float(task_cfg.get("filter_alpha", 0.1))
            velocity_min_std = float(task_cfg.get("velocity_min_std", 1e-6))

            result = identify_from_csv(
                csv_path,
                velocity_column=velocity_column,
                torque_column=torque_column,
                filter_alpha=filter_alpha,
                position_column=position_column,
                velocity_min_std=velocity_min_std,
            )

            task_name = task_cfg.get("name") or f"task_{idx}"
            print(f"[{task_name}]")
            print(_format_result(result))

            output_value = task_cfg.get("output_file")
            if output_value:
                output_path = _resolve_path(base_dir, output_value)
                payload = {
                    "task": task_name,
                    "csv_path": str(csv_path),
                    "velocity_column": velocity_column,
                    "torque_column": torque_column,
                    "position_column": position_column,
                    "filter_alpha": filter_alpha,
                    "velocity_min_std": velocity_min_std,
                    "inertia": result.inertia,
                    "damping": result.damping,
                    "residual_norm": result.residual_norm,
                }
                _write_result_yaml(output_path, payload)
                print(f"[{task_name}] Results written to {output_path}")

        except Exception as exc:
            overall_success = False
            print(f"[task {idx}] Identification failed: {exc}")

    return 0 if overall_success else 1


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Estimate equivalent inertia and damping from logged CSV data."
    )
    parser.add_argument("csv_path", type=Path, nargs="?", help="Path to the CSV log file.")
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML configuration describing identification tasks (default: config/identification_eval.yaml).",
    )
    parser.add_argument(
        "--velocity-column",
        default="output_vel",
        help="Column name for angular velocity (default: output_vel).",
    )
    parser.add_argument(
        "--position-column",
        help="Column name for angular position used to derive velocity when needed.",
    )
    parser.add_argument(
        "--torque-column",
        default="tau_out",
        help="Column name for output torque (default: tau_out).",
    )
    parser.add_argument(
        "--filter-alpha",
        type=float,
        default=0.1,
        help="Low-pass filter alpha for velocity preprocessing (default: 0.1).",
    )
    parser.add_argument(
        "--velocity-min-std",
        type=float,
        default=1e-6,
        help="Threshold on velocity standard deviation to trigger position-based differentiation.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Optional path to write identification results as YAML when using CSV arguments.",
    )
    args = parser.parse_args(argv)

    if args.csv_path is not None:
        try:
            result = identify_from_csv(
                Path(args.csv_path),
                velocity_column=args.velocity_column,
                torque_column=args.torque_column,
                filter_alpha=args.filter_alpha,
                position_column=args.position_column,
                velocity_min_std=float(args.velocity_min_std),
            )
        except Exception as exc:  # pragma: no cover - CLI surface
            print(f"Identification failed: {exc}", file=sys.stderr)
            return 1

        print(_format_result(result))
        if args.output_file:
            payload = {
                "csv_path": str(Path(args.csv_path)),
                "velocity_column": args.velocity_column,
                "position_column": args.position_column,
                "torque_column": args.torque_column,
                "filter_alpha": float(args.filter_alpha),
                "velocity_min_std": float(args.velocity_min_std),
                "inertia": result.inertia,
                "damping": result.damping,
                "residual_norm": result.residual_norm,
            }
            _write_result_yaml(args.output_file, payload)
            print(f"Results written to {args.output_file}")
        return 0

    default_config = (
        args.config
        if args.config is not None
        else Path(__file__).parent / "config" / "identification_eval.yaml"
    )
    return _run_from_config(default_config)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
