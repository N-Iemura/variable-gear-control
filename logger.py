from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None


@dataclass
class LogRecord:
    time: float
    pos_1: float
    vel_1: float
    tau_1: float
    pos_2: float
    vel_2: float
    tau_2: float
    output_pos: float
    output_vel: float
    theta_ref: float
    theta_ctrl: float
    tau_out: float
    loop_dt: float = 0.0


@dataclass
class DataLogger:
    config: Dict[str, object]
    base_path: Path = field(default_factory=lambda: Path("."))
    controller_config: Dict[str, object] = field(default_factory=dict)
    reference: object = None
    records: List[LogRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.filename_decimals = int(self.config.get("filename_decimals", 4))

    def log(
        self,
        time_stamp: float,
        pos_1: float,
        vel_1: float,
        tau_1: float,
        pos_2: float,
        vel_2: float,
        tau_2: float,
        output_pos: float,
        output_vel: float,
        reference_position: float,
        reference_control: float,
        tau_out: float,
        loop_dt: Optional[float] = None,
    ) -> None:
        record = LogRecord(
            time=time_stamp,
            pos_1=float(pos_1),
            vel_1=float(vel_1),
            tau_1=float(tau_1),
            pos_2=float(pos_2),
            vel_2=float(vel_2),
            tau_2=float(tau_2),
            output_pos=float(output_pos),
            output_vel=float(output_vel),
            theta_ref=float(reference_position),
            theta_ctrl=float(reference_control),
            tau_out=float(tau_out),
            loop_dt=float(loop_dt) if loop_dt is not None else 0.0,
        )
        self.records.append(record)

    def save(self, metadata: Optional[Dict[str, str]] = None) -> Dict[str, Optional[Path]]:
        csv_dir = self.base_path / str(self.config.get("csv_dir", "csv"))
        csv_dir.mkdir(parents=True, exist_ok=True)

        prefix = str(self.config.get("data_filename_prefix", "record"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        profile_label = self.reference.get_profile_label() if self.reference else None
        profile_slug = None
        if self.reference and profile_label:
            profile_slug = self.reference.sanitize_label_for_filename(profile_label)
        if profile_slug:
            csv_stem = f"{prefix}_{profile_slug}_{timestamp}"
        else:
            csv_stem = f"{prefix}_{timestamp}"
        csv_path = csv_dir / f"{csv_stem}.csv"

        metadata = metadata or {}

        theta_ref_series = [record.theta_ref for record in self.records]
        command_override = self.config.get("filename_command_values")
        if self.reference and theta_ref_series:
            command_values = self.reference.resolve_command_values(
                theta_ref_series,
                theta_ref_series,
                override=command_override,
                decimals=self.filename_decimals,
            )
        elif theta_ref_series:
            command_values = np.unique(
                np.round(np.asarray(theta_ref_series, dtype=float), self.filename_decimals)
            )
        else:
            command_values = np.array([0.0], dtype=float)
        command_values_text = ", ".join(
            self.reference.format_command_value(val, self.filename_decimals)
            if self.reference
            else f"{float(val):.{self.filename_decimals}f}"
            for val in command_values
        ) or "-"

        with csv_path.open("w", newline="") as f:
            for key, value in self._metadata_lines(command_values_text, metadata):
                f.write(f"# {key}={value}\n")
            writer = csv.writer(f)
            writer.writerow(
                [
                    "time",
                    "pos_1",
                    "vel_1",
                    "tau_1",
                    "pos_2",
                    "vel_2",
                    "tau_2",
                    "output_pos",
                    "output_vel",
                    "theta_ref",
                    "theta_ctrl",
                    "tau_out",
                    "loop_dt",
                ]
            )
            for record in self.records:
                writer.writerow(
                    [
                        record.time,
                        record.pos_1,
                        record.vel_1,
                        record.tau_1,
                        record.pos_2,
                        record.vel_2,
                        record.tau_2,
                        record.output_pos,
                        record.output_vel,
                        record.theta_ref,
                        record.theta_ctrl,
                        record.tau_out,
                        record.loop_dt,
                    ]
                )
        figure_path = self._save_plot(csv_path, command_values)

        return {
            "csv": csv_path,
            "figure": figure_path,
        }

    def _save_plot(self, csv_path: Path, command_values: np.ndarray) -> Optional[Path]:
        figure_dir = self.config.get("figure_dir")
        if figure_dir is None or plt is None or not self.records:
            return None

        figure_dir = self.base_path / str(figure_dir)
        figure_dir.mkdir(parents=True, exist_ok=True)

        suffix = str(self.config.get("plot_filename_suffix", "_plot.pdf"))
        figure_name = f"{csv_path.stem}{suffix}"
        figure_path = figure_dir / figure_name

        time_data = np.asarray([rec.time for rec in self.records], dtype=float)
        tau_1 = np.asarray([rec.tau_1 for rec in self.records], dtype=float)
        tau_2 = np.asarray([rec.tau_2 for rec in self.records], dtype=float)
        output_pos = np.asarray([rec.output_pos for rec in self.records], dtype=float)
        output_vel = np.asarray([rec.output_vel for rec in self.records], dtype=float)
        theta_ref = np.asarray([rec.theta_ref for rec in self.records], dtype=float)

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        turn_to_deg = lambda arr: np.asarray(arr, dtype=float) * 360.0
        theta_out_deg = turn_to_deg(output_pos)
        theta_ref_deg = turn_to_deg(theta_ref)
        omega_out_deg = turn_to_deg(output_vel)

        axes[0].plot(time_data, theta_ref_deg, "--", label="θ_ref")
        axes[0].plot(time_data, theta_out_deg, "-", label="θ_out")
        axes[0].set_ylabel("θ [deg]")
        axes[0].legend(loc="upper right")

        axes[1].plot(time_data, omega_out_deg, "-", label="ω_out")
        axes[1].set_ylabel("ω [deg/s]")
        axes[1].legend(loc="upper right")

        axes[2].plot(time_data, tau_1, label="τ_1")
        axes[2].plot(time_data, tau_2, label="τ_2")
        axes[2].set_ylabel("Torque [Nm]")
        axes[2].set_xlabel("Time [s]")
        axes[2].legend(loc="upper right")

        for ax in axes:
            ax.grid(False)
            ax.tick_params(axis="both", direction="in", length=6, width=0.8)

        fig.tight_layout()
        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        try:
            plt.show()
        except Exception:
            pass
        plt.close(fig)
        return figure_path

    def _metadata_lines(self, command_values_text: str, extra: Dict[str, str]) -> List[tuple[str, str]]:
        lines: List[tuple[str, str]] = []

        profile_name = self.reference.get_active_profile_name() if self.reference else "-"
        profile_label = self.reference.get_profile_label() if self.reference else "-"
        profile_settings = (
            self.reference.format_profile_settings(profile_name) if self.reference else "-"
        )

        def fmt_pid(cfg: Dict[str, float]) -> str:
            if not isinstance(cfg, dict):
                return "-"
            return f"kp={cfg.get('kp', 0.0)},ki={cfg.get('ki', 0.0)},kd={cfg.get('kd', 0.0)},max={cfg.get('max_output', 0.0)}"

        def format_vector(values) -> str:
            arr = np.asarray(list(values), dtype=float).ravel()
            return "[" + ", ".join(f"{val:g}" for val in arr) + "]"

        per_motor_cfg = self.controller_config.get("per_motor_pid", {})
        outer_loop_enabled = bool(per_motor_cfg.get("enable_outer_loop", True))
        control_mode = f"per_motor_pid(outer={'on' if outer_loop_enabled else 'off'})"
        derivative_mode = per_motor_cfg.get("derivative_mode", "error")
        derivative_alpha = per_motor_cfg.get("derivative_filter_alpha", 1.0)

        lines.extend(
            [
                ("Profile", profile_name),
                ("ProfileLabel", profile_label),
                ("CommandValues", command_values_text),
                ("Settings", profile_settings),
                ("ControlMode", control_mode),
                ("OutputPID", fmt_pid(self.controller_config.get("outer_pid", {}))),
                ("MotorPID_1", fmt_pid(per_motor_cfg.get("motor1", {}))),
                ("MotorPID_2", fmt_pid(per_motor_cfg.get("motor2", {}))),
                ("PIDDerivativeMode", derivative_mode),
                ("PIDDerivativeFilterAlpha", f"{derivative_alpha}"),
                ("Nullspace", "disabled"),
            ]
        )

        freeze_cfg = self.controller_config.get("freeze", {})
        freeze_text = (
            f"motor={freeze_cfg.get('motor', 'None')},"
            f"kp={freeze_cfg.get('kp', 0.0)},"
            f"kd={freeze_cfg.get('kd', 0.0)}"
        )
        lines.append(("Freeze", freeze_text))

        torque_limits = self.controller_config.get("torque_limits", {})
        torque_vec = [torque_limits.get("motor1", 0.0), torque_limits.get("motor2", 0.0)]
        lines.append(("TorqueLimits", format_vector(torque_vec)))

        for key, value in extra.items():
            lines.append((str(key), str(value)))

        return lines
