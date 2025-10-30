from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None


@dataclass
class LogRecord:
    time: float
    motor0_pos: float
    motor0_vel: float
    motor0_torque: float
    motor1_pos: float
    motor1_vel: float
    motor1_torque: float
    output_pos: float
    output_vel: float
    theta_ref: float
    theta_ctrl: float
    tau_out: float


@dataclass
class DataLogger:
    config: Dict[str, object]
    base_path: Path = field(default_factory=lambda: Path("."))
    records: List[LogRecord] = field(default_factory=list)

    def log(
        self,
        time_stamp: float,
        motor0: Dict[str, float],
        motor1: Dict[str, float],
        output: Dict[str, float],
        reference: Dict[str, float],
        torques: Dict[str, float],
    ) -> None:
        record = LogRecord(
            time=time_stamp,
            motor0_pos=float(motor0.get("pos", 0.0)),
            motor0_vel=float(motor0.get("vel", 0.0)),
            motor0_torque=float(motor0.get("torque", 0.0)),
            motor1_pos=float(motor1.get("pos", 0.0)),
            motor1_vel=float(motor1.get("vel", 0.0)),
            motor1_torque=float(motor1.get("torque", 0.0)),
            output_pos=float(output.get("pos", 0.0)),
            output_vel=float(output.get("vel", 0.0)),
            theta_ref=float(reference.get("position", 0.0)),
            theta_ctrl=float(reference.get("control", 0.0)),
            tau_out=float(torques.get("output", 0.0)),
        )
        self.records.append(record)

    def save(self, metadata: Optional[Dict[str, str]] = None) -> Dict[str, Optional[Path]]:
        csv_dir = self.base_path / str(self.config.get("csv_dir", "csv"))
        csv_dir.mkdir(parents=True, exist_ok=True)

        prefix = str(self.config.get("data_filename_prefix", "record"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = csv_dir / f"{prefix}_{timestamp}.csv"

        metadata = metadata or {}

        with csv_path.open("w", newline="") as f:
            for key, value in metadata.items():
                f.write(f"# {key}={value}\n")
            writer = csv.writer(f)
            writer.writerow(
                [
                    "time",
                    "motor0_pos",
                    "motor0_vel",
                    "motor0_torque",
                    "motor1_pos",
                    "motor1_vel",
                    "motor1_torque",
                    "output_pos",
                    "output_vel",
                    "theta_ref",
                    "theta_ctrl",
                    "tau_out",
                ]
            )
            for record in self.records:
                writer.writerow(
                    [
                        record.time,
                        record.motor0_pos,
                        record.motor0_vel,
                        record.motor0_torque,
                        record.motor1_pos,
                        record.motor1_vel,
                        record.motor1_torque,
                        record.output_pos,
                        record.output_vel,
                        record.theta_ref,
                        record.theta_ctrl,
                        record.tau_out,
                    ]
                )
        figure_path = self._save_plot(csv_path)

        return {
            "csv": csv_path,
            "figure": figure_path,
        }

    def _save_plot(self, csv_path: Path) -> Optional[Path]:
        figure_dir = self.config.get("figure_dir")
        if figure_dir is None or plt is None or not self.records:
            return None

        figure_dir = self.base_path / str(figure_dir)
        figure_dir.mkdir(parents=True, exist_ok=True)

        suffix = str(self.config.get("plot_filename_suffix", "_plot.pdf"))
        figure_path = figure_dir / f"{csv_path.stem}{suffix}"

        time_data = [rec.time for rec in self.records]
        motor0_torque = [rec.motor0_torque for rec in self.records]
        motor1_torque = [rec.motor1_torque for rec in self.records]
        output_pos = [rec.output_pos for rec in self.records]
        output_vel = [rec.output_vel for rec in self.records]
        theta_ref = [rec.theta_ref for rec in self.records]

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(time_data, theta_ref, label="theta_ref")
        axes[0].plot(time_data, output_pos, label="theta_out")
        axes[0].set_ylabel("Position [turn]")
        axes[0].legend(loc="upper right")
        axes[0].grid(True)

        axes[1].plot(time_data, output_vel, label="omega_out")
        axes[1].set_ylabel("Velocity [turn/s]")
        axes[1].legend(loc="upper right")
        axes[1].grid(True)

        axes[2].plot(time_data, motor0_torque, label="tau_motor0")
        axes[2].plot(time_data, motor1_torque, label="tau_motor1")
        axes[2].set_ylabel("Torque [Nm]")
        axes[2].set_xlabel("Time [s]")
        axes[2].legend(loc="upper right")
        axes[2].grid(True)

        fig.suptitle("Variable Gear Control Log")
        fig.tight_layout()
        fig.savefig(figure_path, bbox_inches="tight")
        plt.close(fig)
        return figure_path
