from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def _load_csv(path: Path) -> dict[str, List[float]]:
    data: dict[str, List[float]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key not in data:
                    data[key] = []
                try:
                    data[key].append(float(value))
                except (TypeError, ValueError):
                    pass
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot motor test CSV log.")
    parser.add_argument("--csv", type=Path, required=True, help="CSV log path.")
    parser.add_argument("--out", type=Path, default=None, help="Output image path.")
    parser.add_argument("--show", action="store_true", help="Show the plot window.")
    args = parser.parse_args()

    data = _load_csv(args.csv)
    if "time_s" not in data:
        raise ValueError("CSV does not contain time_s column.")

    t = data["time_s"]
    target = data.get("target", [])
    position = data.get("position", [])
    velocity = data.get("velocity", [])
    command = data.get("command", [])
    torque_measured = data.get("torque_measured", [])

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(t, position, label="position")
    if target:
        axes[0].plot(t, target, "--", label="target")
    axes[0].set_ylabel("turn")
    axes[0].legend(loc="best")

    axes[1].plot(t, velocity, label="velocity")
    axes[1].set_ylabel("turn/s")
    axes[1].legend(loc="best")

    axes[2].plot(t, command, label="command")
    if torque_measured:
        axes[2].plot(t, torque_measured, label="torque_measured")
    axes[2].set_ylabel("Nm")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(loc="best")

    fig.tight_layout()

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150)
    if args.show or args.out is None:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
