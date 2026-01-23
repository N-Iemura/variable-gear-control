import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from plot_torque_utilization import _plot_combined_figure, _read_metadata


def _iter_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(line for line in f if line.strip() and not line.startswith("#"))
        for row in reader:
            yield row


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot 1-second window (shifted to 0-1s) with torque utilization."
    )
    parser.add_argument("csv", type=Path, help="Path to simulation CSV.")
    parser.add_argument(
        "--start",
        type=float,
        default=1.0,
        help="Window start time [s] to crop (default: 1.0).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Window duration [s] to crop (default: 1.0).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save the plot image to this path (PDF/PNG recommended).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not show the plot window.",
    )
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 22,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.edgecolor": "black",
        }
    )

    limit1, limit2, command_type = _read_metadata(args.csv)

    t0 = float(args.start)
    t1 = t0 + float(args.duration)

    time = []
    output_pos = []
    output_vel = []
    theta_ref = []
    omega_ref = []
    tau_1 = []
    tau_2 = []
    util1 = []
    util2 = []
    margin1 = []
    margin2 = []

    for row in _iter_rows(args.csv):
        t = float(row["time"])
        if t0 <= t <= t1:
            t -= t0
            time.append(t)
            output_pos.append(float(row["output_pos"]))
            output_vel.append(float(row["output_vel"]))
            theta_ref.append(float(row["theta_ref"]))
            omega_ref.append(float(row.get("omega_ref", 0.0)))
            tau1 = float(row["tau_1"])
            tau2 = float(row["tau_2"])
            tau_1.append(tau1)
            tau_2.append(tau2)
            util1.append(abs(tau1) / (limit1 or 1.0))
            util2.append(abs(tau2) / (limit2 or 1.0))
            margin1.append((limit1 or 1.0) - abs(tau1))
            margin2.append((limit2 or 1.0) - abs(tau2))

    if not time:
        print("No samples in the requested time window.")
        return 1
    # Remove leading gap if the first sample is after the window start.
    if time[0] > 0.0:
        offset = time[0]
        time[:] = [t - offset for t in time]

    fig = _plot_combined_figure(
        time,
        output_pos,
        output_vel,
        theta_ref,
        omega_ref,
        tau_1,
        tau_2,
        util1,
        util2,
        margin1,
        margin2,
        command_type,
    )

    save_path = args.save
    if save_path is None:
        fig_dir = Path(__file__).resolve().parent / "fig"
        fig_dir.mkdir(parents=True, exist_ok=True)
        save_path = fig_dir / f"{args.csv.stem}_combined_1s_from1.pdf"
    else:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_path)
    print(save_path)
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
