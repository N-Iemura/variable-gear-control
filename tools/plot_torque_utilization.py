import argparse
import csv
import re
from pathlib import Path


def _read_limits(path: Path) -> tuple[float, float]:
    limit1 = None
    limit2 = None
    with path.open("r", encoding="utf-8", newline="") as f:
        for line in f:
            if not line.startswith("#"):
                break
            if line.startswith("# TorqueLimits="):
                match = re.search(r"\[([^\]]+)\]", line)
                if match:
                    parts = [p.strip() for p in match.group(1).split(",")]
                    if len(parts) >= 2:
                        try:
                            limit1 = float(parts[0])
                            limit2 = float(parts[1])
                        except ValueError:
                            pass
                break
    return (limit1 or 1.0, limit2 or 1.0)


def _iter_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(line for line in f if line.strip() and not line.startswith("#"))
        for row in reader:
            yield row


def compute(path: Path):
    limit1, limit2 = _read_limits(path)
    time = []
    util1 = []
    util2 = []
    margin1 = []
    margin2 = []
    for row in _iter_rows(path):
        t = float(row["time"])
        tau1 = float(row["tau_1"])
        tau2 = float(row["tau_2"])
        u1 = abs(tau1) / (limit1 or 1.0)
        u2 = abs(tau2) / (limit2 or 1.0)
        time.append(t)
        util1.append(u1)
        util2.append(u2)
        margin1.append((limit1 or 1.0) - abs(tau1))
        margin2.append((limit2 or 1.0) - abs(tau2))
    return time, util1, util2, margin1, margin2, (limit1, limit2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot torque utilization and margin from a simulation CSV."
    )
    parser.add_argument("csv", type=Path, help="Path to simulation CSV.")
    parser.add_argument("--no-plot", action="store_true", help="Do not show plots.")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save the plot image to this path (PDF/PNG recommended).",
    )
    args = parser.parse_args()

    time, util1, util2, margin1, margin2, limits = compute(args.csv)

    print(f"limits: motor1={limits[0]}, motor2={limits[1]}")
    print(f"util1: max={max(util1):.3f}, mean={sum(util1)/len(util1):.3f}")
    print(f"util2: max={max(util2):.3f}, mean={sum(util2)/len(util2):.3f}")
    print(f"margin1: min={min(margin1):.3f}")
    print(f"margin2: min={min(margin2):.3f}")

    if args.no_plot and args.save is None:
        return 0

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; rerun with --no-plot or install matplotlib.")
        return 1

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 14,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.edgecolor": "black",
        }
    )

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    axes[0].plot(time, util1, label="motor1")
    axes[0].plot(time, util2, label="motor2")
    axes[0].set_ylabel("torque utilization")
    axes[0].legend()
    axes[0].grid(False)

    axes[1].plot(time, margin1, label="motor1")
    axes[1].plot(time, margin2, label="motor2")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("torque margin")
    axes[1].legend()
    axes[1].grid(False)

    fig.tight_layout()
    save_path = args.save
    if save_path is None:
        fig_dir = Path(__file__).resolve().parent / "fig"
        fig_dir.mkdir(parents=True, exist_ok=True)
        save_path = fig_dir / f"{args.csv.stem}_torque_utilization.pdf"
    else:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_path)
    print(f"saved plot: {save_path}")
    if not args.no_plot:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
