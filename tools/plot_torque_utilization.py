import argparse
import csv
import re
from pathlib import Path


def _read_metadata(path: Path) -> tuple[float, float, str]:
    limit1 = None
    limit2 = None
    command_type = "position"
    with path.open("r", encoding="utf-8", newline="") as f:
        for line in f:
            if not line.startswith("#"):
                break
            if line.startswith("# CommandType="):
                command_type = line.split("=", 1)[1].strip()
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
    return (limit1 or 1.0, limit2 or 1.0, command_type)


def _iter_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(line for line in f if line.strip() and not line.startswith("#"))
        for row in reader:
            yield row


def compute(path: Path):
    limit1, limit2, command_type = _read_metadata(path)
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
    return time, util1, util2, margin1, margin2, (limit1, limit2), command_type


def _plot_combined_figure(
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
    command_type: str,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    legend_font = font_manager.FontProperties(family="Times New Roman", size=22)
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    turn_to_deg = lambda arr: np.asarray(arr, dtype=float) * 360.0
    theta_out_deg = turn_to_deg(output_pos)
    theta_ref_deg = turn_to_deg(theta_ref)
    omega_ref_deg = turn_to_deg(omega_ref)
    omega_out_deg = turn_to_deg(output_vel)

    legend_kwargs = dict(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        prop=legend_font,
    )

    if command_type == "velocity":
        axes[0].plot(time, omega_ref_deg, "--", label=r"$\omega_{\mathrm{ref}}$")
        axes[0].plot(time, omega_out_deg, "-", label=r"$\omega_{\mathrm{out}}$")
        axes[0].set_ylabel(r"$\omega$ [deg/s]")
        axes[0].legend(**legend_kwargs)

        axes[1].plot(time, theta_out_deg, "-", label=r"$\theta_{\mathrm{out}}$")
        axes[1].set_ylabel(r"$\theta$ [deg]")
        axes[1].legend(**legend_kwargs)
    else:
        axes[0].plot(time, theta_ref_deg, "--", label=r"$\theta_{\mathrm{ref}}$")
        axes[0].plot(time, theta_out_deg, "-", label=r"$\theta_{\mathrm{out}}$")
        axes[0].set_ylabel(r"$\theta$ [deg]")
        axes[0].legend(**legend_kwargs)

        axes[1].plot(
            time,
            omega_out_deg,
            "-",
            color="tab:green",
            label=r"$\omega_{\mathrm{out}}$",
        )
        axes[1].set_ylabel(r"$\omega$ [deg/s]")
        axes[1].legend(**legend_kwargs)

    axes[2].plot(time, tau_1, "-", color="tab:blue", label=r"$\tau_1$")
    axes[2].plot(time, tau_2, "-", color="tab:red", label=r"$\tau_2$")
    axes[2].set_ylabel(r"$\tau$ [Nm]")
    axes[2].legend(**legend_kwargs)

    axes[3].plot(time, util1, label="motor1", color="tab:blue")
    axes[3].plot(time, util2, label="motor2", color="tab:red")
    axes[3].set_ylabel("torque\nutilization")
    axes[3].legend(**legend_kwargs)

    axes[4].plot(time, margin1, label="motor1", color="tab:blue")
    axes[4].plot(time, margin2, label="motor2", color="tab:red")
    axes[4].set_xlabel("time [s]")
    axes[4].set_ylabel("torque\nmargin")
    axes[4].legend(**legend_kwargs)

    for ax in axes:
        ax.grid(False)
        ax.tick_params(axis="both", direction="in", length=6, width=0.8)

    if len(time):
        axes[0].set_xlim(time[0], time[-1])

    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    return fig


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

    time, util1, util2, margin1, margin2, limits, command_type = compute(args.csv)

    print(f"limits: motor1={limits[0]}, motor2={limits[1]}")
    print(f"util1: max={max(util1):.3f}, mean={sum(util1)/len(util1):.3f}")
    print(f"util2: max={max(util2):.3f}, mean={sum(util2)/len(util2):.3f}")
    print(f"margin1: min={min(margin1):.3f}")
    print(f"margin2: min={min(margin2):.3f}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; rerun with --no-plot or install matplotlib.")
        return 1

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

    rows = list(_iter_rows(args.csv))
    output_pos = [float(r["output_pos"]) for r in rows]
    output_vel = [float(r["output_vel"]) for r in rows]
    theta_ref = [float(r["theta_ref"]) for r in rows]
    omega_ref = [float(r["omega_ref"]) for r in rows]
    tau_1 = [float(r["tau_1"]) for r in rows]
    tau_2 = [float(r["tau_2"]) for r in rows]

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
        save_path = fig_dir / f"{args.csv.stem}_combined.pdf"
    else:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_path)
    print(f"saved plot: {save_path}")

    if not args.no_plot:
        plt.show()
    else:
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
