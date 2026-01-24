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


def _resolve_columns(fieldnames: list[str]) -> dict:
    """Resolve column names across log variants."""
    # Newer logs
    if "tau_1" in fieldnames and "tau_2" in fieldnames:
        return {
            "tau_1": "tau_1",
            "tau_2": "tau_2",
            "output_pos": "output_pos",
            "output_vel": "output_vel",
            "theta_ref": "theta_ref",
            "omega_ref": "omega_ref",
        }
    # Legacy logs (motor0/motor1)
    if "motor0_torque" in fieldnames and "motor1_torque" in fieldnames:
        return {
            "tau_1": "motor0_torque",
            "tau_2": "motor1_torque",
            "output_pos": "output_pos",
            "output_vel": "output_vel",
            "theta_ref": "theta_ref",
            "omega_ref": "omega_ref",
        }
    return {}


def compute(path: Path):
    limit1, limit2, command_type = _read_metadata(path)
    limit1 = float(getattr(compute, "_limit1_override", limit1))
    limit2 = float(getattr(compute, "_limit2_override", limit2))
    window_start = getattr(compute, "_window_start", None)
    window_end = getattr(compute, "_window_end", None)
    time = []
    util1 = []
    util2 = []
    margin1 = []
    margin2 = []
    torque_offset = float(getattr(compute, "_motor1_offset", 0.0))
    rows_iter = _iter_rows(path)
    first_row = next(rows_iter, None)
    if first_row is None:
        return time, util1, util2, margin1, margin2, (limit1, limit2), command_type
    columns = _resolve_columns(list(first_row.keys()))
    if not columns:
        raise KeyError("Unsupported CSV format: torque columns not found.")
    # include first row
    row = first_row
    while row is not None:
        t = float(row["time"])
        if window_start is not None and t < window_start:
            row = next(rows_iter, None)
            continue
        if window_end is not None and t > window_end:
            row = next(rows_iter, None)
            continue
        if window_start is not None:
            t -= window_start
        tau1 = float(row[columns["tau_1"]]) + torque_offset
        tau2 = float(row[columns["tau_2"]])
        u1 = abs(tau1) / (limit1 or 1.0)
        u2 = abs(tau2) / (limit2 or 1.0)
        time.append(t)
        util1.append(u1)
        util2.append(u2)
        margin1.append((limit1 or 1.0) - abs(tau1))
        margin2.append((limit2 or 1.0) - abs(tau2))
        row = next(rows_iter, None)
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
    axes[3].set_ylabel("\u03bc")
    axes[3].set_ylim(bottom=0.0)
    axes[3].legend(**legend_kwargs)

    axes[4].plot(time, margin1, label="motor1", color="tab:blue")
    axes[4].plot(time, margin2, label="motor2", color="tab:red")
    axes[4].set_xlabel("time [s]")
    axes[4].set_ylabel("m [Nm]")
    axes[4].set_ylim(bottom=0.0)
    axes[4].legend(**legend_kwargs)

    for ax in axes:
        ax.grid(False)
        ax.tick_params(axis="both", direction="in", length=6, width=0.8)
    # Reduce overlap between the lower-left x/y tick labels (e.g., 0 and 0.0).
    axes[4].tick_params(axis="x", pad=8)
    axes[4].tick_params(axis="y", pad=8)

    if len(time):
        for ax in axes:
            ax.set_xlim(0.0, time[-1])

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
    parser.add_argument(
        "--margin-offset-motor1",
        type=float,
        default=0.0,
        help="Add an offset [Nm] to motor1 torque margin (default: 0.0).",
    )
    parser.add_argument(
        "--torque-offset-motor1",
        type=float,
        default=0.0,
        help="Add an offset [Nm] to motor1 torque values before plotting/utilization.",
    )
    parser.add_argument(
        "--limit-motor1",
        type=float,
        default=None,
        help="Override motor1 torque limit [Nm] for utilization/margin.",
    )
    parser.add_argument(
        "--limit-motor2",
        type=float,
        default=None,
        help="Override motor2 torque limit [Nm] for utilization/margin.",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="Window start time [s] to crop (optional).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Window duration [s] to crop (optional).",
    )
    args = parser.parse_args()

    compute._motor1_offset = args.torque_offset_motor1
    if args.limit_motor1 is not None:
        compute._limit1_override = args.limit_motor1
    if args.limit_motor2 is not None:
        compute._limit2_override = args.limit_motor2
    if args.start is not None and args.duration is not None:
        compute._window_start = float(args.start)
        compute._window_end = float(args.start) + float(args.duration)
    time, util1, util2, margin1, margin2, limits, command_type = compute(args.csv)
    if args.margin_offset_motor1:
        margin1 = [m + args.margin_offset_motor1 for m in margin1]

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
    if not rows:
        print("No data rows.")
        return 1
    columns = _resolve_columns(list(rows[0].keys()))
    if not columns:
        print("Unsupported CSV format: torque columns not found.")
        return 1
    if args.start is not None and args.duration is not None:
        t0 = float(args.start)
        t1 = t0 + float(args.duration)
        rows = [r for r in rows if t0 <= float(r["time"]) <= t1]
        # shift time to start at zero for plotting
        for r in rows:
            r["time"] = str(float(r["time"]) - t0)
    output_pos = [float(r[columns["output_pos"]]) for r in rows]
    output_vel = [float(r[columns["output_vel"]]) for r in rows]
    theta_ref = [float(r[columns["theta_ref"]]) for r in rows]
    omega_ref_key = columns.get("omega_ref")
    omega_ref = [float(r.get(omega_ref_key, 0.0)) for r in rows] if omega_ref_key else [0.0] * len(rows)
    tau_1 = [float(r[columns["tau_1"]]) + float(args.torque_offset_motor1) for r in rows]
    tau_2 = [float(r[columns["tau_2"]]) for r in rows]

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
