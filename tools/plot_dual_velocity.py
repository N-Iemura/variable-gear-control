import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib import font_manager
import yaml


def _load_csv(path: Path, start_time: float, k1: float, k2: float) -> Dict[str, List[float]]:
    data: Dict[str, List[float]] = {
        "time_s": [],
        "motor1_target_velocity": [],
        "motor2_target_velocity": [],
        "motor1_velocity": [],
        "motor2_velocity": [],
        "output_velocity": [],
        "output_target_velocity": [],
    }
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row["time_s"])
            if t < start_time:
                continue
            t -= start_time
            t1 = float(row["motor1_target_velocity"])
            t2 = float(row["motor2_target_velocity"])
            data["time_s"].append(t)
            data["motor1_target_velocity"].append(t1)
            data["motor2_target_velocity"].append(t2)
            data["motor1_velocity"].append(float(row["motor1_velocity"]))
            data["motor2_velocity"].append(float(row["motor2_velocity"]))
            data["output_velocity"].append(float(row["output_velocity"]))
            data["output_target_velocity"].append(k1 * t1 + k2 * t2)
    return data


def _legend_kwargs():
    legend_font = font_manager.FontProperties(family="Times New Roman", size=22)
    return dict(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        prop=legend_font,
    )


def _plot_default(
    data: Dict[str, List[float]], out_path: Path, invert_output: bool, show: bool
) -> None:
    if not data["time_s"]:
        raise ValueError("No samples after start_time.")

    t = data["time_s"]
    m1 = data["motor1_velocity"]
    m2 = data["motor2_velocity"]
    t1 = data["motor1_target_velocity"]
    t2 = data["motor2_target_velocity"]
    out = data["output_velocity"]
    out_ref = data["output_target_velocity"]
    if invert_output:
        out = [-v for v in out]
        out_ref = [-v for v in out_ref]

    legend_kwargs = _legend_kwargs()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, m1, color="tab:blue", label=r"$\omega_1$")
    axes[0].plot(t, m2, color="tab:red", label=r"$\omega_2$")
    axes[0].plot(t, t1, "--", color="tab:blue", label=r"$\omega_{1,\mathrm{ref}}$")
    axes[0].plot(t, t2, "--", color="tab:red", label=r"$\omega_{2,\mathrm{ref}}$")
    axes[0].set_ylabel(r"$\omega$ [turn/s]")
    axes[0].legend(**legend_kwargs)

    axes[1].plot(t, out, color="tab:green", label=r"$\omega_{\mathrm{out}}$")
    axes[1].plot(t, out_ref, "--", color="tab:green", label=r"$\omega_{\mathrm{out,ref}}$")
    axes[1].set_ylabel(r"$\omega$ [turn/s]")
    axes[1].set_xlabel("time [s]")
    axes[1].legend(**legend_kwargs)

    for ax in axes:
        ax.grid(False)
        ax.tick_params(axis="both", direction="in", length=6, width=0.8)

    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    if show:
        plt.show()
    plt.close(fig)


def _plot_combined(
    data: Dict[str, List[float]], out_path: Path, invert_output: bool, show: bool
) -> None:
    if not data["time_s"]:
        raise ValueError("No samples after start_time.")

    t = data["time_s"]
    m1 = data["motor1_velocity"]
    m2 = data["motor2_velocity"]
    out = data["output_velocity"]
    out_ref = data["output_target_velocity"]
    t1 = data["motor1_target_velocity"]
    t2 = data["motor2_target_velocity"]
    if invert_output:
        out = [-v for v in out]
        out_ref = [-v for v in out_ref]

    legend_kwargs = _legend_kwargs()
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(t, m1, color="tab:blue", label=r"$\omega_1$")
    ax.plot(t, t1, "--", color="tab:blue", label=r"$\omega_{1,\mathrm{ref}}$")
    ax.plot(t, m2, color="tab:red", label=r"$\omega_2$")
    ax.plot(t, t2, "--", color="tab:red", label=r"$\omega_{2,\mathrm{ref}}$")
    ax.plot(t, out, color="tab:green", label=r"$\omega_{\mathrm{out}}$")
    ax.plot(t, out_ref, "--", color="tab:green", label=r"$\omega_{\mathrm{out,ref}}$")
    ax.set_ylabel(r"$\omega$ [turn/s]")
    ax.set_xlabel("time [s]")
    ax.legend(**legend_kwargs)
    ax.grid(False)
    ax.tick_params(axis="both", direction="in", length=6, width=0.8)
    if len(t):
        ax.set_xlim(0.0, t[-1])

    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    if show:
        plt.show()
    plt.close(fig)


def _plot_separate(
    data: Dict[str, List[float]], out_path: Path, invert_output: bool, show: bool
) -> None:
    if not data["time_s"]:
        raise ValueError("No samples after start_time.")

    t = data["time_s"]
    m1 = data["motor1_velocity"]
    m2 = data["motor2_velocity"]
    out = data["output_velocity"]
    out_ref = data["output_target_velocity"]
    t1 = data["motor1_target_velocity"]
    t2 = data["motor2_target_velocity"]
    if invert_output:
        out = [-v for v in out]
        out_ref = [-v for v in out_ref]

    legend_kwargs = _legend_kwargs()
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, m1, color="tab:blue", label=r"$\omega_1$")
    axes[0].plot(t, t1, "--", color="tab:blue", label=r"$\omega_{1,\mathrm{ref}}$")
    axes[0].set_ylabel(r"$\omega_1$ [turn/s]")
    axes[0].legend(**legend_kwargs)

    axes[1].plot(t, m2, color="tab:red", label=r"$\omega_2$")
    axes[1].plot(t, t2, "--", color="tab:red", label=r"$\omega_{2,\mathrm{ref}}$")
    axes[1].set_ylabel(r"$\omega_2$ [turn/s]")
    axes[1].legend(**legend_kwargs)

    axes[2].plot(t, out, color="tab:green", label=r"$\omega_{\mathrm{out}}$")
    axes[2].plot(t, out_ref, "--", color="tab:green", label=r"$\omega_{\mathrm{out,ref}}$")
    axes[2].set_ylabel(r"$\omega_{\mathrm{out}}$ [turn/s]")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(**legend_kwargs)

    for ax in axes:
        ax.grid(False)
        ax.tick_params(axis="both", direction="in", length=6, width=0.8)
        if len(t):
            ax.set_xlim(0.0, t[-1])

    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    if show:
        plt.show()
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot dual velocity logs to PDF.")
    parser.add_argument("csv", type=Path, nargs="+", help="Dual velocity CSV paths.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("test/fig"),
        help="Output directory for PDF files.",
    )
    parser.add_argument(
        "--controller-config",
        type=Path,
        default=Path("config/controller.yaml"),
        help="Controller config path for kinematic_matrix.",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=0.3,
        help="Start time [s] to shift to 0.",
    )
    parser.add_argument(
        "--invert-output",
        action="store_true",
        default=True,
        help="Invert output velocity sign for plotting.",
    )
    parser.add_argument(
        "--no-invert-output",
        action="store_false",
        dest="invert_output",
        help="Do not invert output velocity sign.",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Generate a single-axis plot with all three velocities.",
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help="Generate a three-axis plot with each velocity separately.",
    )
    parser.add_argument("--show", action="store_true", help="Show plot windows.")
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

    cfg = yaml.safe_load(args.controller_config.read_text(encoding="utf-8"))
    kin = cfg.get("velocity_distribution", {}).get("kinematic_matrix", None)
    if not kin or len(kin) < 2:
        raise ValueError("kinematic_matrix not found in controller config.")
    k1, k2 = float(kin[0]), float(kin[1])

    for csv_path in args.csv:
        data = _load_csv(csv_path, args.start_time, k1, k2)
        if args.combined:
            out_path = args.out_dir / f"{csv_path.stem}_omega_combined.pdf"
            _plot_combined(data, out_path, args.invert_output, args.show)
            print(out_path)
        if args.separate:
            out_path = args.out_dir / f"{csv_path.stem}_omega_separate.pdf"
            _plot_separate(data, out_path, args.invert_output, args.show)
            print(out_path)
        if not args.combined and not args.separate:
            out_path = args.out_dir / f"{csv_path.stem}.pdf"
            _plot_default(data, out_path, args.invert_output, args.show)
            print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
