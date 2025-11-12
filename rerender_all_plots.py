#!/usr/bin/env python3
"\"\"Re-render every CSV log with plot_csv.py, writing PDFs to fig/.\"\""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    csv_dir = repo_root / "csv"
    plot_script = repo_root / "plot_csv.py"
    if not csv_dir.is_dir():
        print(f"CSV directory not found: {csv_dir}", file=sys.stderr)
        return 1
    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        print("No CSV files found to plot.")
        return 0

    for path in csv_files:
        print(f"[plot] {path.name}")
        result = subprocess.run(
            [sys.executable, str(plot_script), str(path)],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            print(f"Failed to plot {path}", file=sys.stderr)
            return result.returncode
        else:
            sys.stdout.write(result.stdout)
    print("All CSV plots regenerated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
