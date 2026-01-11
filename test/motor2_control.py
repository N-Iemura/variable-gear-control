from __future__ import annotations

import sys

from motor_test_common import run_motor_cli


def main() -> int:
    return run_motor_cli("motor2")


if __name__ == "__main__":
    sys.exit(main())
