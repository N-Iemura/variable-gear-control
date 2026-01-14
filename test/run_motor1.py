from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TEST = _ROOT / "test"
CONFIG = _ROOT / "test" / "config" / "test_motor.yaml"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_TEST) not in sys.path:
    sys.path.insert(0, str(_TEST))

from motor1_control import main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--test-config", str(CONFIG)]
    raise SystemExit(main())
