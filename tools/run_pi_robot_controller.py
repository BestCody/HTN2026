"""Run the packaged Pi robot controller from a repository checkout."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from moira.robot_link import controller_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(controller_main())
