"""Same shim as notebooks/notebook_helpers.py — use when Jupyter cwd is repo root."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.notebook_helpers import *  # noqa: F403
