"""Jupyter shim — re-exports from src.notebook_helpers."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.notebook_helpers import *  # noqa: E402, F401, F403
