#!/usr/bin/env python3
"""Plan B pipeline — same entry point as repo root (single orchestration)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
runpy.run_path(str(REPO_ROOT / "run_analysis.py"), run_name="__main__")
