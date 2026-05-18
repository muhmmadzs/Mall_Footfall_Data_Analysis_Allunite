"""Reuse Plan A Task 2 implementation from repo src/."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("plan_a_task2", _REPO / "src" / "task2.py")
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
run_task2 = _mod.run_task2
