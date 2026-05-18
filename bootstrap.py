"""Add repo root to sys.path — run before `import src` in notebooks.

In Jupyter (cwd = notebooks/):
    %run ../bootstrap.py

Or open notebooks/mall_footfall_analysis.ipynb (uses notebook_helpers automatically).
"""
import sys
from pathlib import Path


def find_repo_root() -> Path:
    for root in (Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent):
        if (root / "src" / "notebook_helpers.py").exists():
            return root.resolve()
    raise RuntimeError(
        "Repo root not found. Open the project folder or run from notebooks/ "
        "with Mall_Footfall_Data_Analysis_Allunite as parent."
    )


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
