import os
from pathlib import Path


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default

PLAN_B_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PLAN_B_ROOT.parent
DATA_DIR = _path_from_env("MALL_FOOTFALL_DATA_DIR", REPO_ROOT / "data")
OUTPUT_DIR = _path_from_env("MALL_FOOTFALL_PLAN_B_OUTPUT_DIR", PLAN_B_ROOT / "outputs")
REPORT_DIR = PLAN_B_ROOT / "report"
PLAN_A_OUTPUT = _path_from_env("MALL_FOOTFALL_OUTPUT_DIR", REPO_ROOT / "outputs")

SESSION_CSV = _path_from_env(
    "MALL_FOOTFALL_SESSION_CSV",
    DATA_DIR / "allunite_device_session.csv",
)
FACILITY_CSV = _path_from_env(
    "MALL_FOOTFALL_FACILITY_CSV",
    DATA_DIR / "facility_information - Sheet1.csv",
)
MANUAL_CSV = _path_from_env(
    "MALL_FOOTFALL_MANUAL_CSV",
    DATA_DIR / "Manual Counting - Sheet1.csv",
)

# Facilities with manual counting ground truth
CALIBRATED_FACILITIES = {66330, 66333}
