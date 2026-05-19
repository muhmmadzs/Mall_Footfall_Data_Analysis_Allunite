import os
from pathlib import Path


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = _path_from_env("MALL_FOOTFALL_DATA_DIR", PROJECT_ROOT / "data")
OUTPUT_DIR = _path_from_env("MALL_FOOTFALL_OUTPUT_DIR", PROJECT_ROOT / "outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORT_DIR = PROJECT_ROOT / "report"

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
