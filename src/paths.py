from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORT_DIR = PROJECT_ROOT / "report"

SESSION_CSV = DATA_DIR / "allunite_device_session.csv"
FACILITY_CSV = DATA_DIR / "facility_information - Sheet1.csv"
MANUAL_CSV = DATA_DIR / "Manual Counting - Sheet1.csv"
