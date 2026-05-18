from pathlib import Path

PLAN_B_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PLAN_B_ROOT.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = PLAN_B_ROOT / "outputs"
REPORT_DIR = PLAN_B_ROOT / "report"
PLAN_A_OUTPUT = REPO_ROOT / "outputs"

SESSION_CSV = DATA_DIR / "allunite_device_session.csv"
FACILITY_CSV = DATA_DIR / "facility_information - Sheet1.csv"
MANUAL_CSV = DATA_DIR / "Manual Counting - Sheet1.csv"

# Facilities with manual counting ground truth
CALIBRATED_FACILITIES = {66330, 66333}
