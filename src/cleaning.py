import pandas as pd


def build_quality_mask(df: pd.DataFrame) -> pd.Series:
    """Rows suitable for traffic / journey analysis (per assignment PDF)."""
    return (
        ~df["is_excluded"]
        & ~df["is_anomaly"]
        & ~df["is_fake"]
        & ~df["is_permanent_device"]
    )
