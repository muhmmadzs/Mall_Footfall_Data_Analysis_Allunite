"""Mall floor-plan layout: sensor pin positions on assets/*.png maps."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from src.paths import DATA_DIR, PROJECT_ROOT

MAP_DIR = PROJECT_ROOT / "assets"
POSITIONS_CSV = DATA_DIR / "sensor_map_positions.csv"

MAP_FILES: Dict[str, Path] = {
    "lower_mall": MAP_DIR / "1.png",
    "south_john": MAP_DIR / "2.png",
}

MAP_TITLES: Dict[str, str] = {
    "lower_mall": "Lower Mall",
    "south_john": "South John Street (upper level)",
}


def asset_id_from_facility_name(name: str) -> Optional[str]:
    m = re.search(r"08(\d{3})", str(name))
    if m:
        return str(int(m.group(0)))
    return None


def facilities_with_asset_id(facilities: pd.DataFrame) -> pd.DataFrame:
    """Active sensors with asset_id (8003, …) from facility_name or asset_data JSON."""
    fac = facilities.loc[facilities["box_macs"].astype(str).str.len() > 2].copy()

    def _asset_from_row(row: pd.Series) -> Optional[str]:
        aid = asset_id_from_facility_name(row.get("facility_name", ""))
        if aid:
            return aid
        raw = row.get("asset_data")
        if pd.isna(raw):
            return None
        try:
            data = json.loads(str(raw).replace('""', '"'))
            v = data.get("asset_id")
            return str(int(v)) if v is not None else None
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    fac["asset_id"] = fac.apply(_asset_from_row, axis=1)
    fac["facility_num"] = fac["facility_num"].astype(int)
    return fac.dropna(subset=["asset_id"])


def load_sensor_positions() -> pd.DataFrame:
    pos = pd.read_csv(POSITIONS_CSV)
    pos["facility_num"] = pos["facility_num"].astype(int)
    pos["asset_id"] = pos["asset_id"].astype(str)
    return pos


def sensor_layout(facilities: pd.DataFrame) -> pd.DataFrame:
    """Merge facility metadata with map pin coordinates."""
    fac = facilities_with_asset_id(facilities)
    pos = load_sensor_positions()
    layout = fac.merge(pos, on=["facility_num", "asset_id"], how="inner")
    layout["map_path"] = layout["map_id"].map(MAP_FILES)
    return layout


def pixel_xy(row, img_w: int, img_h: int) -> Tuple[float, float]:
    xf = getattr(row, "x_frac", row["x_frac"])
    yf = getattr(row, "y_frac", row["y_frac"])
    return float(xf) * img_w, float(yf) * img_h


def layout_xy_dict(layout: pd.DataFrame, map_id: str, img_w: int, img_h: int) -> Dict[int, Tuple[float, float]]:
    sub = layout.loc[layout["map_id"] == map_id]
    out: Dict[int, Tuple[float, float]] = {}
    for _, row in sub.iterrows():
        out[int(row["facility_num"])] = pixel_xy(row, img_w, img_h)
    return out


def asset_label(layout: pd.DataFrame, facility_num: int) -> str:
    row = layout.loc[layout["facility_num"] == facility_num]
    if len(row):
        return str(row.iloc[0]["asset_id"])
    return str(facility_num)
