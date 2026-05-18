from __future__ import annotations

from typing import Dict

import pandas as pd

from src.cleaning import build_quality_mask


def run_task2(
    sessions: pd.DataFrame, facilities: pd.DataFrame
) -> tuple[pd.DataFrame, Dict[str, float]]:
    clean = sessions.loc[build_quality_mask(sessions), ["device_id", "facility_num"]]
    device_fac = clean.drop_duplicates()

    facility_counts = (
        device_fac.groupby("facility_num", as_index=False)["device_id"]
        .nunique()
        .rename(columns={"device_id": "devices_at_facility"})
    )
    counts_map = dict(
        zip(facility_counts["facility_num"], facility_counts["devices_at_facility"])
    )

    per_device = device_fac.groupby("device_id")["facility_num"].nunique()
    multi_sensor = int((per_device > 1).sum())
    total_devices = int(per_device.shape[0])

    pairs = device_fac.merge(device_fac, on="device_id")
    pairs = pairs.loc[pairs["facility_num_x"] < pairs["facility_num_y"]]
    pair_counts = (
        pairs.groupby(["facility_num_x", "facility_num_y"], as_index=False)["device_id"]
        .nunique()
        .rename(columns={"device_id": "shared_devices"})
    )
    pair_counts["devices_facility_x"] = pair_counts["facility_num_x"].map(counts_map)
    pair_counts["devices_facility_y"] = pair_counts["facility_num_y"].map(counts_map)
    pair_counts["union_devices"] = (
        pair_counts["devices_facility_x"]
        + pair_counts["devices_facility_y"]
        - pair_counts["shared_devices"]
    )
    pair_counts["jaccard_overlap"] = (
        pair_counts["shared_devices"] / pair_counts["union_devices"]
    ).round(6)
    pair_counts["overlap_pct_smaller_side"] = (
        pair_counts["shared_devices"]
        / pair_counts[["devices_facility_x", "devices_facility_y"]].min(axis=1)
    ).round(6)

    lookup = facilities[["facility_num", "facility_name"]].drop_duplicates()
    pair_counts = pair_counts.merge(
        lookup.rename(
            columns={"facility_num": "facility_num_x", "facility_name": "facility_name_x"}
        ),
        on="facility_num_x",
        how="left",
    ).merge(
        lookup.rename(
            columns={"facility_num": "facility_num_y", "facility_name": "facility_name_y"}
        ),
        on="facility_num_y",
        how="left",
    )
    pair_counts = pair_counts.sort_values("shared_devices", ascending=False)

    summary = {
        "total_unique_devices_clean": total_devices,
        "multi_sensor_devices": multi_sensor,
        "multi_sensor_rate": float(multi_sensor / total_devices if total_devices else 0.0),
    }
    return pair_counts, summary
