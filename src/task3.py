from __future__ import annotations

import pandas as pd

from src.cleaning import build_quality_mask


def run_task3(
    sessions: pd.DataFrame, facilities: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    clean = sessions.loc[
        build_quality_mask(sessions), ["device_id", "facility_num", "session_start"]
    ]
    multi_ids = clean.groupby("device_id")["facility_num"].nunique()
    multi_ids = multi_ids.loc[multi_ids > 1].index
    events = clean.loc[clean["device_id"].isin(multi_ids)].sort_values(
        ["device_id", "session_start"]
    )

    events = events.copy()
    events["prev_facility"] = events.groupby("device_id")["facility_num"].shift()
    events = events.loc[
        events["prev_facility"].isna()
        | (events["facility_num"] != events["prev_facility"])
    ]
    events["next_facility"] = events.groupby("device_id")["facility_num"].shift(-1)

    transitions = events.loc[events["next_facility"].notna(), ["facility_num", "next_facility"]]
    transitions["next_facility"] = transitions["next_facility"].astype(int)
    transition_counts = (
        transitions.groupby(["facility_num", "next_facility"], as_index=False)
        .size()
        .rename(columns={"size": "transition_count"})
        .sort_values("transition_count", ascending=False)
    )

    journey_paths = (
        events.groupby("device_id")
        .agg(
            first_seen=("session_start", "min"),
            last_seen=("session_start", "max"),
            unique_facilities=("facility_num", "nunique"),
            path_steps=("facility_num", "size"),
            path=("facility_num", lambda x: " -> ".join(map(str, x.tolist()))),
        )
        .reset_index()
    )
    top_paths = (
        journey_paths.groupby("path", as_index=False)["device_id"]
        .nunique()
        .rename(columns={"device_id": "device_count"})
        .sort_values("device_count", ascending=False)
    )
    top_paths["path_length"] = top_paths["path"].str.count("->") + 1

    sample = journey_paths.sample(n=min(200, len(journey_paths)), random_state=42)

    lookup = facilities[["facility_num", "facility_name"]].drop_duplicates()
    transition_counts = transition_counts.merge(
        lookup.rename(
            columns={"facility_num": "facility_num", "facility_name": "facility_name_from"}
        ),
        on="facility_num",
        how="left",
    ).merge(
        lookup.rename(
            columns={"facility_num": "next_facility", "facility_name": "facility_name_to"}
        ),
        on="next_facility",
        how="left",
    )

    return top_paths, transition_counts, sample, int(len(journey_paths))
