"""Task 2 & 3 visualizations overlaid on mall floor-plan images (assets/1.png, assets/2.png)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
from PIL import Image

from src.map_layout import (
    MAP_FILES,
    MAP_TITLES,
    asset_label,
    layout_xy_dict,
    sensor_layout,
)
from src.paths import PLOTS_DIR


def _load_map_image(map_id: str) -> Tuple[np.ndarray, int, int]:
    path = MAP_FILES[map_id]
    if not path.exists():
        raise FileNotFoundError(f"Mall map image not found: {path}")
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    return arr, w, h


def _draw_sensor_nodes(
    ax,
    layout: pd.DataFrame,
    map_id: str,
    img_w: int,
    img_h: int,
    node_sizes: Optional[Dict[int, float]] = None,
) -> Dict[int, Tuple[float, float]]:
    xy = layout_xy_dict(layout, map_id, img_w, img_h)
    sub = layout.loc[layout["map_id"] == map_id]
    for _, r in sub.iterrows():
        fac = int(r["facility_num"])
        x, y = xy[fac]
        size = 120.0
        if node_sizes and fac in node_sizes:
            size = 80.0 + 40.0 * np.sqrt(node_sizes[fac] / max(node_sizes.values()))
        ax.scatter(
            x,
            y,
            s=size,
            c="#e74c3c",
            edgecolors="yellow",
            linewidths=2,
            zorder=5,
        )
        ax.text(
            x,
            y - 12,
            str(r["asset_id"]),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.85),
            zorder=6,
        )
    return xy


def plot_task2_overlap_on_maps(
    pair_counts: pd.DataFrame,
    facilities: pd.DataFrame,
    path: Path,
    min_shared: int = 3000,
    top_n: int = 12,
) -> None:
    """Draw shared-device overlap as lines between sensors on each floor plan."""
    layout = sensor_layout(facilities)
    if layout.empty:
        return

    pairs = pair_counts.nlargest(top_n, "shared_devices").copy()
    pairs = pairs.loc[pairs["shared_devices"] >= min_shared]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    map_order = ["lower_mall", "south_john"]

    for ax, map_id in zip(axes, map_order):
        arr, w, h = _load_map_image(map_id)
        ax.imshow(arr, extent=[0, w, 0, h], aspect="auto")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")
        ax.set_title(f"Task 2 — device overlap · {MAP_TITLES[map_id]}", fontsize=11)

        node_sizes: Dict[int, float] = {}
        for fac in layout.loc[layout["map_id"] == map_id, "facility_num"]:
            col_x = f"devices_facility_x" if f"devices_facility_x" in pairs.columns else None
            # use max devices from pair table for this facility
            m = 0.0
            for _, pr in pairs.iterrows():
                if int(pr["facility_num_x"]) == int(fac):
                    m = max(m, float(pr.get("devices_facility_x", 0)))
                if int(pr["facility_num_y"]) == int(fac):
                    m = max(m, float(pr.get("devices_facility_y", 0)))
            if m == 0:
                sub = pair_counts[
                    (pair_counts["facility_num_x"] == fac)
                    | (pair_counts["facility_num_y"] == fac)
                ]
                if len(sub):
                    m = float(
                        sub[["devices_facility_x", "devices_facility_y"]].max().max()
                    )
            node_sizes[int(fac)] = m

        xy = _draw_sensor_nodes(ax, layout, map_id, w, h, node_sizes)

        segments: List[List[Tuple[float, float]]] = []
        widths: List[float] = []
        colors: List[str] = []
        max_shared = float(pairs["shared_devices"].max()) if len(pairs) else 1.0

        for pr in pairs.itertuples(index=False):
            fx, fy = int(pr.facility_num_x), int(pr.facility_num_y)
            if fx in xy and fy in xy:
                segments.append([xy[fx], xy[fy]])
                lw = 1.0 + 5.0 * (pr.shared_devices / max_shared)
                widths.append(lw)
                colors.append(plt.cm.plasma(pr.shared_devices / max_shared))

        if segments:
            lc = LineCollection(
                segments,
                linewidths=widths,
                colors=colors,
                alpha=0.75,
                zorder=4,
            )
            ax.add_collection(lc)

    # Cross-floor strongest pair (8009 ↔ 8023)
    cross = pairs.loc[
        (pairs["facility_num_x"].isin([66333]) & pairs["facility_num_y"].isin([66342]))
        | (pairs["facility_num_x"].isin([66342]) & pairs["facility_num_y"].isin([66333]))
    ]
    if len(cross):
        row = cross.iloc[0]
        fig.text(
            0.5,
            0.02,
            f"Cross-level overlap: {asset_label(layout, int(row.facility_num_x))} ↔ "
            f"{asset_label(layout, int(row.facility_num_y))} — "
            f"{int(row.shared_devices):,} shared devices",
            ha="center",
            fontsize=10,
            style="italic",
        )

    fig.suptitle(
        "Task 2 — Multi-sensor overlap on mall map (line width ∝ shared devices)",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_task3_transitions_on_maps(
    transitions: pd.DataFrame,
    facilities: pd.DataFrame,
    path: Path,
    min_count: int = 1500,
    top_n: int = 15,
) -> None:
    """Draw directed movement arrows between sensors on floor plans."""
    layout = sensor_layout(facilities)
    if layout.empty:
        return

    trans = transitions.nlargest(top_n, "transition_count").copy()
    trans = trans.loc[trans["transition_count"] >= min_count]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    map_order = ["lower_mall", "south_john"]
    max_count = float(trans["transition_count"].max()) if len(trans) else 1.0

    for ax, map_id in zip(axes, map_order):
        arr, w, h = _load_map_image(map_id)
        ax.imshow(arr, extent=[0, w, 0, h], aspect="auto")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")
        ax.set_title(f"Task 3 — journeys · {MAP_TITLES[map_id]}", fontsize=11)

        xy = _draw_sensor_nodes(ax, layout, map_id, w, h)

        for tr in trans.itertuples(index=False):
            f_from, f_to = int(tr.facility_num), int(tr.next_facility)
            if f_from not in xy or f_to not in xy:
                continue
            x1, y1 = xy[f_from]
            x2, y2 = xy[f_to]
            dx, dy = x2 - x1, y2 - y1
            dist = np.hypot(dx, dy) or 1.0
            shrink = 18.0
            x1s = x1 + shrink * dx / dist
            y1s = y1 + shrink * dy / dist
            x2s = x2 - shrink * dx / dist
            y2s = y2 - shrink * dy / dist
            weight = tr.transition_count / max_count
            lw = 1.4 + 5.5 * weight
            arrow = FancyArrowPatch(
                (x1s, y1s),
                (x2s, y2s),
                arrowstyle="-|>",
                mutation_scale=16 + 12 * weight,
                linewidth=lw,
                color="black",
                alpha=0.95,
                zorder=4,
            )
            ax.add_patch(arrow)

    cross = trans.loc[
        ((trans["facility_num"] == 66333) & (trans["next_facility"] == 66342))
        | ((trans["facility_num"] == 66342) & (trans["next_facility"] == 66333))
    ]
    if len(cross):
        row = cross.iloc[0]
        fig.text(
            0.5,
            0.02,
            f"Cross-level flow: {asset_label(layout, int(row.facility_num))} → "
            f"{asset_label(layout, int(row.next_facility))} — "
            f"{int(row.transition_count):,} transitions",
            ha="center",
            fontsize=10,
            style="italic",
        )

    fig.suptitle(
        "Task 3 — Sensor-to-sensor transitions (arrow width ∝ journey count)",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_task3_top_paths_on_map(
    top_paths: pd.DataFrame,
    facilities: pd.DataFrame,
    path: Path,
    n: int = 5,
) -> None:
    """Highlight top multi-sensor path strings on the lower-mall map."""
    layout = sensor_layout(facilities)
    if layout.empty or top_paths.empty:
        return

    fac_to_asset = dict(zip(layout["facility_num"].astype(int), layout["asset_id"]))
    arr, w, h = _load_map_image("lower_mall")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(arr, extent=[0, w, 0, h], aspect="auto")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    _draw_sensor_nodes(ax, layout, "lower_mall", w, h)

    lines = []
    top = top_paths.nlargest(n, "device_count")
    for i, row in enumerate(top.itertuples(index=False)):
        parts = [p.strip() for p in str(row.path).split("->")]
        assets = [fac_to_asset.get(int(p), p) for p in parts if p.strip().isdigit()]
        lines.append(f"{i + 1}. {' → '.join(assets)}  ({int(row.device_count):,} devices)")

    ax.set_title("Task 3 — Top journey paths (lower-mall sensors)", fontsize=11)
    fig.text(
        0.02,
        0.02,
        "\n".join(lines),
        transform=fig.transFigure,
        fontsize=8,
        va="bottom",
        family="monospace",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"),
    )
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_mall_map_plots(
    pair_counts: pd.DataFrame,
    transitions: pd.DataFrame,
    top_paths: pd.DataFrame,
    facilities: pd.DataFrame,
    output_dir: Path | None = None,
) -> Path:
    out = output_dir or PLOTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    plot_task2_overlap_on_maps(
        pair_counts, facilities, out / "task2_overlap_mall_map.png"
    )
    plot_task3_transitions_on_maps(
        transitions, facilities, out / "task3_transitions_mall_map.png"
    )
    plot_task3_top_paths_on_map(
        top_paths, facilities, out / "task3_top_paths_mall_map.png"
    )
    # Legacy filename used in report
    plot_task3_transitions_on_maps(
        transitions, facilities, out / "task3_sensor_map.png"
    )
    return out
