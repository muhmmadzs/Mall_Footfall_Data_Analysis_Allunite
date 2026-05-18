"""Interactive Plotly charts — Plan A vs Plan B, mall visitors, all facilities."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import pandas as pd


def _require_plotly():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:
        raise ImportError("Install plotly: pip install plotly") from e
    return go, make_subplots


def _show(fig) -> None:
    fig.show()


def _facility_labels(facilities: Optional[pd.DataFrame], facility_nums: Sequence[int]) -> Dict[int, str]:
    if facilities is None or facilities.empty:
        return {int(f): str(f) for f in facility_nums}
    lookup = facilities.drop_duplicates("facility_num").set_index("facility_num")
    out: Dict[int, str] = {}
    for f in facility_nums:
        f = int(f)
        if f in lookup.index:
            name = str(lookup.loc[f].get("facility_name", f))
            out[f] = f"{f} — {name}"
        else:
            out[f] = str(f)
    return out


def plot_comparison_interactive(comp: Optional[pd.DataFrame] = None):
    """Daily Plan A vs Plan B + mall visitors."""
    go, make_subplots = _require_plotly()

    if comp is None or comp.empty:
        print("No comparison data.")
        return None

    comp = comp.copy()
    comp["date"] = pd.to_datetime(comp["date"])

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=comp["date"],
            y=comp["footfall_plan_a"],
            name="Plan A (sensor-hour sum)",
            mode="lines+markers",
            line=dict(color="#4a90d9", width=2),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=comp["date"],
            y=comp["footfall_plan_b"],
            name="Plan B (sensor-hour sum)",
            mode="lines+markers",
            line=dict(color="#e67e22", width=2),
        ),
        secondary_y=False,
    )
    if "estimated_mall_visitors" in comp.columns:
        fig.add_trace(
            go.Bar(
                x=comp["date"],
                y=comp["estimated_mall_visitors"],
                name="Mall visitors",
                marker_color="rgba(39, 174, 96, 0.45)",
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title="Daily — Plan A vs Plan B & mall visitors",
        hovermode="x unified",
        height=480,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="Sensor-hour footfall (sum)", secondary_y=False)
    fig.update_yaxes(title_text="Mall visitors", secondary_y=True)
    _show(fig)
    return fig


def plot_mall_visitors_hourly_interactive(mall_hourly: Optional[pd.DataFrame] = None):
    """Hourly mall-wide visitor estimate."""
    go, _ = _require_plotly()

    if mall_hourly is None or mall_hourly.empty:
        print("No mall hourly data.")
        return None

    mh = mall_hourly.sort_values("hour_start")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mh["hour_start"],
            y=mh["estimated_mall_visitors"],
            mode="lines",
            name="Mall visitors",
            line=dict(color="#27ae60", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(39, 174, 96, 0.15)",
        )
    )
    fig.update_layout(
        title="Hourly estimated mall visitors (all locations deduped)",
        xaxis_title="Hour",
        yaxis_title="Visitors",
        height=400,
        template="plotly_white",
        hovermode="x unified",
    )
    _show(fig)
    return fig


def plot_all_facilities_hourly_grid(
    hourly: pd.DataFrame,
    facilities: Optional[pd.DataFrame] = None,
):
    """3×3 grid: hourly Plan A vs Plan B for every sensor."""
    go, make_subplots = _require_plotly()

    facs = sorted(hourly["facility_num"].astype(int).unique())
    labels = _facility_labels(facilities, facs)
    n = len(facs)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[labels[f] for f in facs],
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
    )

    for i, fac in enumerate(facs):
        row, col = i // ncols + 1, i % ncols + 1
        g = hourly.loc[hourly["facility_num"].astype(int) == fac].sort_values("hour_start")
        fig.add_trace(
            go.Scatter(
                x=g["hour_start"],
                y=g["footfall_plan_a"],
                name="Plan A",
                legendgroup="plan_a",
                showlegend=(i == 0),
                line=dict(color="#4a90d9", width=1.2),
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=g["hour_start"],
                y=g["footfall_plan_b"],
                name="Plan B",
                legendgroup="plan_b",
                showlegend=(i == 0),
                line=dict(color="#e67e22", width=1.2),
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title="All locations — hourly footfall (Plan A vs Plan B)",
        height=280 * nrows,
        template="plotly_white",
        hovermode="x unified",
    )
    _show(fig)
    return fig


def plot_all_facilities_daily_heatmaps(
    hourly: pd.DataFrame,
    facilities: Optional[pd.DataFrame] = None,
):
    """Heatmaps: facility × day for Plan A and Plan B."""
    go, make_subplots = _require_plotly()

    h = hourly.copy()
    h["date"] = pd.to_datetime(h["hour_start"]).dt.date
    facs = sorted(h["facility_num"].astype(int).unique())
    labels = _facility_labels(facilities, facs)
    y_labels = [labels[f] for f in facs]

    def pivot(col: str) -> pd.DataFrame:
        p = h.pivot_table(
            index="facility_num", columns="date", values=col, aggfunc="sum", fill_value=0
        )
        return p.reindex(facs)

    pa = pivot("footfall_plan_a")
    pb = pivot("footfall_plan_b")
    x_labels = [str(d) for d in pa.columns]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Plan A — daily footfall per location", "Plan B — daily footfall per location"],
        horizontal_spacing=0.12,
    )
    fig.add_trace(
        go.Heatmap(z=pa.values, x=x_labels, y=y_labels, colorscale="Blues", coloraxis="coloraxis"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(z=pb.values, x=x_labels, y=y_labels, colorscale="Oranges", coloraxis="coloraxis2"),
        row=1,
        col=2,
    )
    fig.update_layout(
        title="All locations — daily footfall heatmap",
        height=420,
        template="plotly_white",
        coloraxis=dict(colorscale="Blues"),
        coloraxis2=dict(colorscale="Oranges"),
    )
    _show(fig)
    return fig


def plot_facility_dropdown_interactive(
    hourly: pd.DataFrame,
    facilities: Optional[pd.DataFrame] = None,
    default_facility: Optional[int] = None,
):
    """Single large chart with dropdown to switch facility."""
    go, _ = _require_plotly()

    facs = sorted(hourly["facility_num"].astype(int).unique())
    labels = _facility_labels(facilities, facs)
    default_facility = int(default_facility or facs[0])

    traces = []
    for fi, fac in enumerate(facs):
        g = hourly.loc[hourly["facility_num"].astype(int) == fac].sort_values("hour_start")
        traces.append(
            go.Scatter(
                x=g["hour_start"],
                y=g["footfall_plan_a"],
                mode="lines",
                name="Plan A",
                line=dict(color="#4a90d9", width=2),
                visible=(fac == default_facility),
            )
        )
        traces.append(
            go.Scatter(
                x=g["hour_start"],
                y=g["footfall_plan_b"],
                mode="lines",
                name="Plan B",
                line=dict(color="#e67e22", width=2),
                visible=(fac == default_facility),
            )
        )

    buttons = []
    for fi, fac in enumerate(facs):
        vis = [False] * (2 * len(facs))
        vis[2 * fi] = True
        vis[2 * fi + 1] = True
        buttons.append(
            dict(
                label=labels[fac],
                method="update",
                args=[{"visible": vis}, {"title": f"Hourly footfall — {labels[fac]}"}],
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Hourly footfall — {labels[default_facility]}",
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.12,
            )
        ],
        height=450,
        template="plotly_white",
        hovermode="x unified",
        yaxis_title="Estimated footfall",
    )
    _show(fig)
    return fig


def plot_footfall_dashboard_interactive(
    comp: Optional[pd.DataFrame] = None,
    hourly: Optional[pd.DataFrame] = None,
    mall_hourly: Optional[pd.DataFrame] = None,
    facilities: Optional[pd.DataFrame] = None,
):
    """Show all interactive views: daily, mall hourly, all-location grid, heatmaps, facility dropdown."""
    if hourly is None or hourly.empty:
        print("Missing hourly data — run run_both_plans() first.")
        return None

    figs = {
        "daily": plot_comparison_interactive(comp),
        "mall_hourly": plot_mall_visitors_hourly_interactive(mall_hourly),
        "all_locations_grid": plot_all_facilities_hourly_grid(hourly, facilities),
        "all_locations_heatmap": plot_all_facilities_daily_heatmaps(hourly, facilities),
        "facility_dropdown": plot_facility_dropdown_interactive(hourly, facilities),
    }
    return figs
