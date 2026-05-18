"""Interactive comparison: Plan A vs Plan B + mall visitors."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def plot_comparison_interactive(comp: Optional[pd.DataFrame] = None):
    """Plotly chart — sensor-hour footfall (A vs B) and mall visitors."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:
        raise ImportError("Install plotly: pip install plotly") from e

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
                name="Mall visitors (dedup × capture)",
                marker_color="rgba(39, 174, 96, 0.45)",
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title="Plan A vs Plan B — daily footfall & mall visitors",
        hovermode="x unified",
        height=480,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="Sensor-hour footfall (sum)", secondary_y=False)
    fig.update_yaxes(title_text="Estimated mall visitors", secondary_y=True)
    fig.show()
    return fig
