"""Plotly charts for the Dash dashboard (pure Python, runs on s390x).

IMPORTANT: every data argument is passed as a plain Python list, never a numpy array or pandas
Series. Plotly 6+ packs numpy arrays into little-endian binary ("bdata") for the browser; on a
big-endian machine such as IBM Z (s390x) the browser then reads garbage numbers. Plain lists are
sent as ordinary JSON numbers and are safe on any architecture.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .appeal_queue import REASON_SHORT

TEAL = "#0F766E"
NAVY = "#12344D"
SLATE = "#64748B"
AMBER = "#D97706"
CORAL = "#C2410C"
LIGHT = "#CBD5E1"
INK = "#1B2A3A"
PALE_TEAL = "#99C9C4"

STAGE_COLORS = {
    "Recovered on appeal": TEAL,
    "Written off": SLATE,
    "Appeal pending": AMBER,
    "No appeal filed": CORAL,
    "Status unclear": LIGHT,
}
LEVEL_COLORS = {"Clean": "#2F855A", "Low": "#2B6CB0", "Medium": "#D69E2E", "High": "#C05621"}

_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=INK, size=13),
    margin=dict(l=10, r=10, t=55, b=15),
    height=330,
)


def _finish(fig: go.Figure, title: str, height: int | None = None) -> go.Figure:
    layout = dict(_LAYOUT)
    if height:
        layout["height"] = height
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=16, color=NAVY)), **layout)
    fig.update_xaxes(gridcolor="#E2E8F0", zeroline=False)
    fig.update_yaxes(gridcolor="#E2E8F0", zeroline=False)
    return fig


def stage_fig(stages: pd.DataFrame) -> go.Figure:
    """Donut: where the denied dollars stand."""
    data = stages[stages["amount"] > 0]
    fig = go.Figure(
        go.Pie(
            labels=data["stage"].tolist(),
            values=data["amount"].tolist(),
            customdata=data["claims"].tolist(),
            hole=0.6,
            sort=False,
            marker=dict(colors=[STAGE_COLORS.get(s, LIGHT) for s in data["stage"]], line=dict(color="white", width=2)),
            textinfo="percent",
            hovertemplate="%{label}<br>$%{value:,.0f}<br>%{customdata:,} claims<extra></extra>",
        )
    )
    fig.update_layout(legend=dict(orientation="v", x=1.0, y=0.5))
    return _finish(fig, "Where denied dollars stand")


def reasons_fig(by_reason: pd.DataFrame, top_n: int = 8) -> go.Figure:
    """Horizontal bars: top denial reasons by dollars."""
    data = by_reason.head(top_n)
    labels = [
        REASON_SHORT.get(code, desc)
        for code, desc in zip(data["denial_reason_code"], data["denial_reason_description"])
    ]
    fig = go.Figure(
        go.Bar(
            y=labels,
            x=data["denied_amount"].tolist(),
            orientation="h",
            marker_color=TEAL,
            customdata=data[["denial_reason_code", "denied"]].values.tolist(),
            hovertemplate="%{y} (%{customdata[0]})<br>$%{x:,.0f} across %{customdata[1]:,} denials<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickprefix="$", tickformat="~s", title="Denied amount")
    fig.update_layout(margin=dict(l=10, r=10, t=55, b=15))
    return _finish(fig, f"Top {len(data)} denial reasons by dollars")


def payer_fig(by_payer: pd.DataFrame, overall_rate_pct: float) -> go.Figure:
    """Denial rate by payer (zero-based) against the overall average."""
    fig = go.Figure(
        go.Bar(
            x=by_payer["insurance_provider"].tolist(),
            y=by_payer["denial_rate_pct"].tolist(),
            marker_color=NAVY,
            text=[f"{r:.1f}" for r in by_payer["denial_rate_pct"]],
            textposition="inside",
            textfont=dict(color="white"),
            customdata=by_payer[["claims", "denied"]].values.tolist(),
            hovertemplate="%{x}<br>%{y:.2f}% (%{customdata[1]:,} of %{customdata[0]:,} claims)<extra></extra>",
        )
    )
    top = float(by_payer["denial_rate_pct"].max()) if len(by_payer) else overall_rate_pct
    fig.add_hline(
        y=overall_rate_pct, line_dash="dash", line_color=AMBER, line_width=2,
        annotation_text=f"Overall {overall_rate_pct:.1f}%", annotation_position="top right",
    )
    fig.update_yaxes(range=[0, max(top, overall_rate_pct) * 1.2], title="Denial rate (%)")
    return _finish(fig, "Denial rate by payer")


def monthly_fig(monthly: pd.DataFrame) -> go.Figure:
    """Denied dollars by month; partly covered months are lighter."""
    labels = monthly["month"].dt.strftime("%b %Y").tolist()
    colors = [PALE_TEAL if p else TEAL for p in monthly["is_partial"]]
    cover = ["Partial month in data" if p else "Full month" for p in monthly["is_partial"]]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=monthly["denied_amount"].tolist(),
            marker_color=colors,
            customdata=[[int(n), c] for n, c in zip(monthly["denied"], cover)],
            hovertemplate="%{x}<br>$%{y:,.0f} across %{customdata[0]:,} denials<br>%{customdata[1]}<extra></extra>",
        )
    )
    fig.update_yaxes(tickprefix="$", tickformat="~s", title="Denied amount")
    return _finish(fig, "Denied dollars by month of denial")


def level_fig(counts: dict[str, int]) -> go.Figure:
    """Claims by risk level."""
    levels = list(LEVEL_COLORS)
    fig = go.Figure(
        go.Bar(
            y=levels,
            x=[counts.get(level, 0) for level in levels],
            orientation="h",
            marker_color=[LEVEL_COLORS[level] for level in levels],
            hovertemplate="%{y}: %{x:,} claims<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Claims")
    return _finish(fig, "Claims by risk level", height=230)
