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
STAGE_COLORS_DARK = {
    "Recovered on appeal": "#2DD4BF",
    "Written off": "#94A3B8",
    "Appeal pending": "#FCD34D",
    "No appeal filed": "#FB923C",
    "Status unclear": "#475569",
}
LEVEL_COLORS = {"Clean": "#2F855A", "Low": "#2B6CB0", "Medium": "#D69E2E", "High": "#C05621"}
LEVEL_COLORS_DARK = {"Clean": "#6EE7B7", "Low": "#93C5FD", "Medium": "#FCD34D", "High": "#FCA5A5"}

_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="-apple-system,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif", color=INK, size=13),
    margin=dict(l=10, r=10, t=50, b=15),
    height=330,
    hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#CDD8E3",
                    font=dict(family="-apple-system,'Segoe UI',Roboto,Arial,sans-serif", size=13, color="#0F2340"), align="left"),
)

_LAYOUT_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="-apple-system,'Segoe UI',Roboto,Arial,sans-serif", color="#E6EDF3", size=13),
    margin=dict(l=10, r=10, t=50, b=15),
    height=330,
    hoverlabel=dict(bgcolor="#21293A", bordercolor="#364155",
                    font=dict(family="-apple-system,'Segoe UI',Roboto,Arial,sans-serif", size=13, color="#E6EDF3"), align="left"),
)


def _finish(fig: go.Figure, title: str, height: int | None = None, dark: bool = False) -> go.Figure:
    layout = dict(_LAYOUT_DARK if dark else _LAYOUT)
    if height:
        layout["height"] = height
    title_color = "#2DD4BF" if dark else "#0F2340"
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=14, color=title_color, weight=700)),
        **layout,
    )
    grid_color = "#2A3547" if dark else "#E2E8F0"
    fig.update_xaxes(gridcolor=grid_color, zeroline=False, tickfont=dict(size=12))
    fig.update_yaxes(gridcolor=grid_color, zeroline=False, tickfont=dict(size=12))
    return fig


def stage_fig(stages: pd.DataFrame, dark: bool = False) -> go.Figure:
    """Donut: where the denied dollars stand."""
    data = stages[stages["amount"] > 0]
    colors_map = STAGE_COLORS_DARK if dark else STAGE_COLORS
    line_color = "#161B22" if dark else "white"
    fig = go.Figure(
        go.Pie(
            labels=data["stage"].tolist(),
            values=data["amount"].tolist(),
            customdata=data["claims"].tolist(),
            hole=0.6,
            sort=False,
            marker=dict(colors=[colors_map.get(s, LIGHT) for s in data["stage"]], line=dict(color=line_color, width=2)),
            textinfo="percent",
            hovertemplate="%{label}<br>$%{value:,.0f}<br>%{customdata:,} claims<extra></extra>",
        )
    )
    fig.update_layout(legend=dict(orientation="v", x=1.0, y=0.5))
    return _finish(fig, "Where denied dollars stand", dark=dark)


def reasons_fig(by_reason: pd.DataFrame, top_n: int = 8, dark: bool = False) -> go.Figure:
    """Horizontal bars: top denial reasons by dollars."""
    data = by_reason.head(top_n)
    labels = [
        REASON_SHORT.get(code, desc)
        for code, desc in zip(data["denial_reason_code"], data["denial_reason_description"])
    ]
    teal_color = "#2DD4BF" if dark else TEAL
    fig = go.Figure(
        go.Bar(
            y=labels,
            x=data["denied_amount"].tolist(),
            orientation="h",
            marker_color=teal_color,
            customdata=data[["denial_reason_code", "denied"]].values.tolist(),
            hovertemplate="%{y} (%{customdata[0]})<br>$%{x:,.0f} across %{customdata[1]:,} denials<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickprefix="$", tickformat="~s", title="Denied amount")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=15))
    return _finish(fig, f"Top {len(data)} denial reasons by dollars", dark=dark)


def payer_fig(by_payer: pd.DataFrame, overall_rate_pct: float, dark: bool = False) -> go.Figure:
    """Denial rate by payer (zero-based) against the overall average."""
    navy_color = "#60A5FA" if dark else NAVY
    amber_color = "#FCD34D" if dark else AMBER
    fig = go.Figure(
        go.Bar(
            x=by_payer["insurance_provider"].tolist(),
            y=by_payer["denial_rate_pct"].tolist(),
            marker_color=navy_color,
            text=[f"{r:.1f}" for r in by_payer["denial_rate_pct"]],
            textposition="inside",
            textfont=dict(color="white" if not dark else "#0D1117"),
            customdata=by_payer[["claims", "denied"]].values.tolist(),
            hovertemplate="%{x}<br>%{y:.2f}% (%{customdata[1]:,} of %{customdata[0]:,} claims)<extra></extra>",
        )
    )
    top = float(by_payer["denial_rate_pct"].max()) if len(by_payer) else overall_rate_pct
    fig.add_hline(
        y=overall_rate_pct, line_dash="dash", line_color=amber_color, line_width=2,
        annotation_text=f"Overall {overall_rate_pct:.1f}%", annotation_position="top right",
    )
    fig.update_yaxes(range=[0, max(top, overall_rate_pct) * 1.2], title="Denial rate (%)")
    return _finish(fig, "Denial rate by payer", dark=dark)


def monthly_fig(monthly: pd.DataFrame, dark: bool = False) -> go.Figure:
    """Denied dollars by month; partly covered months are lighter."""
    labels = monthly["month"].dt.strftime("%b %Y").tolist()
    if dark:
        colors = ["#94A3B8" if p else "#2DD4BF" for p in monthly["is_partial"]]
    else:
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
    return _finish(fig, "Denied dollars by month of denial", dark=dark)


def level_fig(counts: dict[str, int], dark: bool = False) -> go.Figure:
    """Claims by risk level."""
    levels = list(LEVEL_COLORS)
    colors_map = LEVEL_COLORS_DARK if dark else LEVEL_COLORS
    fig = go.Figure(
        go.Bar(
            y=levels,
            x=[counts.get(level, 0) for level in levels],
            orientation="h",
            marker_color=[colors_map[level] for level in levels],
            hovertemplate="%{y}: %{x:,} claims<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Claims")
    return _finish(fig, "Claims by risk level", height=230, dark=dark)
