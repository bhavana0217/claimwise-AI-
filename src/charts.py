"""Altair charts for the dashboard (calm clinical palette, readable labels)."""
from __future__ import annotations

import altair as alt
import pandas as pd

from .appeal_queue import REASON_SHORT

TEAL = "#0F766E"
NAVY = "#12344D"
SLATE = "#64748B"
AMBER = "#D97706"
CORAL = "#C2410C"
LIGHT = "#CBD5E1"
INK = "#1B2A3A"

STAGE_COLORS = {
    "Recovered on appeal": TEAL,
    "Written off": SLATE,
    "Appeal pending": AMBER,
    "No appeal filed": CORAL,
    "Status unclear": LIGHT,
}


def _style(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(labelColor=INK, titleColor=SLATE, gridColor="#E2E8F0", domainColor=LIGHT, labelFontSize=12)
        .configure_legend(labelColor=INK, titleColor=SLATE, labelFontSize=12)
        .configure_title(color=NAVY, fontSize=15, anchor="start", fontWeight=600)
    )


def stage_chart(stages: pd.DataFrame) -> alt.Chart:
    """Donut: where the denied dollars stand."""
    data = stages[stages["amount"] > 0].copy()
    data["share"] = data["amount"] / data["amount"].sum() if len(data) else 0
    domain = list(STAGE_COLORS)
    chart = (
        alt.Chart(data, title="Where denied dollars stand")
        .mark_arc(innerRadius=62, outerRadius=105, stroke="white", strokeWidth=2)
        .encode(
            theta=alt.Theta("amount:Q", stack=True),
            color=alt.Color(
                "stage:N",
                scale=alt.Scale(domain=domain, range=[STAGE_COLORS[d] for d in domain]),
                legend=alt.Legend(title=None, orient="right"),
            ),
            order=alt.Order("amount:Q", sort="descending"),
            tooltip=[
                alt.Tooltip("stage:N", title="Stage"),
                alt.Tooltip("amount:Q", title="Denied amount", format="$,.0f"),
                alt.Tooltip("claims:Q", title="Claims", format=","),
                alt.Tooltip("share:Q", title="Share of denied $", format=".1%"),
            ],
        )
        .properties(height=280)
    )
    return _style(chart)


def reasons_chart(by_reason: pd.DataFrame, top_n: int = 8) -> alt.Chart:
    """Horizontal bars: top denial reasons by dollars, with readable names."""
    data = by_reason.head(top_n).copy()
    data["reason"] = [
        REASON_SHORT.get(code, desc) for code, desc in zip(data["denial_reason_code"], data["denial_reason_description"])
    ]
    chart = (
        alt.Chart(data, title=f"Top {len(data)} denial reasons by dollars")
        .mark_bar(color=TEAL, cornerRadiusEnd=4)
        .encode(
            y=alt.Y("reason:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("denied_amount:Q", title="Denied amount ($)", axis=alt.Axis(format="$,.0s")),
            tooltip=[
                alt.Tooltip("reason:N", title="Reason"),
                alt.Tooltip("denial_reason_code:N", title="Code"),
                alt.Tooltip("denied:Q", title="Denials", format=","),
                alt.Tooltip("denied_amount:Q", title="Denied amount", format="$,.0f"),
            ],
        )
        .properties(height=280)
    )
    return _style(chart)


def payer_chart(by_payer: pd.DataFrame, overall_rate_pct: float) -> alt.Chart:
    """Denial rate by payer (zero-based bars) against the overall average."""
    bars = (
        alt.Chart(by_payer, title="Denial rate by payer")
        .mark_bar(color=NAVY, cornerRadiusEnd=4)
        .encode(
            x=alt.X("insurance_provider:N", sort="-y", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("denial_rate_pct:Q", title="Denial rate (%)", scale=alt.Scale(domainMin=0)),
            tooltip=[
                alt.Tooltip("insurance_provider:N", title="Payer"),
                alt.Tooltip("claims:Q", title="Claims", format=","),
                alt.Tooltip("denied:Q", title="Denied", format=","),
                alt.Tooltip("denial_rate_pct:Q", title="Denial rate (%)", format=".2f"),
            ],
        )
    )
    labels = bars.mark_text(dy=16, color="white", fontSize=12, fontWeight=600).encode(
        text=alt.Text("denial_rate_pct:Q", format=".1f")
    )
    rule_df = pd.DataFrame({"rate": [overall_rate_pct]})
    rule = alt.Chart(rule_df).mark_rule(color=AMBER, strokeDash=[6, 4], strokeWidth=2).encode(y="rate:Q")
    return _style((bars + labels + rule).properties(height=280))


def monthly_chart(monthly: pd.DataFrame) -> alt.Chart:
    """Denied dollars by month; partly covered months are drawn lighter."""
    data = monthly.copy()
    data["month_label"] = data["month"].dt.strftime("%b %Y")
    data["coverage"] = data["is_partial"].map({True: "Partial month in data", False: "Full month"})
    chart = (
        alt.Chart(data, title="Denied dollars by month of denial")
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("month_label:N", sort=data["month_label"].tolist(), title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("denied_amount:Q", title="Denied amount ($)", axis=alt.Axis(format="$,.0s")),
            color=alt.Color(
                "coverage:N",
                scale=alt.Scale(domain=["Full month", "Partial month in data"], range=[TEAL, "#99C9C4"]),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("month_label:N", title="Month"),
                alt.Tooltip("denied:Q", title="Denials", format=","),
                alt.Tooltip("denied_amount:Q", title="Denied amount", format="$,.0f"),
                alt.Tooltip("coverage:N", title="Coverage"),
            ],
        )
        .properties(height=280)
    )
    return _style(chart)
