import pandas as pd

from src import plotly_charts as pc
from src.appeal_queue import REASON_SHORT, build_appeal_queue
from src.financial_analytics import (
    appeal_summary, denial_rate_by_payer, denial_summary, denied_by_month, denied_dollars_by_stage,
    denials_by_reason,
)
from tests.test_financial_analytics import CLAIMS, DENIALS, build


def test_stage_donut_has_one_slice_per_nonzero_stage_and_correct_total():
    stages = denied_dollars_by_stage(appeal_summary(build_appeal_queue(build(CLAIMS, DENIALS))))
    fig = pc.stage_fig(stages)
    pie = fig.data[0]
    assert sorted(pie.labels) == ["No appeal filed", "Recovered on appeal"]
    assert sum(pie.values) == 5000


def test_reasons_chart_uses_readable_names_largest_first():
    fig = pc.reasons_fig(denials_by_reason(build(CLAIMS, DENIALS)))
    bar = fig.data[0]
    assert list(bar.y) == [REASON_SHORT["PR96"], REASON_SHORT["CO18"]]
    assert list(bar.x) == [3000, 2000]


def test_payer_chart_is_zero_based_and_marks_overall_rate():
    merged = build(CLAIMS, DENIALS)
    fig = pc.payer_fig(denial_rate_by_payer(merged), denial_summary(merged)["denial_rate_pct"])
    assert fig.layout.yaxis.range[0] == 0
    assert any(s.y0 == 50.0 for s in fig.layout.shapes)  # dashed overall-rate line


def test_monthly_chart_lightens_partial_months():
    rows = ["B1,P1,E1,Aetna,Insurance,C1,01-01-2025 00:00,100,0,Denied,R",
            "B2,P2,E2,Aetna,Insurance,C2,01-01-2025 00:00,200,0,Denied,R"]
    denials = ["C1,D1,CO18,Dup,100,01-02-2025,No,,,", "C2,D2,CO18,Dup,200,10-03-2025,No,,,"]
    monthly = denied_by_month(build(rows, denials))
    fig = pc.monthly_fig(monthly)
    assert list(fig.data[0].marker.color) == [pc.TEAL, pc.PALE_TEAL]


def test_level_chart_fills_missing_levels_with_zero():
    fig = pc.level_fig({"Clean": 5, "High": 2})
    assert list(fig.data[0].y) == ["Clean", "Low", "Medium", "High"]
    assert list(fig.data[0].x) == [5, 0, 0, 2]


def test_no_chart_uses_binary_encoded_arrays_which_break_on_big_endian_machines():
    """Regression test: numpy arrays become little-endian 'bdata' blobs, unreadable on IBM Z (s390x)."""
    merged = build(CLAIMS, DENIALS)
    queue = build_appeal_queue(merged)
    summary = denial_summary(merged)
    rows = ["B1,P1,E1,Aetna,Insurance,C1,01-01-2025 00:00,100,0,Denied,R"]
    monthly = denied_by_month(build(rows, ["C1,D1,CO18,Dup,100,01-02-2025,No,,,"]))
    figs = [
        pc.stage_fig(denied_dollars_by_stage(appeal_summary(queue))),
        pc.reasons_fig(denials_by_reason(merged)),
        pc.payer_fig(denial_rate_by_payer(merged), summary["denial_rate_pct"]),
        pc.monthly_fig(monthly),
        pc.level_fig({"Clean": 1}),
    ]
    for fig in figs:
        assert "bdata" not in fig.to_json()
