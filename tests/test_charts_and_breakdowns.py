import pandas as pd
import pytest

from src import charts
from src.appeal_queue import REASON_SHORT, build_appeal_queue
from src.claims_processor import load_claims, load_denials, merge_claims_denials
from src.financial_analytics import (
    appeal_summary, denial_rate_by_payer, denial_summary, denied_by_month, denied_dollars_by_stage,
    denials_by_reason,
)
from src.risk_engine import RULES, analyze_claims, breakdown_frame, score_if_corrected
from tests.test_financial_analytics import CLAIMS, DENIALS, build
from tests.test_risk_engine import AS_OF, make


def test_breakdown_lists_each_fired_rule_with_points_and_fix():
    df = breakdown_frame("MISSING_PATIENT_ID;INVALID_AMOUNT", "HIGH_AMOUNT")
    assert df["Item"].tolist() == [RULES["MISSING_PATIENT_ID"].label, RULES["INVALID_AMOUNT"].label, RULES["HIGH_AMOUNT"].label]
    assert df["Points"].tolist() == [30, 40, 10]
    assert df["Type"].tolist() == ["Confirmed data issue", "Confirmed data issue", "Risk indicator"]
    assert df["How to fix"].iloc[0] == RULES["MISSING_PATIENT_ID"].next_step
    assert breakdown_frame("", "").empty


def test_breakdown_points_match_the_engine_score():
    row = analyze_claims(make({"patient_id": "", "insurance_provider": "", "billed_amount": "20000"}), as_of=AS_OF).iloc[0]
    assert breakdown_frame(row["confirmed_issues"], row["risk_indicators"])["Points"].sum() == row["risk_score"]


def test_score_if_corrected_keeps_only_indicators():
    assert score_if_corrected("MISSING_PATIENT_ID;INVALID_AMOUNT", "HIGH_AMOUNT") == 10
    assert score_if_corrected("MISSING_PATIENT_ID", "") == 0
    assert score_if_corrected("", "POSSIBLE_DUPLICATE;HIGH_AMOUNT") == 35


def test_queue_points_columns_add_up_to_priority():
    q = build_appeal_queue(build(CLAIMS, DENIALS), as_of=pd.Timestamp("2025-05-01"))
    total = q["amount_points"] + q["status_points"] + q["age_points"]
    assert (total.round(1) == q["priority_score"]).all()
    assert set(q["reason_label"]) == {REASON_SHORT["CO18"], REASON_SHORT["PR96"]}


def test_denied_by_month_flags_partial_months():
    rows = [
        "B1,P1,E1,Aetna,Insurance,C1,01-01-2025 00:00,100,0,Denied,R",
        "B2,P2,E2,Aetna,Insurance,C2,01-01-2025 00:00,200,0,Denied,R",
        "B3,P3,E3,Aetna,Insurance,C3,01-01-2025 00:00,400,0,Denied,R",
        "B4,P4,E4,Aetna,Insurance,C4,01-01-2025 00:00,800,0,Paid,",
    ]
    denials = [
        "C1,D1,CO18,Dup,100,01-02-2025,No,,,",       # Feb 1 -> Feb is a full month at the start
        "C2,D2,CO18,Dup,200,15-02-2025,No,,,",
        "C3,D3,CO18,Dup,400,06-03-2025,No,,,",       # data ends Mar 6 -> March partial
    ]
    out = denied_by_month(build(rows, denials))
    assert out["denied"].tolist() == [2, 1]
    assert out["denied_amount"].tolist() == [300, 400]
    assert out["is_partial"].tolist() == [False, True]


def test_denied_by_month_empty_and_stage_table():
    assert denied_by_month(build(CLAIMS[:1], [])).empty
    stages = denied_dollars_by_stage(appeal_summary(build_appeal_queue(build(CLAIMS, DENIALS))))
    assert stages.set_index("stage").loc["Recovered on appeal", "amount"] == 2000
    assert stages.set_index("stage").loc["No appeal filed", "amount"] == 3000
    assert stages["amount"].sum() == 5000


def test_every_chart_builds_a_valid_spec():
    merged = build(CLAIMS, DENIALS)
    queue = build_appeal_queue(merged)
    summary = denial_summary(merged)
    specs = [
        charts.stage_chart(denied_dollars_by_stage(appeal_summary(queue))),
        charts.reasons_chart(denials_by_reason(merged)),
        charts.payer_chart(denial_rate_by_payer(merged), summary["denial_rate_pct"]),
        charts.monthly_chart(denied_by_month(merged)),
    ]
    for chart in specs:
        assert chart.to_dict()["$schema"].startswith("https://vega.github.io")
