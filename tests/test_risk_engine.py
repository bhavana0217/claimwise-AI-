import pandas as pd
import pytest

from src.claims_processor import CLAIM_COLUMNS, clean_claims, load_claims
from src.risk_engine import (
    ADDED_COLUMNS,
    CONFIRMED_CODES,
    INDICATOR_CODES,
    RULES,
    analyze_claims,
    risk_level_for,
)

AS_OF = pd.Timestamp("2025-06-30")


def make(*overrides):
    """Build cleaned claims. Each override dict changes fields of a valid base claim."""
    rows = []
    for n, over in enumerate(overrides, start=1):
        row = {
            "billing_id": f"B{n}", "patient_id": f"P{n}", "encounter_id": f"E{n}",
            "insurance_provider": "Aetna", "payment_method": "Insurance", "claim_id": f"C{n}",
            "claim_billing_date": "05-03-2025 00:00", "billed_amount": "1000",
            "paid_amount": "0", "claim_status": "Pending", "denial_reason": "",
        }
        row.update({k: str(v) for k, v in over.items()})
        rows.append(row)
    return clean_claims(pd.DataFrame(rows, columns=CLAIM_COLUMNS))


def run(*overrides):
    return analyze_claims(make(*overrides), as_of=AS_OF)


def test_clean_claim_scores_zero():
    out = run({})
    row = out.iloc[0]
    assert row["risk_score"] == 0 and row["risk_level"] == "Clean"
    assert not row["needs_review"] and row["finding_type"] == "None"
    assert row["explanation"] == "No rule-based issues found."


@pytest.mark.parametrize(
    "code, override",
    [
        ("MISSING_CLAIM_ID", {"claim_id": ""}),
        ("MISSING_PATIENT_ID", {"patient_id": ""}),
        ("MISSING_ENCOUNTER_ID", {"encounter_id": ""}),
        ("MISSING_PAYER", {"insurance_provider": ""}),
        ("INVALID_AMOUNT", {"billed_amount": "0"}),
        ("INVALID_AMOUNT", {"billed_amount": "-5"}),
        ("INVALID_AMOUNT", {"billed_amount": "abc"}),
        ("MISSING_DATE", {"claim_billing_date": ""}),
        ("MISSING_DATE", {"claim_billing_date": "garbage"}),
        ("FUTURE_DATE", {"claim_billing_date": "01-01-2030 00:00"}),
        ("PAID_EXCEEDS_BILLED", {"paid_amount": "1500", "claim_status": "Paid"}),
        ("PAID_STATUS_NO_PAYMENT", {"claim_status": "Paid", "paid_amount": "0"}),
        ("DENIED_STATUS_WITH_PAYMENT", {"claim_status": "Denied", "paid_amount": "50", "denial_reason": "X"}),
        ("DENIED_NO_REASON", {"claim_status": "Denied"}),
        ("HIGH_AMOUNT", {"billed_amount": "10000.01"}),
    ],
)
def test_each_single_rule_fires(code, override):
    row = run(override).iloc[0]
    fired = [c for c in (row["confirmed_issues"] + ";" + row["risk_indicators"]).split(";") if c]
    assert code in fired


def test_status_and_amount_rules_do_not_fire_on_valid_claims():
    out = run(
        {"claim_status": "Paid", "paid_amount": "600"},
        {"claim_status": "Denied", "denial_reason": "Duplicate Claim"},
        {"billed_amount": "10000"},  # exactly at threshold is not "over"
    )
    assert (out["risk_score"] == 0).all()


def test_negative_billed_amount_does_not_also_trigger_paid_exceeds_billed():
    row = run({"billed_amount": "-250", "paid_amount": "0"}).iloc[0]
    assert row["confirmed_issues"] == "INVALID_AMOUNT"


def test_duplicate_claim_id_flags_every_row_sharing_the_id():
    out = run({"claim_id": "SAME"}, {"claim_id": "SAME"}, {"claim_id": "OTHER"})
    assert out["confirmed_issues"].tolist() == ["DUPLICATE_CLAIM_ID", "DUPLICATE_CLAIM_ID", ""]


def test_possible_duplicate_is_an_indicator_not_a_confirmed_issue():
    same = {"patient_id": "PX", "billed_amount": "800"}
    out = run(same, same)
    for _, row in out.iterrows():
        assert row["risk_indicators"] == "POSSIBLE_DUPLICATE"
        assert row["confirmed_issues"] == ""
        assert row["finding_type"] == "Risk indicator only"
        assert row["risk_score"] == RULES["POSSIBLE_DUPLICATE"].weight
        assert row["needs_review"]  # 25 reaches the review threshold


def test_same_claim_id_and_content_reports_only_duplicate_claim_id():
    same = {"patient_id": "PX", "billed_amount": "800", "claim_id": "SAME"}
    out = run(same, same)
    assert out["confirmed_issues"].tolist() == ["DUPLICATE_CLAIM_ID"] * 2
    assert out["risk_indicators"].tolist() == ["", ""]


def test_indicator_only_below_threshold_does_not_need_review():
    row = run({"billed_amount": "20000"}).iloc[0]
    assert row["risk_score"] == RULES["HIGH_AMOUNT"].weight == 10
    assert row["finding_type"] == "Risk indicator only"
    assert not row["needs_review"]


def test_scores_stay_within_0_to_100_and_cap_applies():
    worst = {"patient_id": "", "insurance_provider": "", "billed_amount": "0", "claim_id": "",
             "encounter_id": "", "claim_billing_date": "", "claim_status": "Paid"}
    out = run(worst, {})
    assert out.iloc[0]["risk_score"] == 100  # raw sum exceeds 100, capped
    assert out["risk_score"].between(0, 100).all()


def test_score_equals_sum_of_documented_weights():
    row = run({"patient_id": "", "insurance_provider": ""}).iloc[0]
    assert row["risk_score"] == RULES["MISSING_PATIENT_ID"].weight + RULES["MISSING_PAYER"].weight


def test_explanation_matches_the_flags_that_fired():
    out = run({"patient_id": "", "billed_amount": "20000"})
    row = out.iloc[0]
    assert RULES["MISSING_PATIENT_ID"].label in row["explanation"]
    assert RULES["HIGH_AMOUNT"].label in row["explanation"]
    assert RULES["MISSING_PAYER"].label not in row["explanation"]
    assert "not a prediction" in row["explanation"]  # indicators are labeled as such
    assert RULES["MISSING_PATIENT_ID"].next_step in row["recommended_next_steps"]
    assert RULES["HIGH_AMOUNT"].next_step in row["recommended_next_steps"]


def test_future_date_depends_on_as_of():
    claims = make({"claim_billing_date": "15-07-2025 00:00"})
    assert analyze_claims(claims, as_of=pd.Timestamp("2025-06-30")).iloc[0]["confirmed_issues"] == "FUTURE_DATE"
    assert analyze_claims(claims, as_of=pd.Timestamp("2025-08-01")).iloc[0]["confirmed_issues"] == ""


@pytest.mark.parametrize(
    "score, level",
    [(0, "Clean"), (1, "Low"), (24, "Low"), (25, "Medium"), (49, "Medium"), (50, "High"), (100, "High")],
)
def test_risk_level_boundaries(score, level):
    assert risk_level_for(score) == level


def test_empty_input_returns_expected_columns():
    out = analyze_claims(make(), as_of=AS_OF)
    assert out.empty
    assert all(c in out.columns for c in ADDED_COLUMNS)


def test_rule_registry_is_consistent():
    assert set(CONFIRMED_CODES) | set(INDICATOR_CODES) == set(RULES)
    assert not set(CONFIRMED_CODES) & set(INDICATOR_CODES)
    assert all(0 < r.weight <= 100 for r in RULES.values())


def test_synthetic_demo_set_expected_results():
    claims, _ = load_claims("data/sample_claims.csv")
    out = analyze_claims(claims, as_of=pd.Timestamp("2025-06-30")).set_index("billing_id")
    # six clean claims
    assert (out.loc["SYN-B001":"SYN-B006", "risk_score"] == 0).all()
    assert out.loc["SYN-B007", "confirmed_issues"] == "MISSING_CLAIM_ID"
    assert out.loc["SYN-B013", "confirmed_issues"] == "FUTURE_DATE"
    assert out.loc["SYN-B023", "risk_score"] == 100
    assert out.loc["SYN-B022", "finding_type"] == "Risk indicator only"
    assert not out.loc["SYN-B022", "needs_review"]
    assert int(out["needs_review"].sum()) == 18
