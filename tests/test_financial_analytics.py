import io

import pandas as pd
import pytest

from src.claims_processor import CLAIM_COLUMNS, DENIAL_COLUMNS, load_claims, load_denials, merge_claims_denials
from src.financial_analytics import (
    appeal_summary,
    claim_check_summary,
    denial_rate_by_amount_band,
    denial_rate_by_payer,
    denial_summary,
    denials_by_reason,
)
from src.appeal_queue import build_appeal_queue
from src.risk_engine import analyze_claims

AS_OF = pd.Timestamp("2025-06-30")


def csv(header, rows):
    return io.StringIO("\n".join([",".join(header), *rows]) + "\n")


def build(claim_rows, denial_rows):
    claims, _ = load_claims(csv(CLAIM_COLUMNS, claim_rows))
    denials, _ = load_denials(csv(DENIAL_COLUMNS, denial_rows))
    return merge_claims_denials(claims, denials)


# 4 claims: Aetna 1000 paid, Aetna 2000 denied, Cigna 3000 denied, Cigna 4000 paid
CLAIMS = [
    "B1,P1,E1,Aetna,Insurance,C1,01-03-2025 00:00,1000,600,Paid,",
    "B2,P2,E2,Aetna,Insurance,C2,01-03-2025 00:00,2000,0,Denied,Duplicate Claim",
    "B3,P3,E3,Cigna,Insurance,C3,01-03-2025 00:00,3000,0,Denied,Service Not Covered",
    "B4,P4,E4,Cigna,Insurance,C4,01-03-2025 00:00,4000,2500,Paid,",
]
DENIALS = [
    "C2,D1,CO18,Duplicate claim/service.,2000,01-04-2025,Yes,Approved,10-04-2025,Paid",
    "C3,D2,PR96,Non-covered charges,3000,01-04-2025,No,,,",
]


def test_denial_summary_manual_numbers():
    s = denial_summary(build(CLAIMS, DENIALS))
    assert s["insurance_claims"] == 4
    assert s["total_billed"] == 10000
    assert s["denied_claims"] == 2
    assert s["denial_rate_pct"] == 50.0
    assert s["denied_amount"] == 5000


def test_denial_rate_by_payer_manual_numbers():
    out = denial_rate_by_payer(build(CLAIMS, DENIALS)).set_index("insurance_provider")
    assert out.loc["Aetna", "claims"] == 2 and out.loc["Aetna", "denied"] == 1
    assert out.loc["Aetna", "denial_rate_pct"] == 50.0
    assert out.loc["Cigna", "denied_amount"] == 3000


def test_denials_by_reason_sorted_by_dollars():
    out = denials_by_reason(build(CLAIMS, DENIALS))
    assert out["denial_reason_code"].tolist() == ["PR96", "CO18"]
    assert out["denied_amount"].tolist() == [3000, 2000]


def test_amount_band_table_has_all_claims():
    rows = [f"B{i},P{i},E{i},Aetna,Insurance,C{i},01-03-2025 00:00,{100 * i},0,Paid," for i in range(1, 11)]
    out = denial_rate_by_amount_band(build(rows, []), bands=5)
    assert out["claims"].sum() == 10 and (out["denial_rate_pct"] == 0).all()


def test_claim_check_summary_manual_numbers():
    claims, _ = load_claims(csv(CLAIM_COLUMNS, [
        "B1,P1,E1,Aetna,Insurance,C1,01-03-2025 00:00,1000,0,Pending,",   # clean
        "B2,,E2,Aetna,Insurance,C2,01-03-2025 00:00,2000,0,Pending,",     # missing patient -> review
        "B3,P3,E3,Aetna,Insurance,C3,01-03-2025 00:00,12000,0,Pending,",  # high amount only -> no review
        "B4,P4,E4,Aetna,Insurance,C4,01-03-2025 00:00,-100,0,Pending,",   # invalid amount -> review, $0
    ]))
    s = claim_check_summary(analyze_claims(claims, as_of=AS_OF))
    assert s["total_claims"] == 4
    assert s["total_billed"] == 15000          # 1000 + 2000 + 12000 + 0 (negative counted as 0)
    assert s["flagged_claims"] == 2
    assert s["billed_value_requiring_review"] == 2000
    assert s["confirmed_issue_claims"] == 2 and s["indicator_only_claims"] == 1
    assert s["flagged_pct"] == 50.0


def test_claim_check_summary_empty_has_no_division_error():
    claims, _ = load_claims(csv(CLAIM_COLUMNS, []))
    s = claim_check_summary(analyze_claims(claims, as_of=AS_OF))
    assert s["total_claims"] == 0 and s["flagged_pct"] == 0.0


def test_appeal_summary_and_win_rate():
    s = appeal_summary(build_appeal_queue(build(CLAIMS, DENIALS)))
    assert s["recovered_count"] == 1 and s["recovered_value"] == 2000
    assert s["no_appeal_count"] == 1 and s["no_appeal_value"] == 3000
    assert s["open_count"] == 1 and s["open_value"] == 3000
    assert s["resolved_appeal_win_rate_pct"] == 100.0
