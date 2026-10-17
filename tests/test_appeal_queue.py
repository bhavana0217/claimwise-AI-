import io

import pandas as pd

from src.appeal_queue import GUIDANCE, build_appeal_queue
from src.claims_processor import CLAIM_COLUMNS, DENIAL_COLUMNS, load_claims, load_denials, merge_claims_denials


def csv(header, rows):
    return io.StringIO("\n".join([",".join(header), *rows]) + "\n")


def queue_for(denial_rows, amounts, as_of="2025-03-31"):
    claim_rows = [
        f"B{i},P{i},E{i},Aetna,Insurance,C{i},01-01-2025 00:00,{amt},0,Denied,Duplicate Claim"
        for i, amt in enumerate(amounts, start=1)
    ]
    claims, _ = load_claims(csv(CLAIM_COLUMNS, claim_rows))
    denials, _ = load_denials(csv(DENIAL_COLUMNS, denial_rows))
    return build_appeal_queue(merge_claims_denials(claims, denials), as_of=pd.Timestamp(as_of)).set_index("claim_id")


def test_priority_score_manual_calculation():
    # as_of 2025-03-31
    q = queue_for(
        [
            # 2500 denied, no appeal, denied 45 days earlier (2025-02-14): 25 + 30 + 10 = 65
            "C1,D1,CO18,Dup,2500,14-02-2025,No,,,",
            # 8000 pending, denied exactly 90 days earlier (2024-12-31): 50 + 20 + 20 = 90
            "C2,D2,CO50,Med,8000,31-12-2024,Yes,Pending,,Reprocessed",
        ],
        [2500, 8000],
    )
    assert q.loc["C1", "priority_score"] == 65.0 and q.loc["C1", "priority_level"] == "Medium"
    assert q.loc["C2", "priority_score"] == 90.0 and q.loc["C2", "priority_level"] == "High"
    assert q.index[0] == "C2"  # sorted highest priority first


def test_closed_denials_get_zero_priority_and_no_action():
    q = queue_for(
        [
            "C1,D1,CO18,Dup,1000,14-02-2025,Yes,Approved,20-02-2025,Paid",
            "C2,D2,CO18,Dup,1000,14-02-2025,Yes,Denied,20-02-2025,Written off",
        ],
        [1000, 1000],
    )
    assert q["priority_score"].tolist() == [0.0, 0.0]
    assert set(q["work_status"]) == {"Closed - recovered", "Closed - written off"}
    assert (q["priority_level"] == "Closed").all() and (~q["is_open"]).all()
    assert q["recommended_next_step"].str.startswith("No action needed").all()


def test_scores_are_within_0_to_100_even_for_huge_amounts():
    q = queue_for(["C1,D1,CO18,Dup,900000,01-01-2024,No,,,"], [900000])
    assert q.loc["C1", "priority_score"] == 100.0


def test_guidance_matches_denial_code_and_unknown_code_falls_back():
    q = queue_for(
        [
            "C1,D1,CO197,Prior auth,1000,14-02-2025,No,,,",
            "C2,D2,ZZ999,Something new,1000,14-02-2025,No,,,",
        ],
        [1000, 1000],
    )
    assert q.loc["C1", "action_category"] == GUIDANCE["CO197"][0]
    assert q.loc["C1", "recommended_next_step"] == GUIDANCE["CO197"][1]
    assert q.loc["C2", "action_category"] == "Manual review"


def test_explanation_mentions_amount_code_and_score():
    q = queue_for(["C1,D1,CO18,Dup,2500,14-02-2025,No,,,"], [2500])
    text = q.loc["C1", "explanation"]
    assert "$2,500.00" in text and "CO18" in text and "Priority 65/100" in text
    assert "No appeal has been filed" in text


def test_no_denials_gives_empty_queue():
    q = queue_for([], [1000])
    assert q.empty
