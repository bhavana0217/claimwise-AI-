import io
from pathlib import Path

import pandas as pd
import pytest

from src.appeal_queue import build_appeal_queue
from src.claims_processor import (
    CLAIM_COLUMNS, DENIAL_COLUMNS, SchemaError, load_claims, load_denials, merge_claims_denials,
)
from src.payer_alerts import UPDATE_COLUMNS, alert_impact, load_updates, new_update, suggested_action

AS_OF = pd.Timestamp("2025-06-06")


def csv(header, rows):
    return io.StringIO("\n".join([",".join(header), *rows]) + "\n")


def make_queue(claim_rows, denial_rows):
    claims, _ = load_claims(csv(CLAIM_COLUMNS, claim_rows))
    denials, _ = load_denials(csv(DENIAL_COLUMNS, denial_rows))
    return build_appeal_queue(merge_claims_denials(claims, denials), as_of=AS_OF)


def claim(i, payer, amount):
    return f"B{i},P{i},E{i},{payer},Insurance,C{i},01-03-2025 00:00,{amount},0,Denied,Reason"


# C1 Cigna CO197 open 1000 | C2 Cigna CO197 open 2000 | C3 Cigna CO197 closed (recovered) 500
# C4 Aetna CO197 open 4000 (different payer) | C5 Cigna CO29 open 700 (different code)
CLAIMS = [claim(1, "Cigna", 1000), claim(2, "Cigna", 2000), claim(3, "Cigna", 500),
          claim(4, "Aetna", 4000), claim(5, "Cigna", 700)]
DENIALS = [
    "C1,D1,CO197,Prior auth,1000,01-05-2025,No,,,",
    "C2,D2,CO197,Prior auth,2000,01-05-2025,Yes,Pending,,Reprocessed",
    "C3,D3,CO197,Prior auth,500,01-05-2025,Yes,Approved,10-05-2025,Paid",
    "C4,D4,CO197,Prior auth,4000,01-05-2025,No,,,",
    "C5,D5,CO29,Timely filing,700,01-05-2025,No,,,",
]


def test_bundled_feed_loads_and_is_labeled_synthetic():
    updates = load_updates("data/payer_rule_updates.csv")
    assert len(updates) == 7
    assert set(updates["severity"]) <= {"High", "Medium", "Low"}
    assert (updates["source"] == "SYNTHETIC DEMO").all()
    multi = updates[updates["update_id"] == "SYN-U03"].iloc[0]
    assert multi["codes"] == ["PR27", "PR125"]


def test_invalid_severity_and_missing_column_are_rejected():
    header = ",".join(UPDATE_COLUMNS)
    bad = io.StringIO(f"{header}\nU1,Cigna,T,S,01-07-2025,Urgent,CO197,X\n")
    with pytest.raises(SchemaError):
        load_updates(bad)
    with pytest.raises(SchemaError):
        load_updates(io.StringIO("update_id,payer\nU1,Cigna\n"))


def test_impact_counts_only_matching_payer_and_code_and_only_open_denials():
    queue = make_queue(CLAIMS, DENIALS)
    updates = new_update("Cigna", "Prior auth", "s", "2025-07-01", "High", ["CO197"], "U1")
    out = alert_impact(updates, queue, AS_OF).iloc[0]
    assert out["open_denials_affected"] == 2          # C1 and C2 (C3 closed, C4 other payer, C5 other code)
    assert out["open_value_affected"] == 3000
    assert out["historical_denials_same_codes"] == 3  # C1, C2, C3
    assert out["status"] == "Upcoming"


def test_status_in_effect_and_sorting_by_severity_then_value():
    queue = make_queue(CLAIMS, DENIALS)
    low = new_update("Aetna", "Low one", "s", "2025-05-01", "Low", ["CO197"], "U1")
    high = new_update("Cigna", "High one", "s", "2025-05-01", "High", ["CO29"], "U2")
    med_big = new_update("Cigna", "Med big", "s", "2025-07-01", "Medium", ["CO197"], "U3")
    med_small = new_update("Cigna", "Med small", "s", "2025-07-01", "Medium", ["CO29"], "U4")
    out = alert_impact(pd.concat([low, med_small, high, med_big], ignore_index=True), queue, AS_OF)
    assert out["update_id"].tolist() == ["U2", "U3", "U4", "U1"]
    assert out.set_index("update_id").loc["U2", "status"] == "In effect"


def test_no_matching_denials_gives_zero_impact_and_matching_advice():
    queue = make_queue(CLAIMS, DENIALS)
    updates = new_update("Humana", "Anything", "s", "2025-07-01", "Medium", ["CO50"], "U1")
    row = alert_impact(updates, queue, AS_OF).iloc[0]
    assert row["open_denials_affected"] == 0 and row["open_value_affected"] == 0
    assert "No open denials currently match" in suggested_action(row)


def test_suggested_action_mentions_counts_and_defers_to_payer_policy():
    queue = make_queue(CLAIMS, DENIALS)
    updates = new_update("Cigna", "Prior auth", "s", "2025-07-01", "High", ["CO197"], "U1")
    text = suggested_action(alert_impact(updates, queue, AS_OF).iloc[0])
    assert "2 open Cigna denial" in text and "before it takes effect" in text
    assert "payer's own policy" in text


def test_new_update_rejects_unknown_severity():
    with pytest.raises(ValueError):
        new_update("Cigna", "t", "s", "2025-07-01", "Urgent", ["CO197"], "U1")


@pytest.mark.skipif(not Path("data/denials.csv").exists(), reason="real datathon files not present")
def test_impact_matches_independent_pandas_calculation_on_real_data():
    claims, _ = load_claims("data/claims_and_billing.csv")
    denials, _ = load_denials("data/denials.csv")
    queue = build_appeal_queue(merge_claims_denials(claims, denials))
    updates = load_updates("data/payer_rule_updates.csv")
    impact = alert_impact(updates, queue, queue["denial_date"].max()).set_index("update_id")

    raw_c = pd.read_csv("data/claims_and_billing.csv")
    raw_d = pd.read_csv("data/denials.csv")
    both = raw_d.merge(raw_c[["claim_id", "insurance_provider"]], on="claim_id")
    open_ = both[(both["appeal_filed"] == "No") | (both["appeal_status"] == "Pending")]
    for uid, payer, codes in [("SYN-U01", "Cigna", ["CO197"]), ("SYN-U03", "Medicaid", ["PR27", "PR125"])]:
        sel = open_[(open_["insurance_provider"] == payer) & open_["denial_reason_code"].isin(codes)]
        assert impact.loc[uid, "open_denials_affected"] == len(sel)
        assert impact.loc[uid, "open_value_affected"] == pytest.approx(sel["denied_amount"].sum())
