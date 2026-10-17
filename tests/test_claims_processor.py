import io

import pandas as pd
import pytest

from src.claims_processor import (
    CLAIM_COLUMNS,
    DENIAL_COLUMNS,
    SchemaError,
    load_claims,
    load_denials,
    merge_claims_denials,
)

HEADER = ",".join(CLAIM_COLUMNS)
DENIAL_HEADER = ",".join(DENIAL_COLUMNS)


def claims_csv(*rows):
    return io.StringIO("\n".join([HEADER, *rows]) + "\n")


def denials_csv(*rows):
    return io.StringIO("\n".join([DENIAL_HEADER, *rows]) + "\n")


def test_sample_file_loads_and_excludes_self_pay():
    claims, info = load_claims("data/sample_claims.csv")
    assert info == {"rows_read": 26, "self_pay_excluded": 1}
    assert len(claims) == 25
    assert (claims["payment_method"] != "Selfpay").all()


def test_missing_required_column_raises_clear_error():
    bad = io.StringIO("billing_id,claim_id\nB1,C1\n")
    with pytest.raises(SchemaError) as err:
        load_claims(bad)
    assert "patient_id" in str(err.value)
    assert "claims file" in str(err.value)


def test_empty_file_raises_schema_error():
    with pytest.raises(SchemaError):
        load_claims(io.StringIO(""))


def test_header_only_file_gives_empty_frame():
    claims, info = load_claims(claims_csv())
    assert claims.empty and info["rows_read"] == 0


def test_blanks_become_missing_and_types_are_parsed():
    claims, _ = load_claims(
        claims_csv(
            "B1,P1,E1,Aetna,Insurance,C1,05-03-2025 00:00,100.50,40,Paid,",
            "B2, ,E2,,Insurance,C2,not a date,abc,,Pending,",
        )
    )
    first, second = claims.iloc[0], claims.iloc[1]
    assert first["billed_amount"] == 100.5 and first["paid_amount"] == 40
    # day-first: 05-03-2025 is 5 March
    assert first["claim_billing_date"] == pd.Timestamp("2025-03-05")
    assert pd.isna(second["patient_id"]) and pd.isna(second["insurance_provider"])
    assert pd.isna(second["claim_billing_date"])
    assert pd.isna(second["billed_amount"])
    assert second["paid_amount"] == 0


def test_self_pay_variants_are_excluded():
    claims, info = load_claims(
        claims_csv(
            "B1,P1,E1,,Self-pay,,05-03-2025 00:00,100,0,Paid,",
            "B2,P2,E2,Aetna,Insurance,C2,05-03-2025 00:00,100,0,Pending,",
        )
    )
    assert info["self_pay_excluded"] == 1 and len(claims) == 1


def test_denials_missing_claim_id_and_duplicates_are_handled():
    denials, info = load_denials(
        denials_csv(
            "C1,D1,CO18,Dup,100,01-03-2025,No,,,",
            "C1,D2,CO18,Dup,100,02-03-2025,No,,,",
            ",D3,CO18,Dup,100,02-03-2025,No,,,",
        )
    )
    assert len(denials) == 1
    assert info["denial_rows_without_claim_id"] == 1
    assert info["duplicate_denial_rows_dropped"] == 1


def test_merge_marks_denied_claims_only():
    claims, _ = load_claims(
        claims_csv(
            "B1,P1,E1,Aetna,Insurance,C1,05-03-2025 00:00,100,0,Denied,Duplicate Claim",
            "B2,P2,E2,Aetna,Insurance,C2,05-03-2025 00:00,200,150,Paid,",
        )
    )
    denials, _ = load_denials(denials_csv("C1,D1,CO18,Dup,100,01-03-2025,No,,,"))
    merged = merge_claims_denials(claims, denials)
    assert len(merged) == 2
    assert merged.set_index("claim_id")["has_denial_record"].to_dict() == {"C1": True, "C2": False}
