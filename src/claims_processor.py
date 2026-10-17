"""Load, validate and clean claim data.

This module only prepares data. It does not score or judge claims; that is the
job of risk_engine.py (claim checker) and appeal_queue.py (denial rework).
"""
from __future__ import annotations

from pathlib import Path
from typing import IO, Union

import numpy as np
import pandas as pd

CLAIM_COLUMNS = [
    "billing_id",
    "patient_id",
    "encounter_id",
    "insurance_provider",
    "payment_method",
    "claim_id",
    "claim_billing_date",
    "billed_amount",
    "paid_amount",
    "claim_status",
    "denial_reason",
]

DENIAL_COLUMNS = [
    "claim_id",
    "denial_id",
    "denial_reason_code",
    "denial_reason_description",
    "denied_amount",
    "denial_date",
    "appeal_filed",
    "appeal_status",
    "appeal_resolution_date",
    "final_outcome",
]

CsvSource = Union[str, Path, IO]


class SchemaError(ValueError):
    """Raised when an input file is empty, unreadable, or missing columns."""


def read_csv_strict(source: CsvSource, name: str = "file") -> pd.DataFrame:
    """Read a CSV as text and raise SchemaError with a readable message on failure."""
    try:
        df = pd.read_csv(source, dtype=str)
    except pd.errors.EmptyDataError as exc:
        raise SchemaError(f"The {name} is empty.") from exc
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise SchemaError(f"The {name} could not be read as a CSV file: {exc}") from exc
    df.columns = [str(c).strip() for c in df.columns]
    return df


def validate_columns(df: pd.DataFrame, required: list[str], name: str = "file") -> None:
    """Raise SchemaError if any required column is absent."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"The {name} is missing required column(s): {', '.join(missing)}")


def _blank_to_nan(series: pd.Series) -> pd.Series:
    """Strip whitespace and turn empty strings into NaN (object dtype)."""
    s = series.astype("string").str.strip().replace("", pd.NA)
    return s.astype(object).where(s.notna(), np.nan)


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse day-first dates; anything unreadable becomes NaT."""
    return pd.to_datetime(series, format="mixed", dayfirst=True, errors="coerce")


def clean_claims(df: pd.DataFrame) -> pd.DataFrame:
    """Return a typed copy of a raw claims table (columns already validated)."""
    out = df.copy()
    for col in CLAIM_COLUMNS:
        if col in ("billed_amount", "paid_amount", "claim_billing_date"):
            continue
        out[col] = _blank_to_nan(out[col])
    out["billed_amount"] = pd.to_numeric(out["billed_amount"], errors="coerce")
    # A blank or unreadable paid amount is treated as 0 (nothing paid yet).
    out["paid_amount"] = pd.to_numeric(out["paid_amount"], errors="coerce").fillna(0.0)
    out["claim_billing_date"] = _parse_dates(_blank_to_nan(out["claim_billing_date"]))
    out["claim_status_norm"] = out["claim_status"].astype(object).str.lower()
    return out


def split_self_pay(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Separate self-pay rows (not insurance claims). Returns (insurance, n_self_pay)."""
    method = (
        df["payment_method"].astype(object).str.lower().str.replace(r"[^a-z]", "", regex=True)
    )
    is_self_pay = (method == "selfpay").fillna(False)
    return df[~is_self_pay].reset_index(drop=True), int(is_self_pay.sum())


def load_claims(source: CsvSource) -> tuple[pd.DataFrame, dict]:
    """Load, validate and clean a claims file.

    Returns (insurance_claims, info). Self-pay rows are excluded because they are
    not insurance claims; the count is reported in info["self_pay_excluded"].
    """
    raw = read_csv_strict(source, "claims file")
    validate_columns(raw, CLAIM_COLUMNS, "claims file")
    cleaned = clean_claims(raw)
    insurance, n_self_pay = split_self_pay(cleaned)
    info = {"rows_read": int(len(raw)), "self_pay_excluded": n_self_pay}
    return insurance, info


def clean_denials(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return a typed copy of the denials table plus data-quality counts."""
    out = df.copy()
    for col in DENIAL_COLUMNS:
        if col in ("denied_amount", "denial_date", "appeal_resolution_date"):
            continue
        out[col] = _blank_to_nan(out[col])
    out["denied_amount"] = pd.to_numeric(out["denied_amount"], errors="coerce")
    out["denial_date"] = _parse_dates(_blank_to_nan(out["denial_date"]))
    out["appeal_resolution_date"] = _parse_dates(_blank_to_nan(out["appeal_resolution_date"]))

    n_no_id = int(out["claim_id"].isna().sum())
    out = out[out["claim_id"].notna()]
    n_dupes = int(out["claim_id"].duplicated().sum())
    out = out.drop_duplicates("claim_id", keep="first").reset_index(drop=True)
    return out, {"denial_rows_without_claim_id": n_no_id, "duplicate_denial_rows_dropped": n_dupes}


def load_denials(source: CsvSource) -> tuple[pd.DataFrame, dict]:
    """Load, validate and clean a denials file."""
    raw = read_csv_strict(source, "denials file")
    validate_columns(raw, DENIAL_COLUMNS, "denials file")
    cleaned, info = clean_denials(raw)
    info["rows_read"] = int(len(raw))
    return cleaned, info


def merge_claims_denials(claims: pd.DataFrame, denials: pd.DataFrame) -> pd.DataFrame:
    """Left-join denial details onto claims by claim_id."""
    merged = claims.merge(denials, on="claim_id", how="left", validate="many_to_one")
    merged["has_denial_record"] = merged["denial_id"].notna()
    return merged
