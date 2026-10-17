"""Payer rule-change alerts.

IMPORTANT: the bundled feed (data/payer_rule_updates.csv) is SYNTHETIC and only illustrates
the feature. It is not read from any payer and is not real-time. In production this table
would be filled from payer bulletins (manually or by an integration).

Impact is measured against the loaded data: an update lists the denial codes it relates to;
we count the payer's open denials with those codes (and, for context, all historical
denials with those codes).
"""
from __future__ import annotations

import io
from datetime import date

import pandas as pd

from .claims_processor import SchemaError, read_csv_strict, validate_columns

UPDATE_COLUMNS = [
    "update_id", "payer", "topic", "summary", "effective_date",
    "severity", "related_denial_codes", "source",
]
SEVERITIES = ["High", "Medium", "Low"]
_SEVERITY_ORDER = {s: i for i, s in enumerate(SEVERITIES)}


def _split_codes(value) -> list[str]:
    if pd.isna(value):
        return []
    return [c.strip() for c in str(value).split("|") if c.strip()]


def clean_updates(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    for col in UPDATE_COLUMNS:
        df[col] = df[col].astype("string").str.strip()
    bad = sorted(set(df["severity"].dropna()) - set(SEVERITIES))
    if bad:
        raise SchemaError(f"Payer updates file has invalid severity value(s): {', '.join(bad)}")
    df["effective_date"] = pd.to_datetime(df["effective_date"], format="mixed", dayfirst=True, errors="coerce")
    df["codes"] = df["related_denial_codes"].map(_split_codes)
    return df.astype({c: object for c in UPDATE_COLUMNS if c != "effective_date"}).reset_index(drop=True)


def load_updates(source) -> pd.DataFrame:
    """Load and validate a payer updates CSV."""
    raw = read_csv_strict(source, "payer updates file")
    validate_columns(raw, UPDATE_COLUMNS, "payer updates file")
    return clean_updates(raw)


def new_update(
    payer: str,
    topic: str,
    summary: str,
    effective_date: date | pd.Timestamp,
    severity: str,
    codes: list[str],
    update_id: str,
    source: str = "SIMULATED IN APP",
) -> pd.DataFrame:
    """Build a one-row updates frame (used by the 'simulate an update' form)."""
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}")
    row = {
        "update_id": update_id, "payer": payer.strip(), "topic": topic.strip(),
        "summary": summary.strip(), "effective_date": pd.Timestamp(effective_date),
        "severity": severity, "related_denial_codes": "|".join(codes), "source": source,
    }
    out = pd.DataFrame([row])
    out["codes"] = [list(codes)]
    return out


def alert_impact(updates: pd.DataFrame, queue: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Attach status and impact numbers to each update, most urgent first."""
    as_of = pd.Timestamp(as_of)
    rows = []
    for u in updates.itertuples(index=False):
        same = (queue["insurance_provider"] == u.payer) & queue["denial_reason_code"].isin(u.codes)
        open_mask = same & queue["is_open"]
        eff = u.effective_date
        if pd.isna(eff):
            status = "Date unknown"
        elif eff > as_of:
            status = "Upcoming"
        else:
            status = "In effect"
        rows.append(
            {
                "update_id": u.update_id,
                "severity": u.severity,
                "payer": u.payer,
                "topic": u.topic,
                "effective_date": eff,
                "status": status,
                "related_codes": ", ".join(u.codes),
                "open_denials_affected": int(open_mask.sum()),
                "open_value_affected": float(queue.loc[open_mask, "denied_amount"].fillna(0).sum()),
                "historical_denials_same_codes": int(same.sum()),
                "summary": u.summary,
                "source": u.source,
            }
        )
    cols = [
        "update_id", "severity", "payer", "topic", "effective_date", "status", "related_codes",
        "open_denials_affected", "open_value_affected", "historical_denials_same_codes",
        "summary", "source",
    ]
    out = pd.DataFrame(rows, columns=cols)
    if out.empty:
        return out
    out["_order"] = out["severity"].map(_SEVERITY_ORDER)
    out = out.sort_values(["_order", "open_value_affected"], ascending=[True, False]).drop(columns="_order")
    return out.reset_index(drop=True)


def suggested_action(row: pd.Series) -> str:
    """Plain-language next step for an alert (general guidance, not payer policy)."""
    timing = "before it takes effect" if row["status"] == "Upcoming" else "now that it is in effect"
    base = (
        f"Read the {row['payer']} notice on '{row['topic']}' and confirm what changed. "
        f"Review the {row['open_denials_affected']:,} open {row['payer']} denial(s) with related codes "
        f"({row['related_codes']}), and update front-end checks {timing}."
    )
    if row["open_denials_affected"] == 0:
        base = (
            f"Read the {row['payer']} notice on '{row['topic']}' and confirm what changed. No open denials "
            f"currently match; share the change with registration and billing staff {timing}."
        )
    return base + " Verify details against the payer's own policy."
