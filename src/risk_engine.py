"""Transparent, rules-based claim checker.

Every rule is either:
  * CONFIRMED  - a deterministic data defect that can be seen in the record, or
  * INDICATOR  - a pattern that is *associated with* denials. It is NOT a
                 prediction that the claim will be denied.

Risk score = sum of the weights of the rules that fired, capped at 100.
There is no machine learning in this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CONFIRMED = "confirmed"
INDICATOR = "indicator"

HIGH_AMOUNT_THRESHOLD = 10_000.0
SCORE_CAP = 100
# A claim needs human review if it has any confirmed issue or its score reaches this.
REVIEW_SCORE_THRESHOLD = 25
# (upper bound exclusive, label). Score 0 is always "Clean".
LEVEL_BANDS = [(25, "Low"), (50, "Medium"), (SCORE_CAP + 1, "High")]


@dataclass(frozen=True)
class Rule:
    kind: str
    weight: int
    label: str
    explanation: str
    next_step: str


RULES: dict[str, Rule] = {
    "MISSING_CLAIM_ID": Rule(
        CONFIRMED, 40, "Missing claim ID",
        "No claim ID is recorded, so this claim cannot be tracked reliably.",
        "Look up or assign the claim ID in the billing system.",
    ),
    "MISSING_PATIENT_ID": Rule(
        CONFIRMED, 30, "Missing patient ID",
        "No patient ID is recorded.",
        "Match the claim to the patient record and add the patient ID.",
    ),
    "MISSING_ENCOUNTER_ID": Rule(
        CONFIRMED, 20, "Missing encounter ID",
        "No encounter ID is recorded, so the charge cannot be tied to a visit.",
        "Link the claim to its encounter in the billing system.",
    ),
    "MISSING_PAYER": Rule(
        CONFIRMED, 35, "Missing insurance provider",
        "No insurance provider is recorded.",
        "Verify the patient's coverage and add the correct payer.",
    ),
    "INVALID_AMOUNT": Rule(
        CONFIRMED, 40, "Invalid billed amount",
        "The billed amount is missing, zero, or negative.",
        "Correct the charge amount from the encounter or charge records.",
    ),
    "MISSING_DATE": Rule(
        CONFIRMED, 25, "Missing or unreadable billing date",
        "The billing date is blank or cannot be read as a date.",
        "Confirm the billing date and correct the record.",
    ),
    "FUTURE_DATE": Rule(
        CONFIRMED, 25, "Billing date in the future",
        "The billing date is later than today's date.",
        "Check for a data-entry error in the date.",
    ),
    "DUPLICATE_CLAIM_ID": Rule(
        CONFIRMED, 35, "Duplicate claim ID",
        "This claim ID appears on more than one row.",
        "Decide whether the rows are true duplicates, then remove or re-key them.",
    ),
    "PAID_EXCEEDS_BILLED": Rule(
        CONFIRMED, 30, "Paid amount exceeds billed amount",
        "The paid amount is larger than the billed amount.",
        "Review the payment posting and any adjustments.",
    ),
    "PAID_STATUS_NO_PAYMENT": Rule(
        CONFIRMED, 20, "Status is Paid but no payment recorded",
        "The claim status is Paid, but no payment amount is recorded.",
        "Check whether the payment was posted; correct the status or the payment.",
    ),
    "DENIED_STATUS_WITH_PAYMENT": Rule(
        CONFIRMED, 20, "Status is Denied but a payment is recorded",
        "The claim status is Denied, but a payment amount is recorded.",
        "Check for a partial payment or an incorrect status.",
    ),
    "DENIED_NO_REASON": Rule(
        CONFIRMED, 15, "Denied with no denial reason",
        "The claim status is Denied, but no denial reason is recorded.",
        "Get the denial reason from the payer's remittance advice and record it.",
    ),
    "POSSIBLE_DUPLICATE": Rule(
        INDICATOR, 25, "Possible duplicate submission",
        "Another claim with a different ID has the same patient, payer, billing date, and "
        "amount. Duplicate claims are a common denial reason, but this is only an indicator, "
        "not a confirmed duplicate.",
        "Compare the two claims and void one if they are the same service.",
    ),
    "HIGH_AMOUNT": Rule(
        INDICATOR, 10, f"High billed amount (over ${HIGH_AMOUNT_THRESHOLD:,.0f})",
        "The billed amount is unusually high. High-value claims may receive additional "
        "review, so this is an indicator only.",
        "Double-check charges and supporting documentation before submission.",
    ),
}

CONFIRMED_CODES = [c for c, r in RULES.items() if r.kind == CONFIRMED]
INDICATOR_CODES = [c for c, r in RULES.items() if r.kind == INDICATOR]

ADDED_COLUMNS = [
    "confirmed_issues",
    "risk_indicators",
    "confirmed_issue_count",
    "risk_indicator_count",
    "risk_score",
    "risk_level",
    "needs_review",
    "finding_type",
    "explanation",
    "recommended_next_steps",
]


def _compute_masks(df: pd.DataFrame, as_of: pd.Timestamp, high_amount: float) -> dict[str, pd.Series]:
    billed = df["billed_amount"]
    paid = df["paid_amount"]
    status = df["claim_status_norm"]

    dup_claim_id = df["claim_id"].notna() & df["claim_id"].duplicated(keep=False)

    content_key = ["patient_id", "insurance_provider", "claim_billing_date", "billed_amount"]
    content_complete = df[content_key].notna().all(axis=1)
    same_content = df.duplicated(content_key, keep=False) & content_complete

    masks = {
        "MISSING_CLAIM_ID": df["claim_id"].isna(),
        "MISSING_PATIENT_ID": df["patient_id"].isna(),
        "MISSING_ENCOUNTER_ID": df["encounter_id"].isna(),
        "MISSING_PAYER": df["insurance_provider"].isna(),
        "INVALID_AMOUNT": billed.isna() | (billed <= 0),
        "MISSING_DATE": df["claim_billing_date"].isna(),
        "FUTURE_DATE": (df["claim_billing_date"] > as_of).fillna(False),
        "DUPLICATE_CLAIM_ID": dup_claim_id,
        "PAID_EXCEEDS_BILLED": (billed > 0) & (paid > billed),
        "PAID_STATUS_NO_PAYMENT": (status == "paid") & (paid <= 0),
        "DENIED_STATUS_WITH_PAYMENT": (status == "denied") & (paid > 0),
        "DENIED_NO_REASON": (status == "denied") & df["denial_reason"].isna(),
        # Rows that already share a claim ID are reported by DUPLICATE_CLAIM_ID instead.
        "POSSIBLE_DUPLICATE": same_content & ~dup_claim_id,
        "HIGH_AMOUNT": billed > high_amount,
    }
    return {code: mask.fillna(False).astype(bool) for code, mask in masks.items()}


def _join_codes(masks: dict[str, pd.Series], codes: list[str], index: pd.Index) -> pd.Series:
    out = pd.Series([""] * len(index), index=index, dtype=object)
    for code in codes:
        out = out.mask(masks[code], out + code + ";")
    return out.str.rstrip(";")


def _explain(confirmed: str, indicators: str) -> tuple[str, str]:
    """Build the plain-English explanation and next-step text for one claim."""
    conf_codes = [c for c in confirmed.split(";") if c]
    ind_codes = [c for c in indicators.split(";") if c]
    if not conf_codes and not ind_codes:
        return "No rule-based issues found.", "No action needed from this check."

    parts = []
    steps = []
    for code in conf_codes:
        rule = RULES[code]
        parts.append(f"Confirmed data issue - {rule.label}: {rule.explanation}")
        steps.append(rule.next_step)
    for code in ind_codes:
        rule = RULES[code]
        parts.append(f"Risk indicator (not a prediction) - {rule.label}: {rule.explanation}")
        steps.append(rule.next_step)
    numbered = [f"{i}) {s}" for i, s in enumerate(dict.fromkeys(steps), start=1)]
    return " | ".join(parts), " ".join(numbered)


def breakdown_frame(confirmed: str, indicators: str) -> pd.DataFrame:
    """One row per rule that fired: what it is, how many points it adds, and how to fix it."""
    rows = []
    for text, kind_label in ((confirmed, "Confirmed data issue"), (indicators, "Risk indicator")):
        for code in [c for c in str(text).split(";") if c]:
            rule = RULES[code]
            rows.append(
                {"Item": rule.label, "Type": kind_label, "Points": rule.weight, "How to fix": rule.next_step}
            )
    return pd.DataFrame(rows, columns=["Item", "Type", "Points", "How to fix"])


def score_if_corrected(confirmed: str, indicators: str) -> int:
    """Score that would remain if every confirmed data issue were corrected.

    Risk indicators stay because they cannot be 'corrected' - a person has to check them
    and decide. They lower the score only if that check shows they do not apply.
    """
    total = sum(RULES[c].weight for c in str(indicators).split(";") if c)
    return min(total, SCORE_CAP)


def risk_level_for(score: float) -> str:
    """Map a 0-100 score to a label."""
    if score <= 0:
        return "Clean"
    for upper, label in LEVEL_BANDS:
        if score < upper:
            return label
    return LEVEL_BANDS[-1][1]


def analyze_claims(
    claims: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    high_amount_threshold: float = HIGH_AMOUNT_THRESHOLD,
) -> pd.DataFrame:
    """Apply every rule to every claim and return the claims with result columns added.

    `claims` must come from claims_processor.load_claims (typed, cleaned columns).
    `as_of` is the date used for the future-date rule (defaults to today).
    """
    df = claims.copy().reset_index(drop=True)
    as_of = pd.Timestamp.today().normalize() if as_of is None else pd.Timestamp(as_of)

    masks = _compute_masks(df, as_of, high_amount_threshold)

    df["confirmed_issues"] = _join_codes(masks, CONFIRMED_CODES, df.index)
    df["risk_indicators"] = _join_codes(masks, INDICATOR_CODES, df.index)
    df["confirmed_issue_count"] = sum(
        (masks[c].astype(int) for c in CONFIRMED_CODES), start=pd.Series(0, index=df.index)
    )
    df["risk_indicator_count"] = sum(
        (masks[c].astype(int) for c in INDICATOR_CODES), start=pd.Series(0, index=df.index)
    )

    raw_score = sum(
        (masks[c].astype(int) * RULES[c].weight for c in RULES), start=pd.Series(0, index=df.index)
    )
    df["risk_score"] = raw_score.clip(upper=SCORE_CAP).astype(int)
    df["risk_level"] = df["risk_score"].map(risk_level_for)
    df["needs_review"] = (df["confirmed_issue_count"] > 0) | (df["risk_score"] >= REVIEW_SCORE_THRESHOLD)
    df["finding_type"] = np.select(
        [df["confirmed_issue_count"] > 0, df["risk_indicator_count"] > 0],
        ["Confirmed data issue", "Risk indicator only"],
        default="None",
    )

    explained = [_explain(c, i) for c, i in zip(df["confirmed_issues"], df["risk_indicators"])]
    df["explanation"] = [e[0] for e in explained]
    df["recommended_next_steps"] = [e[1] for e in explained]
    return df
