"""Prioritize denied claims for human rework or appeal.

This works on claims that were ALREADY denied (known facts from the denials file),
not on predictions. Priority score (0-100) for open items:

    amount points  = min(denied_amount / 5000, 1) * 50
    status points  = 30 (no appeal filed) | 20 (appeal pending) | 10 (status unclear)
    age points     = min(days_since_denial / 90, 1) * 20

Closed items (appeal approved / written off) get a priority of 0.
Payer appeal deadlines differ; the age component is only a rough urgency signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AMOUNT_CAP = 5000.0
AMOUNT_MAX_POINTS = 50
AGE_CAP_DAYS = 90
AGE_MAX_POINTS = 20
STATUS_POINTS = {
    "Open - no appeal filed": 30,
    "Open - appeal pending": 20,
    "Open - status unclear": 10,
}
OPEN_STATUSES = list(STATUS_POINTS)
CLOSED_STATUSES = ["Closed - recovered", "Closed - written off"]
HIGH_PRIORITY = 70
MEDIUM_PRIORITY = 45

GUIDANCE_DISCLAIMER = (
    "General guidance only. Always verify the payer's own policy and the remittance advice "
    "before acting."
)

# denial_reason_code -> (category, recommended next step)
GUIDANCE: dict[str, tuple[str, str]] = {
    "CO18": ("Correct and resubmit",
             "Check whether the original claim was paid. If not, confirm this is not a true "
             "duplicate and resubmit a corrected claim or appeal with proof."),
    "CO119": ("Correct and resubmit",
              "Verify the patient's active coverage, rebill the correct payer, and keep proof of "
              "the original submission date."),
    "CO16": ("Correct and resubmit",
             "Find the missing or incorrect fields on the remittance advice, correct the claim, "
             "and resubmit."),
    "CO11": ("Coding review",
             "Have a coder review the diagnosis and procedure codes. Correct and resubmit, or "
             "appeal with documentation if the coding is accurate."),
    "CO197": ("Appeal with documentation",
              "Check whether an authorization was obtained. If so, appeal with it attached. If "
              "not, ask the payer whether retroactive authorization is allowed."),
    "CO22": ("Eligibility and coverage",
             "Verify primary and secondary coverage, update coordination-of-benefits details, and "
             "rebill the payers in the correct order."),
    "CO29": ("Appeal with documentation",
             "Gather proof of timely submission (clearinghouse reports, earlier rejections). "
             "Appeal only if the payer's policy allows exceptions."),
    "CO50": ("Appeal with documentation",
             "Request clinical documentation and a provider letter of medical necessity, then "
             "follow the payer's appeal process."),
    "CO58": ("Correct and resubmit",
             "Check the place-of-service code against the encounter, correct it, and resubmit."),
    "PR109": ("Eligibility and coverage",
              "Confirm which payer covers this service, then rebill the correct payer or follow "
              "policy for patient responsibility."),
    "PR119": ("Patient responsibility review",
              "Confirm the benefit limit with the payer. If the maximum is reached, review "
              "patient-responsibility options under policy."),
    "PR125": ("Eligibility and coverage",
              "Re-verify eligibility and coverage dates, correct patient or insurance details, "
              "and resubmit if coverage was active."),
    "PR27": ("Eligibility and coverage",
             "Verify coverage on the date of service. If coverage was active, appeal with proof of "
             "eligibility; otherwise follow policy for the correct payer or patient."),
    "PR96": ("Patient responsibility review",
             "Confirm the service is non-covered under the plan and check whether any coverage "
             "exception or documentation applies under payer policy."),
}
DEFAULT_GUIDANCE = (
    "Manual review",
    "Review the remittance advice and payer policy to decide between correct-and-resubmit and appeal.",
)

# Short, readable names for charts and tables (denial_reason_code -> label)
REASON_SHORT = {
    "CO18": "Duplicate claim",
    "CO119": "Billed to wrong payer",
    "CO16": "Missing or incorrect information",
    "CO11": "Diagnosis does not match procedure",
    "CO197": "Prior authorization absent",
    "CO22": "Coordination of benefits",
    "CO29": "Filed after timely filing deadline",
    "CO50": "Not medically necessary",
    "CO58": "Invalid place of service",
    "PR109": "Not covered by this payer",
    "PR119": "Benefit maximum reached",
    "PR125": "Patient has no coverage",
    "PR27": "Coverage terminated or not eligible",
    "PR96": "Non-covered charges",
}

STATUS_NOTE = {
    "Open - no appeal filed": "No appeal has been filed yet.",
    "Open - appeal pending": "An appeal is pending with the payer.",
    "Open - status unclear": "The appeal status could not be determined from the data.",
    "Closed - recovered": "The appeal was approved and the claim was paid, so no action is needed.",
    "Closed - written off": "The appeal was denied and the amount was written off, so no action is needed.",
}

QUEUE_COLUMNS = [
    "claim_id", "insurance_provider", "billed_amount", "denied_amount", "denial_reason_code",
    "denial_reason_description", "denial_date", "days_since_denial", "appeal_filed",
    "appeal_status", "final_outcome", "work_status", "is_open", "priority_score",
    "amount_points", "status_points", "age_points", "priority_level", "reason_label",
    "action_category", "recommended_next_step", "explanation",
]


def _work_status(filed: pd.Series, status: pd.Series) -> pd.Series:
    filed_l = filed.astype(object).str.lower()
    status_l = status.astype(object).str.lower()
    conditions = [
        filed_l == "no",
        (filed_l == "yes") & (status_l == "pending"),
        (filed_l == "yes") & (status_l == "approved"),
        (filed_l == "yes") & (status_l == "denied"),
    ]
    choices = [
        "Open - no appeal filed",
        "Open - appeal pending",
        "Closed - recovered",
        "Closed - written off",
    ]
    return pd.Series(np.select(conditions, choices, default="Open - status unclear"), index=filed.index)


def priority_level_for(score: float, is_open: bool) -> str:
    if not is_open:
        return "Closed"
    if score >= HIGH_PRIORITY:
        return "High"
    if score >= MEDIUM_PRIORITY:
        return "Medium"
    return "Low"


def build_appeal_queue(merged: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Return every denied claim with a work status and (for open items) a priority score.

    `merged` is the output of claims_processor.merge_claims_denials. `as_of` is the date
    used to measure denial age; it defaults to the latest denial date in the data so
    results do not change depending on when the app is run.
    """
    d = merged[merged["denial_id"].notna()].copy().reset_index(drop=True)

    if as_of is None:
        latest = d["denial_date"].max() if len(d) else pd.NaT
        as_of = latest if pd.notna(latest) else pd.Timestamp.today().normalize()
    as_of = pd.Timestamp(as_of)

    d["days_since_denial"] = (as_of - d["denial_date"]).dt.days.clip(lower=0).fillna(0).astype(int)
    d["work_status"] = _work_status(d["appeal_filed"], d["appeal_status"])
    d["is_open"] = d["work_status"].isin(OPEN_STATUSES)

    amount_pts = (d["denied_amount"].fillna(0) / AMOUNT_CAP).clip(upper=1) * AMOUNT_MAX_POINTS
    status_pts = d["work_status"].map(STATUS_POINTS).fillna(0)
    age_pts = (d["days_since_denial"] / AGE_CAP_DAYS).clip(upper=1) * AGE_MAX_POINTS
    open_flag = d["is_open"].astype(float)
    d["amount_points"] = (amount_pts * open_flag).round(1)
    d["status_points"] = (status_pts * open_flag).round(1)
    d["age_points"] = (age_pts * open_flag).round(1)
    d["priority_score"] = ((amount_pts + status_pts + age_pts) * open_flag).round(1)
    d["priority_level"] = [
        priority_level_for(s, o) for s, o in zip(d["priority_score"], d["is_open"])
    ]

    guidance = d["denial_reason_code"].map(lambda c: GUIDANCE.get(c, DEFAULT_GUIDANCE))
    d["reason_label"] = [
        REASON_SHORT.get(code, desc) for code, desc in zip(d["denial_reason_code"], d["denial_reason_description"])
    ]
    d["action_category"] = [g[0] for g in guidance]
    d["recommended_next_step"] = np.where(
        d["is_open"], [g[1] for g in guidance], "No action needed - this denial is closed."
    )

    explanations = []
    for row, a_pts, s_pts, g_pts in zip(d.itertuples(index=False), amount_pts, status_pts, age_pts):
        date_txt = row.denial_date.strftime("%Y-%m-%d") if pd.notna(row.denial_date) else "unknown date"
        text = (
            f"Denied ${row.denied_amount:,.2f} on {date_txt} ({row.denial_reason_code}: "
            f"{row.denial_reason_description}). {STATUS_NOTE.get(row.work_status, '')}"
        )
        if row.is_open:
            text += (
                f" Priority {row.priority_score:.0f}/100 = amount {a_pts:.0f}/{AMOUNT_MAX_POINTS} "
                f"+ status {s_pts:.0f}/30 + age {g_pts:.0f}/{AGE_MAX_POINTS}."
            )
        explanations.append(text)
    d["explanation"] = explanations

    d = d.sort_values(["priority_score", "denied_amount"], ascending=False).reset_index(drop=True)
    return d[QUEUE_COLUMNS]
