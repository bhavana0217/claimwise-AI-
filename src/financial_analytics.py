"""Financial summaries.

IMPORTANT WORDING RULES used everywhere in the app:
  * "Billed value requiring review" is an ESTIMATE of the billed dollars attached to claims
    a person should look at. It is not a guaranteed loss and not a forecast of denials.
  * Denial history numbers are descriptive statistics of the loaded data only.
"""
from __future__ import annotations

import pandas as pd


def claim_check_summary(analyzed: pd.DataFrame) -> dict:
    """Summary of the claim checker output (from risk_engine.analyze_claims)."""
    total = int(len(analyzed))
    flagged = analyzed[analyzed["needs_review"]]
    # Negative or missing billed amounts are data errors; they count as $0 in dollar totals.
    total_billed = float(analyzed["billed_amount"].fillna(0).clip(lower=0).sum())
    review_value = float(flagged["billed_amount"].fillna(0).clip(lower=0).sum())
    return {
        "total_claims": total,
        "total_billed": total_billed,
        "flagged_claims": int(len(flagged)),
        "flagged_pct": (len(flagged) / total * 100) if total else 0.0,
        "billed_value_requiring_review": review_value,
        "review_value_pct": (review_value / total_billed * 100) if total_billed else 0.0,
        "confirmed_issue_claims": int((analyzed["confirmed_issue_count"] > 0).sum()),
        "indicator_only_claims": int((analyzed["finding_type"] == "Risk indicator only").sum()),
    }


def denial_summary(merged: pd.DataFrame) -> dict:
    """Historical denial statistics from claims merged with denial records."""
    total = int(len(merged))
    denied = merged[merged["has_denial_record"]]
    return {
        "insurance_claims": total,
        "total_billed": float(merged["billed_amount"].fillna(0).sum()),
        "denied_claims": int(len(denied)),
        "denial_rate_pct": (len(denied) / total * 100) if total else 0.0,
        "denied_amount": float(denied["denied_amount"].fillna(0).sum()),
    }


def denial_rate_by_payer(merged: pd.DataFrame) -> pd.DataFrame:
    """Claims, denials, denial rate and denied dollars per payer."""
    if merged.empty:
        return pd.DataFrame(columns=["insurance_provider", "claims", "denied", "denial_rate_pct", "denied_amount"])
    g = merged.groupby("insurance_provider", dropna=False)
    out = pd.DataFrame(
        {
            "claims": g.size(),
            "denied": g["has_denial_record"].sum().astype(int),
            "denied_amount": g["denied_amount"].sum(),
        }
    ).reset_index()
    out["denial_rate_pct"] = (out["denied"] / out["claims"] * 100).round(2)
    return out.sort_values("denial_rate_pct", ascending=False).reset_index(drop=True)[
        ["insurance_provider", "claims", "denied", "denial_rate_pct", "denied_amount"]
    ]


def denials_by_reason(merged: pd.DataFrame) -> pd.DataFrame:
    """Denied count and dollars per denial reason, largest dollar total first."""
    denied = merged[merged["has_denial_record"]]
    if denied.empty:
        return pd.DataFrame(columns=["denial_reason_code", "denial_reason_description", "denied", "denied_amount"])
    g = denied.groupby(["denial_reason_code", "denial_reason_description"], dropna=False)
    out = pd.DataFrame({"denied": g.size(), "denied_amount": g["denied_amount"].sum()}).reset_index()
    return out.sort_values("denied_amount", ascending=False).reset_index(drop=True)


def denial_rate_by_amount_band(merged: pd.DataFrame, bands: int = 5) -> pd.DataFrame:
    """Denial rate across equal-sized billed-amount bands (to check for any signal)."""
    valid = merged[merged["billed_amount"].notna()]
    if valid["billed_amount"].nunique() < bands:
        return pd.DataFrame(columns=["billed_amount_band", "claims", "denial_rate_pct"])
    band = pd.qcut(valid["billed_amount"], bands, duplicates="drop")
    g = valid.groupby(band, observed=True)["has_denial_record"]
    out = pd.DataFrame({"claims": g.size(), "denial_rate_pct": (g.mean() * 100).round(2)}).reset_index()
    out["billed_amount_band"] = [f"${iv.left:,.0f} to ${iv.right:,.0f}" for iv in out["billed_amount"]]
    return out[["billed_amount_band", "claims", "denial_rate_pct"]]


def appeal_summary(queue: pd.DataFrame) -> dict:
    """Totals for the denial rework queue (from appeal_queue.build_appeal_queue)."""

    def count_value(mask: pd.Series) -> tuple[int, float]:
        return int(mask.sum()), float(queue.loc[mask, "denied_amount"].fillna(0).sum())

    no_appeal = count_value(queue["work_status"] == "Open - no appeal filed")
    pending = count_value(queue["work_status"] == "Open - appeal pending")
    unclear = count_value(queue["work_status"] == "Open - status unclear")
    recovered = count_value(queue["work_status"] == "Closed - recovered")
    written_off = count_value(queue["work_status"] == "Closed - written off")
    resolved = recovered[0] + written_off[0]
    return {
        "open_count": no_appeal[0] + pending[0] + unclear[0],
        "open_value": no_appeal[1] + pending[1] + unclear[1],
        "no_appeal_count": no_appeal[0],
        "no_appeal_value": no_appeal[1],
        "pending_count": pending[0],
        "pending_value": pending[1],
        "unclear_count": unclear[0],
        "unclear_value": unclear[1],
        "recovered_count": recovered[0],
        "recovered_value": recovered[1],
        "written_off_count": written_off[0],
        "written_off_value": written_off[1],
        "resolved_appeal_win_rate_pct": (recovered[0] / resolved * 100) if resolved else 0.0,
    }


def denied_by_month(merged: pd.DataFrame) -> pd.DataFrame:
    """Denied claims and dollars by month of denial; flags months only partly covered by the data."""
    cols = ["month", "denied", "denied_amount", "is_partial"]
    denied = merged[merged["has_denial_record"] & merged["denial_date"].notna()]
    if denied.empty:
        return pd.DataFrame(columns=cols)
    month = denied["denial_date"].dt.to_period("M").dt.to_timestamp()
    g = denied.groupby(month)
    out = pd.DataFrame({"denied": g.size(), "denied_amount": g["denied_amount"].sum()})
    out.index.name = "month"
    out = out.reset_index()
    first_day = denied["denial_date"].min().normalize()
    last_day = denied["denial_date"].max().normalize()
    out["is_partial"] = False
    if first_day.day != 1:
        out.loc[out.index[0], "is_partial"] = True
    if (last_day + pd.offsets.MonthEnd(0)).normalize() != last_day:
        out.loc[out.index[-1], "is_partial"] = True
    return out[cols]


def denied_dollars_by_stage(asum: dict) -> pd.DataFrame:
    """Where the denied dollars stand, from appeal_summary()."""
    return pd.DataFrame(
        [
            {"stage": "Recovered on appeal", "amount": asum["recovered_value"], "claims": asum["recovered_count"]},
            {"stage": "Written off", "amount": asum["written_off_value"], "claims": asum["written_off_count"]},
            {"stage": "Appeal pending", "amount": asum["pending_value"], "claims": asum["pending_count"]},
            {"stage": "No appeal filed", "amount": asum["no_appeal_value"], "claims": asum["no_appeal_count"]},
            {"stage": "Status unclear", "amount": asum["unclear_value"], "claims": asum["unclear_count"]},
        ]
    )
