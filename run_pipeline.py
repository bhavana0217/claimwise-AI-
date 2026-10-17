"""Command-line run: validate the data, print a summary, and write result CSVs.

Usage (from the project folder, with the virtual environment active):
    python run_pipeline.py
    python run_pipeline.py --claims data/claims_and_billing.csv --denials data/denials.csv
    python run_pipeline.py --demo        # check the synthetic demo claims instead
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.appeal_queue import build_appeal_queue
from src.claims_processor import SchemaError, load_claims, load_denials, merge_claims_denials
from src.financial_analytics import appeal_summary, claim_check_summary, denial_summary
from src.risk_engine import analyze_claims


def main() -> int:
    parser = argparse.ArgumentParser(description="ClaimWise AI command-line run")
    parser.add_argument("--claims", default="data/claims_and_billing.csv")
    parser.add_argument("--denials", default="data/denials.csv")
    parser.add_argument("--out", default="output")
    parser.add_argument("--demo", action="store_true", help="check data/sample_claims.csv (synthetic)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.demo:
            claims, info = load_claims("data/sample_claims.csv")
            denials = None
        else:
            claims, info = load_claims(args.claims)
            denials, denial_info = load_denials(args.denials)
    except (SchemaError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Claims file OK: {info['rows_read']:,} rows read, "
          f"{info['self_pay_excluded']:,} self-pay rows excluded, {len(claims):,} insurance claims.")

    analyzed = analyze_claims(claims)
    s = claim_check_summary(analyzed)
    print("\nCLAIM CHECKER (rules-based)")
    print(f"  Claims checked:                      {s['total_claims']:,}")
    print(f"  Total billed:                        ${s['total_billed']:,.2f}")
    print(f"  Flagged for review:                  {s['flagged_claims']:,} ({s['flagged_pct']:.1f}%)")
    print(f"  Billed value requiring review (est): ${s['billed_value_requiring_review']:,.2f}")
    print(f"  Confirmed-issue claims: {s['confirmed_issue_claims']:,}   Indicator-only claims: {s['indicator_only_claims']:,}")
    analyzed.drop(columns=["claim_status_norm"]).to_csv(out_dir / "analyzed_claims.csv", index=False)
    print(f"  Wrote {out_dir / 'analyzed_claims.csv'}")

    if denials is not None:
        merged = merge_claims_denials(claims, denials)
        d = denial_summary(merged)
        queue = build_appeal_queue(merged)
        a = appeal_summary(queue)
        print("\nDENIAL HISTORY (from the data)")
        print(f"  Denied claims: {d['denied_claims']:,} ({d['denial_rate_pct']:.2f}%), denied amount ${d['denied_amount']:,.2f}")
        print(f"  Open denials to work: {a['open_count']:,} worth ${a['open_value']:,.2f} "
              f"({a['no_appeal_count']:,} with no appeal filed, {a['pending_count']:,} pending)")
        queue.to_csv(out_dir / "appeal_queue.csv", index=False)
        print(f"  Wrote {out_dir / 'appeal_queue.csv'}")

    print("\nDecision support only: results need human review and payer policy verification.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
