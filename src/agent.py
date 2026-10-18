"""ClaimWise AI - bounded, read-only AI investigator.

Exposes a single function: investigate(claim_id) -> dict

Requires OPENAI_API_KEY to call the LLM.  If the key is absent the function
returns immediately with status="unavailable" so the rest of the app can
degrade gracefully.

DO NOT send PHI to external APIs.  Use only de-identified demonstration
datasets or an approved privacy arrangement.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = APP_DIR / "data"

SYSTEM = (
    "You are Couverte AI, a cautious healthcare billing investigation assistant. "
    "Investigate a single claim using the provided read-only tools. Choose relevant "
    "tools and cite their observed outputs by tool name in your final response. "
    "Distinguish review findings, missing verification, and descriptive payer history. "
    "Never claim an anomaly is a denial prediction. Never invent eligibility, "
    "authorization, payer policy, clinical necessity, or actual corrections. "
    "Never use the claim's own historical outcome to infer what was knowable before "
    "submission. Do not request or disclose PHI.\n\n"
    "Format your final answer in Markdown using exactly these four headings in order:\n\n"
    "## Summary\n"
    "2-3 sentences describing what the investigation found.\n\n"
    "## Key findings\n"
    "Bullet list of concrete observations from the tool outputs. "
    "Cite the tool name for each finding.\n\n"
    "## Recommended actions\n"
    "Numbered list of specific, actionable verification steps a human reviewer should take. "
    "Each step must be verifiable by a person without additional data access.\n\n"
    "## Limitations\n"
    "Bullet list of what this investigation cannot determine: missing data, "
    "payer-specific policy, clinical necessity, and any other relevant caveats. "
    "Always include: this is a risk indicator tool, not a denial prediction."
)

MODEL_DEFAULT = "gpt-4.1-mini"


# ------------------------------------------------------------------ data helpers

@lru_cache(maxsize=1)
def _load_data():
    """Load and merge claims + denials lazily (cached after first call)."""
    from src.claims_processor import load_claims, load_denials, merge_claims_denials
    claims, _ = load_claims(DATA_DIR / "claims_and_billing.csv")
    denials, _ = load_denials(DATA_DIR / "denials.csv")
    merged = merge_claims_denials(claims, denials)
    return claims, denials, merged


def _get_claim_row(claim_id: str) -> pd.Series:
    claims, _denials, _merged = _load_data()
    row = claims[claims["claim_id"].astype(str) == str(claim_id)]
    if row.empty:
        raise ValueError("Claim ID not found: " + str(claim_id))
    return row.iloc[0]


def _safe(value):
    """Convert pandas NA and numpy scalars to plain Python types."""
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


# ------------------------------------------------------------------ tool functions

def get_claim_details(claim_id: str) -> dict:
    """Read permitted claim billing fields (no patient identifiers)."""
    row = _get_claim_row(claim_id)
    keys = ("claim_id", "insurance_provider", "payment_method", "billed_amount", "claim_billing_date")
    return _safe({k: row.get(k) for k in keys})


def get_denial_history(claim_id: str) -> dict:
    """Return denial record for this claim if one exists."""
    _claims, denials, _merged = _load_data()
    match = denials[denials["claim_id"].astype(str) == str(claim_id)]
    if match.empty:
        return {"claim_id": claim_id, "has_denial": False}
    row = match.iloc[0]
    return _safe({
        "claim_id": claim_id,
        "has_denial": True,
        "denial_reason_code": row.get("denial_reason_code"),
        "denial_reason_description": row.get("denial_reason_description"),
        "denied_amount": row.get("denied_amount"),
        "denial_date": str(row.get("denial_date")) if pd.notna(row.get("denial_date")) else None,
        "appeal_filed": row.get("appeal_filed"),
        "appeal_status": row.get("appeal_status"),
        "final_outcome": row.get("final_outcome"),
    })


def get_payer_rules(claim_id: str) -> dict:
    """Return historical denial rate and top denial reasons for the payer on this claim."""
    from src.financial_analytics import denial_rate_by_payer, denials_by_reason
    _claims, _denials, merged = _load_data()
    row = _get_claim_row(claim_id)
    payer = row.get("insurance_provider")
    by_payer = denial_rate_by_payer(merged)
    payer_row = by_payer[by_payer["insurance_provider"] == payer]
    if payer_row.empty:
        rate_info = {"payer": str(payer), "denial_rate_pct": None, "claims": 0, "denied": 0}
    else:
        pr = payer_row.iloc[0]
        rate_info = _safe({
            "payer": pr["insurance_provider"],
            "claims": pr["claims"],
            "denied": pr["denied"],
            "denial_rate_pct": pr["denial_rate_pct"],
        })
    by_reason = denials_by_reason(merged)
    payer_merged = merged[merged["insurance_provider"] == payer]
    payer_reasons = denials_by_reason(payer_merged).head(4)
    top_reasons = [
        _safe({"code": r["denial_reason_code"], "description": r["denial_reason_description"], "count": r["denied"]})
        for _, r in payer_reasons.iterrows()
    ]
    return {
        "payer_stats": rate_info,
        "top_denial_reasons": top_reasons,
        "limitation": "Historical patterns do not establish this claim will be denied.",
    }


def get_similar_claims(claim_id: str) -> dict:
    """Compare billed amount with same-payer records (descriptive, not a prediction)."""
    claims, _denials, _merged = _load_data()
    row = _get_claim_row(claim_id)
    payer = row.get("insurance_provider")
    peers = claims[(claims["insurance_provider"] == payer) &
                   (claims["claim_id"].astype(str) != str(claim_id))].copy()
    amounts = pd.to_numeric(peers["billed_amount"], errors="coerce").dropna()
    value = pd.to_numeric(row.get("billed_amount"), errors="coerce")
    return _safe({
        "comparison": "Same insurance provider (not clinically matched)",
        "peer_count": int(len(amounts)),
        "peer_median_billed_amount": round(float(amounts.median()), 2) if len(amounts) else None,
        "selected_billed_amount": float(value) if pd.notna(value) else None,
        "limitation": "Different procedures and services can have different legitimate charges.",
    })


TOOLS = {
    "get_claim_details": get_claim_details,
    "get_denial_history": get_denial_history,
    "get_payer_rules": get_payer_rules,
    "get_similar_claims": get_similar_claims,
}

TOOL_DESCRIPTIONS = {
    "get_claim_details": "Read permitted claim billing fields (no patient identifiers).",
    "get_denial_history": "Return the denial record for this claim, if one exists.",
    "get_payer_rules": "Summarize historical denial rate and top denial reasons for the payer.",
    "get_similar_claims": "Compare billed amount with same-payer records.",
}

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": TOOL_DESCRIPTIONS[name],
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    }
    for name in TOOLS
]

MAX_ROUNDS = 5


# ------------------------------------------------------------------ LLM helpers

def _call_openai_sdk(client, model: str, messages: list, tools: list) -> object:
    """Call the OpenAI SDK and return the response message."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0,
    )
    return response.choices[0].message


def _call_openai_rest(api_key: str, model: str, messages: list, tools: list) -> dict:
    """Fallback: call the OpenAI REST endpoint directly using requests."""
    import requests as req
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0,
    }
    resp = req.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]


# ------------------------------------------------------------------ public API

def investigate(claim_id: str) -> dict:
    """Run the bounded AI investigation for claim_id.

    Returns a dict with keys:
      summary    : str   - AI-generated summary or error message
      tool_trace : list  - list of {"tool": str, "result": dict}
      claim_id   : str   - the claim_id that was investigated
      status     : str   - "ok" | "unavailable" | "error" | "limit_reached"
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {
            "summary": "AI investigator unavailable: OPENAI_API_KEY is not configured.",
            "tool_trace": [],
            "claim_id": claim_id,
            "status": "unavailable",
        }

    model = os.environ.get("COUVERTE_AGENT_MODEL", MODEL_DEFAULT)

    try:
        # Validate claim exists before calling the LLM
        _get_claim_row(claim_id)
    except ValueError as exc:
        return {
            "summary": str(exc),
            "tool_trace": [],
            "claim_id": claim_id,
            "status": "error",
        }
    except Exception as exc:
        return {
            "summary": "Data load error: " + str(exc),
            "tool_trace": [],
            "claim_id": claim_id,
            "status": "error",
        }

    # Try the SDK first; fall back to requests if openai is not installed
    try:
        import openai as _openai_module
        use_sdk = True
        client = _openai_module.OpenAI(api_key=api_key)
    except ImportError:
        use_sdk = False
        client = None

    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                "Investigate claim " + str(claim_id) + ". "
                "Use tools to gather evidence, then recommend human review actions."
            ),
        },
    ]
    trace: list[dict] = []

    try:
        for _round in range(MAX_ROUNDS):
            if use_sdk:
                msg = _call_openai_sdk(client, model, messages, TOOL_SPECS)
                msg_dict = msg.model_dump(exclude_none=True)
                messages.append(msg_dict)
                tool_calls = msg.tool_calls or []
                finish = not tool_calls
                content = msg.content or ""
                if finish:
                    return {
                        "summary": content or "No summary generated.",
                        "tool_trace": trace,
                        "claim_id": claim_id,
                        "status": "ok",
                    }
                for call in tool_calls[:4]:
                    name = call.function.name
                    if name not in TOOLS:
                        result = {"error": "Tool not allowed"}
                    else:
                        result = TOOLS[name](claim_id)
                    trace.append({"tool": name, "result": result})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result, default=str)[:16000],
                    })
            else:
                # REST fallback
                raw = _call_openai_rest(api_key, model, messages, TOOL_SPECS)
                messages.append(raw)
                tool_calls = raw.get("tool_calls") or []
                finish = not tool_calls
                content = raw.get("content") or ""
                if finish:
                    return {
                        "summary": content or "No summary generated.",
                        "tool_trace": trace,
                        "claim_id": claim_id,
                        "status": "ok",
                    }
                for call in tool_calls[:4]:
                    name = call.get("function", {}).get("name", "")
                    if name not in TOOLS:
                        result = {"error": "Tool not allowed"}
                    else:
                        result = TOOLS[name](claim_id)
                    trace.append({"tool": name, "result": result})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": json.dumps(result, default=str)[:16000],
                    })

        return {
            "summary": "Investigation tool-call limit reached; review evidence manually.",
            "tool_trace": trace,
            "claim_id": claim_id,
            "status": "limit_reached",
        }

    except Exception as exc:
        return {
            "summary": "Investigation failed: " + str(exc),
            "tool_trace": trace,
            "claim_id": claim_id,
            "status": "error",
        }
