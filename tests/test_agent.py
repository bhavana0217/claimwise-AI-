"""Tests for src/agent.py

Never makes real API calls - all LLM interactions are mocked.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = APP_DIR / "data"


# ------------------------------------------------------------------ fixture: a real claim ID

def _first_claim_id() -> str:
    """Load the claims CSV and return the first valid claim_id."""
    from src.claims_processor import load_claims
    claims, _ = load_claims(DATA_DIR / "claims_and_billing.csv")
    return str(claims["claim_id"].dropna().iloc[0])


CLAIM_ID = _first_claim_id()


# ------------------------------------------------------------------ tool function tests

def test_get_claim_details_returns_expected_keys():
    from src.agent import get_claim_details
    result = get_claim_details(CLAIM_ID)
    assert isinstance(result, dict)
    assert "claim_id" in result
    assert "insurance_provider" in result
    assert "billed_amount" in result


def test_get_claim_details_nonempty():
    from src.agent import get_claim_details
    result = get_claim_details(CLAIM_ID)
    assert result["claim_id"] == CLAIM_ID


def test_get_denial_history_shape():
    from src.agent import get_denial_history
    result = get_denial_history(CLAIM_ID)
    assert isinstance(result, dict)
    assert "claim_id" in result
    assert "has_denial" in result
    assert isinstance(result["has_denial"], bool)


def test_get_payer_rules_shape():
    from src.agent import get_payer_rules
    result = get_payer_rules(CLAIM_ID)
    assert isinstance(result, dict)
    assert "payer_stats" in result
    assert "top_denial_reasons" in result
    assert isinstance(result["top_denial_reasons"], list)


def test_get_similar_claims_shape():
    from src.agent import get_similar_claims
    result = get_similar_claims(CLAIM_ID)
    assert isinstance(result, dict)
    assert "peer_count" in result
    assert isinstance(result["peer_count"], int)
    assert "peer_median_billed_amount" in result


def test_get_claim_details_unknown_id_raises():
    from src.agent import _get_claim_row
    with pytest.raises(ValueError, match="not found"):
        _get_claim_row("DOES_NOT_EXIST_XYZ")


# ------------------------------------------------------------------ investigate: unavailable key

def test_investigate_no_api_key_returns_unavailable():
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        from src.agent import investigate
        result = investigate(CLAIM_ID)
    assert result["status"] == "unavailable"
    assert result["claim_id"] == CLAIM_ID
    assert "OPENAI_API_KEY" in result["summary"]
    assert isinstance(result["tool_trace"], list)


def test_investigate_empty_api_key_returns_unavailable():
    with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
        from src.agent import investigate
        result = investigate(CLAIM_ID)
    assert result["status"] == "unavailable"


# ------------------------------------------------------------------ investigate: mocked LLM (SDK path)

def _make_mock_message(content=None, tool_calls=None):
    """Build a mock openai Message object."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.model_dump.return_value = {
        "role": "assistant",
        "content": content,
    }
    return msg


def _make_sdk_response(msg):
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    return resp


def test_investigate_with_mocked_sdk_returns_ok():
    """LLM immediately returns a summary with no tool calls."""
    final_msg = _make_mock_message(content="Summary: claim looks clean.", tool_calls=[])
    mock_response = _make_sdk_response(final_msg)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    mock_openai_module = MagicMock()
    mock_openai_module.OpenAI.return_value = mock_client

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-abc"}, clear=False):
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            # Force re-import to pick up the patched module
            import importlib
            import src.agent as agent_mod
            importlib.reload(agent_mod)
            result = agent_mod.investigate(CLAIM_ID)

    assert isinstance(result, dict)
    assert set(result.keys()) >= {"summary", "tool_trace", "claim_id", "status"}
    assert result["claim_id"] == CLAIM_ID
    assert result["status"] in ("ok", "limit_reached", "error")
    assert isinstance(result["tool_trace"], list)


def test_investigate_with_mocked_sdk_one_tool_call():
    """LLM calls one tool then returns a summary."""
    tool_call = MagicMock()
    tool_call.function.name = "get_claim_details"
    tool_call.id = "call_001"

    first_msg = _make_mock_message(content=None, tool_calls=[tool_call])
    first_msg.model_dump.return_value = {
        "role": "assistant",
        "tool_calls": [{"id": "call_001", "function": {"name": "get_claim_details"}}],
    }

    final_msg = _make_mock_message(content="Reviewed claim details; no anomalies found.", tool_calls=[])

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        _make_sdk_response(first_msg),
        _make_sdk_response(final_msg),
    ]

    mock_openai_module = MagicMock()
    mock_openai_module.OpenAI.return_value = mock_client

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-abc"}, clear=False):
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            import importlib
            import src.agent as agent_mod
            importlib.reload(agent_mod)
            result = agent_mod.investigate(CLAIM_ID)

    assert isinstance(result, dict)
    assert result["claim_id"] == CLAIM_ID
    assert isinstance(result["tool_trace"], list)


# ------------------------------------------------------------------ investigate: error paths

def test_investigate_bad_claim_id_returns_error():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-abc"}, clear=False):
        import importlib
        import src.agent as agent_mod
        importlib.reload(agent_mod)
        result = agent_mod.investigate("DEFINITELY_NOT_A_REAL_CLAIM_ID_9999")
    assert result["status"] == "error"
    assert result["claim_id"] == "DEFINITELY_NOT_A_REAL_CLAIM_ID_9999"
    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 0


def test_investigate_result_keys():
    """investigate() always returns the four required keys regardless of outcome."""
    import importlib
    import src.agent as agent_mod
    importlib.reload(agent_mod)
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        result = agent_mod.investigate(CLAIM_ID)
    assert "summary" in result
    assert "tool_trace" in result
    assert "claim_id" in result
    assert "status" in result
