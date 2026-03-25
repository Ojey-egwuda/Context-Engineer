"""
test_evaluation.py — End-to-end quality evaluation for Context Engineer.

WHAT THIS TESTS
---------------
Unlike unit tests (which test a single function in isolation), evaluation
tests verify that the SYSTEM works correctly end-to-end:
  - Does the agent recall facts after they've been offloaded?
  - Does classification work correctly for different message types?
  - Does the cost tracker accumulate correctly?
  - Does scratchpad compression trigger at the right threshold?
  - Does multi-user session isolation work correctly?
  - Does memory poisoning protection hold?

WHY NO API KEY IS REQUIRED
---------------------------
All Claude API calls are mocked. This means:
  1. Tests run in CI without credentials
  2. Tests are deterministic (no LLM randomness)
  3. Tests are fast (no network latency)

HOW TO RUN
----------
  pytest tests/test_evaluation.py -v
"""

import pytest
import time
import uuid
from unittest.mock import patch, MagicMock


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_mock_response(text: str, input_tokens: int = 50, output_tokens: int = 30):
    """Create a deterministic mock Anthropic API response."""
    block = MagicMock()
    block.text  = text
    block.type  = "text"
    resp = MagicMock()
    resp.content                = [block]
    resp.usage.input_tokens     = input_tokens
    resp.usage.output_tokens    = output_tokens
    resp.stop_reason            = "end_turn"
    return resp


def make_state(token_budget: int = 5000, pre_rot_threshold: float = 0.70):
    """Fresh AgentState with no API calls needed."""
    from src.agents.main_agent import create_session
    return create_session(
        system_prompt="You are a helpful assistant.",
        token_budget=token_budget,
        pre_rot_threshold=pre_rot_threshold,
    )


# ── Test 1: Full pipeline runs end-to-end ────────────────────────────────────

@patch("src.graph.nodes._client")
@patch("src.context.layer_manager._haiku_client")
def test_full_pipeline_completes(mock_haiku, mock_client):
    mock_client.messages.create.return_value = make_mock_response("Hello! How can I help?")
    mock_haiku.messages.create.return_value  = make_mock_response("WORKING")

    from src.agents.main_agent import chat
    state = make_state()
    new_state, response = chat(state, "Hello")

    assert response == "Hello! How can I help?"
    assert len(new_state["messages"]) >= 2      # user + assistant
    assert new_state["agent_mode"] == "responding"


# ── Test 2: Cost accumulates across turns ────────────────────────────────────

@patch("src.graph.nodes._client")
@patch("src.context.layer_manager._haiku_client")
def test_cost_accumulates_across_turns(mock_haiku, mock_client):
    mock_client.messages.create.return_value = make_mock_response(
        "Got it.", input_tokens=100, output_tokens=20
    )
    mock_haiku.messages.create.return_value = make_mock_response("WORKING")

    from src.agents.main_agent import chat
    state = make_state()

    state, _ = chat(state, "First message")
    cost_1   = state["total_cost_usd"]
    in_tok_1 = state["session_input_tokens"]
    assert cost_1   > 0, "Cost should be non-zero after first turn"
    assert in_tok_1 > 0

    state, _ = chat(state, "Second message")
    assert state["total_cost_usd"]          > cost_1,   "Cost must grow each turn"
    assert state["session_input_tokens"]    > in_tok_1, "Input tokens must grow each turn"


# ── Test 3: Offloading triggers at threshold ─────────────────────────────────

@patch("src.graph.nodes._client")
@patch("src.context.layer_manager._haiku_client")
def test_offloading_triggers_at_threshold(mock_haiku, mock_client):
    mock_client.messages.create.return_value = make_mock_response("Okay.")
    mock_haiku.messages.create.return_value  = make_mock_response("WORKING")

    from src.agents.main_agent import chat
    state    = make_state(token_budget=500, pre_rot_threshold=0.50)
    long_msg = "This is a moderately long message to consume tokens. " * 5

    for _ in range(6):
        state, _ = chat(state, long_msg)

    assert state["offloaded_count"]  > 0, "Offloading should have triggered"
    assert state["offloaded_tokens"] > 0


# ── Test 4: Offloaded messages are retrievable ───────────────────────────────

@patch("src.graph.nodes._client")
@patch("src.context.layer_manager._haiku_client")
def test_offloaded_messages_are_retrievable(mock_haiku, mock_client):
    mock_client.messages.create.return_value = make_mock_response("Noted.")
    mock_haiku.messages.create.return_value  = make_mock_response("WORKING")

    from src.agents.main_agent import chat
    from src.context.offload_store import retrieve_relevant

    state = make_state(token_budget=400, pre_rot_threshold=0.50)
    state, _ = chat(state, "My favourite programming language is Rust.")

    filler = "Tell me about software engineering practices. " * 8
    for _ in range(5):
        state, _ = chat(state, filler)

    if state["offloaded_count"] > 0:
        results = retrieve_relevant(state["session_id"], "programming language Rust")
        contents = " ".join(r["content"] for r in results)
        assert "Rust" in contents or "programming" in contents, \
            "Offloaded Rust fact should be retrievable"


# ── Test 5: Heuristic CRITICAL classification ────────────────────────────────

def test_heuristic_classifies_critical_signals():
    from src.context.layer_manager import classify_layer, ContextLayer
    phrases = [
        "My name is Alice",
        "Please remember that I prefer Python",
        "Remember this: always use metric units",
        "Never forget that I'm allergic to shellfish",
    ]
    for phrase in phrases:
        assert classify_layer(phrase, "user") == ContextLayer.CRITICAL, \
            f"Expected CRITICAL for: {phrase!r}"


# ── Test 6: Heuristic BACKGROUND classification ──────────────────────────────

def test_heuristic_classifies_background():
    from src.context.layer_manager import classify_layer, ContextLayer
    assert classify_layer("Ok", "user")                     == ContextLayer.BACKGROUND
    assert classify_layer("Sure", "user")                   == ContextLayer.BACKGROUND
    assert classify_layer("By the way, also note", "user")  == ContextLayer.BACKGROUND


# ── Test 7: System messages always CRITICAL ──────────────────────────────────

def test_system_messages_always_critical():
    from src.context.layer_manager import classify_layer, ContextLayer
    for content in ["Hello", "by the way", "Hi", ""]:
        assert classify_layer(content, "system") == ContextLayer.CRITICAL


# ── Test 8: Memory poisoning guard ──────────────────────────────────────────

@patch("src.graph.nodes._client")
@patch("src.context.layer_manager._haiku_client")
def test_memory_poisoning_downgraded(mock_haiku, mock_client):
    """Second CRITICAL user message is downgraded to WORKING."""
    mock_client.messages.create.return_value = make_mock_response("Understood.")
    mock_haiku.messages.create.return_value  = make_mock_response("CRITICAL")

    from src.agents.main_agent import chat
    state = make_state()

    state, _ = chat(state, "My name is Alice — please remember this.")
    state, _ = chat(state, "My name is actually Bob — remember this instead.")

    user_messages   = [m for m in state["messages"] if m.get("role") == "user"]
    critical_count  = sum(1 for m in user_messages if m.get("layer") == "critical")
    assert critical_count <= 1, \
        f"Memory poisoning guard failed — {critical_count} CRITICAL user messages"


# ── Test 9: Scratchpad populated each turn ───────────────────────────────────

@patch("src.graph.nodes._client")
@patch("src.context.layer_manager._haiku_client")
def test_scratchpad_populated(mock_haiku, mock_client):
    mock_client.messages.create.return_value = make_mock_response("Response.")
    mock_haiku.messages.create.return_value  = make_mock_response("WORKING")

    from src.agents.main_agent import chat
    state = make_state()
    state, _ = chat(state, "Hello there")

    scratchpad = state.get("scratchpad", "")
    assert scratchpad.strip(),          "Scratchpad should have entries after a turn"
    assert "active_tokens=" in scratchpad


# ── Test 10: Session isolation ───────────────────────────────────────────────

@patch("src.graph.nodes._client")
@patch("src.context.layer_manager._haiku_client")
def test_session_isolation(mock_haiku, mock_client):
    """Session B cannot see Session A's offloaded messages."""
    mock_client.messages.create.return_value = make_mock_response("OK.")
    mock_haiku.messages.create.return_value  = make_mock_response("WORKING")

    from src.agents.main_agent import chat
    from src.context.offload_store import retrieve_relevant

    state_a = make_state(token_budget=300, pre_rot_threshold=0.40)
    state_b = make_state(token_budget=300, pre_rot_threshold=0.40)

    for _ in range(4):
        state_a, _ = chat(state_a, "Secret information in session A only. " * 3)

    results = retrieve_relevant(state_b["session_id"], "secret session A")
    assert len(results) == 0, "Session B must not see Session A's offloaded messages"


# ── Test 11: Health dict includes cost fields ────────────────────────────────

def test_health_includes_cost_fields():
    from src.agents.main_agent import get_context_health
    state  = make_state()
    health = get_context_health(state)

    assert "total_cost_usd"        in health
    assert "session_input_tokens"  in health
    assert "session_output_tokens" in health
    assert health["total_cost_usd"]       == 0.0
    assert health["session_input_tokens"] == 0


# ── Test 12: Active tokens stay within budget ────────────────────────────────

@patch("src.graph.nodes._client")
@patch("src.context.layer_manager._haiku_client")
def test_active_tokens_within_budget(mock_haiku, mock_client):
    mock_client.messages.create.return_value = make_mock_response("OK.")
    mock_haiku.messages.create.return_value  = make_mock_response("WORKING")

    from src.agents.main_agent import chat
    state = make_state(token_budget=800, pre_rot_threshold=0.60)

    for _ in range(8):
        state, _ = chat(state, "A moderately long message about various things. " * 3)

    budget  = state["token_budget"]
    current = state["current_tokens"]
    # Allow 10% overshoot — offload fires slightly after threshold
    assert current <= budget * 1.10, \
        f"Active tokens {current} exceed budget {budget} by more than 10%"


# ── Test 13: LLM classifier falls back to heuristic on failure ───────────────

def test_llm_classifier_falls_back():
    """classify_layer_llm() returns a valid layer even when Haiku is unavailable."""
    from src.context.layer_manager import classify_layer_llm, ContextLayer

    with patch("src.context.layer_manager._get_haiku", return_value=None):
        layer = classify_layer_llm("My name is Alex", "user")
        assert layer in list(ContextLayer), f"Expected a valid ContextLayer, got {layer}"


# ── Test 14: Cost calculation uses correct pricing ───────────────────────────

def test_cost_calculation():
    """Verify cost per token constants match expected pricing."""
    from src.config import (
        SONNET_INPUT_COST_PER_TOKEN, SONNET_OUTPUT_COST_PER_TOKEN,
        HAIKU_INPUT_COST_PER_TOKEN,  HAIKU_OUTPUT_COST_PER_TOKEN,
    )
    # $3 per 1M input tokens
    assert abs(SONNET_INPUT_COST_PER_TOKEN  - 3.0  / 1_000_000) < 1e-12
    # $15 per 1M output tokens
    assert abs(SONNET_OUTPUT_COST_PER_TOKEN - 15.0 / 1_000_000) < 1e-12
    # Haiku should be cheaper than Sonnet
    assert HAIKU_INPUT_COST_PER_TOKEN  < SONNET_INPUT_COST_PER_TOKEN
    assert HAIKU_OUTPUT_COST_PER_TOKEN < SONNET_OUTPUT_COST_PER_TOKEN


# ── Test 15: Scratchpad compression triggers at threshold ─────────────────────

@patch("src.context.context_manager._get_haiku_client")
def test_scratchpad_compression_triggers(mock_get_haiku):
    """summarize_scratchpad() fires when entries exceed the threshold."""
    mock_client = MagicMock()
    mock_client.messages.create.return_value = make_mock_response(
        "Summary: 10 turns, 3 offloads, 1 retrieval hit."
    )
    mock_get_haiku.return_value = mock_client

    from src.context.context_manager import summarize_scratchpad

    # Build a scratchpad with 25 entries
    entries = [f"[12:0{i%10}:00] active_tokens=1,000  offloaded=0  retrieved=no  no_tools"
               for i in range(25)]
    long_scratchpad = "\n".join(entries)

    result = summarize_scratchpad(long_scratchpad)

    # Should be shorter than the original
    assert len(result.split("\n")) < 25, "Compressed scratchpad should have fewer lines"
    # Recent entries should still be present
    for entry in entries[-3:]:
        assert entry in result, "Last 3 entries should be preserved verbatim"
