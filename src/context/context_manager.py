"""
context_manager.py — Ties all context engineering techniques together.

WHY THIS EXISTS
---------------
Individual techniques (token counting, layering, offloading, retrieval)
are powerful on their own. This module provides the functions that
COMBINE them — specifically for building the optimal context window
to send to Claude at each step.

Think of it as the scheduler in an operating system:
  - It knows what's in RAM (active context)
  - It knows what's on disk (offload store)
  - It decides what to load, what to evict, and what to compress

KEY FUNCTION: build_context_window()
  This is called by reason_node before every Claude API call.
  It assembles the best possible context window from:
    1. CRITICAL messages (always included)
    2. Retrieved memory (compressed, injected as a system note)
    3. WORKING messages (current task context)
  BACKGROUND messages are excluded — they've been offloaded.

TECHNIQUE 7: Retrieval-Augmented Compression
  When we retrieve messages from the offload store, we don't just
  paste them in verbatim. We compress them to their most information-
  dense form first. This maximises how much past context we can
  re-inject within a fixed token budget.
"""

import time
import anthropic
from src.context.token_counter import count_tokens
from src.context.layer_manager import ContextLayer
from src.context.offload_store import initialise_db
from src.config import ANTHROPIC_API_KEY, HAIKU_MODEL_NAME

_haiku_client: anthropic.Anthropic | None = None


def _get_haiku_client() -> anthropic.Anthropic | None:
    """Lazy-init the Haiku client (only created if summarisation is needed)."""
    global _haiku_client
    if _haiku_client is None and ANTHROPIC_API_KEY:
        _haiku_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _haiku_client

# Initialise DB on module load — safe, idempotent
initialise_db()


def build_context_window(
    messages: list[dict],
    retrieved_context: str = "",
) -> list[dict]:
    """
    Assemble the optimised context window for a Claude API call.

    Structure (in order):
      1. CRITICAL messages  — Always present, never removed
      2. Retrieved context  — Injected as a system note (if any)
      3. WORKING messages   — Current task conversation

    BACKGROUND messages are excluded — they've been offloaded.

    Args:
        messages:          The full active message list from agent state.
        retrieved_context: Compressed text from the offload store.

    Returns:
        Ordered list of message dicts ready for the Anthropic API.
    """
    critical_msgs = [
        m for m in messages
        if m.get("layer") == ContextLayer.CRITICAL.value
        and m.get("message_id") not in ("retrieved", "scratchpad")
    ]
    working_msgs = [
        m for m in messages
        if m.get("layer") == ContextLayer.WORKING.value
    ]

    context: list[dict] = []

    # Layer 1: Critical (always first — the model sees these before anything)
    context.extend(critical_msgs)

    # Layer 2: Retrieved memory (injected as a system note with clear markers)
    if retrieved_context:
        context.append({
            "role": "system",
            "content": (
                "[RETRIEVED FROM LONG-TERM MEMORY]\n"
                "The following was retrieved from past conversation that "
                "was offloaded to storage. Use it if relevant:\n\n"
                f"{retrieved_context}\n"
                "[END RETRIEVED MEMORY]"
            ),
            "layer": ContextLayer.CRITICAL.value,
            "token_count": count_tokens(retrieved_context),
            "message_id": "retrieved",
            "timestamp": time.time(),
        })

    # Layer 3: Working context (most recent task-relevant messages)
    context.extend(working_msgs)

    return context


def compress_retrieved(
    retrieved_messages: list[dict],
    max_tokens: int = 1500,
) -> str:
    """
    Technique 7: Retrieval-Augmented Compression.

    Takes retrieved messages and compresses them into a dense summary
    using Claude Haiku — preserving key facts, names, and decisions
    at a fraction of the original token cost.

    WHY COMPRESS?
    Verbatim injection of retrieved messages can cost 3,000+ tokens.
    LLM summarisation condenses the same information to ~300-400 tokens
    while retaining semantic fidelity — far better than naive truncation.

    WHY HAIKU?
    Haiku is ~18× cheaper than Sonnet for this task. Summarisation is
    a mechanical, low-difficulty operation that doesn't need Sonnet's
    reasoning power. This is a classic "right model for the job" decision.

    FALLBACK
    If the Haiku API call fails (key missing, network error, etc.),
    we fall back to the original 350-char truncation approach so the
    system never breaks silently.

    Args:
        retrieved_messages: Messages from retrieve_relevant().
        max_tokens:         Token budget for the compressed output.

    Returns:
        Compressed string ready to inject into the context window.
    """
    if not retrieved_messages:
        return ""

    # Build the raw text to summarise
    raw_parts = [
        f"[{msg['role'].upper()}]: {msg['content']}"
        for msg in retrieved_messages
    ]
    raw_text = "\n\n".join(raw_parts)

    # ── Primary: LLM summarisation with Haiku ──────────────────────────────
    client = _get_haiku_client()
    if client:
        try:
            response = client.messages.create(
                model=HAIKU_MODEL_NAME,
                max_tokens=400,
                messages=[{
                    "role": "user",
                    "content": (
                        "Compress the following conversation excerpts into a concise "
                        "summary. Preserve all key facts, names, numbers, decisions, "
                        "and context. Be maximally dense — every word must carry "
                        "information. Output only the summary, no preamble.\n\n"
                        f"{raw_text}"
                    ),
                }],
            )
            return response.content[0].text
        except Exception:
            pass  # Fall through to truncation fallback

    # ── Fallback: 350-char truncation ──────────────────────────────────────
    lines = []
    used = 0
    for msg in retrieved_messages:
        content_snippet = msg["content"]
        if len(content_snippet) > 350:
            content_snippet = content_snippet[:350] + "..."
        line = f"[{msg['role'].upper()}]: {content_snippet}"
        line_tokens = count_tokens(line)
        if used + line_tokens > max_tokens:
            break
        lines.append(line)
        used += line_tokens

    return "\n\n".join(lines)


def summarize_scratchpad(scratchpad: str) -> str:
    """
    Compress the scratchpad when it grows beyond the configured threshold.

    The scratchpad is an append-only reasoning trace. Without periodic
    compression it grows unbounded, consuming tokens and becoming
    unreadable. This function keeps the last 3 entries verbatim
    (most relevant) and summarises everything older with Haiku.

    Falls back to keeping the last 10 lines if the API call fails.
    """
    lines = [l for l in scratchpad.split("\n") if l.strip()]
    recent      = lines[-3:]       # Always keep last 3 entries raw
    to_summarise = lines[:-3]

    client = _get_haiku_client()
    if client and to_summarise:
        try:
            raw = "\n".join(to_summarise)
            response = client.messages.create(
                model=HAIKU_MODEL_NAME,
                max_tokens=150,
                messages=[{
                    "role": "user",
                    "content": (
                        "Summarise these agent reasoning trace entries into 2 sentences. "
                        "Focus on: total offloads, retrieval hits, tool calls, token trends.\n\n"
                        f"{raw}"
                    ),
                }],
            )
            summary = response.content[0].text.strip()
            return "\n".join([f"[SCRATCHPAD SUMMARY] {summary}"] + recent)
        except Exception:
            pass

    return "\n".join(lines[-10:])  # Fallback: keep last 10 lines


def calculate_window_stats(messages: list[dict]) -> dict:
    """
    Analyse the current context window composition.

    Returns stats used by the Streamlit token dashboard.
    """
    total_tokens = sum(m.get("token_count", 0) for m in messages)
    by_layer = {layer.value: 0 for layer in ContextLayer}
    by_role = {"user": 0, "assistant": 0, "system": 0}

    for msg in messages:
        layer = msg.get("layer", ContextLayer.WORKING.value)
        role = msg.get("role", "user")
        tokens = msg.get("token_count", 0)

        if layer in by_layer:
            by_layer[layer] += tokens
        if role in by_role:
            by_role[role] += tokens

    return {
        "total_tokens": total_tokens,
        "by_layer": by_layer,
        "by_role": by_role,
        "message_count": len(messages),
    }
