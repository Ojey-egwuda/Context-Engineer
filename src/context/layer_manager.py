"""
layer_manager.py — Layered Action Space (Technique 2).

WHY THIS EXISTS
---------------
Not all context is equally important. A user's name stated at the
start of a conversation should never be forgotten. A passing remark
from three hours ago can safely be offloaded. Without layers,
you'd either keep everything (runs out of space) or lose important
things (bad responses).

THE THREE LAYERS
----------------
  CRITICAL   — Must stay in context at all times.
               Examples: system prompt, user name, key constraints,
               explicit "remember this" statements.
               Never offloaded, ever.

  WORKING    — Relevant to the current task.
               In context while the task is active.
               Offloaded after BACKGROUND when space is needed.

  BACKGROUND — Older conversation turns, peripheral information.
               First to be offloaded when Pre-Rot Threshold hits.

HOW CLASSIFICATION WORKS
------------------------
We use heuristic rules (keyword matching, message length, role).
This is fast and free — no API call required.

In production you'd train a lightweight classifier to do this, or
use a small LLM call with a structured output. For this project,
heuristics are transparent and debuggable.
"""

from enum import Enum
import anthropic as _anthropic
from src.config import ANTHROPIC_API_KEY, HAIKU_MODEL_NAME, USE_LLM_CLASSIFICATION

_haiku_client: "_anthropic.Anthropic | None" = None


def _get_haiku():
    global _haiku_client
    if _haiku_client is None and ANTHROPIC_API_KEY:
        _haiku_client = _anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _haiku_client


class ContextLayer(str, Enum):
    """
    The three tiers of the Layered Action Space.

    We inherit from str so that layer values serialise cleanly
    to/from JSON and the SQLite database without extra conversion.
    """
    CRITICAL   = "critical"
    WORKING    = "working"
    BACKGROUND = "background"


# Classification Signals

# These phrases in a message suggest it contains critical information
# that the agent must not lose.
_CRITICAL_SIGNALS = [
    "please remember", "don't forget", "remember this",
    "my name is", "i am called", "you must always",
    "key constraint", "hard rule:", "never forget",
    "please keep in mind", "important constraint",
    "remember this instead", "is important:",
]

# These phrases suggest the message is peripheral / background
_BACKGROUND_SIGNALS = [
    "by the way", "also ", "additionally", "incidentally",
    "just to mention", "quick note", "side note", "off topic",
    "not important", "just wondering", "curious about",
]


def classify_layer(content: str, role: str) -> ContextLayer:
    """
    Classify a message into a context layer.

    Args:
        content: The message text.
        role:    "user", "assistant", or "system".

    Returns:
        ContextLayer enum value.

    Logic:
        1. System messages are always CRITICAL.
        2. Messages with critical signals are CRITICAL.
        3. Very short messages (< 15 words) are BACKGROUND.
        4. Messages with background signals are BACKGROUND.
        5. Everything else is WORKING.
    """
    # Rule 1: System messages are always critical (system prompt, injected facts)
    if role == "system":
        return ContextLayer.CRITICAL

    content_lower = content.lower()

    # Rule 2: Explicit critical signals in the message
    if any(signal in content_lower for signal in _CRITICAL_SIGNALS):
        return ContextLayer.CRITICAL

    # Rule 3: Very short messages are low-value background
    word_count = len(content.split())
    if word_count < 10:
        return ContextLayer.BACKGROUND

    # Rule 4: Explicit background signals
    if any(signal in content_lower for signal in _BACKGROUND_SIGNALS):
        return ContextLayer.BACKGROUND

    # Rule 5: Default — working layer (relevant to current task)
    return ContextLayer.WORKING


def classify_layer_llm(content: str, role: str) -> "ContextLayer":
    """
    Classify message importance using Claude Haiku.

    More accurate than heuristics — understands implication and nuance
    that keyword matching misses. For example:
      "I go by Alex" → CRITICAL (identity, not caught by exact keyword)
      "Interesting, go on" → BACKGROUND (filler, heuristic might miss it)

    Falls back to heuristic classify_layer() on any failure.
    Cost: ~$0.00008 per message — negligible at Haiku pricing.
    """
    if role == "system":
        return ContextLayer.CRITICAL

    client = _get_haiku()
    if client is None:
        return classify_layer(content, role)

    try:
        response = client.messages.create(
            model=HAIKU_MODEL_NAME,
            max_tokens=5,
            system=(
                "Classify the importance of this message for an AI assistant's memory. "
                "Reply with exactly one word:\n"
                "CRITICAL — identity facts, explicit constraints, 'remember this', "
                "permanent user preferences\n"
                "WORKING — questions, answers, ongoing discussion relevant to the task\n"
                "BACKGROUND — filler, small talk, peripheral remarks, easily forgettable\n"
                "Reply with only: CRITICAL, WORKING, or BACKGROUND"
            ),
            messages=[{"role": "user", "content": content[:600]}],
        )
        label = response.content[0].text.strip().upper()
        if "CRITICAL" in label:
            return ContextLayer.CRITICAL
        elif "BACKGROUND" in label:
            return ContextLayer.BACKGROUND
        return ContextLayer.WORKING
    except Exception:
        return classify_layer(content, role)


def get_offload_candidates(
    messages: list[dict],
    tokens_to_free: int,
) -> list[str]:
    """
    Choose which messages to offload to free up a target token count.

    CRITICAL messages are NEVER selected, no matter how much space
    we need. This is a hard guarantee, not a preference.

    Selection order:
      1. BACKGROUND messages, oldest first
      2. WORKING messages, oldest first
      (CRITICAL: never)

    Args:
        messages:       Active message list from agent state.
        tokens_to_free: Target number of tokens to free.

    Returns:
        List of message_ids to offload (may be fewer than needed
        if not enough non-critical messages exist).
    """
    # Priority: 0 = offload first, 1 = offload second, 999 = never offload
    layer_priority = {
        ContextLayer.BACKGROUND.value: 0,
        ContextLayer.WORKING.value:    1,
        ContextLayer.CRITICAL.value:   999,
    }

    # Filter out critical messages — they're ineligible
    eligible = [
        m for m in messages
        if m.get("layer") != ContextLayer.CRITICAL.value
        and m.get("message_id") not in ("retrieved", "scratchpad")
    ]

    # Sort: background first, then by age (oldest timestamp first)
    eligible.sort(
        key=lambda m: (
            layer_priority.get(m.get("layer", "working"), 1),
            m.get("timestamp", 0),
        )
    )

    selected_ids = []
    freed = 0

    for msg in eligible:
        if freed >= tokens_to_free:
            break
        selected_ids.append(msg["message_id"])
        freed += msg.get("token_count", 0)

    return selected_ids


def layer_summary(messages: list[dict]) -> dict:
    """
    Count and token-sum messages by layer. Used by the dashboard.

    Returns a dict like:
        {
          "critical":   {"count": 2, "tokens": 450},
          "working":    {"count": 8, "tokens": 3200},
          "background": {"count": 1, "tokens": 120},
        }
    """
    summary = {
        layer.value: {"count": 0, "tokens": 0}
        for layer in ContextLayer
    }

    for msg in messages:
        layer = msg.get("layer", ContextLayer.WORKING.value)
        if layer in summary:
            summary[layer]["count"]  += 1
            summary[layer]["tokens"] += msg.get("token_count", 0)

    return summary
