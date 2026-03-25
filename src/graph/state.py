"""
state.py — The TypedDict schema for the LangGraph agent.

WHY TYPEDDICT MATTERS
---------------------
LangGraph passes state between nodes as a Python dictionary.
Without TypedDict, any node can write any key and you won't
discover the bug until runtime — often in production.

TypedDict makes the state contract explicit:
  - Every field has a name, a type, and a clear purpose
  - Nodes can be tested in isolation (mock the state dict)
  - The pipeline is self-documenting — read this file to understand
    exactly what information flows through the system
  - Type checkers (mypy, pyright) can catch misuse early

WHAT'S IN THE STATE
-------------------
The state carries everything the agent needs across all nodes:
  1. The conversation (messages)
  2. Token tracking metadata (for Pre-Rot Threshold)
  3. Offloading metadata (what's been moved to storage)
  4. Retrieval results (compressed past context)
  5. The scratchpad (reasoning trace, separate from conversation)
  6. Agent mode (for the dashboard to show what's happening)
  7. The final response (written by respond_node)
"""

from typing import TypedDict


class AgentState(TypedDict):
    """
    Complete state schema for the context engineering agent.

    This dict is passed into the graph at the START node and
    returned from the END node after all nodes have processed it.
    Each node receives the full state and returns only the keys
    it modified — LangGraph merges the updates.
    """

    # Conversation
    messages: list[dict]
    # Full list of active messages. Each message is a dict with:
    #   role:         "user" | "assistant" | "system"
    #   content:      The message text
    #   layer:        "critical" | "working" | "background"
    #   token_count:  Token count including safety buffer
    #   message_id:   UUID string for tracking / offloading
    #   timestamp:    Unix timestamp (float) for age-based offloading

    session_id: str
    # Unique identifier for this conversation session.
    # Used to namespace offloaded messages in SQLite.

    # Token tracking (Technique 1: Pre-Rot Threshold)
    token_budget: int
    # Total context window budget for this session (e.g. 100_000).
    # Set at session creation, never changes.

    current_tokens: int
    # Running total of tokens in the ACTIVE context window.
    # Updated by classify_input_node and respond_node.
    # Reduced by offload_context_node when messages are moved out.

    pre_rot_threshold: float
    # Fraction of token_budget at which to trigger offloading (e.g. 0.70).
    # When current_tokens / token_budget >= this value, needs_offload = True.

    needs_offload: bool
    # Flag set by monitor_tokens_node.
    # The conditional edge after monitor_tokens reads this flag
    # to decide whether to route to offload_context or retrieve_context.

    # Offloading metadata (Technique 3: Context Offloading) ────────────────
    offloaded_count: int
    # Cumulative number of messages moved to SQLite this session.
    # Shown on the dashboard — demonstrates the system is working.

    offloaded_tokens: int
    # Cumulative tokens freed through offloading this session.
    # Also shown on dashboard.

    #  Retrieval (Technique 7: Retrieval-Augmented Compression) ─────────────
    latest_query: str
    # The user's most recent message text.
    # Used by retrieve_context_node to find relevant offloaded messages.

    retrieved_context: str
    # Compressed text retrieved from the offload store.
    # Empty string if nothing was retrieved (no offloads yet, or no match).
    # Injected into the context window as a system note in reason_node.

    # Scratchpad (Technique 6: Scratchpad Management)
    scratchpad: str
    # A running log of the agent's reasoning trace.
    # Stored SEPARATELY from the messages list so that:
    #   - The reasoning trace doesn't pollute the factual context
    #   - The scratchpad can be cleared independently
    #   - It's inspectable in the dashboard for debugging
    # Format: timestamped entries appended with "\n".

    # Agent mode (for dashboard visibility)
    agent_mode: str
    # Current processing state. One of:
    #   "idle"        — Waiting for input
    #   "classifying" — classify_input_node is running
    #   "monitoring"  — monitor_tokens_node is running
    #   "offloading"  — offload_context_node is running
    #   "retrieving"  — retrieve_context_node is running
    #   "reasoning"   — reason_node is running (Claude API call)
    #   "responding"  — respond_node is updating message history

    # Output
    final_response: str
    # The assistant's response text.
    # Written by reason_node, added to messages by respond_node.
    # Also displayed directly by the Streamlit app.

    # Cost tracking
    session_input_tokens: int
    # Cumulative input tokens sent to Claude (Sonnet) this session.
    # Incremented by reason_node after every API call (including tool loops).

    session_output_tokens: int
    # Cumulative output tokens received from Claude (Sonnet) this session.

    total_cost_usd: float
    # Running session cost in USD, calculated from token usage and model pricing.
    # Displayed on the dashboard so users can see the economics of context management.
