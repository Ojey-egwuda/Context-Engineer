"""
config.py — Central configuration for Context Engineer.

All tunable values live here. Change them in one place,
they update everywhere.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Context Window Settings
# TOKEN_BUDGET: The total context window we're managing.
# Claude Sonnet supports up to 200K tokens, but we use 100K for this demo
# so you can actually observe the offloading behaviour in a normal session.
# Reduce to 5_000 during development to trigger offloading quickly.
TOKEN_BUDGET: int = int(os.getenv("TOKEN_BUDGET", "100000"))

# PRE_ROT_THRESHOLD: Fraction of TOKEN_BUDGET at which we trigger cleanup.
# 0.70 means "start offloading when we've used 70% of the window".
# Proactive, not reactive — we clean before quality degrades.
PRE_ROT_THRESHOLD: float = float(os.getenv("PRE_ROT_THRESHOLD", "0.70"))

# CRITICAL_RESERVE: Fraction of budget permanently reserved for CRITICAL messages.
# These are never offloaded (system prompt, key facts, constraints).
CRITICAL_RESERVE: float = 0.20

# Model Settings
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
MODEL_NAME: str = "claude-sonnet-4-5"
HAIKU_MODEL_NAME: str = "claude-haiku-4-5-20251001"   # Used for cheap summarisation
MAX_RESPONSE_TOKENS: int = 1024

# Pricing — used by the cost tracker ($ per token)
# Source: Anthropic pricing page (claude-sonnet-4-5 / claude-haiku-4-5)
SONNET_INPUT_COST_PER_TOKEN: float  = 3.0  / 1_000_000   # $3 / 1M input tokens
SONNET_OUTPUT_COST_PER_TOKEN: float = 15.0 / 1_000_000   # $15 / 1M output tokens
HAIKU_INPUT_COST_PER_TOKEN: float   = 0.80 / 1_000_000   # $0.80 / 1M input tokens
HAIKU_OUTPUT_COST_PER_TOKEN: float  = 4.0  / 1_000_000   # $4 / 1M output tokens

# LangSmith Observability
# LangSmith logs every graph execution — nodes, edges, state changes.
# Without this, debugging a multi-agent pipeline is guesswork.
# Your EU instance is at eu.smith.langchain.com
LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_ENDPOINT: str = os.getenv(
    "LANGCHAIN_ENDPOINT", "https://eu.smith.langchain.com"
)
LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "context-engineer")

# Set LangSmith env vars so LangGraph picks them up automatically
if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"]  = LANGCHAIN_TRACING_V2
    os.environ["LANGCHAIN_ENDPOINT"]    = LANGCHAIN_ENDPOINT
    os.environ["LANGCHAIN_API_KEY"]     = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"]     = LANGCHAIN_PROJECT

# Offload Store
# SQLite database path. We resolve to an ABSOLUTE path so it works
# regardless of where Streamlit sets the working directory at runtime.
# On Windows, relative paths can resolve to the wrong location depending
# on how the process was launched.
import pathlib
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DEFAULT_DB   = str(_PROJECT_ROOT / "data" / "offload_store.db")
OFFLOAD_DB_PATH: str = os.getenv("OFFLOAD_DB_PATH", _DEFAULT_DB)

# Token Counting 
# tiktoken encoding that gives the closest approximation to Claude's tokeniser.
# Claude uses its own internal tokeniser, but cl100k_base is within ~5%.
# We add a 10% safety buffer on top to be conservative.
TIKTOKEN_ENCODING: str = "cl100k_base"
TOKEN_SAFETY_BUFFER: float = 1.10   # Add 10% to all counts
CHARS_PER_TOKEN_FALLBACK: int = 4   # Fallback if tiktoken unavailable

# LLM Classification
# When True, classify_layer_llm() (Haiku) is used instead of heuristics.
# More accurate but adds ~1 API call per user message. Set False to use
# heuristics only (faster, free, less accurate).
USE_LLM_CLASSIFICATION: bool = os.getenv("USE_LLM_CLASSIFICATION", "true").lower() == "true"

# Scratchpad summarization threshold — compress when entries exceed this
SCRATCHPAD_SUMMARIZE_AFTER: int = int(os.getenv("SCRATCHPAD_SUMMARIZE_AFTER", "20"))