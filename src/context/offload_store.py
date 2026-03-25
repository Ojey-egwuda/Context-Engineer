"""
offload_store.py — Long-term memory for the agent.

WHY THIS EXISTS
---------------
When a message exits the active context window, it doesn't disappear.
It gets written here. The agent can retrieve it later when relevant.

This is what makes the agent's EFFECTIVE memory unbounded, even though
the ACTIVE context window remains fixed. You give the agent a long-term
memory that the LLM itself doesn't natively have.

Think of it like the difference between RAM and a hard disk:
  - Active context window = RAM (fast, limited)
  - Offload store = hard disk (slower to access, but unlimited)

HOW IT WORKS
------------
SQLite database with a simple messages table. Each offloaded message
is stored with:
  - Its content and metadata
  - A keywords list (for retrieval scoring)
  - The session_id it belongs to

Retrieval uses keyword overlap scoring — simple but effective for a
portfolio demo. In production you'd replace keyword matching with
vector similarity search (ChromaDB or similar).

WHY SQLITE OVER REDIS
---------------------
Redis is faster and more production-appropriate, but requires a
running server. SQLite is built into Python, requires zero setup,
and is perfectly adequate for this project. The architecture is
identical — you could swap in Redis with a single file change.
"""

import sqlite3
import json
import time
from pathlib import Path
from src.config import OFFLOAD_DB_PATH
from src.context import vector_store


# Database Setup

def _get_conn() -> sqlite3.Connection:
    """Open a database connection, creating the file if needed."""
    Path(OFFLOAD_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(OFFLOAD_DB_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dicts
    return conn


def initialise_db() -> None:
    """
    Create the offload_messages table if it doesn't exist.

    Call this once at startup (already called in context_manager.py).
    Idempotent — safe to call multiple times.
    """
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS offloaded_messages (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id    TEXT    NOT NULL,
                session_id    TEXT    NOT NULL,
                role          TEXT    NOT NULL,
                content       TEXT    NOT NULL,
                layer         TEXT    NOT NULL,
                token_count   INTEGER NOT NULL,
                keywords      TEXT    NOT NULL,
                timestamp     REAL    NOT NULL,
                offloaded_at  REAL    DEFAULT (unixepoch('now'))
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_session ON offloaded_messages(session_id)"
        )
        # User sessions table — enables cross-session persistence ──────────
        # Maps user_id → session history so returning users get their context back
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       TEXT    NOT NULL,
                session_id    TEXT    NOT NULL,
                created_at    REAL    DEFAULT (unixepoch('now')),
                last_active   REAL    DEFAULT (unixepoch('now')),
                message_count INTEGER DEFAULT 0,
                summary       TEXT    DEFAULT ''
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_user ON user_sessions(user_id)"
        )
        # Critical memory table — persists CRITICAL messages across sessions
        # These are the facts that must survive session resets:
        # user identity, key constraints, important preferences
        conn.execute("""
            CREATE TABLE IF NOT EXISTS critical_memory (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       TEXT    NOT NULL,
                message_id    TEXT    NOT NULL,
                role          TEXT    NOT NULL,
                content       TEXT    NOT NULL,
                token_count   INTEGER NOT NULL,
                keywords      TEXT    NOT NULL,
                created_at    REAL    DEFAULT (unixepoch('now')),
                session_id    TEXT    NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_critical_user ON critical_memory(user_id)"
        )
        conn.commit()


# Keyword Extraction

# Words we ignore when building keyword sets
_STOPWORDS = {
    "this", "that", "with", "have", "from", "they", "been", "were",
    "will", "what", "when", "which", "their", "there", "about",
    "would", "could", "should", "than", "then", "into", "also",
    "just", "like", "some", "your", "more", "very", "over", "here",
    "only", "well", "back", "even", "such", "much", "make", "most",
}


def _extract_keywords(text: str) -> list[str]:
    """
    Extract meaningful keywords from text for retrieval scoring.

    Simple approach: lowercase words longer than 4 chars, excluding
    common stopwords. Returns up to 20 unique keywords.

    In production: replace with an embedding model for semantic matching.
    For this project: keyword overlap is fast and interpretable — you
    can SEE exactly why a retrieval matched.
    """
    words = text.lower().split()
    keywords = [
        w.strip(".,!?;:\"'()[]{}") for w in words
        if len(w) > 4 and w.lower().strip(".,!?;:") not in _STOPWORDS
    ]
    # Unique, preserve order, max 20
    seen = set()
    unique = []
    for kw in keywords:
        if kw and kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique[:20]


# Core Operations

def offload_message(
    message_id: str,
    session_id: str,
    role: str,
    content: str,
    layer: str,
    token_count: int,
    timestamp: float,
) -> None:
    """
    Write a message to the offload store.

    Called by offload_context_node when context exceeds Pre-Rot Threshold.
    The message is removed from active state and lives here until retrieved.
    """
    keywords = _extract_keywords(content)
    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO offloaded_messages
              (message_id, session_id, role, content, layer, token_count, keywords, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            message_id, session_id, role, content,
            layer, token_count, json.dumps(keywords), timestamp
        ))
        conn.commit()

    # Add to semantic search index (alongside SQLite for retrieval ranking)
    vector_store.add_to_index(message_id, session_id, content)


def retrieve_relevant(
    session_id: str,
    query: str,
    max_results: int = 5,
    max_tokens: int = 2000,
) -> list[dict]:
    """
    Find offloaded messages relevant to a query using semantic vector search.

    PRIMARY: Vector similarity search via ChromaDB — finds semantically
    related messages even when they share no keywords with the query.
    For example, "What did I say about Python?" will match a message
    about "programming" or "coding" because the concepts are close in
    embedding space.

    FALLBACK: Keyword overlap scoring if vector search returns nothing
    (e.g. first session before ChromaDB index is warm, or on import error).

    SQLite remains the source of truth — ChromaDB only provides the ranked
    message IDs; actual content is always fetched from SQLite.

    Args:
        session_id:  Only retrieve from this session's messages.
        query:       The user's current message (used as the search vector).
        max_results: Cap on number of messages returned.
        max_tokens:  Token budget for the retrieved set.

    Returns:
        List of message dicts, most relevant first.
    """
    # ── Primary: semantic search ────────────────────────────────────────────
    semantic_ids = vector_store.semantic_search(session_id, query, n_results=max_results)

    if semantic_ids:
        placeholders = ",".join("?" * len(semantic_ids))
        with _get_conn() as conn:
            rows = conn.execute(f"""
                SELECT message_id, role, content, layer, token_count, keywords, timestamp
                FROM offloaded_messages
                WHERE session_id = ? AND message_id IN ({placeholders})
            """, [session_id] + semantic_ids).fetchall()

        if rows:
            # Preserve semantic ranking order returned by ChromaDB
            id_rank = {mid: i for i, mid in enumerate(semantic_ids)}
            ordered = sorted(
                [dict(r) for r in rows],
                key=lambda r: id_rank.get(r["message_id"], 999),
            )
            results, used = [], 0
            for msg in ordered:
                if used + msg["token_count"] > max_tokens:
                    break
                results.append(msg)
                used += msg["token_count"]
            return results

    # ── Fallback: keyword overlap ───────────────────────────────────────────
    query_keywords = set(_extract_keywords(query))
    if not query_keywords:
        return []

    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT message_id, role, content, layer, token_count, keywords, timestamp
            FROM offloaded_messages
            WHERE session_id = ?
            ORDER BY timestamp DESC
        """, (session_id,)).fetchall()

    if not rows:
        return []

    scored = []
    for row in rows:
        stored_keywords = set(json.loads(row["keywords"]))
        overlap = len(query_keywords & stored_keywords)
        if overlap > 0:
            scored.append((overlap, dict(row)))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    results, used_tokens = [], 0
    for _, msg in scored[:max_results]:
        if used_tokens + msg["token_count"] > max_tokens:
            break
        results.append(msg)
        used_tokens += msg["token_count"]

    return results


def get_session_stats(session_id: str) -> dict:
    """
    Statistics about a session's offloaded messages.
    Used by the Streamlit dashboard.
    """
    with _get_conn() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*)        AS message_count,
                SUM(token_count) AS total_tokens,
                MIN(timestamp)  AS earliest,
                MAX(timestamp)  AS latest
            FROM offloaded_messages
            WHERE session_id = ?
        """, (session_id,)).fetchone()
    return dict(row) if row else {
        "message_count": 0, "total_tokens": 0,
        "earliest": None, "latest": None
    }


def clear_session(session_id: str) -> None:
    """Delete all offloaded messages for a session. Used when resetting."""
    with _get_conn() as conn:
        conn.execute(
            "DELETE FROM offloaded_messages WHERE session_id = ?",
            (session_id,)
        )
        conn.commit()
    vector_store.clear_session_index(session_id)


# Cross-Session Persistence

def register_session(user_id: str, session_id: str) -> None:
    """
    Register a new session for a user.
    Called when create_session() is called with a user_id.
    """
    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO user_sessions (user_id, session_id)
            VALUES (?, ?)
        """, (user_id, session_id))
        conn.commit()


def update_session_activity(session_id: str, message_count: int) -> None:
    """Update last_active and message_count for a session."""
    with _get_conn() as conn:
        conn.execute("""
            UPDATE user_sessions
            SET last_active = unixepoch('now'), message_count = ?
            WHERE session_id = ?
        """, (message_count, session_id))
        conn.commit()


def save_critical_memory(
    user_id: str,
    session_id: str,
    message_id: str,
    role: str,
    content: str,
    token_count: int,
) -> None:
    """
    Persist a CRITICAL message to long-term user memory.

    CRITICAL messages are the facts that must survive session resets:
    user identity, key constraints, explicit instructions like
    "please remember this".

    These are written here AND kept in active context — this is the
    cross-session backup copy.
    """
    keywords = _extract_keywords(content)
    with _get_conn() as conn:
        # Avoid duplicates — same content for same user
        existing = conn.execute("""
            SELECT id FROM critical_memory
            WHERE user_id = ? AND content = ?
        """, (user_id, content)).fetchone()

        if not existing:
            conn.execute("""
                INSERT INTO critical_memory
                  (user_id, session_id, message_id, role, content, token_count, keywords)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, session_id, message_id,
                role, content, token_count,
                json.dumps(keywords)
            ))
            conn.commit()


def load_critical_memory(user_id: str) -> list[dict]:
    """
    Load all CRITICAL messages for a returning user.

    Called at session start when user_id is provided.
    Returns messages sorted by creation time (oldest first) so
    the agent reconstructs context in the right order.

    This is what makes the system truly persistent — a returning
    user never has to re-introduce themselves.
    """
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT message_id, role, content, token_count, created_at
            FROM critical_memory
            WHERE user_id = ?
            ORDER BY created_at ASC
        """, (user_id,)).fetchall()
    return [dict(row) for row in rows] if rows else []


def load_prior_session_messages(user_id: str, max_sessions: int = 3) -> list[dict]:
    """
    Load offloaded messages from a user's most recent prior sessions.

    These go into the offload store under the new session_id so the
    retrieve_from_memory tool can find them during the new session.
    This gives the agent access to prior conversation context on demand
    without flooding the active context window.

    Args:
        user_id:      The user to load history for.
        max_sessions: How many prior sessions to pull from.
    """
    with _get_conn() as conn:
        # Get the most recent session_ids for this user
        prior_sessions = conn.execute("""
            SELECT session_id FROM user_sessions
            WHERE user_id = ?
            ORDER BY last_active DESC
            LIMIT ?
        """, (user_id, max_sessions)).fetchall()

        if not prior_sessions:
            return []

        session_ids = [row["session_id"] for row in prior_sessions]
        placeholders = ",".join("?" * len(session_ids))

        rows = conn.execute(f"""
            SELECT message_id, session_id, role, content, layer,
                   token_count, keywords, timestamp
            FROM offloaded_messages
            WHERE session_id IN ({placeholders})
            ORDER BY timestamp ASC
        """, session_ids).fetchall()

    return [dict(row) for row in rows] if rows else []


def get_user_session_count(user_id: str) -> int:
    """Return how many sessions this user has had. Used to detect returning users."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM user_sessions WHERE user_id = ?",
            (user_id,)
        ).fetchone()
    return row["cnt"] if row else 0


def get_user_last_active(user_id: str) -> float | None:
    """Return timestamp of user's most recent session."""
    with _get_conn() as conn:
        row = conn.execute("""
            SELECT MAX(last_active) AS last_active
            FROM user_sessions WHERE user_id = ?
        """, (user_id,)).fetchone()
    return row["last_active"] if row else None


def flush_session_messages(session_id: str, messages: list[dict]) -> int:
    """
    Write all active messages for a session to the offload store.

    Called at session end (logout / new session button) so that WORKING
    and BACKGROUND messages — which were never offloaded during the session
    because the token budget wasn't hit — are persisted to the DB.

    Without this, only CRITICAL messages survive between sessions. With it,
    the full conversation history is available to returning users via
    retrieve_from_memory.

    Args:
        session_id: The ending session's ID.
        messages:   The full list of active messages from AgentState.

    Returns:
        Number of messages written.
    """
    written = 0
    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        layer   = msg.get("layer", "working")

        # Skip system messages — they're rebuilt from the system_prompt arg
        if role == "system":
            continue
        # Skip empty content
        if not content or not content.strip():
            continue

        try:
            offload_message(
                message_id=  msg.get("message_id", str(__import__("uuid").uuid4())),
                session_id=  session_id,
                role=        role,
                content=     content,
                layer=       layer,
                token_count= msg.get("token_count", len(content) // 4),
                timestamp=   msg.get("timestamp", __import__("time").time()),
            )
            written += 1
        except Exception:
            pass  # Don't let a single message failure break the flush

    return written