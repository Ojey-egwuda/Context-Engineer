"""
vector_store.py — Semantic search index for offloaded messages.

WHY THIS EXISTS
---------------
The original retrieval used keyword overlap scoring — fast and transparent,
but it misses semantic matches. "What did I say about Python?" won't match
a message that only mentions "programming" or "snake language".

Vector search embeds every offloaded message into a high-dimensional space
where semantically similar text lands near each other. Retrieval becomes
"find the nearest neighbours to this query" — robust to paraphrase,
synonyms, and topic drift.

HOW IT WORKS
------------
ChromaDB is a file-based vector database — zero infrastructure, just a
directory on disk alongside the SQLite store. It handles:
  - Embedding text using a local ONNX model (no external API calls)
  - HNSW approximate nearest-neighbour search (fast at scale)
  - Metadata filtering (restrict search to a single session_id)

SQLite remains the source of truth for message content and metadata.
ChromaDB is purely a search index — the actual message data is always
fetched from SQLite after a vector lookup returns IDs.

FALLBACK
--------
If ChromaDB is unavailable (import error, disk issue), all functions
are no-ops or return empty results. The keyword fallback in
offload_store.retrieve_relevant() handles the rest.
"""

import pathlib
from src.config import OFFLOAD_DB_PATH

# Store ChromaDB index alongside the SQLite database
_CHROMA_PATH = str(pathlib.Path(OFFLOAD_DB_PATH).parent / "chroma")

_collection = None


def _get_collection():
    """Lazy-initialise the ChromaDB collection (once per process)."""
    global _collection
    if _collection is not None:
        return _collection

    try:
        import chromadb
        client = chromadb.PersistentClient(path=_CHROMA_PATH)
        _collection = client.get_or_create_collection(
            name="offloaded_messages",
            metadata={"hnsw:space": "cosine"},
        )
        return _collection
    except Exception:
        return None


def add_to_index(message_id: str, session_id: str, content: str) -> None:
    """
    Add an offloaded message to the semantic search index.

    Called alongside offload_store.offload_message() so the vector index
    stays in sync with the SQLite store.

    Uses upsert so re-offloading the same message_id is idempotent.
    """
    col = _get_collection()
    if col is None:
        return
    try:
        col.upsert(
            ids=[message_id],
            documents=[content],
            metadatas=[{"session_id": session_id}],
        )
    except Exception:
        pass  # Never let indexing break the main flow


def semantic_search(
    session_id: str,
    query: str,
    n_results: int = 5,
    distance_threshold: float = 0.85,
) -> list[str]:
    """
    Find message IDs semantically similar to the query.

    Returns a ranked list of message_ids (most relevant first),
    filtered to the given session. Fetch actual content from SQLite
    using these IDs.

    The distance_threshold filters out genuinely unrelated documents.
    ChromaDB uses cosine distance (0 = identical, 1 = orthogonal).
    Documents with distance >= threshold are considered too dissimilar
    to be useful and are dropped.

    Args:
        session_id:         Restrict search to this session's messages.
        query:              The current user message used as the search vector.
        n_results:          Maximum number of IDs to return.
        distance_threshold: Drop results with cosine distance >= this value.

    Returns:
        List of message_id strings, best match first. Empty on failure.
    """
    col = _get_collection()
    if col is None:
        return []

    try:
        total = col.count()
        if total == 0:
            return []

        results = col.query(
            query_texts=[query],
            n_results=min(n_results, total),
            where={"session_id": session_id},
            include=["ids", "distances"],
        )
        ids       = results.get("ids",       [[]])[0]
        distances = results.get("distances", [[]])[0]

        # Filter by similarity — drop anything too dissimilar to be useful
        filtered = [
            mid for mid, dist in zip(ids, distances)
            if dist < distance_threshold
        ]
        return filtered
    except Exception:
        return []


def clear_session_index(session_id: str) -> None:
    """Remove all indexed messages for a session. Called on session clear."""
    col = _get_collection()
    if col is None:
        return
    try:
        col.delete(where={"session_id": session_id})
    except Exception:
        pass
