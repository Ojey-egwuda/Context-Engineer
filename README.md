# 🧠 Context Engineer

### Advanced Context Engineering for Production AI Agents

🚀 <a href="https://context-engineer.streamlit.app/" target="_blank" rel="noopener noreferrer">Live App</a>

👉 https://context-engineer.streamlit.app/

> **Production-grade context management for LLM agents** — A LangGraph system that actively monitors, classifies, offloads, and retrieves context so your AI agent never loses track of what matters — even across multiple sessions.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_AI-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io)
[![LangSmith](https://img.shields.io/badge/LangSmith-Observability-purple.svg)](https://smith.langchain.com)
[![SQLite](https://img.shields.io/badge/SQLite-Persistence-lightblue.svg)](https://sqlite.org)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Search-teal.svg)](https://www.trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [The Problem](#-the-problem)
- [The 7 Techniques](#-the-7-techniques)
- [What's New](#-whats-new)
- [Cross-Session Persistent Memory](#-cross-session-persistent-memory)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Observability](#-observability)
- [Red-Team Testing](#-red-team-testing)
- [Key Design Decisions](#-key-design-decisions)
- [Author](#-author)

---

## 🎯 The Problem

Most AI agents break **silently** when their context window fills up. The model starts hallucinating earlier facts, forgetting key constraints, and degrading in quality — with no warning to the user. By the time you notice, the conversation is already corrupted.

The standard fix (truncate old messages) is destructive. You throw away context the agent might still need.

**This project solves it differently:**

The agent monitors its own context health, classifies every message by importance, offloads low-priority content to a SQLite + ChromaDB store before the window fills, and retrieves relevant history on demand using semantic vector search. The result is **effective unbounded memory with a fixed active context window**, with real-time streaming and full observability via LangSmith.

---

## ⚙️ The 7 Techniques

| #   | Technique                    | What It Does                                                                                          | Where in Code                     |
| --- | ---------------------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------- |
| T1  | 🔴 **Pre-Rot Threshold**     | Triggers cleanup at 70% capacity — proactively, before quality degrades                               | `nodes.py → monitor_tokens_node`  |
| T2  | 🟡 **Layered Action Space**  | Classifies every message as CRITICAL / WORKING / BACKGROUND — CRITICAL messages are never evicted     | `layer_manager.py`                |
| T3  | 🟠 **Context Offloading**    | Moves BACKGROUND messages out of active state into SQLite + ChromaDB long-term storage                | `nodes.py → offload_context_node` |
| T4  | 🔵 **Agent-as-Tool**         | Sub-agents (`retrieve_from_memory`, `summarise_context`) are tools the reasoning node calls on demand | `sub_agents.py`                   |
| T5  | 🟢 **Token Budgeting**       | Fixed token allowances tracked in state — the agent always knows exactly how full its window is       | `token_counter.py`                |
| T6  | 🟣 **Scratchpad Management** | Reasoning trace maintained separately from conversation — summarised by Haiku when it grows too long  | `nodes.py → reason_node`          |
| T7  | ⚪ **RAG Compression**       | Retrieved chunks are compressed to their most information-dense form before re-injection              | `context_manager.py`              |

---

## 🆕 What's New

### Streaming Responses
Claude's output streams token-by-token into the Streamlit UI via `stream_chat()` — a generator that runs all preprocessing nodes (classify → monitor → offload → retrieve) synchronously, then streams the API response. No spinners, no waiting for the full response.

### Semantic Vector Search (ChromaDB)
Retrieval upgraded from keyword overlap scoring to **cosine similarity search** via ChromaDB. Offloaded messages are embedded using a local ONNX model — no external API calls. Queries like "programming language" now match messages that mention "Python" or "Rust" even without exact keyword overlap. A cosine distance threshold (0.85) filters out genuinely unrelated documents.

### LLM-Based Layer Classification
A second classification mode using **Claude Haiku** replaces heuristic keyword matching when enabled. Haiku understands implication — "I go by Alex" is correctly marked CRITICAL even without matching any keyword. Cost: ~$0.00008 per message. Falls back to heuristics automatically on any failure.

### Cost Tracking
Every turn tracks `session_input_tokens`, `session_output_tokens`, and `total_cost_usd` in agent state. The Streamlit dashboard shows live session cost in the metrics row.

### Async Graph
`build_async_graph()` wires an async version of the graph using `AsyncAnthropic`. Use `async_chat()` for concurrent serving in FastAPI or aiohttp — multiple sessions can be awaited simultaneously.

### Scratchpad Summarisation
When the reasoning trace exceeds 20 entries, Haiku automatically compresses older entries into a summary while preserving the 3 most recent verbatim. Keeps the scratchpad useful without letting it bloat.

### Evaluation Suite
15 end-to-end evaluation tests covering the full pipeline — offload triggering, retrieval accuracy, layer classification, cost accumulation, session isolation, memory poisoning, scratchpad compression, and budget enforcement. Fully mocked — no API key required, deterministic, fast.

---

## 💾 Cross-Session Persistent Memory

Beyond within-session context management, the system maintains a **true long-term memory store** across sessions:

| Feature                 | How It Works                                                                                                                    |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| 🔐 **Critical Memory**  | CRITICAL messages (identity, key facts, explicit instructions) written to SQLite `critical_memory` table and restored on return |
| 📦 **Session Flush**    | All conversation messages flushed to the offload store at session close — not just when memory pressure forces it               |
| 👤 **Returning Users**  | Never have to re-introduce themselves — prior history is semantically searchable via `retrieve_from_memory` from message one    |
| 🕐 **Session Tracking** | Every session registered with timestamp and message count — full audit trail of user activity                                   |
| 🛡️ **Poisoning Guard** | Second CRITICAL user message is automatically downgraded to WORKING — prevents identity override attacks                        |

This is the architectural pattern used in production customer support agents, legal document assistants, and long-running research pipelines.

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER MESSAGE                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              🏷️  CLASSIFY INPUT NODE  (T2)                      │
│  • Labels message: CRITICAL / WORKING / BACKGROUND              │
│  • Uses Claude Haiku (LLM mode) or keyword heuristics           │
│  • CRITICAL = identity facts, key constraints, system prompt    │
│  • Detects and blocks memory poisoning attacks                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              📊  MONITOR TOKENS NODE  (T1, T5)                  │
│  • Counts tokens across all active messages                     │
│  • Checks Pre-Rot Threshold (default 70%)                       │
│  • Sets needs_offload flag — deterministically, not via LLM     │
│  • Records token snapshot to scratchpad (T6)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
          needs_offload=True           needs_offload=False
              │                               │
              ▼                               │
┌─────────────────────────┐                  │
│  📤 OFFLOAD CONTEXT  (T3)│                  │
│  • Moves BACKGROUND msgs │                  │
│    from state to SQLite  │                  │
│  • Indexes in ChromaDB   │                  │
│  • Preserves all CRITICAL│                  │
│  • Updates token count   │                  │
└─────────────────────────┘                  │
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              🔍  RETRIEVE CONTEXT NODE  (T7)                    │
│  • ChromaDB cosine similarity search over offloaded messages    │
│  • Falls back to keyword overlap if vector store unavailable    │
│  • Compresses retrieved chunks via Haiku RAG compression        │
│  • Injects compressed context back into reasoning window        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              🧠  REASON NODE  (T4, T6)                          │
│  • Streams Claude response token by token                       │
│  • Appends timestamped entry to scratchpad on every turn        │
│  • Summarises scratchpad via Haiku when it exceeds 20 entries   │
│  • Tracks input/output tokens + cost per turn                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              💬  RESPOND NODE                                    │
│  • Writes final response to message history                     │
│  • Persists CRITICAL messages to cross-session store            │
│  • Updates session activity counter                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FINAL RESPONSE                            │
│     Streamed live · Layer badge · Scratchpad trace · Cost       │
└─────────────────────────────────────────────────────────────────┘
```

### Sub-Agents (T4 — Agent-as-Tool)

The reasoning node has access to two tools it calls autonomously:

| Tool                      | Purpose                                                                                                         |
| ------------------------- | --------------------------------------------------------------------------------------------------------------- |
| 🔎 `retrieve_from_memory` | Semantic vector search over offloaded messages — pulls relevant history back into context when needed           |
| 📝 `summarise_context`    | Produces a compressed summary of a set of messages when the full text is too large to re-inject                 |

---

## 🛠 Tech Stack

| Component              | Technology                                        |
| ---------------------- | ------------------------------------------------- |
| **LLM**                | Anthropic Claude Sonnet (responses) + Haiku (classification, compression) |
| **Agent Framework**    | LangGraph 0.2                                     |
| **Persistence**        | SQLite (built-in Python)                          |
| **Vector Search**      | ChromaDB (local ONNX embeddings, cosine similarity) |
| **Token Counting**     | tiktoken `cl100k_base` + 10% safety buffer        |
| **Observability**      | LangSmith (EU instance)                           |
| **Frontend**           | Streamlit (streaming via `st.write_stream`)       |
| **Testing**            | pytest — 15 evaluation tests, no API key required |
| **Language**           | Python 3.11+                                      |

---

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- An [Anthropic API key](https://console.anthropic.com)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Ojey-egwuda/Context-Engineer
cd Context-Engineer
```

### Step 2: Create Virtual Environment

```bash
# Windows (PowerShell)
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the project root:

```env
# Required
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Demo mode — set low to trigger offloading quickly
TOKEN_BUDGET=3000

# Optional — use Claude Haiku for smarter layer classification
USE_LLM_CLASSIFICATION=true

# Optional — observability (recommended)
LANGSMITH_API_KEY=ls__your_key_here
LANGCHAIN_PROJECT=context-engineer
LANGCHAIN_ENDPOINT=https://eu.api.smith.langchain.com
LANGCHAIN_TRACING_V2=true
```

**Get API Keys:**

- Anthropic API Key: [console.anthropic.com](https://console.anthropic.com)
- LangSmith Key: [smith.langchain.com](https://smith.langchain.com) → Settings → API Keys

---

## 🚀 Usage

### Streamlit App (Recommended)

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`. Enter your name to start a session. Claude's response streams token-by-token into the chat. The token dashboard updates in real time — every message shows its layer classification, the scratchpad records the full reasoning trace, and the Session Cost metric tracks spend per turn.

**Demo tip:** Set `TOKEN_BUDGET=3000` in `.env` to trigger offloading within a few messages and watch T1, T2, T3, and T7 all activate live.

### Running Tests

```bash
# Full evaluation suite — no API key required, fully deterministic
pytest tests/test_evaluation.py -v

# All tests
pytest tests/ -v
```

### Running the Red-Team Evaluator

```bash
# Requires LANGSMITH_API_KEY in .env
python evaluators/red_team_evaluator.py
```

Results appear in LangSmith under `Datasets → red-team-cases → Experiments`.

---

## 📁 Project Structure

```
Context-Engineer/
├── app.py                          # Streamlit UI — streaming chat, token dashboard, cost metrics
├── requirements.txt
├── .env.example
├── evaluators/
│   └── red_team_evaluator.py       # LangSmith automated evaluation suite
├── src/
│   ├── config.py                   # All tunable values — budget, thresholds, model, pricing
│   ├── context/
│   │   ├── token_counter.py        # tiktoken counting with 10% safety buffer
│   │   ├── layer_manager.py        # CRITICAL / WORKING / BACKGROUND classification (heuristic + Haiku)
│   │   ├── offload_store.py        # SQLite long-term memory + cross-session persistence
│   │   ├── vector_store.py         # ChromaDB semantic search index for offloaded messages
│   │   └── context_manager.py      # Context window assembly + Haiku RAG compression
│   ├── graph/
│   │   ├── state.py                # AgentState TypedDict schema (inc. cost tracking fields)
│   │   ├── nodes.py                # All graph nodes + async_reason_node
│   │   └── graph.py                # Graph wiring + async graph build
│   └── agents/
│       ├── main_agent.py           # Public interface: create_session, chat, stream_chat, async_chat
│       └── sub_agents.py           # retrieve_from_memory + summarise_context tools
└── tests/
    └── test_evaluation.py          # 15 end-to-end eval tests — fully mocked, no API key needed
```

---

## 📡 Observability

Every graph execution is traced in **LangSmith** (EU instance). Each run captures:

| Signal                      | What It Tells You                                 |
| --------------------------- | ------------------------------------------------- |
| 🔢 Token usage per turn     | How fast the window is filling                    |
| 📤 Offload events           | When and what was evicted from active context     |
| 🔧 Tool calls               | Which sub-agents were invoked and with what query |
| 🏷️ Layer classifications    | How each message was categorised                  |
| 🔍 Retrieved context length | Whether prior memory was successfully recalled    |
| 💰 Cost per turn            | Input/output tokens and USD cost accumulated      |

---

## 🛡️ Red-Team Testing

The system was tested against **20 adversarial attack categories** across two rounds. All 20 passed.

Additional protections added:
- **Memory poisoning guard** — a second CRITICAL user message is downgraded to WORKING, preventing identity override attacks
- **Cosine distance threshold** — vector search filters documents with similarity < 0.85 to prevent false positive retrievals

---

## 🔑 Key Design Decisions

**ChromaDB over keyword search.** The original retrieval used token overlap scoring — fast but brittle. "What did I say about Python?" would miss a message that only mentions "programming". ChromaDB embeds every offloaded message into a high-dimensional space using a local ONNX model (no external API calls) and returns semantically similar results. SQLite remains the source of truth — ChromaDB is purely a search index.

**Tool use removed from streaming path.** When Claude responds to a CRITICAL message (e.g. a long self-introduction), it may start with a tool call block rather than text. The streaming `text_stream` iterator yields nothing in that case, producing a blank response. Since `retrieve_context_node` already runs retrieval *before* the API call, Claude-initiated tool calls during streaming are redundant — removing them ensures Claude always responds with direct text that streams correctly.

**SQLite over Redis.** SQLite is built into Python and requires zero infrastructure. The `offload_store.py` interface is designed so you can swap in Redis or a managed vector database with a single file change — no other code changes needed.

**Two-tier classification.** Heuristic keyword matching is fast, transparent, and always available. Claude Haiku adds semantic understanding for messages that don't match any keyword but are clearly important — like "I go by Alex". Both modes use the same `ContextLayer` enum; switching is a single config flag.

**10% token safety buffer.** Claude uses its own internal tokeniser. tiktoken's `cl100k_base` is within ~5% on most content, but the buffer ensures we never undercount and hit hard API limits unexpectedly.

**Deterministic offload decision.** The graph decides when to offload — not the LLM. This is critical for production reliability. A model cannot talk itself out of or into an offload cycle.

---

## 👨‍💻 Author

**Ojonugwa Egwuda** — AI Engineer, Oxford UK

- LinkedIn: [linkedin.com/in/egwudaojonugwa](https://www.linkedin.com/in/egwudaojonugwa/)
- GitHub: [github.com/Ojey-egwuda](https://github.com/Ojey-egwuda)
- Portfolio: [ojey-egwuda.github.io](https://ojey-egwuda.github.io)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built for engineers who care about production reliability**

🧠 🔧 🚀

</div>
