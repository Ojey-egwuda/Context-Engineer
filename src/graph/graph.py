"""
graph.py — Wire all nodes into the LangGraph execution graph.

WHY THE GRAPH MATTERS
---------------------
The graph structure IS the control flow. The LLM cannot decide to skip
a step. If the graph says validation runs, validation runs — always.

This is what "deterministic composition" means:
  - Sequence is defined by edges, not by LLM output
  - Conditional branches are explicit (should_offload function)
  - You can trace any execution in LangSmith and replay it exactly

GRAPH FLOW
----------
  START
    → classify_input    (label the new message with a context layer)
    → monitor_tokens    (check if Pre-Rot Threshold is hit)
    → [conditional]
        if needs_offload → offload_context → retrieve_context
        else             →                  retrieve_context
    → reason            (Claude API call with optimised context)
    → respond           (add response to message history)
    → END

The conditional edge is the Pre-Rot Threshold in action:
  - monitor_tokens sets state["needs_offload"] = True/False
  - should_offload() reads that flag and returns the next node name
  - LangGraph routes to the correct node
"""

from langgraph.graph import StateGraph, END, START

from src.graph.state import AgentState
from src.graph.nodes import (
    classify_input_node,
    monitor_tokens_node,
    offload_context_node,
    retrieve_context_node,
    reason_node,
    respond_node,
)


def should_offload(state: AgentState) -> str:
    """
    Conditional edge function: route after monitor_tokens.

    Returns the NAME of the next node to execute.
    LangGraph uses the return value to select the edge.

    This is where the Pre-Rot Threshold decision happens.
    The LLM has no say in this routing — it's a pure Python function.
    """
    if state.get("needs_offload", False):
        return "offload_context"
    return "retrieve_context"


def build_graph():
    """
    Construct and compile the context engineering graph.

    Returns a compiled LangGraph that can be invoked with:
        result = agent_graph.invoke(state)
    """
    graph = StateGraph(AgentState)

    # Register all nodes
    graph.add_node("classify_input",   classify_input_node)
    graph.add_node("monitor_tokens",   monitor_tokens_node)
    graph.add_node("offload_context",  offload_context_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("reason",           reason_node)
    graph.add_node("respond",          respond_node)

    # Wire the edges

    # Fixed sequence: start → classify → monitor
    graph.add_edge(START,             "classify_input")
    graph.add_edge("classify_input",  "monitor_tokens")

    # Conditional branch: offload (if needed) or skip straight to retrieve
    graph.add_conditional_edges(
        "monitor_tokens",
        should_offload,
        {
            "offload_context":  "offload_context",
            "retrieve_context": "retrieve_context",
        }
    )

    # After offloading, always retrieve (there's now something to retrieve)
    graph.add_edge("offload_context",  "retrieve_context")

    # Fixed sequence: retrieve → reason → respond → done
    graph.add_edge("retrieve_context", "reason")
    graph.add_edge("reason",           "respond")
    graph.add_edge("respond",          END)

    return graph.compile()


# Build once at module level — reused across all sessions
agent_graph = build_graph()


def build_async_graph():
    """
    Async variant using async_reason_node.

    Use for production deployments where multiple users are served
    concurrently (FastAPI, aiohttp). The awaitable Claude API call
    lets other requests proceed while waiting for the LLM response.

    Usage:
        result = await async_agent_graph.ainvoke(state)
    """
    from src.graph.nodes import async_reason_node

    graph = StateGraph(AgentState)
    graph.add_node("classify_input",   classify_input_node)
    graph.add_node("monitor_tokens",   monitor_tokens_node)
    graph.add_node("offload_context",  offload_context_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("reason",           async_reason_node)
    graph.add_node("respond",          respond_node)

    graph.add_edge(START,            "classify_input")
    graph.add_edge("classify_input", "monitor_tokens")
    graph.add_conditional_edges(
        "monitor_tokens", should_offload,
        {"offload_context": "offload_context", "retrieve_context": "retrieve_context"},
    )
    graph.add_edge("offload_context",  "retrieve_context")
    graph.add_edge("retrieve_context", "reason")
    graph.add_edge("reason",           "respond")
    graph.add_edge("respond",          END)
    return graph.compile()


async_agent_graph = build_async_graph()
