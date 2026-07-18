"""Full Document Analyst graph (Tasks 1.5 + 1.7).

Public API
----------
load_mcp_tools(server_path)
    Connect the GIVEN MCP server over stdio and return its LangChain tools.
    Uses `langchain-mcp-adapters` (`MultiServerMCPClient`).

make_mcp_node(tools, llm)
    Return a LangGraph node that handles one calculation step: the LLM
    selects and calls exactly one MCP tool, and the result is appended to
    `step_results`.

build_graph(llm, retriever, tools)
    Assemble and compile the full Document Analyst graph with dependency
    injection so the graph can be unit-tested offline with fakes.

MCP notes (Task 1.5 caveat)
---------------------------
The MCP server runs as a *stdio subprocess* bundled inside the serving
container — it launches when `load_mcp_tools()` is called (at graph-build
time) and lives for the lifetime of the serving process. Tools are loaded
once and reused for all requests, which keeps tool invocation synchronous
and avoids repeated subprocess startup overhead.

Reference: wk5_langgraph/9.subgraphs.ipynb, wk4_agents_mcp/
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from agent.planner import make_planner
from agent.prompts import MCP_STEP_PROMPT
from agent.rag_agent import make_rag_agent
from agent.state import AnalystState
from agent.supervisor import MCP, RAG, SYNTH, make_supervisor, route_from_supervisor
from agent.synthesizer import make_synthesizer

logger = logging.getLogger(__name__)


# ─── MCP integration (Task 1.5) ──────────────────────────────────────────────


def load_mcp_tools(server_path: str | None = None) -> list[Any]:
    """Connect to the GIVEN MCP server over stdio and return its LangChain tools.

    The server is launched as a subprocess; the async MCP client is run
    synchronously using a fresh event loop so the graph builder can call this
    from ordinary (non-async) code.

    Parameters
    ----------
    server_path:
        Absolute path to ``tools/mcp_server.py``.  Defaults to the sibling
        ``tools/mcp_server.py`` relative to this file's package root.

    Returns
    -------
    A list of LangChain ``BaseTool`` objects derived from the MCP tool schemas.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    if server_path is None:
        # Resolve relative to the project root (parent of the agent/ package).
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        server_path = os.path.join(root, "tools", "mcp_server.py")

    python_exe = sys.executable  # use same interpreter as the current process

    async def _get_tools():
        client = MultiServerMCPClient(
            {
                "analyst": {
                    "command": python_exe,
                    "args": [server_path],
                    "transport": "stdio",
                }
            }
        )
        return await client.get_tools()

    # Jupyter/Databricks notebooks already have a running event loop, so
    # asyncio.run() raises "cannot be called from a running event loop".
    # Running the coroutine in a fresh daemon thread with its own loop works
    # in both notebook and plain-script contexts without nest_asyncio.
    tools = _run_async(_get_tools())
    logger.info("Loaded %d MCP tools from %s", len(tools), server_path)
    return tools


def _run_async(coro):
    """Run *coro* in a fresh event loop on a daemon thread and return its result.

    This avoids the ``RuntimeError: asyncio.run() cannot be called from a
    running event loop`` that occurs inside Jupyter / Databricks notebooks,
    which already own an event loop on the main thread.
    """
    result: dict = {}

    def _target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result["value"] = loop.run_until_complete(coro)
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc
        finally:
            loop.close()

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join()

    if "error" in result:
        raise result["error"]
    return result["value"]


# ─── MCP tool node (Task 1.5) ────────────────────────────────────────────────


def make_mcp_node(tools: list[Any], llm):
    """Return a LangGraph node that executes one calculation step via MCP tools.

    The node binds the MCP tools to the LLM with `bind_tools`, invokes the
    LLM to select the right tool, then calls the tool and appends the result
    to `step_results`.

    Parameters
    ----------
    tools:
        LangChain tools returned by `load_mcp_tools()` (real or fake).
    llm:
        Any LangChain chat model.
    """
    # Bind tools to the LLM so it can emit tool-call messages.
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    tool_map = {t.name: t for t in tools}

    def mcp_tools(state: AnalystState) -> dict:
        plan = state.get("plan", [])
        idx = state.get("current_step_index", 0)
        step = plan[idx] if idx < len(plan) else "perform a calculation"

        # Collect prior results for context so the LLM can substitute values.
        prior = state.get("step_results", [])
        prior_context = (
            "\n".join(f"Step {i + 1} result: {r}" for i, r in enumerate(prior))
            if prior
            else "(no prior results)"
        )

        logger.debug("MCP node: handling step %d: %r", idx, step)

        messages = [
            SystemMessage(content=MCP_STEP_PROMPT),
            HumanMessage(
                content=(
                    f"Prior step results:\n{prior_context}\n\n"
                    f"Current step to execute: {step}"
                )
            ),
        ]

        # First LLM call → should emit a tool-call message.
        ai_msg = llm_with_tools.invoke(messages)

        result_text: str

        if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
            tool_call = ai_msg.tool_calls[0]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call.get("id", "call_0")

            if tool_name in tool_map:
                try:
                    # MCP tools from langchain-mcp-adapters are async-only
                    # (they implement _arun but not _run).  Use ainvoke()
                    # via _run_async() so we can call them from this sync node.
                    raw = _run_async(tool_map[tool_name].ainvoke(tool_args))
                    # ainvoke returns a list of content blocks: [{'type':'text','text':'...'}]
                    if isinstance(raw, list):
                        result_text = " ".join(
                            block.get("text", str(block))
                            for block in raw
                            if isinstance(block, dict)
                        ) or str(raw)
                    else:
                        result_text = str(raw)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Tool %r error: %s", tool_name, exc)
                    result_text = f"Tool error: {exc}"
            else:
                result_text = f"Unknown tool requested: {tool_name}"

            # Run a follow-up LLM call to get the formatted answer (optional —
            # for this assignment returning the raw tool result is sufficient).
            _ = llm_with_tools.invoke(
                messages
                + [
                    ai_msg,
                    ToolMessage(content=result_text, tool_call_id=tool_id),
                ]
            )
        else:
            # LLM answered without calling a tool (e.g. in fake-LLM tests).
            result_text = ai_msg.content.strip()

        logger.debug("MCP node result: %r", result_text)

        current_results = list(state.get("step_results", []))
        current_results.append(result_text)

        return {
            "step_results": current_results,
            "current_step_index": idx + 1,
        }

    return mcp_tools


# ─── Graph assembly (Task 1.7) ────────────────────────────────────────────────


def build_graph(llm=None, retriever=None, tools=None):
    """Assemble and compile the full Document Analyst graph.

    All dependencies are injected so the graph can be tested offline with
    fakes (no Databricks calls) or wired to real Databricks services in
    production.

    Parameters
    ----------
    llm:
        LangChain chat model.  If ``None``, the default from
        ``config.get_chat_llm()`` is used.
    retriever:
        LangChain retriever.  If ``None``, ``rag.store.get_retriever()`` is
        used — which requires VECTOR_SEARCH_* env vars.
    tools:
        List of LangChain tools for the MCP node.  If ``None``,
        ``load_mcp_tools()`` is called — which requires the MCP server.

    Returns
    -------
    A compiled ``CompiledGraph`` ready to ``.invoke()`` or ``.stream()``.
    """
    # ── Resolve defaults (lazy so tests with explicit fakes never hit network)
    if llm is None:
        from config import get_chat_llm

        llm = get_chat_llm()

    if retriever is None:
        from rag.store import get_retriever

        retriever = get_retriever()

    if tools is None:
        tools = load_mcp_tools()

    # ── Build nodes
    planner = make_planner(llm)
    supervisor = make_supervisor(llm)
    rag_agent = make_rag_agent(retriever, llm)
    mcp_node = make_mcp_node(tools, llm)
    synthesizer = make_synthesizer(llm)

    # ── Wire the graph
    builder = StateGraph(AnalystState)

    builder.add_node("planner", planner)
    builder.add_node("supervisor", supervisor)
    builder.add_node(RAG, rag_agent)
    builder.add_node(MCP, mcp_node)
    builder.add_node(SYNTH, synthesizer)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {RAG: RAG, MCP: MCP, SYNTH: SYNTH},
    )
    builder.add_edge(RAG, "supervisor")
    builder.add_edge(MCP, "supervisor")
    builder.add_edge(SYNTH, END)

    graph = builder.compile()
    logger.info("Document Analyst graph compiled successfully.")
    return graph
