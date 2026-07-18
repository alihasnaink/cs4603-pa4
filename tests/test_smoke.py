"""Offline smoke test for the Document Analyst graph (Task 1.7 / Bonus A).

This test builds the graph with a mocked LLM, retriever, and tools — no
Databricks calls, no network, no credentials required.  It proves:

  1. The graph module imports cleanly.
  2. The graph compiles without errors.
  3. A combined retrieval + calculation query returns a non-empty `messages`
     list with the synthesized answer in the last message.
  4. The planner produced a plan (at least one step).
  5. Both specialist nodes ran (step_results contains ≥2 entries for the
     combined query).

Run:
    uv run pytest -q
    # or
    python -m pytest tests/test_smoke.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage


# ─── Fake dependencies ────────────────────────────────────────────────────────


class _FakeLLM:
    """Minimal LangChain-compatible fake LLM.

    Returns canned responses based on which node is calling it:
      - Planner call → JSON plan list.
      - Supervisor call → routing decision.
      - RAG extract call → a fact string.
      - MCP / synthesizer call → a plain answer string.
    """

    def __init__(self):
        self._call_count = 0

    # Implement bind_tools so make_mcp_node doesn't crash.
    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self._call_count += 1
        # Inspect the last human message to pick a canned reply.
        last_content = ""
        for m in reversed(messages):
            if hasattr(m, "content"):
                last_content = m.content
                break

        # Planner → JSON plan (two steps: one retrieval, one calculation).
        if "User question:" in last_content and "decompose" not in last_content.lower():
            content = (
                '["Find net revenue in the annual report", '
                '"Calculate 8% compound growth over 3 years on that figure"]'
            )
        # Supervisor → classify a step.
        elif "Step to classify:" in last_content:
            step = last_content.replace("Step to classify:", "").strip().lower()
            content = "rag_agent" if "find" in step or "look" in step else "mcp_tools"
        # RAG extract → return a fact.
        elif "STEP:" in last_content and "CONTEXT:" in last_content:
            content = "Net revenue in FY2023 was ¥16.91 trillion [source: annual_report.pdf, p.4]."
        # Synthesizer → return a combined answer.
        elif "Step results:" in last_content or "step results" in last_content.lower():
            content = (
                "Meridian's net revenue in FY2023 was ¥16.91 trillion. "
                "After 3 years at 8% CAGR the projected revenue is ¥21.30 trillion."
            )
        # MCP node (tool result already in context or fallback).
        elif "Prior step results:" in last_content:
            content = "16.91 at 8% CAGR for 3 years = 21.30"
        else:
            content = "OK"

        return AIMessage(content=content)


class _FakeRetriever:
    """Returns a single fake document regardless of query."""

    def invoke(self, query: str):  # noqa: ARG002
        return [
            Document(
                page_content="Net revenue: ¥16.91 trillion",
                metadata={"source": "annual_report.pdf", "page": 4},
            )
        ]


class _FakeTool:
    """Minimal fake tool that echoes its args."""

    name = "calculate"
    description = "Evaluate a math expression."

    def invoke(self, args):
        expr = args.get("expression", "0")
        return f"{expr} = 21.30 (fake)"


# ─── Tests ────────────────────────────────────────────────────────────────────


def test_graph_module_imports():
    """Minimal collection guard: the graph module must import cleanly."""
    from agent.graph import build_graph  # noqa: F401


def test_graph_compiles_with_fakes():
    """build_graph must return a compiled graph with fake dependencies."""
    from agent.graph import build_graph

    graph = build_graph(
        llm=_FakeLLM(),
        retriever=_FakeRetriever(),
        tools=[_FakeTool()],
    )
    assert graph is not None


def test_graph_runs_combined_query():
    """Full end-to-end run with a combined retrieval + calculation query.

    Asserts:
      - messages[-1] is a non-empty AIMessage (serving contract).
      - A plan was produced.
      - step_results has at least 2 entries (both specialists ran).
    """
    from agent.graph import build_graph

    graph = build_graph(
        llm=_FakeLLM(),
        retriever=_FakeRetriever(),
        tools=[_FakeTool()],
    )

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "What was Meridian's net revenue in FY2023, "
                        "and what would it be after 3 years of 8% compound annual growth?"
                    )
                )
            ]
        }
    )

    # ── Serving contract: last message must be a non-empty AI answer.
    assert "messages" in result, "State must contain 'messages'"
    assert len(result["messages"]) >= 1, "messages must not be empty"
    last_msg = result["messages"][-1]
    assert isinstance(last_msg, AIMessage), f"Last message must be AIMessage, got {type(last_msg)}"
    assert last_msg.content.strip(), "Final AIMessage must have non-empty content"

    # ── Plan was produced.
    assert "plan" in result and len(result["plan"]) >= 1, "A plan must have been produced"

    # ── Both specialists ran (step_results ≥ 2 for a combined query).
    assert "step_results" in result and len(result["step_results"]) >= 2, (
        "Both RAG and MCP specialists must have contributed a step result"
    )
