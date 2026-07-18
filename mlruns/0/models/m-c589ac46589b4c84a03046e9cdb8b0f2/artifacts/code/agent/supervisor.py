"""Supervisor node + routing edge (Task 1.3).

`make_supervisor(llm)` returns a LangGraph node that inspects the current
plan step and decides which specialist agent should handle it:

  • "rag_agent"   — document retrieval steps
  • "mcp_tools"   — calculation / numerical steps
  • "synthesizer" — when all plan steps have been completed

`route_from_supervisor(state)` is the conditional edge function: it simply
reads `state["next_agent"]` so LangGraph can dispatch to the right node.

Reference: wk6_agent_design/1.multi_agent.ipynb
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import SUPERVISOR_PROMPT
from agent.state import AnalystState

logger = logging.getLogger(__name__)

# Constants used both here and in graph.py for edge mapping.
RAG = "rag_agent"
MCP = "mcp_tools"
SYNTH = "synthesizer"

_VALID = {RAG, MCP, SYNTH}


def make_supervisor(llm):
    """Return a supervisor node closed over *llm*.

    Parameters
    ----------
    llm:
        Any LangChain chat model.

    Returns
    -------
    A callable compatible with `StateGraph.add_node`.
    """

    def supervisor(state: AnalystState) -> dict:
        plan = state.get("plan", [])
        idx = state.get("current_step_index", 0)

        # All steps done → route to synthesizer.
        if idx >= len(plan):
            logger.debug("Supervisor: all %d steps done → synthesizer", len(plan))
            return {"next_agent": SYNTH}

        current_step = plan[idx]
        logger.debug(
            "Supervisor: classifying step %d/%d: %r", idx + 1, len(plan), current_step
        )

        # Ask the LLM to classify the current step.
        response = llm.invoke(
            [
                SystemMessage(content=SUPERVISOR_PROMPT),
                HumanMessage(
                    content=f"Step to classify: {current_step}"
                ),
            ]
        )
        raw = response.content.strip().strip('"').strip("'").lower()

        # Clean up potential JSON / markdown wrapping.
        if raw.startswith('"') and raw.endswith('"'):
            raw = raw[1:-1]

        # Validate and fall back to rag_agent on ambiguous output.
        if raw not in {RAG, MCP}:
            logger.warning(
                "Supervisor returned unexpected route %r for step %r; defaulting to rag_agent",
                raw,
                current_step,
            )
            raw = RAG

        logger.debug("Supervisor → %s", raw)
        return {"next_agent": raw}

    return supervisor


def route_from_supervisor(state: AnalystState) -> str:
    """Conditional edge: return the supervisor's routing decision.

    LangGraph calls this after the supervisor node runs and uses the returned
    string to select the next node.  The value must match a key in the mapping
    passed to `add_conditional_edges`.
    """
    return state.get("next_agent", SYNTH)
