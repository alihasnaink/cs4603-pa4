"""Planner node (Task 1.2).

`make_planner(llm)` returns a LangGraph node function that decomposes the
user's analytical question into 2–5 ordered atomic steps (JSON list).  The
plan is stored in state so downstream nodes can iterate through it.

Failure handling
----------------
If the LLM response is not valid JSON (or not a list), the planner falls back
gracefully to a single-step plan containing the original user question, so the
graph can still run rather than crashing.

Reference: wk6_agent_design/2.plan_and_execute.ipynb
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import PLANNER_PROMPT
from agent.state import AnalystState

logger = logging.getLogger(__name__)


def make_planner(llm):
    """Return a planner node closed over *llm*.

    Parameters
    ----------
    llm:
        Any LangChain chat model.  In tests this is replaced by a fake.

    Returns
    -------
    A callable compatible with `StateGraph.add_node`.
    """

    def planner(state: AnalystState) -> dict:
        # Extract the last human message as the user question.
        messages = state.get("messages", [])
        if not messages:
            question = "No question provided."
        else:
            last = messages[-1]
            question = last.content if hasattr(last, "content") else str(last)

        # Ask the LLM to decompose the question into an ordered step list.
        response = llm.invoke(
            [
                SystemMessage(content=PLANNER_PROMPT),
                HumanMessage(content=f"User question: {question}"),
            ]
        )
        raw = response.content.strip()

        # Parse the JSON list; fall back to a single-step plan on failure.
        plan = _parse_plan(raw, fallback=question)

        logger.debug("Planner produced %d steps: %s", len(plan), plan)

        return {
            "plan": plan,
            "current_step_index": 0,
            "step_results": [],
        }

    return planner


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _parse_plan(raw: str, *, fallback: str) -> list[str]:
    """Try to parse *raw* as a JSON list of strings.

    Strips optional markdown code fences before parsing.  Returns a
    single-element list containing *fallback* if parsing fails.
    """
    # Strip ```json ... ``` or ``` ... ``` fences if present.
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Drop first line (``` or ```json) and last line (```)
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        cleaned = "\n".join(inner).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list) and all(isinstance(s, str) for s in parsed):
            # Clamp to 2–5 steps as per spec.
            if len(parsed) < 1:
                raise ValueError("Empty plan")
            return parsed[:5]
        raise ValueError("Plan is not a list of strings")
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Planner parse failure (%s); using single-step fallback.", exc)
        return [fallback]
