"""State schema for the Document Analyst graph (Task 1.1).

`AnalystState` is a TypedDict that flows through every node of the LangGraph
multi-agent graph. The `messages` channel uses the built-in `add_messages`
reducer so incoming messages are *appended* rather than replaced — this is the
contract the deployed serving endpoint reads (last message = final answer).

All other fields are plain assignments (last-writer-wins), which is fine because
only one node writes each field at a time in this linear-with-loop topology.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AnalystState(TypedDict):
    """Shared state flowing through the Document Analyst graph.

    Fields
    ------
    messages:
        Full conversation history with `add_messages` reducer (append-only).
        The serving endpoint reads ``state["messages"][-1].content`` as the
        final answer, so the synthesizer **must** append an AIMessage here.
    plan:
        Ordered list of 2–5 atomic step strings produced by the planner.
    current_step_index:
        Index into `plan` pointing to the step currently being executed.
        Specialist nodes increment this after completing their step.
    step_results:
        Accumulated results from all completed steps (one string per step).
    next_agent:
        Routing decision written by the supervisor: ``"rag_agent"``,
        ``"mcp_tools"``, or ``"synthesizer"``.
    final_answer:
        The synthesized, cited answer string.  Also written to `messages`
        as an AIMessage so the serving contract is satisfied.
    """

    messages: Annotated[list, add_messages]
    plan: list[str]
    current_step_index: int
    step_results: list[str]
    next_agent: str
    final_answer: str
