"""Synthesizer node (Task 1.6).

`make_synthesizer(llm)` returns a LangGraph node that:
  1. Collects all `step_results` from the state.
  2. Calls the LLM to produce a single coherent, cited answer.
  3. Writes the answer to BOTH `final_answer` AND the `messages` channel as
     an AIMessage.

Writing to `messages` is critical for the deployed serving contract: the
endpoint is expected to return the last message in the messages list, so if the
synthesizer only sets `final_answer` the endpoint response will be empty even
though the local graph looks correct.

Reference: Task 1.6 spec note, wk5_langgraph/11b.langgraph_rag.ipynb
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.prompts import SYNTHESIZER_PROMPT
from agent.state import AnalystState

logger = logging.getLogger(__name__)


def make_synthesizer(llm):
    """Return a synthesizer node closed over *llm*.

    Parameters
    ----------
    llm:
        Any LangChain chat model.

    Returns
    -------
    A callable compatible with `StateGraph.add_node`.
    """

    def synthesizer(state: AnalystState) -> dict:
        # Recover original user question from first human message.
        messages = state.get("messages", [])
        user_question = ""
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "human":
                user_question = msg.content
                break
            # Also handle raw dicts (from the serving endpoint input)
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_question = msg.get("content", "")
                break

        step_results = state.get("step_results", [])
        plan = state.get("plan", [])

        # Format step results with their plan step labels.
        context_lines = []
        for i, (step, result) in enumerate(zip(plan, step_results), 1):
            context_lines.append(f"Step {i}: {step}\nResult: {result}")
        # Handle any extra results beyond the plan (shouldn't happen, but safe).
        for i, result in enumerate(step_results[len(plan):], len(plan) + 1):
            context_lines.append(f"Step {i}: (extra)\nResult: {result}")

        context = "\n\n".join(context_lines) if context_lines else "(no step results)"

        logger.debug(
            "Synthesizer: combining %d step results for question: %r",
            len(step_results),
            user_question,
        )

        response = llm.invoke(
            [
                SystemMessage(content=SYNTHESIZER_PROMPT),
                HumanMessage(
                    content=(
                        f"User question: {user_question}\n\n"
                        f"Step results:\n{context}"
                    )
                ),
            ]
        )
        answer = response.content.strip()

        logger.debug("Synthesizer answer: %r", answer[:200])

        # Write answer to messages channel AND final_answer so both the
        # serving endpoint contract and local state inspection work.
        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    return synthesizer
