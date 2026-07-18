"""RAG agent node (Task 1.4) — retrieves from Databricks Vector Search.

`make_rag_agent(retriever, llm)` returns a LangGraph node that:
  1. Retrieves top-k chunks from the Vector Search index for the current step.
  2. Formats them with [source: file, p.N] citations.
  3. Asks the LLM to extract a single cited fact (or "not found in documents").
  4. Appends the result to `step_results` and increments `current_step_index`.

`format_docs(docs)` converts a list of LangChain Documents into a
citation-annotated context string.

The retriever is injected so the same node works in tests with a fake, and in
production with the real `rag/store.py::get_retriever()`.

Reference: PA2 Part 1, wk5_langgraph/11b.langgraph_rag.ipynb
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import RAG_EXTRACT_PROMPT
from agent.state import AnalystState

logger = logging.getLogger(__name__)


# ─── Document formatting ──────────────────────────────────────────────────────


def format_docs(docs) -> str:
    """Format a list of LangChain Documents into a cited context string.

    Each chunk is rendered as:

        [Chunk N — source: <source>, p.<page>]
        <text>

    The index was built with ``chunk_to_embed`` as the embedding source, so
    ``page_content`` may contain the embed text.  We prefer ``chunk_to_retrieve``
    from metadata (fetched explicitly via ``columns``) when available, as it holds
    the human-readable retrieval text intended for the LLM.
    """
    if not docs:
        return "(no documents retrieved)"

    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        # Prefer the explicit retrieval text column if present.
        text = (
            meta.get("chunk_to_retrieve")
            or (doc.page_content if hasattr(doc, "page_content") else None)
            or str(doc)
        )
        parts.append(f"[Chunk {i} — source: {source}, p.{page}]\n{text}")

    return "\n\n".join(parts)



# ─── Node factory ─────────────────────────────────────────────────────────────


def make_rag_agent(retriever, llm):
    """Return a RAG retrieval node closed over *retriever* and *llm*.

    Parameters
    ----------
    retriever:
        A LangChain ``BaseRetriever`` (real or fake).
    llm:
        Any LangChain chat model.

    Returns
    -------
    A callable compatible with `StateGraph.add_node`.
    """

    def rag_agent(state: AnalystState) -> dict:
        plan = state.get("plan", [])
        idx = state.get("current_step_index", 0)
        step = plan[idx] if idx < len(plan) else "retrieve relevant information"

        logger.debug("RAG agent: retrieving for step %d: %r", idx, step)

        # Retrieve top-k chunks.
        try:
            docs = retriever.invoke(step)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Retriever error: %s", exc)
            docs = []

        context = format_docs(docs)

        # Extract the relevant fact via the LLM.
        if not docs:
            fact = "not found in documents"
        else:
            response = llm.invoke(
                [
                    SystemMessage(content=RAG_EXTRACT_PROMPT),
                    HumanMessage(
                        content=(
                            f"STEP: {step}\n\n"
                            f"CONTEXT:\n{context}"
                        )
                    ),
                ]
            )
            fact = response.content.strip()

        logger.debug("RAG agent result: %r", fact)

        current_results = list(state.get("step_results", []))
        current_results.append(fact)

        return {
            "step_results": current_results,
            "current_step_index": idx + 1,
        }

    return rag_agent
