"""Vector Search retriever factory (Task 1.4 support / rag/store.py).

`get_retriever(k)` returns a LangChain retriever over the Databricks Vector
Search index built by `ingest.py`.  Both the local agent and the deployed
serving container call this identical function — no separate embedding path
is needed because the index lives in Databricks and is reachable from anywhere
with a valid DATABRICKS_HOST / DATABRICKS_TOKEN.
"""

from __future__ import annotations

from config import get_settings

# The index was built with chunk_to_embed as the embedding source column.
# Do NOT pass text_column= to DatabricksVectorSearch — it infers it from
# the index spec automatically.  We still fetch chunk_to_retrieve so the
# rag_agent can surface it as citation text.
CITATION_COLUMNS = ["chunk_id", "chunk_to_retrieve", "source", "page"]


def get_vector_store():
    """Return a DatabricksVectorSearch handle over the configured index.

    Credentials and index coordinates are read from environment variables via
    `config.get_settings()` so the same code runs locally and in the serving
    container (where they are supplied as endpoint environment variables).
    """
    from databricks_langchain import DatabricksVectorSearch

    s = get_settings()
    return DatabricksVectorSearch(
        endpoint=s["vs_endpoint"],
        index_name=s["vs_index"],
        # text_column is intentionally omitted — the SDK reads it from the
        # index spec (embedding_source_column = "chunk_to_embed").
        columns=CITATION_COLUMNS,
    )


def get_retriever(k: int = 4):
    """Return a top-k LangChain retriever over the Vector Search index.

    Parameters
    ----------
    k:
        Number of chunks to retrieve per query.  Default is 4 to balance
        recall against context-window cost.

    Returns
    -------
    A LangChain ``BaseRetriever`` that can be called with `.invoke(query)`.
    """
    vs = get_vector_store()
    return vs.as_retriever(search_kwargs={"k": k})
