"""MLflow models-from-code definition (Task 2.1).

TODO: Make this file self-contained so MLflow can serialise it:
  - validate DATABRICKS_HOST/TOKEN/MODEL at import time (clear error if missing),
  - rebuild the graph with production clients (LLM, Vector Search retriever,
    MCP tools),
  - end with `mlflow.models.set_model(graph)`.

Must import cleanly:  python -c "import deployment.agent_model"
"""

from __future__ import annotations

import os
import mlflow

from config import get_settings

# Validate env vars at import time (will raise OSError if missing)
get_settings()

from agent.graph import build_graph, load_mcp_tools
from config import get_chat_llm
from rag.store import get_retriever

# Resolve MCP server path relative to the packaged code
_mcp_server = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools", "mcp_server.py")
tools = load_mcp_tools(_mcp_server)

# Build the graph
graph = build_graph(llm=get_chat_llm(), retriever=get_retriever(), tools=tools)

# Tell MLflow what to serve
mlflow.models.set_model(graph)
