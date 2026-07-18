"""Log, register, and serve the Document Analyst (Tasks 2.2 + 2.3).

Run:  uv run python deployment/deploy.py

TODO:
  - `log_and_register()`: set registry uri to 'databricks-uc', log the model via
    `mlflow.langchain.log_model(lc_model="deployment/agent_model.py", name=...,
    code_paths=[...], pip_requirements=[...], input_example={...})`, then
    `mlflow.register_model(...)` into $UC_CATALOG.$UC_SCHEMA.<model>.
  - `create_or_update_endpoint(uc_name, version)`: create/update a Model Serving
    endpoint with `WorkspaceClient().serving_endpoints`, workload_size='Small',
    scale_to_zero_enabled=True, and environment_vars supplied as secret refs
    ({{secrets/cs4603-deploy/...}}). Wait for READY and print the URL.
"""

from __future__ import annotations

import os
import time
import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from dotenv import load_dotenv

def log_and_register():
    load_dotenv(override=True)
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    
    # Set experiment to avoid local mlruns with spaces
    w = WorkspaceClient()
    try:
        email = w.current_user.me().user_name
    except Exception:
        email = "alihasnainkiani@gmail.com"
    mlflow.set_experiment(f"/Users/{email}/cs4603-pa4-deployment")
    
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    with mlflow.start_run():
        model_info = mlflow.langchain.log_model(
            lc_model=os.path.join(root, "deployment", "agent_model.py"),
            name="agent",
            code_paths=[
                os.path.join(root, "agent"),
                os.path.join(root, "rag"),
                os.path.join(root, "tools"),
                os.path.join(root, "config.py"),
            ],
            pip_requirements=[
                "mlflow",
                "langgraph",
                "langchain-openai",
                "databricks-langchain",
                "databricks-vectorsearch",
                "langchain-mcp-adapters",
                "mcp",
                "openai"
            ],
            input_example={"messages": [{"role": "user", "content": "What was the revenue?"}]},
        )
    
    catalog = os.environ.get("UC_CATALOG", "main")
    schema = os.environ.get("UC_SCHEMA", "default")
    user = os.environ.get("USER", "student")
    model_name = f"{user}_pa4_analyst_model"
    uc_name = f"{catalog}.{schema}.{model_name}"
    
    registered = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=uc_name
    )
    
    print(f"Registered model version: {registered.version}")
    return uc_name, str(registered.version)


def create_or_update_endpoint(uc_name: str, version: str) -> str:
    load_dotenv(override=True)
    w = WorkspaceClient()
    user = os.environ.get("USER", "student")
    endpoint_name = f"{user}-pa4-analyst-endpoint"
    
    print(f"Deploying endpoint: {endpoint_name}...")
    
    env_vars = {
        "DATABRICKS_HOST": "{{secrets/cs4603-deploy/DATABRICKS_HOST}}",
        "DATABRICKS_TOKEN": "{{secrets/cs4603-deploy/DATABRICKS_TOKEN}}",
        "DATABRICKS_MODEL": "{{secrets/cs4603-deploy/DATABRICKS_MODEL}}",
        "VECTOR_SEARCH_ENDPOINT": os.environ.get("VECTOR_SEARCH_ENDPOINT", "pa4-vs-endpoint"),
        "VECTOR_SEARCH_INDEX": os.environ.get("VECTOR_SEARCH_INDEX", f"main.default.{user}_analyst_index"),
        "EMBEDDINGS_ENDPOINT": os.environ.get("EMBEDDINGS_ENDPOINT", "databricks-gte-large-en"),
    }
    
    # Delete existing endpoint if there
    try:
        w.serving_endpoints.get(endpoint_name)
        print("Endpoint exists. Deleting...")
        w.serving_endpoints.delete(endpoint_name)
        time.sleep(5)
    except Exception:
        pass
        
    print("Creating endpoint...")
    response = w.serving_endpoints.create(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            name=endpoint_name,
            served_entities=[
                ServedEntityInput(
                    entity_name=uc_name,
                    entity_version=version,
                    workload_size="Small",
                    scale_to_zero_enabled=True,
                    environment_vars=env_vars,
                )
            ],
        ),
    )
    
    print("Waiting for endpoint to reach READY state...")
    # Polling for ready state (can also use .result() if response has it, but manual polling is safe)
    if hasattr(response, "result"):
        response.result()
    else:
        while True:
            status = w.serving_endpoints.get(endpoint_name)
            state = status.state.ready.value
            print(f"Status: {state}")
            if state == "READY":
                break
            elif "FAILED" in state:
                raise RuntimeError(f"Endpoint failed: {state}")
            time.sleep(30)
    
    url = f"{os.environ.get('DATABRICKS_HOST', '')}/serving-endpoints/{endpoint_name}/invocations"
    print(f"Endpoint '{endpoint_name}' is READY at: {url}")
    return url


if __name__ == "__main__":
    name, ver = log_and_register()
    create_or_update_endpoint(name, ver)
