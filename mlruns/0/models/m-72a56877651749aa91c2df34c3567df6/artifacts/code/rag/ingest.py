"""Corpus ingestion into Databricks Vector Search (Task 0.3 / rag/ingest.py).

Run inside a Databricks notebook (needs Spark + ai_parse_document/ai_prep_search).
Mirror PA2 Part 1:

TODO:
  - `build_chunks_table(spark, volume_path, chunks_table)`: parse the PDF with
    ai_parse_document, chunk with ai_prep_search into a Delta table with columns
    chunk_id, chunk_to_retrieve, chunk_to_embed, source, page. Enable Change Data
    Feed on the table.
  - `create_index()`: create a STANDARD Vector Search endpoint and a TRIGGERED
    Delta Sync index (primary_key='chunk_id',
    embedding_source_column='chunk_to_retrieve',
    embedding_model_endpoint_name=$EMBEDDINGS_ENDPOINT).
"""

from __future__ import annotations
import time


def build_chunks_table(spark, volume_path: str, chunks_table: str) -> None:
    df = spark.read.format("binaryFile").load(volume_path)
    df.createOrReplaceTempView("raw_docs")

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {chunks_table} (
            chunk_id STRING,
            chunk_to_retrieve STRING,
            chunk_to_embed STRING,
            source STRING,
            page STRING
        ) TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)

    spark.sql(f"""
        INSERT INTO {chunks_table}
        WITH parsed AS (
            SELECT path AS source, ai_parse_document(content) AS parsed
            FROM raw_docs
        ),
        prepped AS (
            SELECT source, ai_prep_search(parsed) AS result
            FROM parsed
        )
        SELECT
            chunk.value:chunk_id::STRING AS chunk_id,
            chunk.value:chunk_to_retrieve::STRING AS chunk_to_retrieve,
            chunk.value:chunk_to_embed::STRING AS chunk_to_embed,
            prepped.source AS source,
            CAST(NULL AS STRING) AS page
        FROM prepped,
            LATERAL variant_explode(prepped.result:document.contents) AS chunk
    """)


def create_index(
    chunks_table: str,
    endpoint_name: str = "ali-vs-endpoint",
    index_name: str = "ali_pa4.rag.ali_analyst_index",
    embedding_endpoint: str = "databricks-gte-large-en",
) -> None:
    from databricks.ai_search.client import AISearchClient

    client = AISearchClient()

    existing_endpoints = [e["name"] for e in client.list_endpoints().get("endpoints", [])]
    if endpoint_name not in existing_endpoints:
        print(f"Creating Vector Search endpoint: {endpoint_name}...")
        client.create_endpoint(name=endpoint_name, endpoint_type="STANDARD")
    else:
        print(f"Endpoint '{endpoint_name}' already exists.")

    print(f"Waiting for endpoint {endpoint_name} to be ready...")
    for _ in range(60):  
        state = client.get_endpoint(endpoint_name)["endpoint_status"]["state"]
        if state in ("ONLINE", "READY"):
            break
        time.sleep(10)
    else:
        raise TimeoutError(f"Endpoint {endpoint_name} did not become ready in time.")

    existing_indexes = [
        idx["name"] for idx in client.list_indexes(name=endpoint_name).get("vector_indexes", [])
    ]
    if index_name not in existing_indexes:
        print(f"Creating Vector Search index: {index_name}...")
        client.create_delta_sync_index(
            endpoint_name=endpoint_name,
            source_table_name=chunks_table,
            index_name=index_name,
            pipeline_type="TRIGGERED",
            primary_key="chunk_id",
            embedding_source_column="chunk_to_embed",
            embedding_model_endpoint_name=embedding_endpoint,
        )
    else:
        print(f"Index '{index_name}' already exists. Syncing...")
        index = client.get_index(endpoint_name=endpoint_name, index_name=index_name)
        index.sync()
