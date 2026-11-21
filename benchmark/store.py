"""SQLite and Neo4j settings for a benchmark graph."""

import os
from pathlib import Path

from pydantic import SecretStr

from raganite.config import Neo4jConfig, RaganiteConfig, RetrievalConfig, SQLiteConfig


def prepare_connection(options: dict) -> dict:
    """Fill Neo4j connection settings from the environment."""
    from benchmark.hotpotqa_flow import load_benchmark_env

    load_benchmark_env()
    resolved = dict(options)
    if resolved.get("neo4j_password") in (None, "raganite-local") and os.getenv("NEO4J_PASSWORD"):
        resolved["neo4j_password"] = os.environ["NEO4J_PASSWORD"]
    if resolved.get("neo4j_username") in (None, "neo4j") and os.getenv("NEO4J_USERNAME"):
        resolved["neo4j_username"] = os.environ["NEO4J_USERNAME"]
    if resolved.get("neo4j_database") in (None, "neo4j") and os.getenv("NEO4J_DATABASE"):
        resolved["neo4j_database"] = os.environ["NEO4J_DATABASE"]
    return resolved


def store_config(
    backend: str,
    *,
    sqlite_path: Path,
    neo4j_uri: str,
    username: str,
    password: str,
    database: str,
    dimensions: int,
    namespace: str = "default",
    retrieval: RetrievalConfig | None = None,
) -> RaganiteConfig:
    if backend == "sqlite":
        store = SQLiteConfig(path=str(sqlite_path))
    elif backend == "neo4j":
        store = Neo4jConfig(
            uri=neo4j_uri, username=username, password=SecretStr(password), database=database
        )
    else:
        raise ValueError("backend must be sqlite or neo4j")
    return RaganiteConfig(
        backend=store,
        namespace=namespace,
        embedding_dimensions=dimensions,
        retrieval=retrieval or RetrievalConfig(),
    )


def maybe_index(indexer, documents: list, *, source: str, skip_index: bool):
    """Index ``documents`` unless the caller asked to reuse an existing graph."""
    if skip_index:
        return None
    return indexer.index_documents(documents, source=source)
