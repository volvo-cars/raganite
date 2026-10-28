"""SQLite reads stay concurrent, and entity degree is stored at index time."""

import math
import threading
from pathlib import Path

from raganite.backends.base import DocumentGraph, GraphEdge, GraphNode
from raganite.backends.models import FactHit
from raganite.backends.sqlite import SQLiteBackend
from raganite.config import SQLiteConfig


def test_indexed_entities_store_their_degree():
    backend = SQLiteBackend(SQLiteConfig(path=":memory:"), namespace="demo", dimensions=2)
    chunk = GraphNode(
        id="chunk", label="Chunk", kind="chunk", properties={"text": "Paris", "path": "a.md"}
    )
    paris = GraphNode(id="paris", label="City", kind="entity", properties={"name": "Paris"})
    france = GraphNode(id="france", label="Country", kind="entity", properties={"name": "France"})
    lone = GraphNode(id="lone", label="Person", kind="entity", properties={"name": "Ada"})
    edges = [
        GraphEdge(id="rel", source="paris", target="france", type="CAPITAL"),
        GraphEdge(id="mention", source="paris", target="chunk", type="MENTIONS"),
    ]
    try:
        backend.replace_document(
            DocumentGraph(
                source="docs",
                path="a.md",
                content_hash="h",
                nodes=[chunk, paris, france, lone],
                edges=edges,
            )
        )
        stored = {
            node.id: node.properties["semantic_degree"] for node in backend.nodes(kind="entity")
        }
        assert stored == {"france": 1, "lone": 0, "paris": 1}
    finally:
        backend.close()


def test_opening_a_graph_fills_missing_degrees(tmp_path: Path):
    path = tmp_path / "graph.sqlite3"
    backend = SQLiteBackend(SQLiteConfig(path=str(path)), namespace="demo", dimensions=2)
    paris = GraphNode(id="paris", label="City", kind="entity", properties={"name": "Paris"})
    france = GraphNode(id="france", label="Country", kind="entity", properties={"name": "France"})
    backend.replace_document(
        DocumentGraph(
            source="docs",
            path="a.md",
            content_hash="h",
            nodes=[paris, france],
            edges=[GraphEdge(id="rel", source="paris", target="france", type="CAPITAL")],
        )
    )
    backend._connection.execute(
        "UPDATE nodes SET properties = json_remove(properties, '$.semantic_degree') WHERE kind='entity'",
    )
    backend._connection.commit()
    backend.close()
    opened = SQLiteBackend(SQLiteConfig(path=str(path)), namespace="demo", dimensions=2)
    try:
        stored = {
            node.id: node.properties["semantic_degree"] for node in opened.nodes(kind="entity")
        }
        assert stored == {"france": 1, "paris": 1}
    finally:
        opened.close()


def test_graph_walks_overlap(tmp_path: Path, monkeypatch):
    backend = SQLiteBackend(
        SQLiteConfig(path=str(tmp_path / "graph.sqlite3")), namespace="demo", dimensions=2
    )
    chunk = GraphNode(
        id="chunk", label="Chunk", kind="chunk", properties={"text": "Paris", "path": "a.md"}
    )
    paris = GraphNode(id="paris", label="City", kind="entity", properties={"name": "Paris"})
    fact = GraphNode(
        id="fact",
        label="__Fact__",
        kind="fact",
        properties={"text": "Paris", "relation_type": "CAPITAL", "relationship_id": "rel"},
        embedding=[1.0, 0.0],
    )
    edges = [
        GraphEdge(id="subj", source="fact", target="paris", type="FACT_SUBJECT"),
        GraphEdge(id="mention", source="paris", target="chunk", type="FROM_CHUNK"),
    ]
    hit = FactHit(
        "fact",
        "rel",
        "Paris",
        "CAPITAL",
        1.0,
        "paris",
        "Paris",
        1.0,
        "france",
        "France",
        1.0,
    )
    barrier = threading.Barrier(2, timeout=2)
    entered: set[int] = set()
    guard = threading.Lock()
    original_log1p = math.log1p

    def blocking_log1p(value: float) -> float:
        ident = threading.get_ident()
        with guard:
            first = ident not in entered
            entered.add(ident)
        if first:
            barrier.wait()
        return original_log1p(value)

    monkeypatch.setattr("raganite.backends.sqlite.sqlite.math.log1p", blocking_log1p)
    errors: list[BaseException] = []

    def walk() -> None:
        try:
            backend.expand_facts([hit], depth=0, limit=10, degree_penalty_weight=0.25)
        except BaseException as exc:
            errors.append(exc)

    try:
        backend.replace_document(
            DocumentGraph(
                source="docs",
                path="a.md",
                content_hash="h",
                nodes=[chunk, paris, fact],
                edges=edges,
            )
        )
        threads = [threading.Thread(target=walk) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        assert errors == []
        assert not any(thread.is_alive() for thread in threads)
    finally:
        backend.close()
