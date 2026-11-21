"""SQLite reads stay concurrent, and entity degree is stored at index time."""

from pathlib import Path

from raganite.backends.base import DocumentGraph, GraphEdge, GraphNode
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
