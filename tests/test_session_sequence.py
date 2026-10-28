"""Ordered documents become session entities the two-step walk can cross."""

from neo4j_graphrag.components.types import Neo4jGraph, Neo4jNode, Neo4jRelationship

from raganite.backends.sqlite import SQLiteBackend
from raganite.config import RaganiteConfig, SQLiteConfig
from raganite.indexing.indexer import KnowledgeGraphIndexer


class _Embeddings:
    async def aembed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]


class _Extractor:
    def __init__(self, *args, **kwargs):
        return None

    async def run(self, chunks, document_info, schema=None, examples="", **kwargs):
        person = "Ada" if document_info.path == "session-1.txt" else "Bea"
        marker = "Monday" if person == "Ada" else "Tuesday"
        person_id = f"person-{person}"
        marker_id = f"marker-{marker}"
        nodes = [
            Neo4jNode(
                id=document_info.uid, label="Document", properties={"path": document_info.path}
            ),
            Neo4jNode(id=person_id, label="Person", properties={"name": person}),
            Neo4jNode(id=marker_id, label="Day", properties={"name": marker}),
        ]
        nodes.extend(
            Neo4jNode(
                id=chunk.uid, label="Chunk", properties={"text": chunk.text, "index": chunk.index}
            )
            for chunk in chunks.chunks
        )
        relationships = [
            Neo4jRelationship(
                start_node_id=person_id, end_node_id=marker_id, type="KNOWS", properties={}
            ),
        ]
        return Neo4jGraph(nodes=nodes, relationships=relationships)


def _graph(monkeypatch):
    monkeypatch.setattr("raganite.indexing.indexer.LLMEntityRelationExtractor", _Extractor)
    backend = SQLiteBackend(SQLiteConfig(path=":memory:"), namespace="conv-26", dimensions=2)
    indexer = KnowledgeGraphIndexer(
        backend,
        object(),
        _Embeddings(),
        RaganiteConfig(namespace="conv-26", embedding_dimensions=2),
    )
    return backend, indexer


def test_ordered_sessions_stay_linked_for_the_walk(monkeypatch):
    from langchain_core.documents import Document

    backend, indexer = _graph(monkeypatch)
    sessions = [
        Document(
            page_content="Ada left on Monday.",
            metadata={"path": "session-1.txt", "order": 1, "time": "1:56 pm on 8 May, 2023"},
        ),
        Document(
            page_content="The group met.",
            metadata={"path": "session-2.txt", "order": 2, "time": "7:55 pm on 9 June, 2023"},
        ),
        Document(page_content="A loose note.", metadata={"path": "note.txt"}),
    ]
    try:
        indexer.index_documents(sessions, source="conv-26")
        linked = [edge for edge in backend.edges() if edge.type == "NEXT"]
        assert len(linked) == 1
        assert linked[0].properties == {
            "previous": "1:56 pm on 8 May, 2023",
            "next": "7:55 pm on 9 June, 2023",
        }
        owners = backend._connection.execute(
            "SELECT path FROM document_edges WHERE namespace=? AND edge_id=? ORDER BY path",
            (backend.namespace, linked[0].id),
        ).fetchall()
        assert [row["path"] for row in owners] == ["session-1.txt", "session-2.txt"]
        sessions_for_note = [
            node
            for node in backend.nodes(kind="entity")
            if node.label == "Session" and node.properties.get("path") == "note.txt"
        ]
        assert sessions_for_note == []
        ada = next(
            hit for hit in backend.search_facts([1.0, 0.0], top_k=10) if hit.subject == "Ada"
        )
        reached = backend.expand_facts([ada], depth=2, limit=20)
        assert any("The group met." in row.text for row in reached)

        indexer.index_documents([sessions[1]], source="conv-26", force=True)
        linked_again = [edge for edge in backend.edges() if edge.type == "NEXT"]
        assert len(linked_again) == 1
        owners_again = backend._connection.execute(
            "SELECT path FROM document_edges WHERE namespace=? AND edge_id=? ORDER BY path",
            (backend.namespace, linked_again[0].id),
        ).fetchall()
        assert [row["path"] for row in owners_again] == ["session-1.txt", "session-2.txt"]
        reached_again = backend.expand_facts([ada], depth=2, limit=20)
        assert any("The group met." in row.text for row in reached_again)
    finally:
        backend.close()
