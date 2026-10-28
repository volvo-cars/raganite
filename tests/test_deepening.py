"""Deepened retrieve, topology, and Neo4j layout, tested through their interfaces."""

from unittest.mock import Mock

import pytest
from langchain_core.embeddings import Embeddings

from raganite.backends.base import DocumentGraph, GraphEdge, GraphNode
from raganite.backends.models import FactHit
from raganite.backends.neo4j import Neo4jBackend
from raganite.backends.sqlite import SQLiteBackend
from raganite.config import Neo4jConfig, RaganiteConfig, RetrievalConfig, SQLiteConfig
from raganite.graph import KnowledgeGraph
from raganite.indexing.indexer import apply_topology
from raganite.schema.topology import GraphTopology


class FixedEmbeddings(Embeddings):
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


class CountingBackend(SQLiteBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scans = 0

    def nodes(self, **kwargs):
        self.scans += 1
        return super().nodes(**kwargs)

    def edges(self, **kwargs):
        self.scans += 1
        return super().edges(**kwargs)


def _store(topology: GraphTopology | None = None) -> CountingBackend:
    backend = CountingBackend(SQLiteConfig(path=":memory:"), namespace="demo", dimensions=2)
    if topology is not None:
        backend.topology = topology
    return backend


def _replace(backend, nodes: list[GraphNode], edges: list[GraphEdge]) -> None:
    backend.replace_document(
        DocumentGraph(
            source="docs",
            path="a.md",
            content_hash="h",
            nodes=nodes,
            edges=edges,
        )
    )


def test_expand_follows_the_store_topology():
    topology = GraphTopology(chunk_entity="MENTIONS", fact_subject="SUBJ", fact_object="OBJ")
    backend = _store(topology)
    chunk = GraphNode(
        id="chunk",
        label="Chunk",
        kind="chunk",
        properties={"text": "Paris is the capital", "path": "a.md"},
    )
    paris = GraphNode(id="paris", label="City", kind="entity", properties={"name": "Paris"})
    france = GraphNode(id="france", label="Country", kind="entity", properties={"name": "France"})
    fact = GraphNode(
        id="fact",
        label="__Fact__",
        kind="fact",
        properties={
            "text": "Paris | capital | France",
            "relation_type": "CAPITAL",
            "relationship_id": "rel",
        },
        embedding=[1.0, 0.0],
    )
    edges = [
        GraphEdge(id="subj", source="fact", target="paris", type="SUBJ"),
        GraphEdge(id="obj", source="fact", target="france", type="OBJ"),
        GraphEdge(id="mention", source="paris", target="chunk", type="MENTIONS"),
    ]
    _replace(backend, [chunk, paris, france, fact], edges)
    hit = FactHit(
        "fact",
        "rel",
        "Paris | capital | France",
        "CAPITAL",
        1.0,
        "paris",
        "Paris",
        1.0,
        "france",
        "France",
        1.0,
    )
    try:
        expanded = backend.expand_facts([hit], depth=0, limit=10, degree_penalty_weight=0)
        assert [row.chunk_id for row in expanded] == ["chunk"]
        assert expanded[0].text == "Paris is the capital"
    finally:
        backend.close()


def test_apply_topology_connects_each_entity_to_its_chunk_and_document():
    chunk = GraphNode(id="c", label="Chunk", kind="chunk", properties={"text": "Paris"})
    entity = GraphNode(id="e", label="City", kind="entity", properties={"name": "Paris"})
    document = GraphNode(id="d", label="Document", kind="document", properties={"path": "a.md"})
    nodes = {node.id: node for node in (chunk, entity, document)}
    edges = {"old": GraphEdge(id="old", source="e", target="c", type="MENTIONS")}
    apply_topology(nodes, edges, GraphTopology(), document_id="d")
    linked = {(edge.type, edge.source, edge.target) for edge in edges.values()}
    assert ("FROM_CHUNK", "e", "c") in linked or ("FROM_CHUNK", "c", "e") in linked
    assert ("FROM_DOCUMENT", "c", "d") in linked or ("FROM_DOCUMENT", "d", "c") in linked
    assert "MENTIONS" not in {edge.type for edge in edges.values()}


def test_retrieve_returns_passages_without_scanning_the_store():
    backend = _store()
    chunk = GraphNode(
        id="chunk",
        label="Chunk",
        kind="chunk",
        properties={"text": "Paris is the capital of France", "path": "a.md"},
        embedding=[1.0, 0.0],
    )
    _replace(backend, [chunk], [])
    config = RaganiteConfig(
        backend=SQLiteConfig(path=":memory:"),
        namespace="demo",
        embedding_dimensions=2,
        retrieval=RetrievalConfig(
            use_triples=False,
            filter_triples=False,
            rerank_passages=False,
            workers=1,
        ),
    )
    kg = KnowledgeGraph(embeddings=FixedEmbeddings(), backend=backend, config=config)
    try:
        passages = kg.retrieve("What is the capital of France?")
        assert passages[0].text == "Paris is the capital of France"
        assert passages[0].source == "a.md"
        assert passages[0].score > 0
        assert backend.scans == 0
    finally:
        kg.close()


def test_retrieve_rejects_an_empty_question():
    backend = _store()
    config = RaganiteConfig(
        backend=SQLiteConfig(path=":memory:"),
        namespace="demo",
        embedding_dimensions=2,
        retrieval=RetrievalConfig(
            use_triples=False,
            filter_triples=False,
            rerank_passages=False,
            workers=1,
        ),
    )
    kg = KnowledgeGraph(embeddings=FixedEmbeddings(), backend=backend, config=config)
    try:
        with pytest.raises(ValueError):
            kg.retrieve("   ")
    finally:
        kg.close()


def test_native_neo4j_layout_is_the_same_adapter_and_read_only():
    driver = Mock()
    backend = Neo4jBackend(
        Neo4jConfig(read_only=False), dimensions=2, driver=driver, layout="native"
    )
    try:
        assert type(backend) is Neo4jBackend
        assert backend.layout == "native"
        assert backend.read_only is True
        with pytest.raises(PermissionError):
            backend.replace_document(
                DocumentGraph(source="s", path="p", content_hash="h", nodes=[], edges=[])
            )
    finally:
        backend.close()
        driver.close.assert_not_called()
