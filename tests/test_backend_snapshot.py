"""Backend snapshots. Set RAGANITE_TEST_DOCKER_NEO4J=1 to include Neo4j 5.26."""

import os
import subprocess
import time
from unittest.mock import Mock
from uuid import uuid4

import pytest
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired
from pydantic import SecretStr

from raganite.backends.base import DocumentGraph, GraphEdge, GraphNode
from raganite.backends.models import ChunkRecord, FactEdge
from raganite.backends.neo4j import Neo4jBackend
from raganite.backends.sqlite import SQLiteBackend
from raganite.config import Neo4jConfig, SQLiteConfig
from raganite.retrieval import Retriever
from raganite.retrieval.fact_walk.graph import FactWalkGraph

VECTOR = (0.12345678901234567, -0.9876543210987654)


@pytest.fixture(scope="module")
def disposable_neo4j():
    if os.environ.get("RAGANITE_TEST_DOCKER_NEO4J") != "1":
        pytest.skip("Set RAGANITE_TEST_DOCKER_NEO4J=1 to launch a disposable Neo4j container")
    name = f"raganite-snapshot-test-{uuid4().hex}"
    password = uuid4().hex

    def docker(*args):
        return subprocess.check_output(["docker", *args], text=True, timeout=60).strip()

    try:
        docker(
            "run",
            "--detach",
            "--pull=never",
            "--name",
            name,
            "-p",
            "127.0.0.1::7687",
            "-e",
            f"NEO4J_AUTH=neo4j/{password}",
            "neo4j:5.26",
        )
        port = int(docker("port", name, "7687/tcp").rsplit(":", 1)[1])
        assert port not in {7687, 7688, 7689}
        uri = f"bolt://127.0.0.1:{port}"
        with GraphDatabase.driver(uri, auth=("neo4j", password), connection_timeout=2) as driver:
            deadline = time.monotonic() + 90
            while True:
                try:
                    driver.verify_connectivity()
                    break
                except (ServiceUnavailable, SessionExpired):
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(1)
        yield Neo4jConfig(uri=uri, password=SecretStr(password))
    finally:
        subprocess.run(["docker", "rm", "-fv", name], capture_output=True, check=False, timeout=30)


@pytest.fixture(params=["sqlite", "neo4j"])
def backend(request):
    options = {"namespace": uuid4().hex, "dimensions": 2}
    if request.param == "sqlite":
        store = SQLiteBackend(SQLiteConfig(path=":memory:"), **options)
    else:
        store = Neo4jBackend(request.getfixturevalue("disposable_neo4j"), **options)
    with store:
        store.ensure_indexes()
        yield store


def _node(id_, kind, label, embedding=None, **properties):
    return GraphNode(id=id_, kind=kind, label=label, embedding=embedding, properties=properties)


def _edge(id_, source, type_, target):
    return GraphEdge(id=id_, source=source, type=type_, target=target)


def _graph(path, nodes, edges=(), *, content_hash="v1", source="corpus"):
    return DocumentGraph(
        source=source, path=path, content_hash=content_hash, nodes=nodes, edges=edges
    )


def test_native_snapshot_uses_index_label_and_vector_property(disposable_neo4j):
    config = disposable_neo4j.model_copy(update={"vector_index": "native_passages"})
    with GraphDatabase.driver(
        config.uri, auth=(config.username, config.password.get_secret_value())
    ) as driver:
        driver.execute_query(
            "CREATE (a:Article {content:'Pump manual', path:'manual.txt', semanticVector:$vector}), "
            "(b:Article {text:'Pump biography', path:'bio.txt', semanticVector:$vector}), "
            "(:Article {text:'Not embedded'}), "
            "(:Distractor {text:'Wrong label', semanticVector:$vector})",
            vector=list(VECTOR),
        )
        driver.execute_query(
            "CREATE VECTOR INDEX native_passages FOR (n:Article) ON (n.semanticVector) "
            "OPTIONS {indexConfig:{`vector.dimensions`:2, `vector.similarity_function`:'cosine'}}"
        )
        driver.execute_query("CALL db.awaitIndex('native_passages')")
        with Neo4jBackend(config, driver=driver, dimensions=2, layout="native") as store:
            chunks = store.chunk_records()
            assert {chunk.path for chunk in chunks} == {"manual.txt", "bio.txt"}
            assert store.chunk_embeddings([*(chunk.chunk_id for chunk in chunks), "missing"]) == {
                chunk.chunk_id: VECTOR for chunk in chunks
            }
            embeddings = Mock()
            embeddings.embed_query.return_value = list(VECTOR)
            retriever = Retriever(store, embeddings)
            try:
                result = retriever.retrieve("Which pump?")
                assert {passage.source for passage in result} == {"manual.txt", "bio.txt"}
                assert all(passage.metadata["beam_candidate_count"] == 2 for passage in result)
            finally:
                retriever.close()


def test_owned_snapshot_preserves_records_provenance_and_precision(backend):
    # Siblings share document entities even where the physical link is to only one chunk.
    alice = _node("alice", "entity", "Person", name=" Alice ")
    bob = _node("bob", "entity", "Person", name=" ", title="Bob")
    duplicate_alice = _node("alias", "entity", "Alias", title="Alice")
    doc_z = _node("z-doc", "document", "Document", title="Other title")
    doc_a = _node(
        "a-doc", "document", "Document", path="owner.txt", metadata={"title": "  Alice  "}
    )
    chunk = _node("z", "chunk", "Chunk", VECTOR, path="z.txt", text="Alice passage")
    alias_chunk = _node(
        "a", "chunk", "Chunk", file_path="a.txt", content="Alias passage", title=" ", name=" Alice "
    )
    fact = _node(
        "fact",
        "fact",
        "__Fact__",
        VECTOR,
        text="Alice knows Bob",
        relation_type="KNOWS",
        relationship_id="rel",
    )
    backend.replace_document(
        _graph(
            "owner.txt",
            [doc_z, chunk, bob, doc_a, fact, alice, alias_chunk, duplicate_alice],
            [
                _edge("rel", "alice", "KNOWS", "bob"),
                _edge("subject", "fact", "FACT_SUBJECT", "alice"),
                _edge("object", "fact", "FACT_OBJECT", "bob"),
                _edge("mention", "alice", "FROM_CHUNK", "z"),
                _edge("provenance", "z", "FROM_DOCUMENT", "a-doc"),
            ],
        )
    )
    backend.replace_document(
        _graph(
            "bob.txt",
            [bob, _node("b", "chunk", "Chunk", [1, 0], title="Bob", text="Bob passage")],
            source="other",
        )
    )
    owners = [("other", "bob.txt"), ("corpus", "owner.txt")]
    shared = _node(
        "z-fact",
        "fact",
        "__Fact__",
        [1, 0],
        text="Bob sees Alice",
        relation_type="SEES",
        relationship_id="a-rel",
    )
    backend.store_shared_nodes([(shared, owners)])
    backend.store_shared_edges(
        [
            (_edge("a-rel", "bob", "SEES", "alice"), owners),
            (_edge("z-subject", "z-fact", "FACT_SUBJECT", "bob"), owners),
            (_edge("z-object", "z-fact", "FACT_OBJECT", "alice"), owners),
        ]
    )
    assert backend.document_hash("corpus", "owner.txt") == "v1"
    assert backend.document_hash("other", "owner.txt") is None
    facts = backend.fact_edges()
    chunks = backend.chunk_records()
    assert facts == [
        FactEdge("Bob", "Alice", "Bob sees Alice", "SEES", "a-rel", (1.0, 0.0)),
        FactEdge("Alice", "Bob", "Alice knows Bob", "KNOWS", "rel", VECTOR),
    ]
    assert chunks == [
        ChunkRecord("a", "a.txt", "Alias passage", "Alice", ("Alice", "Bob")),
        ChunkRecord("b", "bob.txt", "Bob passage", "Bob", ("Bob",)),
        ChunkRecord("z", "z.txt", "Alice passage", "Alice", ("Alice", "Bob")),
    ]
    vectors = backend.chunk_embeddings(["z", "missing", "alice", "fact", "a", "b", "z"])
    assert list(vectors.items()) == [("b", (1.0, 0.0)), ("z", VECTOR)]
    assert backend.chunk_embeddings([]) == {}

    graph = FactWalkGraph.from_edges(facts, chunks)
    walk = graph.walk(vectors["z"], tau=0.1, damping=0.85, seeds={"Alice": 1.0})
    assert walk.seeds == 1
    assert walk.incoming["Bob"][1].relationship_id == "rel"
    assert [item["chunk_id"] for item in walk.homes] == ["a", "b"]


def test_owned_replacement_clears_old_properties_and_refreshes_retained_entities(backend):
    alice = _node("alice", "entity", "Person", name="Alice")
    bob = _node("bob", "entity", "Person", name="Bob")
    chunk = _node("chunk", "chunk", "Chunk", VECTOR, title="Old title", text="oldterm")
    backend.replace_document(_graph("other.txt", [alice, bob], source="other"))
    backend.replace_document(
        _graph("file.txt", [chunk, alice, bob], [_edge("rel", "alice", "KNOWS", "bob")])
    )
    assert backend.lexical_passages("oldterm", top_k=5)[0].chunk_id == "chunk"
    assert [n.properties["semantic_degree"] for n in backend.nodes(kind="entity")] == [1, 1]

    backend.replace_document(
        _graph(
            "file.txt",
            [_node("chunk", "chunk", "Chunk", content="New passage")],
            content_hash="v2",
        )
    )
    assert backend.document_hash("corpus", "file.txt") == "v2"
    assert backend.edges() == []
    assert backend.chunk_records() == [ChunkRecord("chunk", "file.txt", "New passage")]
    assert backend.chunk_embeddings(["chunk"]) == {}
    assert backend.lexical_passages("oldterm", top_k=5) == []
    assert [n.properties["semantic_degree"] for n in backend.nodes(kind="entity")] == [0, 0]
    assert backend.delete_documents("other") == 1
    assert [n.id for n in backend.nodes()] == ["chunk"]
