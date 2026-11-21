from unittest.mock import Mock

import numpy as np
import pytest

from raganite.backends.base import DocumentGraph, GraphBackend, GraphNode
from raganite.backends.models import ChunkRecord, PassageHit
from raganite.backends.sqlite import SQLiteBackend
from raganite.config import RaganiteConfig, SQLiteConfig
from raganite.graph import KnowledgeGraph
from raganite.retrieval import BeamRetriever
from raganite.retrieval.beam.policy import load_checkpoint
from raganite.retrieval.beam.retriever import LiveBeamPolicy


@pytest.mark.parametrize("missing", ["fact_edges", "chunk_records", "chunk_embeddings"])
def test_default_retrieval_supports_backends_without_a_complete_snapshot(missing):
    backend = Mock(spec=GraphBackend)
    backend.graph_generation = 0
    backend.fact_edges.return_value = []
    backend.chunk_records.return_value = [ChunkRecord("one", "one.txt", "Pump")]
    backend.chunk_embeddings.return_value = {"one": (1.0, 0.0)}
    getattr(backend, missing).side_effect = getattr(GraphBackend, missing).__get__(backend)
    backend.direct_passages.return_value = [PassageHit("one", "one.txt", "Pump", 1.0)]
    backend.lexical_passages.return_value = []
    embeddings = Mock()
    embeddings.embed_query.return_value = [1.0, 0.0]
    retriever = BeamRetriever(backend, embeddings)
    try:
        assert [passage.text for passage in retriever.retrieve("Which pump?")] == ["Pump"]
        backend.direct_passages.assert_called_once()
        embeddings.embed_query.assert_called_once_with("Which pump?")
    finally:
        retriever.close()


def test_live_passage_matrix_is_normalized_and_title_aligned():
    chunks = [
        {"chunk_id": "b", "path": "b.txt", "text": "Blue"},
        {"chunk_id": "a", "path": "a.txt", "text": "Red"},
    ]
    policy = LiveBeamPolicy(
        chunks,
        [],
        {"a": (3, 4), "b": (0, 0)},
        {"a.txt": "Red", "b.txt": "Blue"},
        dimensions=2,
    )
    assert [chunk["title"] for chunk in policy.graph.chunks] == ["Blue", "Red"]
    np.testing.assert_allclose(policy.passages, [[0, 0], [0.6, 0.8]])


def test_beam_can_keep_multiple_chunks_from_one_document():
    chunks = [
        {"chunk_id": "a", "path": "session.txt", "text": "first"},
        {"chunk_id": "b", "path": "session.txt", "text": "second"},
    ]
    policy = LiveBeamPolicy(
        chunks,
        [],
        {"a": (1, 0), "b": (0, 1)},
        {"session.txt": "Session 1"},
        dimensions=2,
        distinct_chunks=True,
    )
    assert len(policy.index.title_index) == 2


def test_knowledge_graph_uses_bundled_beam_without_chat_or_reranker():
    config = RaganiteConfig(
        embedding_dimensions=2,
        backend=SQLiteConfig(path=":memory:"),
    )

    class Embeddings:
        def embed_query(self, text):
            return [1.0, 0.0]

    backend = SQLiteBackend(config.backend, namespace=config.namespace, dimensions=2)
    try:
        for key, text in (("manual", "Check the pump manual."), ("pump", "The pump is ready.")):
            path = f"{key}.txt"
            backend.replace_document(
                DocumentGraph(
                    source="notes",
                    path=path,
                    content_hash=key,
                    nodes=[
                        GraphNode(
                            id=key,
                            label="Chunk",
                            kind="chunk",
                            properties={"text": text, "path": path},
                            embedding=[1.0, 0.0],
                        ),
                        GraphNode(
                            id=f"{key}-fact",
                            label="__Fact__",
                            kind="fact",
                            properties={"text": text},
                            embedding=[1.0, 0.0],
                        ),
                    ],
                    edges=[],
                )
            )
        with KnowledgeGraph(embeddings=Embeddings(), config=config, backend=backend) as graph:
            assert isinstance(graph.retriever, BeamRetriever)
            result = graph.retrieve("Which pump is ready?", top_k=2)
            assert {passage.source for passage in result} == {"manual.txt", "pump.txt"}
            assert "retrieval.rerank" not in result.usage.stages
    finally:
        backend.close()


def test_checkpoint_loader_rejects_incompatible_schema(tmp_path):
    checkpoint = tmp_path / "legacy.npz"
    np.savez(
        checkpoint,
        weights=[0.0],
        center=[0.0],
        spread=[1.0],
        feature_names=["fact_forward_member_of"],
    )
    with pytest.raises(ValueError, match="Retrain.*relation-agnostic"):
        load_checkpoint(checkpoint.read_bytes())
