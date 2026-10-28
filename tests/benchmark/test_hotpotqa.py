"""Offline regression checks for the evaluation-only HotpotQA harness."""

from unittest.mock import Mock

import pytest
from benchmark.metrics import answer_metrics, compare, normalize, retrieval_metrics
from benchmark.results import passages_from_rows
from langchain_core.embeddings import Embeddings

from raganite.backends import GraphBackend, Neo4jBackend
from raganite.backends.base import DocumentGraph
from raganite.backends.models import PassageHit
from raganite.config import Neo4jConfig, RetrievalConfig
from raganite.retrieval import Retriever


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The Eiffel Tower!", "eiffel tower"),
        ("An A.B. example", "ab example"),
        ("YES", "yes"),
        (" a  bridge ", "bridge"),
    ],
)
def test_official_normalization(text, expected):
    assert normalize(text) == expected


def test_answer_scoring_matches_original_conventions():
    assert answer_metrics("The Eiffel Tower.", ["Eiffel Tower"])["exact_match"] == 1
    assert answer_metrics("yes", ["yes indeed"])["f1"] == 0
    assert answer_metrics("red blue", ["red green"])["f1"] == 0.5
    assert answer_metrics("second", ["first", "second"])["f1"] == 1
    assert answer_metrics("", [""])["exact_match"] == 1
    assert answer_metrics("", [""])["f1"] == 0  # Matches the original scorer.


def test_retrieval_scoring_uses_distinct_support_titles():
    result = retrieval_metrics(["The Alpha", "Alpha", "Beta"], ["Alpha", "Beta"])
    assert result["recall@1"] == 0.5
    assert result["all_support@2"] == 0
    assert result["recall@5"] == result["all_support@5"] == 1


def sample_result(qid="one"):
    return {
        "id": qid,
        "question": "Question?",
        "gold_answers": ["yes"],
        "gold_titles": ["Alpha"],
        "answer": "yes",
        "retrieved": [{"title": "Alpha"}],
        "answer_metrics": answer_metrics("yes", ["yes"]),
        "retrieval_metrics": retrieval_metrics(["Alpha"], ["Alpha"]),
    }


def test_paired_comparison_matches_ids_not_result_order():
    result = compare(
        [sample_result("two"), sample_result("one")], [sample_result("one"), sample_result("two")]
    )
    assert result["questions_compared"] == 2
    assert result["identical_answer_count"] == 2
    assert all(value == 0 for value in result["delta"]["qa"].values())
    with pytest.raises(ValueError):
        compare([sample_result("missing")], [sample_result()])
    changed = {**sample_result(), "gold_answers": ["no"]}
    with pytest.raises(ValueError):
        compare([changed], [sample_result()])
    with pytest.raises(ValueError):
        compare([sample_result(), sample_result()], [sample_result()])


def test_passage_mapping_deduplicates_chunks_but_never_invents_titles():
    row = {"node_id": "chunk", "source": "/corpus/doc_00001.txt", "text": "Evidence", "score": 1}
    passages = passages_from_rows([row, row], {"doc_00001.txt": "Alpha"}, 200)
    assert len(passages) == 1
    assert passages[0]["title"] == "Alpha"
    with pytest.raises(ValueError):
        passages_from_rows([row], {}, 200)


def test_existing_backend_forces_read_only_and_rejects_writes():
    driver = Mock()
    backend = Neo4jBackend(
        Neo4jConfig(read_only=False),
        dimensions=2,
        driver=driver,
        layout="native",
    )
    assert backend.read_only is True
    with pytest.raises(PermissionError):
        backend.replace_document(
            DocumentGraph(source="s", path="p", content_hash="h", nodes=[], edges=[])
        )
    with pytest.raises(PermissionError):
        backend.delete_documents()
    driver.session.assert_not_called()
    backend.close()
    driver.close.assert_not_called()  # Injected resource remains caller-owned.


def test_existing_backend_executes_reads_and_preserves_provenance():
    driver = Mock()
    backend = Neo4jBackend(Neo4jConfig(), dimensions=2, driver=driver, layout="native")
    row = {
        "id": "native-id",
        "labels": ["Chunk"],
        "properties": {"text": "Evidence", "embedding": [1.0, 0.0]},
        "document_path": "doc_00001.txt",
        "score": 1.0,
    }
    backend._read = Mock(return_value=[row])
    hit = backend.vector_search([1.0, 0.0])[0]
    assert hit.node.id == "native-id"
    assert hit.node.properties["path"] == "doc_00001.txt"
    assert "embedding" not in hit.node.properties
    assert backend._read.call_args.kwargs["index"] == backend.config.vector_index
    with pytest.raises(ValueError):
        backend.vector_search([1.0, 0.0], source="tenant")
    with pytest.raises(ValueError):
        backend.vector_search([1.0])


def test_direct_retrieval_does_not_query_graph_edges():
    backend, embeddings = Mock(spec=GraphBackend), Mock(spec=Embeddings)
    embeddings.embed_query.return_value = [1.0, 0.0]
    backend.direct_passages.return_value = [PassageHit("c", "doc.txt", "Evidence", 1)]
    retriever = Retriever(
        backend,
        embeddings,
        config=RetrievalConfig(
            workers=1,
            use_triples=False,
            filter_triples=False,
            rerank_passages=False,
        ),
    )
    try:
        assert retriever.retrieve("question")[0].source == "doc.txt"
        backend.edges.assert_not_called()
        backend.nodes.assert_not_called()
        backend.search_facts.assert_not_called()
        backend.expand_facts.assert_not_called()
    finally:
        retriever.close()
