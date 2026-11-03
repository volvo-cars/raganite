"""A caller can leave passages in fusion order, ask the chat model, or use a cross-encoder."""

import logging
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from raganite.backends.base import DocumentGraph, GraphNode
from raganite.backends.sqlite import SQLiteBackend
from raganite.config import RetrievalConfig, SQLiteConfig
from raganite.retrieval import Retriever
from raganite.retrieval.rerank import langchain_cross_encoder, order_by_documents


class _Embeddings(Embeddings):
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


class _PreferDoor:
    """A LangChain compressor: local and API rerankers both implement this method."""

    def compress_documents(self, documents, query):
        return sorted(documents, key=lambda doc: 0 if "door" in doc.page_content else 1)


def test_the_reranker_name_selects_the_step():
    assert RetrievalConfig().active_reranker == "none"
    assert RetrievalConfig(rerank_passages=True).active_reranker == "llm"
    assert (
        RetrievalConfig(rerank_passages=True, reranker="cross-encoder").active_reranker
        == "cross-encoder"
    )
    assert RetrievalConfig(reranker="compressor").active_reranker == "compressor"


def test_a_compressor_reorders_the_window_and_keeps_the_rest():
    passages = [{"text": "first"}, {"text": "second"}, {"text": "tail"}]
    compressed = [
        Document(page_content="second", metadata={"raganite_index": 1}),
        Document(page_content="first", metadata={"raganite_index": 0}),
    ]
    ordered = order_by_documents(passages, compressed, top_k=2)
    assert [item["text"] for item in ordered] == ["second", "first", "tail"]


def test_a_compressor_reranker_needs_an_object():
    backend = SQLiteBackend(SQLiteConfig(path=":memory:"), namespace="demo", dimensions=2)
    try:
        Retriever(
            backend,
            _Embeddings(),
            None,
            RetrievalConfig(use_triples=False, reranker="compressor", workers=1),
        )
    except ValueError as exc:
        assert "compress_documents" in str(exc)
    else:
        raise AssertionError("A missing compressor should be rejected")
    finally:
        backend.close()


def test_a_cross_encoder_reorders_without_a_chat_model():
    backend = SQLiteBackend(SQLiteConfig(path=":memory:"), namespace="demo", dimensions=2)
    backend.replace_document(
        DocumentGraph(
            source="docs",
            path="story.txt",
            content_hash="h",
            nodes=[
                GraphNode(
                    id="chunk-ball",
                    label="Chunk",
                    kind="chunk",
                    properties={"text": "The boy plays with a ball.", "path": "story.txt"},
                    embedding=[1.0, 0.0],
                ),
                GraphNode(
                    id="chunk-door",
                    label="Chunk",
                    kind="chunk",
                    properties={"text": "The boy opens a door.", "path": "story.txt"},
                    embedding=[0.2, 0.0],
                ),
            ],
            edges=[],
        )
    )
    retriever = Retriever(
        backend,
        _Embeddings(),
        None,
        RetrievalConfig(
            use_triples=False,
            use_fulltext=False,
            reranker="compressor",
            rerank_top_k=5,
            top_k=5,
            workers=1,
        ),
        passage_reranker=_PreferDoor(),
    )
    try:
        passages = retriever.retrieve("What does the boy open?")
        assert "door" in passages[0].text
        assert passages[0].metadata["passage_rerank_status"] == "compressor"
    finally:
        retriever.close()
        backend.close()


def test_loading_a_cross_encoder_is_logged_without_verbose_mode(caplog):
    with (
        caplog.at_level(logging.WARNING, logger="raganite.retrieval.rerank"),
        patch(
            "langchain_community.cross_encoders.HuggingFaceCrossEncoder",
            return_value="model",
        ) as model,
        patch(
            "langchain_classic.retrievers.document_compressors.CrossEncoderReranker",
            return_value="reranker",
        ) as built,
    ):
        result = langchain_cross_encoder("cross-encoder/example", 4)
    assert result == "reranker"
    model.assert_called_once_with(model_name="cross-encoder/example")
    built.assert_called_once_with(model="model", top_n=4)
    assert "Loading cross-encoder cross-encoder/example" in caplog.text
    assert "Loaded cross-encoder cross-encoder/example" in caplog.text


def test_a_failed_cross_encoder_download_is_logged(caplog):
    with (
        caplog.at_level(logging.WARNING, logger="raganite.retrieval.rerank"),
        patch(
            "langchain_community.cross_encoders.HuggingFaceCrossEncoder",
            side_effect=OSError("certificate verify failed"),
        ),
        pytest.raises(OSError, match="certificate verify failed"),
    ):
        langchain_cross_encoder("cross-encoder/example", 4)
    assert "Loading cross-encoder cross-encoder/example failed" in caplog.text
    assert "certificate verify failed" in caplog.text
