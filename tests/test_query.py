"""Query retrieves passages and answers from those passages."""

import asyncio
import threading

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from raganite.backends.base import DocumentGraph, GraphBackend, GraphNode
from raganite.backends.models import PassageHit
from raganite.backends.sqlite import SQLiteBackend
from raganite.config import RaganiteConfig, RetrievalConfig, SQLiteConfig
from raganite.graph import KnowledgeGraph
from raganite.retrieval import Retriever

_USAGE = {
    "input_tokens": 8,
    "output_tokens": 10,
    "total_tokens": 18,
    "input_token_details": {"audio": 0, "cache_read": 0},
    "output_token_details": {"audio": 0, "reasoning": 0},
}


class FixedEmbeddings(Embeddings):
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


class AnsweringChat(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "answering"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self._seen = messages[-1].content
        self._options = kwargs
        message = AIMessage(
            content="Paris",
            usage_metadata=_USAGE,
            response_metadata={"model_name": "answering"},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def _backend() -> SQLiteBackend:
    backend = SQLiteBackend(SQLiteConfig(path=":memory:"), namespace="demo", dimensions=2)
    backend.replace_document(
        DocumentGraph(
            source="docs",
            path="a.md",
            content_hash="h",
            nodes=[
                GraphNode(
                    id="chunk",
                    label="Chunk",
                    kind="chunk",
                    properties={"text": "Paris is the capital of France", "path": "a.md"},
                    embedding=[1.0, 0.0],
                )
            ],
            edges=[],
        )
    )
    return backend


def test_chunks_of_one_file_stay_separate_passages():
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
                    properties={
                        "text": "The boy plays with a ball.",
                        "path": "story.txt",
                        "source": "docs",
                    },
                    embedding=[1.0, 0.0],
                ),
                GraphNode(
                    id="chunk-door",
                    label="Chunk",
                    kind="chunk",
                    properties={
                        "text": "The boy opens a door.",
                        "path": "story.txt",
                        "source": "docs",
                    },
                    embedding=[0.0, 1.0],
                ),
            ],
            edges=[],
        )
    )
    retriever = Retriever(
        backend,
        FixedEmbeddings(),
        config=RetrievalConfig(
            use_triples=False, filter_triples=False, rerank_passages=False, workers=1, top_k=5
        ),
    )
    try:
        passages = retriever.retrieve("What does the boy do?")
        assert [passage.text for passage in passages] == [
            "The boy plays with a ball.",
            "The boy opens a door.",
        ]
        assert {passage.source for passage in passages} == {"story.txt"}
    finally:
        retriever.close()
        backend.close()


def test_query_answers_from_the_retrieved_passages():
    backend = _backend()
    model = AnsweringChat()
    retriever = Retriever(
        backend,
        FixedEmbeddings(),
        model,
        RetrievalConfig(use_triples=False, filter_triples=False, rerank_passages=False, workers=1),
    )
    try:
        result = retriever.query("What is the capital of France?")
        assert result.answer == "Paris"
        assert result.passages[0].source == "a.md"
        assert "Paris is the capital of France" in model._seen
        assert "response_format" not in model._options
        assert result.usage.stages["query.answer"].total_tokens == 18
    finally:
        retriever.close()
        backend.close()


def test_graph_query_uses_the_same_answer():
    backend = _backend()
    config = RaganiteConfig(
        backend=SQLiteConfig(path=":memory:"),
        namespace="demo",
        embedding_dimensions=2,
        retrieval=RetrievalConfig(
            use_triples=False, filter_triples=False, rerank_passages=False, workers=1
        ),
    )
    graph = KnowledgeGraph(
        embeddings=FixedEmbeddings(), llm=AnsweringChat(), backend=backend, config=config
    )
    try:
        result = graph.query("What is the capital of France?")
        assert result.answer == "Paris"
        assert result.passages[0].text == "Paris is the capital of France"
    finally:
        graph.close()


@pytest.mark.asyncio
async def test_aquery_answers_from_the_retrieved_passages():
    backend = _backend()
    retriever = Retriever(
        backend,
        FixedEmbeddings(),
        AnsweringChat(),
        RetrievalConfig(use_triples=False, filter_triples=False, rerank_passages=False, workers=1),
    )
    try:
        result = await retriever.aquery("What is the capital of France?")
        assert result.answer == "Paris"
        assert result.passages[0].source == "a.md"
    finally:
        retriever.close()
        backend.close()


@pytest.mark.asyncio
async def test_concurrent_aqueries_include_retrieval_usage():
    backend = _backend()
    retriever = Retriever(
        backend,
        FixedEmbeddings(),
        AnsweringChat(),
        RetrievalConfig(use_triples=False, filter_triples=False, rerank_passages=True, workers=2),
    )
    try:
        results = await asyncio.gather(
            retriever.aquery("What is the capital of France?"),
            retriever.aquery("Which city is the capital of France?"),
        )
        assert results[0].usage is not results[1].usage
        for result in results:
            assert result.answer == "Paris"
            assert result.usage.stages["retrieval.rerank"].total_tokens == 18
            assert result.usage.stages["retrieval.rerank"].calls == 1
            assert result.usage.stages["query.answer"].total_tokens == 18
            assert result.usage.stages["query.answer"].calls == 1
            assert result.usage.total_tokens == 36
    finally:
        retriever.close()
        backend.close()


def test_query_text_output_prints_only_the_answer(capsys):
    from raganite.cli import _emit

    _emit(
        {
            "answer": "Paris",
            "passages": [{"title": "notes", "source": "paris.txt", "score": 1.0, "text": "Paris is the capital."}],
        },
        "text",
    )
    assert capsys.readouterr().out == "Paris\n"


def test_query_help_describes_answering():
    from click.testing import CliRunner

    from raganite.cli import main

    result = CliRunner().invoke(main, ["query", "--help"])
    assert result.exit_code == 0
    assert "answer" in result.output.lower()


def test_default_retrieval_skips_the_chat_ranking_steps():
    config = RetrievalConfig()
    assert config.filter_triples is False
    assert config.rerank_passages is False
    assert config.use_triples is True
    assert config.triple_top_k == 10


class _ParallelSearches(GraphBackend):
    """Block each search until the other two have started."""

    def __init__(self):
        super().__init__(namespace="demo", dimensions=2)
        self._gate = threading.Barrier(3)
        self.fact_top_k = None

    def _arrive(self) -> None:
        self._gate.wait(timeout=2)

    def direct_passages(self, embedding, *, top_k):
        self._arrive()
        return [PassageHit("chunk", "notes.txt", "Paris is the capital.", 1.0)]

    def lexical_passages(self, query, *, top_k):
        self._arrive()
        return []

    def search_facts(self, embedding, *, top_k=10):
        self.fact_top_k = top_k
        self._arrive()
        return []

    def expand_facts(self, facts, *, depth=2, limit=5000, degree_penalty_weight=0.25):
        return []

    def ensure_indexes(self) -> None:
        return None

    def prepare(self) -> None:
        return None

    def replace_document(self, graph):
        raise NotImplementedError

    def store_shared_edges(self, *args, **kwargs):
        raise NotImplementedError

    def store_shared_nodes(self, *args, **kwargs):
        raise NotImplementedError

    def has_sequence(self, source: str) -> bool:
        return False

    def delete_sequence(self, source: str) -> None:
        return None

    def document_hash(self, source: str, path: str):
        return None

    def delete_documents(self, source=None, *, keep_paths=()):
        return 0

    def nodes(self, **kwargs):
        return []

    def edges(self, **kwargs):
        return []

    def vector_search(self, *args, **kwargs):
        return []

    def fulltext_search(self, *args, **kwargs):
        return []

    def close(self) -> None:
        return None


def test_the_three_searches_run_together_and_fewer_facts_start_the_walk():
    backend = _ParallelSearches()
    retriever = Retriever(
        backend,
        FixedEmbeddings(),
        None,
        RetrievalConfig(filter_triples=False, rerank_passages=False, workers=1),
    )
    try:
        passages = retriever.retrieve("What is the capital of France?")
    finally:
        retriever.close()
    assert passages[0].text == "Paris is the capital."
    assert backend.fact_top_k == 10


def test_text_output_names_a_ranking_step_that_ran():
    from raganite.cli import _ranking_note

    assert _ranking_note([]) is None
    assert (
        _ranking_note(
            [
                {
                    "metadata": {
                        "triple_filter_status": "disabled",
                        "passage_rerank_status": "llm",
                    }
                }
            ]
        )
        == "passage rerank: llm"
    )
    assert (
        _ranking_note([{"metadata": {"triple_filter_status": "disabled"}}]) is None
    )


def test_query_requires_an_llm():
    backend = _backend()
    retriever = Retriever(
        backend,
        FixedEmbeddings(),
        None,
        RetrievalConfig(use_triples=False, filter_triples=False, rerank_passages=False, workers=1),
    )
    try:
        with pytest.raises(ValueError, match="LLM"):
            retriever.query("What is the capital of France?")
    finally:
        retriever.close()
        backend.close()
