"""Run hooks record stage time and token usage from real model calls."""

import asyncio
import json
import time
from pathlib import Path

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from raganite.backends.base import DocumentGraph, GraphNode
from raganite.backends.sqlite import SQLiteBackend
from raganite.clients.llm.usage import (
    register_run_hook,
    track_run,
    unregister_run_hook,
    usage_stage,
)
from raganite.config import RetrievalConfig, SQLiteConfig
from raganite.retrieval import Retriever

_USAGE = {
    "input_tokens": 8,
    "output_tokens": 10,
    "total_tokens": 18,
    "input_token_details": {"audio": 0, "cache_read": 0},
    "output_token_details": {"audio": 0, "reasoning": 0},
}


class ScriptedChat(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        message = AIMessage(
            content="ok", usage_metadata=_USAGE, response_metadata={"model_name": "scripted"}
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def test_registered_hook_totals_time_and_tokens_for_each_stage():
    seen: list[str] = []

    class Recording(BaseCallbackHandler):
        def __init__(self, report):
            super().__init__()
            self.report = report

        def on_llm_end(self, response, **kwargs):
            seen.append(self.report.name)

    def factory(report):
        return Recording(report)

    register_run_hook(factory)
    model = ScriptedChat()
    try:
        with track_run("benchmark") as report:
            with usage_stage("benchmark.answer"):
                time.sleep(0.02)
                model.invoke("Hello")
            with usage_stage("benchmark.judge"):
                model.invoke("Judge")
            with track_run("retrieval"):
                with usage_stage("retrieval.filter"):
                    model.invoke("Filter")
    finally:
        unregister_run_hook(factory)

    assert seen == ["benchmark", "benchmark", "benchmark"]
    answer = report.stages["benchmark.answer"]
    judge = report.stages["benchmark.judge"]
    filtered = report.stages["retrieval.filter"]
    assert answer.calls == 1
    assert answer.input_tokens == 8
    assert answer.output_tokens == 10
    assert answer.total_tokens == 18
    assert answer.input_token_details == {"audio": 0, "cache_read": 0}
    assert answer.output_token_details == {"audio": 0, "reasoning": 0}
    assert answer.elapsed_seconds >= 0.02
    assert answer.model_seconds >= 0
    assert judge.calls == 1
    assert filtered.calls == 1
    assert report.total_tokens == 54
    assert report.elapsed_seconds >= answer.elapsed_seconds
    outside = ScriptedChat()
    before = report.total_tokens
    outside.invoke("later")
    assert report.total_tokens == before


class FixedEmbeddings(Embeddings):
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


def test_retrieval_result_reports_rerank_token_usage():
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
    retriever = Retriever(
        backend,
        FixedEmbeddings(),
        ScriptedChat(),
        RetrievalConfig(use_triples=False, filter_triples=False, rerank_passages=True, workers=1),
    )
    try:
        result = retriever.retrieve("What is the capital of France?")
        assert result[0].source == "a.md"
        stage = result.usage.stages["retrieval.rerank"]
        assert stage.calls == 1
        assert stage.total_tokens == 18
        assert result.usage.elapsed_seconds >= 0
    finally:
        retriever.close()
        backend.close()


@pytest.mark.asyncio
async def test_benchmark_row_includes_stage_usage(tmp_path: Path):
    from benchmark.locomo_flow import _answer_one

    class Passages:
        async def aretrieve(self, question: str):
            with usage_stage("retrieval.rerank"):
                await ScriptedChat().ainvoke(question)
            return []

    output = tmp_path / "rows.jsonl"
    await _answer_one(
        "sample",
        0,
        {"question": "Where?", "answer": "Paris", "category": 2, "evidence": []},
        Passages(),
        answer_model=ScriptedChat(),
        output=output,
        lock=asyncio.Lock(),
        done=set(),
        llm_judge=False,
    )
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["usage"]["stages"]["retrieval.rerank"]["calls"] == 1
    assert row["usage"]["stages"]["benchmark.answer"]["total_tokens"] == 18
    assert row["usage"]["total_tokens"] == 36
