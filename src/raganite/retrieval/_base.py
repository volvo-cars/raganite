#
# Copyright 2026 Volvo Car Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared search, optional reranking, and answering for the beam retriever."""

import asyncio
import contextvars
import json
import logging
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from raganite.backends.base import GraphBackend
from raganite.clients.llm import EmbeddingClient, LLMClient
from raganite.clients.llm.usage import track_run, usage_stage
from raganite.config import RetrievalConfig

from .fact_walk.graph import FactWalkGraph
from .models import Answer, FactEvidence, Passage, RetrievalResult
from .rerank import as_compressor, langchain_cross_encoder, order_by_documents

logger = logging.getLogger(__name__)

ANSWER_PROMPT = (
    "Answer the question using only the passages. "
    "If the passages do not contain the answer, say you do not know. "
    "Return only the short answer."
)

RERANK_PROMPT = (
    "Rank passages by how useful they are for answering the question. "
    'Return JSON exactly as {"indices": [1, 2]} containing every provided '
    "1-based passage index once."
)


def parse_ranked_indices(content: str, *, max_index: int, limit: int) -> list[int] | None:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip() if len(lines) >= 3 else ""
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(payload, dict):
        payload = next(
            (
                payload[k]
                for k in ("indices", "ranked_indices", "selected", "ranking")
                if k in payload
            ),
            None,
        )
    if not isinstance(payload, list):
        return None
    indices = []
    for value in payload:
        if isinstance(value, dict):
            value = value.get("index", value.get("id"))
        if isinstance(value, str) and value.isdigit():
            value = int(value)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 1 <= value <= max_index
            or value in indices
        ):
            return None
        indices.append(value)
    return indices[:limit] if indices else None


class RetrievalBase:
    """Shared retrieval operations; clients remain caller-owned."""

    def __init__(
        self,
        backend: GraphBackend,
        embeddings: EmbeddingClient | Embeddings,
        llm: LLMClient | BaseChatModel | None = None,
        config: RetrievalConfig | None = None,
        *,
        title_by_source: Mapping[str, str] | None = None,
        passage_rerank_prompt: str = RERANK_PROMPT,
        passage_reranker: Any | None = None,
    ):
        if not isinstance(backend, GraphBackend):
            raise TypeError("The backend must implement GraphBackend")
        self.backend = backend
        self.embeddings = (
            embeddings.model if isinstance(embeddings, EmbeddingClient) else embeddings
        )
        self.llm = llm.model if isinstance(llm, LLMClient) else llm
        self.config = config or RetrievalConfig()
        if self.llm is None and self.config.active_reranker == "llm":
            raise ValueError("An LLM is required for passage reranking")
        if self.config.active_reranker == "compressor" and passage_reranker is None:
            raise ValueError(
                "reranker='compressor' needs a passage reranker with compress_documents"
            )
        self._titles = dict(title_by_source or {})
        self._rerank_prompt = passage_rerank_prompt
        self._passage_reranker = as_compressor(passage_reranker)
        if self._passage_reranker is None and self.config.active_reranker == "cross-encoder":
            self._passage_reranker = as_compressor(
                langchain_cross_encoder(self.config.reranker_model, self.config.rerank_top_k)
            )
        self._fact_graph: tuple[int, FactWalkGraph] | None = None
        self._fact_graph_lock = Lock()
        self._slots = BoundedSemaphore(self.config.workers)
        self._pool = ThreadPoolExecutor(
            max_workers=self.config.workers, thread_name_prefix="raganite-retrieval"
        )
        self._search_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="raganite-search")

    def _title(self, source: str) -> str:
        return self._titles.get(source) or self._titles.get(Path(source).name, Path(source).stem)

    def _chat_json(self, system: str, user: str, *, stage: str) -> str:
        if self.llm is None:
            raise ValueError("A ranking LLM is not configured")
        with usage_stage(stage):
            response = self.llm.invoke(
                [("system", system), ("user", user)], **self.config.llm_options
            )
        return response.text.strip()

    def _rerank(
        self, query: str, passages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], str]:
        candidates = passages[: self.config.rerank_top_k]
        kind = self.config.active_reranker
        if not candidates or kind == "none":
            logger.info("Reranking disabled for %s passages", len(passages))
            return passages, "disabled"
        logger.info("Reranking %s of %s passages with %s", len(candidates), len(passages), kind)
        if kind in {"cross-encoder", "compressor"}:
            try:
                compressed = self._passage_reranker.compress(query, candidates)
                logger.info("Rerank status %s", kind)
                return order_by_documents(passages, compressed, top_k=len(candidates)), kind
            except Exception:
                logger.info("Rerank status request_error_fallback")
                logger.debug("Passage rerank failed", exc_info=True)
                return passages, "request_error_fallback"
        text = "\n\n".join(
            f"{i}. {p['title']}\n{p['text'][:4000]}" for i, p in enumerate(candidates, 1)
        )
        try:
            content = self._chat_json(
                self._rerank_prompt,
                f"Question: {query}\n\nPassages:\n{text}",
                stage="retrieval.rerank",
            )
            indices = parse_ranked_indices(
                content, max_index=len(candidates), limit=len(candidates)
            )
            if indices is None:
                logger.info("Rerank status malformed_response_fallback")
                return passages, "malformed_response_fallback"
            selected = set(indices)
            ranked = [candidates[i - 1] for i in indices]
            ranked.extend(p for i, p in enumerate(candidates, 1) if i not in selected)
            ranked.extend(passages[len(candidates) :])
            status = "llm" if len(indices) == len(candidates) else "llm_partial"
            logger.info("Rerank status %s", status)
            return ranked, status
        except Exception:
            logger.info("Rerank status request_error_fallback")
            logger.debug("Passage rerank failed", exc_info=True)
            return passages, "request_error_fallback"

    def _loaded_fact_graph(self) -> FactWalkGraph | None:
        """Read facts and chunks once. Later questions share that matrix."""
        generation = self.backend.graph_generation
        cached = self._fact_graph
        if cached is not None and cached[0] == generation:
            logger.debug(
                "Reusing %s fact edges and %s chunks",
                cached[1].edge_count,
                cached[1].chunk_count,
            )
            return cached[1]
        with self._fact_graph_lock:
            generation = self.backend.graph_generation
            cached = self._fact_graph
            if cached is not None and cached[0] == generation:
                logger.debug(
                    "Reusing %s fact edges and %s chunks",
                    cached[1].edge_count,
                    cached[1].chunk_count,
                )
                return cached[1]
            started = time.perf_counter()
            try:
                edges = self.backend.fact_edges()
                chunks = self.backend.chunk_records()
            except NotImplementedError as exc:
                logger.info("Backend does not provide a beam snapshot: %s", exc)
                return None
            graph = FactWalkGraph.from_edges(edges, chunks)
            logger.info(
                "Loaded %s fact edges and %s chunks in %.1fs",
                graph.edge_count,
                graph.chunk_count,
                time.perf_counter() - started,
            )
            if self.backend.graph_generation == generation:
                self._fact_graph = (generation, graph)
            else:
                logger.info("Fact graph changed while loading; this copy will not be reused")
            return graph

    def _beam_passages(self, query: str, embedding: list[float]) -> list[dict[str, Any]]:
        raise NotImplementedError

    def _retrieve(self, query: str) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("Query must not be empty")
        cfg = self.config
        embedding = self.embeddings.embed_query(query)
        if cfg.use_triples:
            ranked = self._beam_passages(query, embedding)
            if ranked:
                return ranked
            logger.info("Beam returned no passages; falling back to direct search")
        found = self._search_in_parallel(query, embedding)
        direct = found["direct"]
        passages: dict[str, dict[str, Any]] = {}

        def passage(chunk_id: str, source: str, text: str) -> dict[str, Any]:
            # Chunks of one file share a path. The private chunk id keeps them apart.
            return passages.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "source": source,
                    "title": self._title(source),
                    "text": text,
                    "direct_score": 0.0,
                    "lexical_score": 0.0,
                    "direct_rank": None,
                    "lexical_rank": None,
                    "graph_evidence": [],
                },
            )

        for rank, hit in enumerate(direct, 1):
            if hit.chunk_id:
                current = passage(hit.chunk_id, hit.source, hit.text)
                current["direct_score"] = max(current["direct_score"], hit.score)
                current["direct_rank"] = min(current["direct_rank"] or rank, rank)
        if cfg.use_fulltext:
            for rank, hit in enumerate(found["lexical"], 1):
                if not hit.chunk_id:
                    continue
                current = passage(hit.chunk_id, hit.source, hit.text)
                current["lexical_score"] = max(current["lexical_score"], hit.score)
                current["lexical_rank"] = min(current["lexical_rank"] or rank, rank)
        direct_order = sorted(
            (p for p in passages.values() if p["direct_rank"] is not None),
            key=lambda p: (p["direct_rank"], p["source"]),
        )
        lexical_order = sorted(
            (p for p in passages.values() if p["lexical_rank"] is not None),
            key=lambda p: (p["lexical_rank"], p["source"]),
        )
        scores: dict[str, float] = {}
        for weight, order in (
            (1.0, direct_order),
            (cfg.fulltext_weight, lexical_order),
        ):
            for rank, item in enumerate(order, 1):
                chunk_id = item["chunk_id"]
                scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (cfg.rrf_k + rank)
        for chunk_id, item in passages.items():
            item.update(
                score=scores.get(chunk_id, 0.0),
                fusion_score=scores.get(chunk_id, 0.0),
            )
        fused = sorted(
            passages.values(),
            key=lambda p: (
                -p["fusion_score"],
                p["direct_rank"] or cfg.top_k + 1,
                p["lexical_rank"] or cfg.top_k + 1,
                p["title"],
            ),
        )[: cfg.top_k]
        for rank, item in enumerate(fused, 1):
            item["fusion_rank"] = rank
        ranked, status = self._rerank(query, fused)
        for rank, item in enumerate(ranked, 1):
            item.update(final_rank=rank, passage_rerank_status=status)
        return ranked

    @staticmethod
    def _passage(row: dict[str, Any]) -> Passage:
        evidence = tuple(
            FactEvidence(
                fact=item["fact"],
                seed_entity=item["seed_entity"],
                related_entity=item["related_entity"],
                path=tuple(item.get("path") or ()),
                score=float(item.get("score") or 0.0),
                relationship_id=item.get("relationship_id", ""),
                path_degrees=tuple(item.get("path_degrees") or ()),
            )
            for item in row["graph_evidence"]
        )
        metadata = {
            key: value
            for key, value in row.items()
            if key not in {"source", "title", "text", "score", "graph_evidence"}
        }
        return Passage(row["source"], row["title"], row["text"], row["score"], evidence, metadata)

    def _answer_messages(self, question: str, passages: Sequence[Passage]) -> list[tuple[str, str]]:
        context = "\n\n".join(
            f"[{index}] {passage.title}\n{passage.text}"
            for index, passage in enumerate(passages, 1)
        )
        return [
            ("system", ANSWER_PROMPT),
            ("user", f"Passages:\n{context}\n\nQuestion: {question}"),
        ]

    def _answer_options(self) -> dict[str, Any]:
        # Ranking calls request JSON. An answer is plain text, so that format stays off.
        options = dict(self.config.llm_options)
        options.pop("response_format", None)
        return options

    def query(self, question: str, *, top_k: int | None = None) -> Answer:
        """Retrieve passages for a question, then answer from those passages."""
        if self.llm is None:
            raise ValueError("An LLM is required to answer a question")
        with track_run("query") as usage:
            passages = self.retrieve(question, top_k=top_k)
            with usage_stage("query.answer"):
                response = self.llm.invoke(
                    self._answer_messages(question, passages), **self._answer_options()
                )
            return Answer(question, response.text.strip(), tuple(passages), usage)

    async def aquery(self, question: str, *, top_k: int | None = None) -> Answer:
        """Retrieve passages without blocking, then answer from those passages."""
        if self.llm is None:
            raise ValueError("An LLM is required to answer a question")
        with track_run("query") as usage:
            passages = await self.aretrieve(question, top_k=top_k)
            with usage_stage("query.answer"):
                response = await self.llm.ainvoke(
                    self._answer_messages(question, passages), **self._answer_options()
                )
            return Answer(question, response.text.strip(), tuple(passages), usage)

    def retrieve(self, query: str, *, top_k: int | None = None) -> RetrievalResult:
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive")
        with track_run("retrieval") as usage:
            with self._slots:
                rows = self._retrieve(query)
                return RetrievalResult(
                    [self._passage(row) for row in rows[:top_k]],
                    usage,
                )

    def retrieve_many(
        self, queries: Sequence[str], *, top_k: int | None = None
    ) -> list[RetrievalResult]:
        # Each worker needs its own copy. One context cannot run on two threads.
        jobs = [(query, contextvars.copy_context()) for query in queries]

        def retrieve_one(job: tuple[str, contextvars.Context]) -> RetrievalResult:
            query, context = job
            return context.run(self.retrieve, query, top_k=top_k)

        return list(self._pool.map(retrieve_one, jobs))

    async def aretrieve(self, query: str, *, top_k: int | None = None) -> RetrievalResult:
        context = contextvars.copy_context()
        return await asyncio.get_running_loop().run_in_executor(
            self._pool,
            lambda: context.run(self.retrieve, query, top_k=top_k),
        )

    def _search_in_parallel(self, query: str, embedding: list[float]) -> dict[str, Any]:
        """Run chunk vector and full-text searches concurrently."""
        cfg = self.config
        futures: dict[str, Future[Any]] = {
            "direct": self._search_pool.submit(
                self.backend.direct_passages, embedding, top_k=cfg.top_k
            )
        }
        if cfg.use_fulltext:
            futures["lexical"] = self._search_pool.submit(
                self.backend.lexical_passages, query, top_k=cfg.top_k
            )
        try:
            return {name: future.result() for name, future in futures.items()}
        except Exception:
            for future in futures.values():
                future.cancel()
            raise

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        self._search_pool.shutdown(wait=True)
