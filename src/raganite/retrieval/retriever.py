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

"""Filter facts, traverse semantic paths, fuse passages, and rerank with an LLM."""

import asyncio
import json
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import BoundedSemaphore
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from raganite.backends.base import GraphBackend
from raganite.backends.models import FactHit
from raganite.clients.llm import EmbeddingClient, LLMClient
from raganite.config import RetrievalConfig

from .models import FactEvidence, Passage

FILTER_PROMPT = (
    "Rank only the triples that are useful for answering the question. "
    'Return JSON exactly as {"indices": [1, 2]} with at most the requested '
    "number of 1-based indices."
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


class Retriever:
    """Reference-compatible passage ranking; answer generation stays separate.

    All query state is local and clients remain caller-owned. The backend does
    indexed fact lookup and scored expansion, never full-graph Python loading.
    Provider options/retries are delegated to native LangChain model calls.
    """

    def __init__(
        self,
        backend: GraphBackend,
        embeddings: EmbeddingClient | Embeddings,
        llm: LLMClient | BaseChatModel | None = None,
        config: RetrievalConfig | None = None,
        *,
        title_by_source: Mapping[str, str] | None = None,
        triple_filter_prompt: str = FILTER_PROMPT,
        passage_rerank_prompt: str = RERANK_PROMPT,
    ):
        if not isinstance(backend, GraphBackend):
            raise TypeError("The backend must implement GraphBackend")
        self.backend = backend
        self.embeddings = (
            embeddings.model if isinstance(embeddings, EmbeddingClient) else embeddings
        )
        self.llm = llm.model if isinstance(llm, LLMClient) else llm
        self.config = config or RetrievalConfig()
        if self.llm is None and (
            self.config.rerank_passages or (self.config.use_triples and self.config.filter_triples)
        ):
            raise ValueError("An LLM is required for triple filtering and passage reranking")
        self._titles = dict(title_by_source or {})
        self._filter_prompt = triple_filter_prompt
        self._rerank_prompt = passage_rerank_prompt
        if self.config.use_triples:
            self.backend.prepare()
        self._slots = BoundedSemaphore(self.config.workers)
        self._pool = ThreadPoolExecutor(
            max_workers=self.config.workers, thread_name_prefix="raganite-retrieval"
        )

    def _title(self, source: str) -> str:
        return self._titles.get(source) or self._titles.get(Path(source).name, Path(source).stem)

    def _chat_json(self, system: str, user: str) -> str:
        if self.llm is None:
            raise ValueError("A ranking LLM is not configured")
        response = self.llm.invoke([("system", system), ("user", user)], **self.config.llm_options)
        return response.text.strip()

    def _select_facts(self, query: str, facts: list[FactHit]) -> tuple[list[FactHit], str]:
        fallback = facts[: self.config.filtered_triple_k]
        if not facts or not self.config.filter_triples:
            return fallback, "disabled"
        candidates = "\n".join(f"{i}. {fact.text}" for i, fact in enumerate(facts, 1))
        try:
            content = self._chat_json(
                self._filter_prompt,
                f"Question: {query}\n\nCandidate triples:\n{candidates}\n\nSelect and rank at most {self.config.filtered_triple_k}.",
            )
            indices = parse_ranked_indices(
                content, max_index=len(facts), limit=self.config.filtered_triple_k
            )
            if indices is None:
                return fallback, "malformed_response_fallback"
            return [facts[i - 1] for i in indices], "llm"
        except Exception:
            return fallback, "request_error_fallback"

    def _rerank(
        self, query: str, passages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], str]:
        candidates = passages[: self.config.rerank_top_k]
        if not candidates or not self.config.rerank_passages:
            return passages, "disabled"
        text = "\n\n".join(
            f"{i}. {p['title']}\n{p['text'][:4000]}" for i, p in enumerate(candidates, 1)
        )
        try:
            content = self._chat_json(
                self._rerank_prompt, f"Question: {query}\n\nPassages:\n{text}"
            )
            indices = parse_ranked_indices(
                content, max_index=len(candidates), limit=len(candidates)
            )
            if indices is None:
                return passages, "malformed_response_fallback"
            selected = set(indices)
            ranked = [candidates[i - 1] for i in indices]
            ranked.extend(p for i, p in enumerate(candidates, 1) if i not in selected)
            ranked.extend(passages[len(candidates) :])
            return ranked, "llm" if len(indices) == len(candidates) else "llm_partial"
        except Exception:
            return passages, "request_error_fallback"

    def _retrieve(self, query: str) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("Query must not be empty")
        cfg = self.config
        embedding = self.embeddings.embed_query(query)
        direct = self.backend.direct_passages(embedding, top_k=cfg.top_k)
        passages: dict[str, dict[str, Any]] = {}

        def passage(source: str, text: str) -> dict[str, Any]:
            return passages.setdefault(
                source,
                {
                    "source": source,
                    "title": self._title(source),
                    "text": text,
                    "direct_score": 0.0,
                    "graph_score": 0.0,
                    "direct_rank": None,
                    "graph_rank": None,
                    "graph_evidence": [],
                },
            )

        for rank, hit in enumerate(direct, 1):
            if hit.source:
                current = passage(hit.source, hit.text)
                current["direct_score"] = max(current["direct_score"], hit.score)
                current["direct_rank"] = min(current["direct_rank"] or rank, rank)
        filter_status = "disabled"
        expanded = []
        if cfg.use_triples:
            facts = self.backend.search_facts(embedding, top_k=cfg.triple_top_k)
            selected, filter_status = self._select_facts(query, facts)
            expanded = self.backend.expand_facts(
                selected,
                depth=cfg.expansion_depth,
                limit=cfg.expansion_limit,
                degree_penalty_weight=cfg.degree_penalty_weight,
            )
        seen = set()
        for row in expanded:
            if not row.source:
                continue
            current = passage(row.source, row.text)
            key = (
                row.source,
                row.relationship_id,
                row.seed_entity,
                row.related_entity,
                tuple(row.graph_path),
            )
            if key in seen:
                continue
            seen.add(key)
            current["graph_score"] += row.expansion_score
            current["graph_evidence"].append(
                {
                    "relationship_id": row.relationship_id,
                    "fact": row.fact_text,
                    "seed_entity": row.seed_entity,
                    "related_entity": row.related_entity,
                    "path": row.graph_path,
                    "path_degrees": row.path_degrees,
                    "score": row.expansion_score,
                }
            )
        direct_order = sorted(
            (p for p in passages.values() if p["direct_rank"] is not None),
            key=lambda p: (p["direct_rank"], p["source"]),
        )
        graph_order = sorted(
            (p for p in passages.values() if p["graph_score"] > 0),
            key=lambda p: (-p["graph_score"], p["source"]),
        )
        scores: dict[str, float] = {}
        for weight, order in ((1.0, direct_order), (cfg.graph_weight, graph_order)):
            for rank, item in enumerate(order, 1):
                source = item["source"]
                scores[source] = scores.get(source, 0.0) + weight / (cfg.rrf_k + rank)
        for rank, item in enumerate(graph_order, 1):
            item["graph_rank"] = rank
        for source, item in passages.items():
            item.update(
                score=scores.get(source, 0.0),
                fusion_score=scores.get(source, 0.0),
                triple_filter_status=filter_status,
            )
        fused = sorted(
            passages.values(),
            key=lambda p: (
                -p["fusion_score"],
                p["direct_rank"] or cfg.top_k + 1,
                p["graph_rank"] or cfg.expansion_limit + 1,
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

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[Passage]:
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive")
        with self._slots:
            rows = self._retrieve(query)
            return [self._passage(row) for row in rows[:top_k]]

    def retrieve_many(
        self, queries: Sequence[str], *, top_k: int | None = None
    ) -> list[list[Passage]]:
        return list(self._pool.map(lambda query: self.retrieve(query, top_k=top_k), queries))

    async def aretrieve(self, query: str, *, top_k: int | None = None) -> list[Passage]:
        return await asyncio.get_running_loop().run_in_executor(
            self._pool,
            lambda: self.retrieve(query, top_k=top_k),
        )

    def close(self) -> None:
        self._pool.shutdown(wait=True)
