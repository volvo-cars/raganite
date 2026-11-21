"""Live two-hop beam retrieval."""

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Any

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from raganite.backends.base import GraphBackend
from raganite.clients.llm import EmbeddingClient, LLMClient
from raganite.config import RetrievalConfig

from .._base import RERANK_PROMPT, RetrievalBase
from ..fact_walk.graph import FactLink, FactWalkGraph
from .candidates import BeamCandidates, CandidateIndex
from .policy import load_default_checkpoint, select_passages
from .transitions import TransitionIndex

logger = logging.getLogger(__name__)


class LiveBeamPolicy:
    """Passage embeddings and feature indexes for one graph snapshot."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        links: Sequence[FactLink],
        vectors: Mapping[str, Sequence[float]],
        titles: Mapping[str, str],
        *,
        dimensions: int,
        distinct_chunks: bool = False,
    ):
        passage_titles = [titles.get(chunk["path"], Path(chunk["path"]).stem) for chunk in chunks]
        counts = Counter(passage_titles)
        self.graph = SimpleNamespace(
            chunks=[
                {
                    **chunk,
                    "title": f"{title} (chunk {i + 1})"
                    if distinct_chunks and counts[title] > 1
                    else title,
                }
                for i, (chunk, title) in enumerate(zip(chunks, passage_titles, strict=True))
            ],
        )
        self.passages = np.zeros((len(chunks), dimensions), dtype=np.float32)
        for i, chunk in enumerate(chunks):
            vector = vectors.get(chunk["chunk_id"])
            if vector is not None:
                self.passages[i] = np.asarray(vector, dtype=np.float32)
        norms = np.linalg.norm(self.passages, axis=1, keepdims=True)
        missing = int(np.count_nonzero(norms == 0))
        if missing:
            logger.warning("%s/%s graph passages have no nonzero embedding", missing, len(chunks))
        self.passages /= np.where(norms > 0, norms, 1)
        self.index = CandidateIndex(self.graph.chunks)
        self.transitions = TransitionIndex(self.graph.chunks, [link._asdict() for link in links])

    def prepare(
        self,
        graph: FactWalkGraph,
        query: str,
        vector: Sequence[float],
        config: RetrievalConfig,
        *,
        start_candidates: int,
        bridge_candidates: int,
    ) -> BeamCandidates:
        unit = np.asarray(vector, dtype=np.float32)
        if unit.shape != (self.passages.shape[1],):
            raise ValueError("Passage embedding width does not match the question")
        norm = np.linalg.norm(unit)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("Beam retrieval requires a finite nonzero query embedding")
        return self.index.prepare(
            graph,
            self.passages,
            query,
            unit / norm,
            start_limit=start_candidates,
            bridge_limit=bridge_candidates,
            seed_limit=config.fact_seed_count,
            seed_weight=config.passage_seed_weight,
            tau=config.fact_temperature,
            damping=config.fact_damping,
        )


class Retriever(RetrievalBase):
    """Two-hop retriever using a relation-agnostic checkpoint."""

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
        distinct_chunks: bool = True,
        beam_limit: int | None = None,
        beam_width: int | None = None,
        start_candidates: int | None = None,
        bridge_candidates: int | None = None,
    ):
        if beam_limit is not None and beam_limit < 1:
            raise ValueError("Beam passage limit must be positive")
        config = config or RetrievalConfig()
        self._beam_width = config.beam_width if beam_width is None else beam_width
        self._start_candidates = (
            config.beam_start_candidates if start_candidates is None else start_candidates
        )
        self._bridge_candidates = (
            config.beam_bridge_candidates if bridge_candidates is None else bridge_candidates
        )
        if min(self._beam_width, self._start_candidates) < 1 or self._bridge_candidates < 0:
            raise ValueError(
                "Beam width and start candidates must be positive; bridge count cannot be negative"
            )
        self._beam_snapshot: tuple[FactWalkGraph, LiveBeamPolicy] | None = None
        self._beam_lock = Lock()
        self._distinct_chunks = distinct_chunks
        self._beam_limit = config.top_k if beam_limit is None else beam_limit
        super().__init__(
            backend,
            embeddings,
            llm,
            config,
            title_by_source=title_by_source,
            passage_rerank_prompt=passage_rerank_prompt,
            passage_reranker=passage_reranker,
        )

    def _loaded_beam(self) -> tuple[FactWalkGraph, LiveBeamPolicy] | None:
        graph = self._loaded_fact_graph()
        if graph is None:
            return None
        cached = self._beam_snapshot
        if cached is not None and cached[0] is graph:
            return cached
        with self._beam_lock:
            graph = self._loaded_fact_graph()
            if graph is None:
                return None
            cached = self._beam_snapshot
            if cached is not None and cached[0] is graph:
                return cached
            if not graph.chunks:
                return None
            ids = list(graph.chunk_by_id)
            vectors: dict[str, Sequence[float]] = {}
            try:
                for start in range(0, len(ids), 200):
                    vectors.update(self.backend.chunk_embeddings(ids[start : start + 200]))
            except NotImplementedError as exc:
                logger.info("Backend does not provide snapshot embeddings: %s", exc)
                return None
            if not vectors:
                return None
            policy = LiveBeamPolicy(
                graph.chunks,
                graph.links,
                vectors,
                {
                    chunk["path"]: self._titles.get(chunk["path"])
                    or chunk.get("title")
                    or self._title(chunk["path"])
                    for chunk in graph.chunks
                },
                dimensions=len(next(iter(vectors.values()))),
                distinct_chunks=self._distinct_chunks,
            )
            self._beam_snapshot = (graph, policy)
            return self._beam_snapshot

    def _beam_passages(self, query: str, embedding: list[float]) -> list[dict[str, Any]]:
        snapshot = self._loaded_beam()
        if snapshot is None:
            return []
        graph, policy = snapshot
        candidates = policy.prepare(
            graph,
            query,
            embedding,
            self.config,
            start_candidates=self._start_candidates,
            bridge_candidates=self._bridge_candidates,
        )
        selected = select_passages(
            load_default_checkpoint(),
            candidates,
            policy.index,
            policy.transitions,
            policy.passages,
            query,
            width=self._beam_width,
            limit=min(self._beam_limit, self.config.top_k),
        )
        rows = []
        for rank, candidate in enumerate(selected, 1):
            chunk = policy.graph.chunks[candidate]
            rows.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk["path"],
                    "title": chunk["title"],
                    "text": chunk["text"],
                    "score": 1.0 / rank,
                    "final_rank": rank,
                    "graph_evidence": [],
                    "beam_start_candidate_count": candidates.base_count,
                    "beam_bridge_candidate_count": len(candidates.indexes) - candidates.base_count,
                    "beam_candidate_count": len(candidates.indexes),
                }
            )
        ranked, status = self._rerank(query, rows)
        for rank, row in enumerate(ranked, 1):
            row.update(final_rank=rank, passage_rerank_status=status)
        return ranked
