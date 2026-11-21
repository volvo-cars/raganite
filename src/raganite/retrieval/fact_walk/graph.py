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

"""In-memory fact graph scored with one matrix product and a sparse PageRank.

Embeddings stay in one normalized array so fact cosine scores come from
a matrix-vector product instead of a Python loop.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from scipy.sparse import csr_matrix

from raganite.backends.models import ChunkRecord, FactEdge

from .scoring import HomeScore, home_scores


class FactLink(NamedTuple):
    source: str
    target: str
    text: str
    relation_type: str
    relationship_id: str


@dataclass(frozen=True)
class FactWalk:
    homes: list[HomeScore]
    incoming: dict[str, tuple[float, FactLink]]
    seeds: int


class FactWalkGraph:
    """Facts, normalized embeddings, and passage homes loaded once."""

    def __init__(
        self,
        *,
        names: list[str],
        embeddings: np.ndarray,
        sources: np.ndarray,
        targets: np.ndarray,
        group_starts: np.ndarray,
        dangling: np.ndarray,
        links: list[FactLink],
        chunks: list[dict[str, Any]],
    ):
        self.names = names
        self._index = {name: index for index, name in enumerate(names)}
        self.embeddings = embeddings
        self.sources = sources
        self.targets = targets
        self.group_starts = group_starts
        self.dangling = dangling
        self.links = links
        self.chunks = chunks
        self.chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        self.producer_counts: dict[str, int] = {}
        for chunk in chunks:
            for name in dict.fromkeys(chunk.get("entities") or ()):
                self.producer_counts[name] = self.producer_counts.get(name, 0) + 1
        self.edge_count = len(links)
        self.chunk_count = len(chunks)
        self.dimensions = int(embeddings.shape[1]) if embeddings.size else 0

    @classmethod
    def from_edges(
        cls, edges: Sequence[FactEdge], chunks: Sequence[ChunkRecord]
    ) -> "FactWalkGraph":
        width = len(edges[0].embedding) if edges else 0
        first_source: dict[str, int] = {}
        for edge in edges:
            if edge.source not in first_source:
                first_source[edge.source] = len(first_source)
        order = sorted(range(len(edges)), key=lambda index: first_source[edges[index].source])
        grouped = [edges[index] for index in order]
        if grouped:
            matrix = np.asarray([edge.embedding for edge in grouped], dtype=np.float64)
            if matrix.ndim != 2 or matrix.shape[1] != width:
                raise ValueError("Fact embeddings have different widths")
            norms = np.linalg.norm(matrix, axis=1)
            positive = norms > 0
            matrix[positive] /= norms[positive, None]
        else:
            matrix = np.zeros((0, width), dtype=np.float64)

        index: dict[str, int] = {}

        def node(name: str) -> int:
            found = index.get(name)
            if found is None:
                found = len(index)
                index[name] = found
            return found

        sources = np.asarray([node(edge.source) for edge in grouped], dtype=np.int32)
        targets = np.asarray([node(edge.target) for edge in grouped], dtype=np.int32)
        names = [""] * len(index)
        for name, position in index.items():
            names[position] = name
        starts = [0]
        for position in range(1, len(grouped)):
            if grouped[position].source != grouped[position - 1].source:
                starts.append(position)
        is_source = np.zeros(len(names), dtype=bool)
        if len(sources):
            is_source[sources] = True
        return cls(
            names=names,
            embeddings=matrix,
            sources=sources,
            targets=targets,
            group_starts=np.asarray(starts, dtype=np.int32),
            dangling=np.flatnonzero(~is_source),
            links=[
                FactLink(
                    edge.source,
                    edge.target,
                    edge.text,
                    edge.relation_type,
                    edge.relationship_id,
                )
                for edge in grouped
            ],
            chunks=[
                {
                    "chunk_id": chunk.chunk_id,
                    "path": chunk.path,
                    "text": chunk.text,
                    "title": chunk.title,
                    "entities": list(chunk.entities),
                }
                for chunk in chunks
            ],
        )

    def walk(
        self,
        embedding: Sequence[float] | np.ndarray,
        *,
        tau: float,
        damping: float,
        seeds: Mapping[str, float],
        cosines: np.ndarray | None = None,
        evidence: bool = True,
    ) -> FactWalk:
        """Walk from ``seeds``. Values are restart weights on entity names."""
        if tau <= 0:
            raise ValueError("Temperature must be positive")
        if not 0 <= damping < 1:
            raise ValueError("Damping must be in [0, 1)")
        if self.edge_count and len(embedding) != self.dimensions:
            raise ValueError(f"Fact embedding width {self.dimensions} does not match the question")
        chosen = [
            (name, float(weight))
            for name, weight in seeds.items()
            if weight > 0 and name in self._index
        ]
        if not chosen or self.edge_count == 0:
            return FactWalk(homes=[], incoming={}, seeds=0)
        if cosines is None:
            cosines = self._cosines(embedding)
        weights = _grouped_softmax(cosines, self.group_starts, tau)
        mass = _sparse_pagerank(
            self.sources,
            self.targets,
            weights,
            np.asarray([self._index[name] for name, _weight in chosen], dtype=np.int32),
            np.asarray([weight for _name, weight in chosen], dtype=np.float64),
            len(self.names),
            self.dangling,
            damping,
        )
        homes = home_scores(
            {name: float(value) for name, value in zip(self.names, mass, strict=True)},
            self.chunks,
        )
        return FactWalk(
            homes=homes,
            incoming=self._incoming(mass, weights) if evidence else {},
            seeds=len(chosen),
        )

    def fact_cosines(self, embedding: Sequence[float] | np.ndarray) -> np.ndarray:
        """Cosine of every fact sentence with the question. Links share this order."""
        return self._cosines(embedding)

    def _cosines(self, embedding: Sequence[float] | np.ndarray) -> np.ndarray:
        if not self.edge_count:
            return np.empty(0, dtype=np.float64)
        query = np.asarray(embedding, dtype=np.float64)
        if query.shape != (self.dimensions,):
            raise ValueError(f"Fact embedding width {self.dimensions} does not match the question")
        norm = np.linalg.norm(query)
        if norm == 0:
            return np.zeros(self.edge_count, dtype=np.float64)
        return self.embeddings @ (query / norm)

    def _incoming(self, mass: np.ndarray, weights: np.ndarray) -> dict[str, tuple[float, FactLink]]:
        carried = mass[self.sources] * weights
        order = np.lexsort((np.arange(self.edge_count), -carried, self.targets))
        ordered_targets = self.targets[order]
        first = np.empty(self.edge_count, dtype=bool)
        first[0] = True
        if self.edge_count > 1:
            first[1:] = ordered_targets[1:] != ordered_targets[:-1]
        incoming: dict[str, tuple[float, FactLink]] = {}
        for edge in order[first]:
            link = self.links[int(edge)]
            incoming[link.target] = (float(carried[edge]), link)
        return incoming


def _grouped_softmax(values: np.ndarray, starts: np.ndarray, tau: float) -> np.ndarray:
    """Softmax inside each contiguous source group."""
    scaled = values / tau
    lengths = np.diff(np.append(starts, len(values)))
    peaks = np.maximum.reduceat(scaled, starts)
    weights = np.exp(scaled - np.repeat(peaks, lengths))
    totals = np.add.reduceat(weights, starts)
    return weights / np.repeat(totals, lengths)


def _sparse_pagerank(
    sources: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    seeds: np.ndarray,
    seed_weights: np.ndarray,
    node_count: int,
    dangling: np.ndarray,
    damping: float,
) -> np.ndarray:
    """Personalized PageRank. Columns are sources and already sum to one."""
    matrix = csr_matrix((weights, (targets, sources)), shape=(node_count, node_count))
    bias = np.zeros(node_count, dtype=np.float64)
    bias[seeds] = seed_weights
    total = float(bias.sum())
    if total <= 0:
        return bias
    bias /= total
    mass = bias.copy()
    for _ in range(100):
        incoming = matrix @ mass
        dangling_mass = float(mass[dangling].sum()) if len(dangling) else 0.0
        updated = (1 - damping) * bias + damping * (incoming + dangling_mass * bias)
        if np.abs(updated - mass).sum() < 1e-12:
            return updated
        mass = updated
    return mass
