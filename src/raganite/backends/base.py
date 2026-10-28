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

"""Storage-neutral graph contracts and records."""

import hashlib
import json
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from raganite.schema.topology import OWNED, GraphTopology

from .models import FactHit, GraphExpansion, PassageHit

NodeKind = Literal["document", "chunk", "entity", "source", "namespace", "fact"]


def identity(*parts: Any) -> str:
    return hashlib.sha256(
        json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


class GraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    label: str
    kind: NodeKind
    properties: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None

    @property
    def labels(self) -> list[str]:
        return ["__Entity__", self.label] if self.kind == "entity" else [self.label]


class GraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    source: str
    target: str
    type: str
    properties: dict[str, Any] = Field(default_factory=dict)


class DocumentGraph(BaseModel):
    """A complete document replacement, including provenance and content hash."""

    source: str
    path: str
    content_hash: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class SearchHit(BaseModel):
    node: GraphNode
    score: float


def validate_vector(vector: Sequence[float], dimensions: int) -> None:
    if len(vector) != dimensions:
        raise ValueError(f"Expected {dimensions} embedding dimensions, got {len(vector)}")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("Embeddings must contain only finite values")
    if not any(vector):
        raise ValueError("Cosine search requires a non-zero embedding")


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    return dot / norm if norm else 0.0


class GraphBackend(ABC):
    """Fact-capable graph store; implementations must be thread-safe.

    IDs and returned records have the same meaning in every backend. Writes
    replace a whole document transactionally; graph memberships preserve shared
    entities and relationships until their last owning document is removed.
    Expansion follows undirected entity-only relationship trails, with no
    repeated edge within a path. Limits apply after scoring candidates.
    """

    def __init__(
        self,
        *,
        namespace: str,
        dimensions: int,
        read_only: bool = False,
        topology: GraphTopology | None = None,
    ):
        self.namespace = namespace
        self.dimensions = dimensions
        self.read_only = read_only
        self.topology = topology or OWNED

    def require_writable(self) -> None:
        if self.read_only:
            raise PermissionError("The backend is configured read-only")

    def validate_graph(self, graph: DocumentGraph) -> None:
        ids = {node.id for node in graph.nodes}
        if len(ids) != len(graph.nodes):
            raise ValueError("Graph node IDs must be unique within a document")
        if len({edge.id for edge in graph.edges}) != len(graph.edges):
            raise ValueError("Graph relationship IDs must be unique within a document")
        for node in graph.nodes:
            if node.embedding is not None:
                validate_vector(node.embedding, self.dimensions)
        for edge in graph.edges:
            if edge.source not in ids or edge.target not in ids:
                raise ValueError("Relationship endpoints must be included in the document graph")

    @abstractmethod
    def ensure_indexes(self) -> None:
        """Initialize storage and indexes, idempotently."""

    @abstractmethod
    def replace_document(self, graph: DocumentGraph) -> None:
        """Atomically replace the document, its hash, and graph memberships."""

    @abstractmethod
    def store_shared_edges(
        self, edges: Sequence[tuple[GraphEdge, Sequence[tuple[str, str]]]]
    ) -> None:
        """Write edges owned by every listed document. Each owner is a source and path."""

    @abstractmethod
    def document_hash(self, source: str, path: str) -> str | None:
        """Return the hash of the last successfully committed document."""

    @abstractmethod
    def delete_documents(self, source: str | None = None, *, keep_paths: Sequence[str] = ()) -> int:
        """Delete documents except keep_paths; None means all sources in this namespace."""

    @abstractmethod
    def nodes(
        self,
        *,
        ids: Sequence[str] | None = None,
        term: str | None = None,
        kind: NodeKind | None = None,
        limit: int | None = None,
    ) -> list[GraphNode]:
        """Find nodes by ID or case-insensitive scalar property substring."""

    @abstractmethod
    def edges(self, *, node_ids: Sequence[str] | None = None) -> list[GraphEdge]:
        """Read edges incident on any requested node, or all namespace edges."""

    @abstractmethod
    def vector_search(
        self,
        embedding: list[float],
        *,
        top_k: int = 5,
        source: str | None = None,
        kind: Literal["chunk", "fact"] = "chunk",
    ) -> list[SearchHit]:
        """Rank scoped embeddings by cosine similarity."""

    @abstractmethod
    def fulltext_search(
        self, query: str, *, top_k: int = 5, source: str | None = None
    ) -> list[SearchHit]:
        """Rank chunks using the backend's full-text index."""

    def search_chunks(
        self,
        embedding: list[float],
        *,
        top_k: int = 5,
        source: str | None = None,
    ) -> list[SearchHit]:
        """Rank chunks by embedding. Retrieval uses this query, not nodes()."""
        return self.vector_search(embedding, top_k=top_k, source=source, kind="chunk")

    def search_fulltext(
        self, query: str, *, top_k: int = 5, source: str | None = None
    ) -> list[SearchHit]:
        """Rank chunks by full text. Retrieval uses this query, not edges()."""
        return self.fulltext_search(query, top_k=top_k, source=source)

    @abstractmethod
    def prepare(self) -> None:
        """Validate fact storage and cache traversal topology without writing data."""

    def direct_passages(self, embedding: list[float], *, top_k: int) -> list[PassageHit]:
        hits = self.search_chunks(embedding, top_k=top_k)
        rows = []
        for hit in hits:
            properties = hit.node.properties
            source = next(
                (
                    properties[k]
                    for k in ("path", "file_path", "source", "uri")
                    if isinstance(properties.get(k), str)
                ),
                "",
            )
            text = next(
                (
                    properties[k]
                    for k in ("text", "content", "body", "description")
                    if isinstance(properties.get(k), str)
                ),
                "",
            )
            rows.append(PassageHit(hit.node.id, source, text, hit.score))
        return rows

    @abstractmethod
    def search_facts(self, embedding: list[float], *, top_k: int = 30) -> list[FactHit]:
        """Return ranked facts with both endpoints and semantic degrees."""

    @abstractmethod
    def expand_facts(
        self,
        facts: Sequence[FactHit],
        *,
        depth: int = 2,
        limit: int = 5000,
        degree_penalty_weight: float = 0.25,
    ) -> list[GraphExpansion]:
        """Rank entity-only paths to chunks with the reference degree penalty."""

    @abstractmethod
    def close(self) -> None:
        """Release owned database connections."""

    def __enter__(self):
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
