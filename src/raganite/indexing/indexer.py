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

"""Backend-independent implementation of the sample knowledge-graph pipeline."""

import asyncio
import fnmatch
import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TextSplitter,
)
from neo4j_graphrag.components.entity_relation_extractor import LLMEntityRelationExtractor, OnError
from neo4j_graphrag.components.schema import SchemaFromTextExtractor
from neo4j_graphrag.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.components.types import DocumentInfo, TextChunk, TextChunks
from neo4j_graphrag.llm import LLMBase, LLMResponse

from raganite.backends.base import (
    DocumentGraph,
    GraphBackend,
    GraphEdge,
    GraphNode,
    NodeKind,
    identity,
)
from raganite.clients.llm import EmbeddingClient, LLMClient
from raganite.config import RaganiteConfig
from raganite.schema.topology import GraphTopology

logger = logging.getLogger(__name__)


def sequence_position(metadata: dict) -> tuple[int, str] | None:
    """Return a document's order and time when both were supplied."""
    order = metadata.get("order") if isinstance(metadata, dict) else None
    time = metadata.get("time") if isinstance(metadata, dict) else None
    if type(order) is not int or not isinstance(time, str) or not time.strip():
        return None
    return order, time.strip()


def _connect(
    edges: dict[str, GraphEdge], start: str, rel_type: str, end: str, properties: dict | None = None
) -> None:
    edge_id = identity("edge", start, rel_type, end)
    edges[edge_id] = GraphEdge(
        id=edge_id,
        source=start,
        target=end,
        type=rel_type,
        properties=properties or {},
    )


def attach_session(
    nodes: dict[str, GraphNode],
    edges: dict[str, GraphEdge],
    *,
    metadata: dict,
    namespace: str,
    source: str,
    path: str,
    topology: GraphTopology,
) -> None:
    """Add a session entity for an ordered document and tie its chunks and entities to it."""
    position = sequence_position(metadata)
    if position is None:
        return
    order, time = position
    session_id = identity("entity", namespace, "Session", source, order)
    nodes[session_id] = GraphNode(
        id=session_id,
        label="Session",
        kind="entity",
        properties={
            "name": f"session {order}",
            "order": order,
            "time": time,
            "source": source,
            "path": path,
        },
    )
    for node in list(nodes.values()):
        if node.id == session_id:
            continue
        if node.kind == "chunk":
            _connect(edges, session_id, topology.chunk_entity, node.id)
        elif node.kind == "entity":
            _connect(edges, node.id, topology.entity_session, session_id)


def sequence_edges(
    sessions: list[GraphNode], topology: GraphTopology
) -> list[tuple[GraphEdge, tuple[tuple[str, str], tuple[str, str]]]]:
    """Link session entities in order. Each edge is owned by both sessions."""
    ordered = [
        node
        for node in sessions
        if node.kind == "entity"
        and node.label == "Session"
        and sequence_position(node.properties) is not None
        and isinstance(node.properties.get("source"), str)
        and isinstance(node.properties.get("path"), str)
    ]
    ordered.sort(key=lambda node: (node.properties["order"], node.properties["path"]))
    linked: list[tuple[GraphEdge, tuple[tuple[str, str], tuple[str, str]]]] = []
    for earlier, later in zip(ordered, ordered[1:], strict=False):
        if later.properties["order"] <= earlier.properties["order"]:
            continue
        edge_id = identity("edge", earlier.id, topology.next_session, later.id)
        linked.append(
            (
                GraphEdge(
                    id=edge_id,
                    source=earlier.id,
                    target=later.id,
                    type=topology.next_session,
                    properties={
                        "previous": earlier.properties["time"],
                        "next": later.properties["time"],
                    },
                ),
                (
                    (earlier.properties["source"], earlier.properties["path"]),
                    (later.properties["source"], later.properties["path"]),
                ),
            )
        )
    return linked


def kept_relationships(relationships, mapped: dict[str, str], *, path: str) -> list[GraphEdge]:
    """Keep links whose endpoints were extracted. Warn about the ones the model left dangling."""
    kept: list[GraphEdge] = []
    for rel in relationships:
        if rel.start_node_id not in mapped or rel.end_node_id not in mapped:
            logger.warning(
                "Dropping relationship %s from %r to %r in %s because an endpoint was not extracted",
                rel.type,
                rel.start_node_id,
                rel.end_node_id,
                path,
            )
            continue
        start, end = mapped[rel.start_node_id], mapped[rel.end_node_id]
        kept.append(
            GraphEdge(
                id=identity("edge", start, rel.type, end),
                source=start,
                target=end,
                type=rel.type,
                properties=rel.properties or {},
            )
        )
    return kept


def apply_topology(
    nodes: dict[str, GraphNode],
    edges: dict[str, GraphEdge],
    topology: GraphTopology,
    *,
    document_id: str,
) -> None:
    """Make chunk links use the store topology so expansion reads what indexing wrote."""
    for edge in list(edges.values()):
        if edge.source not in nodes or edge.target not in nodes:
            continue
        kinds = {nodes[edge.source].kind, nodes[edge.target].kind}
        if kinds != {"chunk", "entity"} or edge.type == topology.chunk_entity:
            continue
        edges.pop(edge.id)
        edge_id = identity("edge", edge.source, topology.chunk_entity, edge.target)
        edges[edge_id] = GraphEdge(
            id=edge_id,
            source=edge.source,
            target=edge.target,
            type=topology.chunk_entity,
            properties=edge.properties,
        )
    if document_id not in nodes:
        return
    for node in list(nodes.values()):
        if node.kind != "chunk":
            continue
        if any(
            item.type == topology.chunk_document
            and {item.source, item.target} == {node.id, document_id}
            for item in edges.values()
        ):
            continue
        edge_id = identity("edge", node.id, topology.chunk_document, document_id)
        edges[edge_id] = GraphEdge(
            id=edge_id, source=node.id, target=document_id, type=topology.chunk_document
        )


class _ExtractionLLM(LLMBase):
    """Adapt LangChain to the upstream extractor, without replacing its prompts."""

    def __init__(self, model: BaseChatModel, semaphore: asyncio.Semaphore):
        super().__init__(model_name=type(model).__name__)
        self.model = model
        self.semaphore = semaphore

    def _input(self, input, message_history=None, system_instruction=None):
        messages = []
        if system_instruction:
            messages.append(("system", system_instruction))
        history = getattr(message_history, "messages", message_history) or []
        messages.extend(m.model_dump(exclude_none=True) for m in history)
        if isinstance(input, str):
            messages.append(("user", input))
        else:
            messages.extend(m.model_dump(exclude_none=True) for m in input)
        return messages

    def invoke(
        self, input, message_history=None, system_instruction=None, response_format=None, **kwargs
    ) -> LLMResponse:
        if response_format is not None:
            kwargs["response_format"] = response_format
        response = self.model.invoke(
            self._input(input, message_history, system_instruction), **kwargs
        )
        return LLMResponse(content=response.text)

    async def ainvoke(
        self, input, message_history=None, system_instruction=None, response_format=None, **kwargs
    ) -> LLMResponse:
        if response_format is not None:
            kwargs["response_format"] = response_format
        async with self.semaphore:
            response = await self.model.ainvoke(
                self._input(input, message_history, system_instruction), **kwargs
            )
        return LLMResponse(content=response.text)


@dataclass
class IndexingReport:
    indexed_files: list[str] = field(default_factory=list)
    skipped_files: list[tuple[str, str]] = field(default_factory=list)
    failed_files: list[tuple[str, str]] = field(default_factory=list)
    deleted_files: int = 0

    @property
    def total(self) -> int:
        return len(self.indexed_files) + len(self.skipped_files) + len(self.failed_files)


class KnowledgeGraphIndexer:
    """Shared algorithm; only storage operations vary by backend.

    A fresh extractor and schema-inference component are created per document.
    The supplied clients/models are shared. Workers bound both document tasks
    and in-flight model calls, rather than multiplying the two limits.
    """

    def __init__(
        self,
        backend: GraphBackend,
        llm: LLMClient | BaseChatModel,
        embeddings: EmbeddingClient | Embeddings,
        config: RaganiteConfig | None = None,
        *,
        text_splitter: TextSplitter | None = None,
    ):
        self.backend = backend
        self.config = config or RaganiteConfig(
            namespace=backend.namespace, embedding_dimensions=backend.dimensions
        )
        if (self.config.namespace, self.config.embedding_dimensions) != (
            backend.namespace,
            backend.dimensions,
        ):
            raise ValueError(
                "Indexer configuration must match the backend namespace and dimensions"
            )
        self.llm = llm.model if isinstance(llm, LLMClient) else llm
        self.embeddings = (
            embeddings.model if isinstance(embeddings, EmbeddingClient) else embeddings
        )
        self.text_splitter = text_splitter

    async def _chunks(self, document: Document, source: str, path: str) -> TextChunks:
        config = self.config.indexing.chunking
        splitter = self.text_splitter
        if splitter is None and config.strategy != "fixed":
            cls = (
                RecursiveCharacterTextSplitter
                if config.strategy == "recursive"
                else CharacterTextSplitter
            )
            splitter = cls(
                chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap, **config.options
            )
        if splitter is None:
            if config.options:
                raise ValueError("Fixed chunking does not accept LangChain splitter options")
            chunks = await FixedSizeSplitter(
                chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
            ).run(document.page_content)
        else:
            split_docs = await asyncio.to_thread(splitter.split_documents, [document])
            chunks = TextChunks(
                chunks=[
                    TextChunk(text=d.page_content, index=i, metadata=d.metadata)
                    for i, d in enumerate(split_docs)
                ]
            )
        for chunk in chunks.chunks:
            chunk.uid = identity("chunk", self.backend.namespace, source, path, chunk.index)
            # Upstream lexical nodes accept scalar/array metadata, not arbitrary
            # nested LangChain metadata. The full original metadata is retained
            # separately on the portable Document node.
            chunk.metadata = {
                k: v
                for k, v in (chunk.metadata or {}).items()
                if isinstance(v, (str, int, float, bool))
                and k not in {"text", "index", "embedding"}
            }
            chunk.metadata.update({"source": source, "path": path})
        return chunks

    async def _build_graph(
        self,
        document: Document,
        source: str,
        path: str,
        content_hash: str,
        semaphore: asyncio.Semaphore,
    ) -> DocumentGraph:
        cfg = self.config.indexing
        chunks = await self._chunks(document, source, path)
        async with semaphore:
            vectors = await self.embeddings.aembed_documents([c.text for c in chunks.chunks])
        if len(vectors) != len(chunks.chunks):
            raise ValueError("The embedding model returned an unexpected number of vectors")
        for chunk, vector in zip(chunks.chunks, vectors, strict=True):
            chunk.metadata = {**(chunk.metadata or {}), "embedding": vector}
        adapter = _ExtractionLLM(self.llm, semaphore)
        schema = cfg.graph_schema if cfg.schema_mode == "fixed" else None
        if cfg.schema_mode == "extracted":
            schema = await SchemaFromTextExtractor(llm=adapter).run(text=document.page_content)
        options: dict[str, Any] = {}
        if cfg.extraction_prompt is not None:
            options["prompt_template"] = cfg.extraction_prompt
        extractor = LLMEntityRelationExtractor(
            llm=adapter,
            max_concurrency=cfg.workers,
            on_error=OnError(cfg.on_error),
            **options,
        )
        document_id = identity("document", self.backend.namespace, source, path)
        extracted = await extractor.run(
            chunks=chunks,
            document_info=DocumentInfo(path=path, uid=document_id),
            schema=schema,
            examples=cfg.examples,
        )
        mapped = {}
        nodes: dict[str, GraphNode] = {}
        for node in extracted.nodes:
            kind: NodeKind = (
                "document"
                if node.label == "Document"
                else "chunk"
                if node.label == "Chunk"
                else "entity"
            )
            node_id = node.id
            name = node.properties.get("name")
            if kind == "entity" and cfg.perform_entity_resolution and name is not None:
                # The sample resolver merges the same label + name, not fuzzy matches.
                node_id = identity("entity", self.backend.namespace, node.label, name)
            mapped[node.id] = node_id
            properties: dict[str, Any] = dict(node.properties)
            if kind in {"document", "chunk"}:
                properties.update({"source": source, "path": path})
            if kind == "document":
                properties.update({"metadata": document.metadata, "content_hash": content_hash})
            nodes[node_id] = GraphNode(
                id=node_id,
                label=node.label,
                kind=kind,
                properties=properties,
                embedding=node.embedding_properties.get("embedding"),
            )
        edges: dict[str, GraphEdge] = {}
        for edge in kept_relationships(extracted.relationships, mapped, path=path):
            edges[edge.id] = edge
        source_id = identity("source", self.backend.namespace, source)
        namespace_id = identity("namespace", self.backend.namespace)
        nodes[source_id] = GraphNode(
            id=source_id, label="Source", kind="source", properties={"name": source}
        )
        nodes[namespace_id] = GraphNode(
            id=namespace_id,
            label="Namespace",
            kind="namespace",
            properties={"name": self.backend.namespace},
        )
        topo = self.backend.topology
        for start, rel_type, end in (
            (source_id, topo.source_document, document_id),
            (document_id, topo.document_namespace, namespace_id),
        ):
            edge_id = identity("edge", start, rel_type, end)
            edges[edge_id] = GraphEdge(id=edge_id, source=start, target=end, type=rel_type)
        if cfg.index_facts:
            semantic = [
                e
                for e in edges.values()
                if nodes[e.source].kind == nodes[e.target].kind == "entity"
            ]
            texts = []
            for edge in semantic:
                subject = nodes[edge.source].properties.get("name", nodes[edge.source].label)
                object_ = nodes[edge.target].properties.get("name", nodes[edge.target].label)
                suffix = (
                    " | " + json.dumps(edge.properties, ensure_ascii=False, sort_keys=True)
                    if edge.properties
                    else ""
                )
                texts.append(
                    f"{subject} | {edge.type.replace('_', ' ').lower()} | {object_}{suffix}"
                )
            if texts:
                async with semaphore:
                    fact_vectors = await self.embeddings.aembed_documents(texts)
                for edge, text, vector in zip(semantic, texts, fact_vectors, strict=True):
                    fact_id = identity("fact", edge.id, text)
                    nodes[fact_id] = GraphNode(
                        id=fact_id,
                        label="__Fact__",
                        kind="fact",
                        properties={
                            "text": text,
                            "relation_type": edge.type,
                            "relationship_id": edge.id,
                        },
                        embedding=vector,
                    )
                    for rel_type, target in (
                        (topo.fact_subject, edge.source),
                        (topo.fact_object, edge.target),
                    ):
                        edge_id = identity("edge", fact_id, rel_type, target)
                        edges[edge_id] = GraphEdge(
                            id=edge_id, source=fact_id, target=target, type=rel_type
                        )
        apply_topology(nodes, edges, topo, document_id=document_id)
        attach_session(
            nodes,
            edges,
            metadata=document.metadata,
            namespace=self.backend.namespace,
            source=source,
            path=path,
            topology=topo,
        )
        return DocumentGraph(
            source=source,
            path=path,
            content_hash=content_hash,
            nodes=list(nodes.values()),
            edges=list(edges.values()),
        )

    async def index_documents_async(
        self,
        documents: Sequence[Document],
        *,
        source: str = "documents",
        force: bool = False,
    ) -> IndexingReport:
        self.backend.require_writable()
        await asyncio.to_thread(self.backend.ensure_indexes)
        cfg = self.config.indexing
        calls = asyncio.Semaphore(cfg.workers)
        paths = [
            str(doc.metadata.get("path", doc.metadata.get("source", doc.id or str(i))))
            for i, doc in enumerate(documents)
        ]
        if len(set(paths)) != len(paths):
            raise ValueError("Document paths must be unique within one source/batch")
        report = IndexingReport()

        async def index_one(document: Document, path: str):
            try:
                if not document.page_content.strip():
                    return "skipped", path, "empty"
                digest = hashlib.sha256(document.page_content.encode()).hexdigest()
                previous = await asyncio.to_thread(self.backend.document_hash, source, path)
                if not force and previous == digest:
                    return "skipped", path, "unchanged"
                graph = await self._build_graph(document, source, path, digest, calls)
                await asyncio.to_thread(self.backend.replace_document, graph)
                return "indexed", path, ""
            except Exception as exc:
                logger.debug("Indexing %s failed: %s", path, exc, exc_info=True)
                return "failed", path, str(exc)

        for offset in range(0, len(documents), cfg.workers):
            results = await asyncio.gather(
                *(
                    index_one(doc, path)
                    for doc, path in zip(
                        documents[offset : offset + cfg.workers],
                        paths[offset : offset + cfg.workers],
                        strict=True,
                    )
                )
            )
            for status, path, reason in results:
                if status == "indexed":
                    report.indexed_files.append(path)
                elif status == "skipped":
                    report.skipped_files.append((path, reason))
                else:
                    report.failed_files.append((path, reason))
        if any(sequence_position(document.metadata) is not None for document in documents):
            await self._link_sequence(source)
        return report

    async def _link_sequence(self, source: str) -> None:
        stored = await asyncio.to_thread(self.backend.nodes, kind="entity")
        shared = sequence_edges(
            [node for node in stored if node.properties.get("source") == source],
            self.backend.topology,
        )
        if shared:
            await asyncio.to_thread(self.backend.store_shared_edges, shared)

    def index_documents(
        self, documents: Sequence[Document], *, source="documents", force=False
    ) -> IndexingReport:
        return asyncio.run(self.index_documents_async(documents, source=source, force=force))

    async def index_text_async(
        self, text: str, *, source: str, path: str = "text", force=False
    ) -> IndexingReport:
        return await self.index_documents_async(
            [Document(page_content=text, metadata={"path": path})], source=source, force=force
        )

    def index_text(self, text: str, *, source: str, path="text", force=False) -> IndexingReport:
        return asyncio.run(self.index_text_async(text, source=source, path=path, force=force))

    def iter_files(self, root: Path):
        cfg = self.config.indexing
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            if any(
                fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(path.as_posix(), p)
                for p in cfg.exclude_globs
            ):
                continue
            if (
                any(fnmatch.fnmatch(path.name, p) for p in cfg.include_globs)
                and path.stat().st_size <= cfg.max_file_bytes
            ):
                yield path

    async def index_repository_async(
        self, repo_root: str | Path, *, source_name: str | None = None, force=False
    ) -> IndexingReport:
        self.backend.require_writable()
        root = Path(repo_root).resolve()
        if not root.is_dir():
            raise NotADirectoryError(root)
        source = source_name or root.name
        paths = await asyncio.to_thread(lambda: list(self.iter_files(root)))
        report = IndexingReport()
        workers = self.config.indexing.workers
        for offset in range(0, len(paths), workers):
            documents = []
            for path in paths[offset : offset + workers]:
                rel = path.relative_to(root).as_posix()
                try:
                    text = await asyncio.to_thread(
                        path.read_text, encoding="utf-8", errors="replace"
                    )
                    if not text.strip():
                        report.skipped_files.append((rel, "empty"))
                        continue
                    documents.append(
                        Document(page_content=f"# Source: {rel}\n\n{text}", metadata={"path": rel})
                    )
                except OSError as exc:
                    logger.debug("Indexing %s failed: %s", rel, exc, exc_info=True)
                    report.failed_files.append((rel, str(exc)))
            batch = await self.index_documents_async(documents, source=source, force=force)
            report.indexed_files.extend(batch.indexed_files)
            report.skipped_files.extend(batch.skipped_files)
            report.failed_files.extend(batch.failed_files)
        report.deleted_files = await asyncio.to_thread(
            self.backend.delete_documents,
            source,
            keep_paths=[p.relative_to(root).as_posix() for p in paths],
        )
        return report

    def index_repository(
        self, repo_root: str | Path, *, source_name=None, force=False
    ) -> IndexingReport:
        return asyncio.run(
            self.index_repository_async(repo_root, source_name=source_name, force=force)
        )

    def clear_source(self, source_name: str) -> int:
        return self.backend.delete_documents(source_name)

    def clear_graph(self) -> int:
        """Clear only the active namespace, never another tenant's graph."""
        return self.backend.delete_documents()
