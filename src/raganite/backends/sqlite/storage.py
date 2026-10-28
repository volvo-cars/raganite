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

"""Internal SQLite persistence, connection management, and vector ranking."""

import json
import re
import sqlite3
import threading
import uuid
from collections.abc import Sequence
from pathlib import Path
from threading import RLock
from typing import Literal

from raganite.config import SQLiteConfig

from ..base import (
    DocumentGraph,
    GraphBackend,
    GraphEdge,
    GraphNode,
    NodeKind,
    SearchHit,
    cosine,
    validate_vector,
)

_DDL = """
CREATE TABLE IF NOT EXISTS graph_metadata (
    namespace TEXT PRIMARY KEY, dimensions INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    namespace TEXT, source TEXT, path TEXT, content_hash TEXT NOT NULL,
    PRIMARY KEY(namespace, source, path)
);
CREATE TABLE IF NOT EXISTS nodes (
    namespace TEXT, id TEXT, label TEXT NOT NULL, kind TEXT NOT NULL,
    properties TEXT NOT NULL, embedding TEXT,
    PRIMARY KEY(namespace, id)
);
CREATE TABLE IF NOT EXISTS edges (
    namespace TEXT, id TEXT, source TEXT NOT NULL, target TEXT NOT NULL,
    type TEXT NOT NULL, properties TEXT NOT NULL,
    PRIMARY KEY(namespace, id),
    FOREIGN KEY(namespace, source) REFERENCES nodes(namespace, id) ON DELETE CASCADE,
    FOREIGN KEY(namespace, target) REFERENCES nodes(namespace, id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS document_nodes (
    namespace TEXT, source TEXT, path TEXT, node_id TEXT,
    PRIMARY KEY(namespace, source, path, node_id),
    FOREIGN KEY(namespace, source, path) REFERENCES documents ON DELETE CASCADE,
    FOREIGN KEY(namespace, node_id) REFERENCES nodes(namespace, id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS document_edges (
    namespace TEXT, source TEXT, path TEXT, edge_id TEXT,
    PRIMARY KEY(namespace, source, path, edge_id),
    FOREIGN KEY(namespace, source, path) REFERENCES documents ON DELETE CASCADE,
    FOREIGN KEY(namespace, edge_id) REFERENCES edges(namespace, id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS nodes_kind ON nodes(namespace, kind);
CREATE INDEX IF NOT EXISTS edge_source ON edges(namespace, source);
CREATE INDEX IF NOT EXISTS edge_target ON edges(namespace, target);
CREATE INDEX IF NOT EXISTS node_owners ON document_nodes(namespace, node_id);
CREATE INDEX IF NOT EXISTS edge_owners ON document_edges(namespace, edge_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(namespace UNINDEXED, node_id UNINDEXED, text);
"""


class _SQLiteStorage(GraphBackend):
    def __init__(self, config: SQLiteConfig | None = None, *, namespace="default", dimensions=1536):
        config = config or SQLiteConfig()
        super().__init__(namespace=namespace, dimensions=dimensions, read_only=config.read_only)
        self._lock = RLock()
        self._local = threading.local()
        self._readers: list[sqlite3.Connection] = []
        self._timeout = config.timeout
        if config.path == ":memory:":
            if config.read_only:
                raise ValueError("An in-memory SQLite database cannot be opened read-only")
            # A private name keeps each backend's memory database separate, and the
            # shared cache lets reader connections see the writer's tables.
            self._database = f"file:raganite-{uuid.uuid4().hex}?mode=memory&cache=shared"
            self._uri = True
            self._wal = False
        elif config.read_only:
            self._database = Path(config.path).resolve().as_uri() + "?mode=ro"
            self._uri = True
            self._wal = False
        else:
            Path(config.path).parent.mkdir(parents=True, exist_ok=True)
            self._database = config.path
            self._uri = False
            self._wal = True
        self._connection = self._open()
        try:
            self.ensure_indexes()
        except Exception:
            self._connection.close()
            raise

    def _open(self, *, autocommit: bool = False) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._database,
            uri=self._uri,
            timeout=self._timeout,
            check_same_thread=False,
            isolation_level=None if autocommit else "DEFERRED",
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        if self._wal:
            connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _reader(self) -> sqlite3.Connection:
        """Return this thread's connection so reads do not queue on the writer."""
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = self._open(autocommit=True)
            self._local.connection = connection
            with self._lock:
                self._readers.append(connection)
        return connection

    def ensure_indexes(self) -> None:
        with self._lock, self._connection:
            if not self.read_only:
                self._connection.executescript(_DDL)
                self._connection.execute(
                    "INSERT OR IGNORE INTO graph_metadata VALUES (?, ?)",
                    (self.namespace, self.dimensions),
                )
            row = self._connection.execute(
                "SELECT dimensions FROM graph_metadata WHERE namespace=?",
                (self.namespace,),
            ).fetchone()
            if row is None or row[0] != self.dimensions:
                raise ValueError("Stored namespace embedding dimensions do not match configuration")
            if not self.read_only:
                missing = self._connection.execute(
                    "SELECT 1 FROM nodes WHERE namespace=? AND kind='entity' "
                    "AND json_extract(properties, '$.semantic_degree') IS NULL LIMIT 1",
                    (self.namespace,),
                ).fetchone()
                if missing is not None:
                    self._refresh_degrees()

    @staticmethod
    def _node(row) -> GraphNode:
        return GraphNode(
            id=row["id"],
            label=row["label"],
            kind=row["kind"],
            properties=json.loads(row["properties"]),
            embedding=json.loads(row["embedding"]) if row["embedding"] is not None else None,
        )

    def _cleanup(self) -> None:
        self._connection.execute(
            "DELETE FROM edges WHERE namespace=? AND NOT EXISTS "
            "(SELECT 1 FROM document_edges o WHERE o.namespace=edges.namespace AND o.edge_id=edges.id)",
            (self.namespace,),
        )
        self._connection.execute(
            "DELETE FROM nodes WHERE namespace=? AND NOT EXISTS "
            "(SELECT 1 FROM document_nodes o WHERE o.namespace=nodes.namespace AND o.node_id=nodes.id)",
            (self.namespace,),
        )
        self._connection.execute(
            "DELETE FROM chunks_fts WHERE namespace=? AND node_id NOT IN "
            "(SELECT id FROM nodes WHERE namespace=?)",
            (self.namespace, self.namespace),
        )

    def replace_document(self, graph: DocumentGraph) -> None:
        self.require_writable()
        self.validate_graph(graph)
        key = (self.namespace, graph.source, graph.path)
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM documents WHERE namespace=? AND source=? AND path=?", key
            )
            self._connection.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?)", (*key, graph.content_hash)
            )
            for node in graph.nodes:
                self._connection.execute(
                    "INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(namespace, id) "
                    "DO UPDATE SET label=excluded.label, kind=excluded.kind, "
                    "properties=excluded.properties, embedding=excluded.embedding",
                    (
                        self.namespace,
                        node.id,
                        node.label,
                        node.kind,
                        json.dumps(node.properties),
                        json.dumps(node.embedding) if node.embedding is not None else None,
                    ),
                )
                self._connection.execute(
                    "INSERT INTO document_nodes VALUES (?, ?, ?, ?)", (*key, node.id)
                )
                if node.kind == "chunk":
                    self._connection.execute(
                        "DELETE FROM chunks_fts WHERE namespace=? AND node_id=?",
                        (self.namespace, node.id),
                    )
                    self._connection.execute(
                        "INSERT INTO chunks_fts VALUES (?, ?, ?)",
                        (self.namespace, node.id, str(node.properties.get("text", ""))),
                    )
            for edge in graph.edges:
                self._connection.execute(
                    "INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(namespace, id) "
                    "DO UPDATE SET properties=excluded.properties",
                    (
                        self.namespace,
                        edge.id,
                        edge.source,
                        edge.target,
                        edge.type,
                        json.dumps(edge.properties),
                    ),
                )
                self._connection.execute(
                    "INSERT INTO document_edges VALUES (?, ?, ?, ?)", (*key, edge.id)
                )
            self._cleanup()
            self._refresh_degrees()

    def store_shared_edges(
        self, edges: Sequence[tuple[GraphEdge, Sequence[tuple[str, str]]]]
    ) -> None:
        """Write edges owned by every listed document. Each owner is a source and path."""
        self.require_writable()
        if not edges:
            return
        with self._lock, self._connection:
            for edge, owners in edges:
                self._connection.execute(
                    "INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(namespace, id) DO UPDATE SET "
                    "source=excluded.source, target=excluded.target, type=excluded.type, properties=excluded.properties",
                    (
                        self.namespace,
                        edge.id,
                        edge.source,
                        edge.target,
                        edge.type,
                        json.dumps(edge.properties),
                    ),
                )
                self._connection.executemany(
                    "INSERT OR IGNORE INTO document_edges VALUES (?, ?, ?, ?)",
                    [(self.namespace, source, path, edge.id) for source, path in owners],
                )
            self._refresh_degrees()

    def _refresh_degrees(self) -> None:
        """Store each entity's neighbor count so retrieval does not scan edges per step."""
        self._connection.execute(
            """
            UPDATE nodes SET properties = json_set(properties, '$.semantic_degree', (
                SELECT count(*) FROM (
                    SELECT id, target AS neighbor FROM edges
                    WHERE namespace=nodes.namespace AND source=nodes.id
                    UNION
                    SELECT id, source AS neighbor FROM edges
                    WHERE namespace=nodes.namespace AND target=nodes.id
                ) adjacency
                JOIN nodes other ON other.namespace=nodes.namespace
                    AND other.id=adjacency.neighbor AND other.kind='entity'
            ))
            WHERE namespace=? AND kind='entity'
            """,
            (self.namespace,),
        )

    def document_hash(self, source: str, path: str) -> str | None:
        row = (
            self._reader()
            .execute(
                "SELECT content_hash FROM documents WHERE namespace=? AND source=? AND path=?",
                (self.namespace, source, path),
            )
            .fetchone()
        )
        return row[0] if row else None

    def delete_documents(self, source: str | None = None, *, keep_paths: Sequence[str] = ()) -> int:
        self.require_writable()
        with self._lock, self._connection:
            rows = self._connection.execute(
                "SELECT source, path FROM documents WHERE namespace=? AND (? IS NULL OR source=?)",
                (self.namespace, source, source),
            ).fetchall()
            keep = set(keep_paths)
            stale = [
                (self.namespace, row["source"], row["path"])
                for row in rows
                if row["path"] not in keep
            ]
            self._connection.executemany(
                "DELETE FROM documents WHERE namespace=? AND source=? AND path=?", stale
            )
            self._cleanup()
            self._refresh_degrees()
            return len(stale)

    def nodes(
        self,
        *,
        ids: Sequence[str] | None = None,
        term: str | None = None,
        kind: NodeKind | None = None,
        limit: int | None = None,
    ) -> list[GraphNode]:
        if ids is not None and not ids:
            return []
        query = "SELECT * FROM nodes WHERE namespace=?"
        params: list = [self.namespace]
        if kind is not None:
            query += " AND kind=?"
            params.append(kind)
        if ids is not None:
            query += " AND id IN (" + ",".join("?" for _ in ids) + ")"
            params.extend(ids)
        rows = self._reader().execute(query + " ORDER BY id", params).fetchall()
        results = [self._node(row) for row in rows]
        if term is not None:
            results = [
                node
                for node in results
                if any(
                    term.casefold() in str(value).casefold()
                    for value in node.properties.values()
                    if isinstance(value, (str, int, float, bool))
                )
            ]
        return results[:limit] if limit is not None else results

    def edges(self, *, node_ids: Sequence[str] | None = None) -> list[GraphEdge]:
        if node_ids is not None and not node_ids:
            return []
        rows = (
            self._reader()
            .execute(
                "SELECT * FROM edges WHERE namespace=? ORDER BY id",
                (self.namespace,),
            )
            .fetchall()
        )
        ids = set(node_ids) if node_ids is not None else None
        return [
            GraphEdge(
                id=row["id"],
                source=row["source"],
                target=row["target"],
                type=row["type"],
                properties=json.loads(row["properties"]),
            )
            for row in rows
            if ids is None or row["source"] in ids or row["target"] in ids
        ]

    def vector_search(
        self,
        embedding: list[float],
        *,
        top_k: int = 5,
        source: str | None = None,
        kind: Literal["chunk", "fact"] = "chunk",
    ) -> list[SearchHit]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        validate_vector(embedding, self.dimensions)
        rows = (
            self._reader()
            .execute(
                "SELECT n.* FROM nodes n WHERE n.namespace=? AND n.kind=? AND n.embedding IS NOT NULL "
                "AND (? IS NULL OR EXISTS (SELECT 1 FROM document_nodes o WHERE "
                "o.namespace=n.namespace AND o.node_id=n.id AND o.source=?))",
                (self.namespace, kind, source, source),
            )
            .fetchall()
        )
        hits = []
        for row in rows:
            node = self._node(row)
            if node.embedding is not None:
                hits.append(SearchHit(node=node, score=(1 + cosine(embedding, node.embedding)) / 2))
        return sorted(hits, key=lambda hit: (-hit.score, hit.node.id))[:top_k]

    def fulltext_search(
        self, query: str, *, top_k: int = 5, source: str | None = None
    ) -> list[SearchHit]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        tokens = re.findall(r"\w+", query, flags=re.UNICODE)
        if not tokens:
            return []
        expression = " OR ".join('"' + token + '"' for token in tokens)
        rows = (
            self._reader()
            .execute(
                "SELECT n.*, -bm25(chunks_fts) AS score FROM chunks_fts "
                "JOIN nodes n ON n.namespace=chunks_fts.namespace AND n.id=chunks_fts.node_id "
                "WHERE chunks_fts MATCH ? AND n.namespace=? AND (? IS NULL OR EXISTS "
                "(SELECT 1 FROM document_nodes o WHERE o.namespace=n.namespace AND o.node_id=n.id AND o.source=?)) "
                "ORDER BY score DESC, n.id LIMIT ?",
                (expression, self.namespace, source, source, top_k),
            )
            .fetchall()
        )
        return [SearchHit(node=self._node(row), score=row["score"]) for row in rows]

    def close(self) -> None:
        with self._lock:
            readers = list(self._readers)
            self._readers.clear()
            writer = self._connection
        for connection in readers:
            connection.close()
        writer.close()
