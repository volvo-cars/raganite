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

"""Internal persistence and connection management for the Neo4j backend."""

import json
import re
from collections.abc import Sequence
from typing import Any, Literal

from neo4j import GraphDatabase

from raganite.config import Neo4jConfig

from ..base import (
    DocumentGraph,
    GraphBackend,
    GraphEdge,
    GraphNode,
    NodeKind,
    SearchHit,
    identity,
    validate_vector,
)


def quote_identifier(value: str) -> str:
    return "`" + value.replace("`", "``") + "`"


class _Neo4jStorage(GraphBackend):
    """Persistence shared by owned graphs and read-only native graph layouts."""

    def __init__(
        self,
        config: Neo4jConfig | None = None,
        *,
        namespace="default",
        dimensions=1536,
        driver: Any = None,
        layout: Literal["owned", "native"] | None = None,
    ):
        self.config = config or Neo4jConfig()
        self.layout: Literal["owned", "native"] = self.config.layout if layout is None else layout
        if self.layout not in {"owned", "native"}:
            raise ValueError("Neo4j layout must be owned or native")
        if self.layout == "native":
            self.config = self.config.model_copy(update={"read_only": True})
            namespace = "existing-database"
        super().__init__(
            namespace=namespace, dimensions=dimensions, read_only=self.config.read_only
        )
        self.native_graph = self.layout == "native"
        self._owns_driver = driver is None
        self._driver = (
            driver
            if driver is not None
            else GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password.get_secret_value())
                if self.config.password
                else None,
                **self.config.driver_options,
            )
        )

    def _index(self, name: str) -> str:
        if self.layout == "native":
            return name
        return f"{name}_{identity(self.namespace)[:12]}"

    def _lookup_index(self, name: str) -> str:
        """Return the index Neo4j will actually use for this label and property.

        A namespace-scoped name is unique, but Neo4j allows only one vector or
        fulltext index on a label and property. Later namespaces reuse the
        index the first namespace created. Queries still filter by namespace.
        """
        if self.layout == "native":
            return name
        resolved = getattr(self, "_resolved_indexes", {})
        if name in resolved:
            return resolved[name]
        scoped = self._index(name)
        signature = {
            self.config.vector_index: ("VECTOR", "Chunk", "embedding"),
            self.config.fact_vector_index: ("VECTOR", "__Fact__", "embedding"),
            self.config.fulltext_index: ("FULLTEXT", "Chunk", "text"),
        }.get(name)
        rows = self._read(
            "SHOW INDEXES YIELD name, type, state, labelsOrTypes, properties "
            "WHERE state = 'ONLINE' AND (name = $scoped OR type IN ['VECTOR', 'FULLTEXT']) "
            "RETURN name, type, labelsOrTypes, properties",
            scoped=scoped,
        )
        chosen = scoped
        saw_scoped = False
        equivalent = None
        for row in rows:
            if row["name"] == scoped:
                saw_scoped = True
                break
            if signature is None:
                continue
            kind, label, prop = signature
            labels = list(row.get("labelsOrTypes") or [])
            props = list(row.get("properties") or [])
            if row["type"] == kind and labels == [label] and prop in props:
                equivalent = row["name"]
        if not saw_scoped and equivalent:
            chosen = equivalent
        self._resolved_indexes = {**resolved, name: chosen}
        return chosen

    def run(self, cypher: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute native Cypher. This escape hatch is intentionally Neo4j-only.

        Custom queries must handle namespace scoping themselves. In read-only
        mode they execute in a server-side read transaction.
        """
        with self._driver.session(database=self.config.database) as session:

            def execute(tx):
                return [record.data() for record in tx.run(cypher, parameters or {})]

            return (
                session.execute_read(execute) if self.read_only else session.execute_write(execute)
            )

    def _read(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        with self._driver.session(database=self.config.database) as session:
            return session.execute_read(
                lambda tx: [
                    r.data() for r in tx.run(cypher, {"namespace": self.namespace, **params})
                ],
            )

    def ensure_indexes(self) -> None:
        if self.layout == "native":
            self._ensure_native_indexes()
            return
        if not self.read_only:
            self.run(
                "CREATE CONSTRAINT raganite_node_id IF NOT EXISTS FOR (n:RaganiteNode) REQUIRE (n.namespace, n.raganite_id) IS UNIQUE"
            )
            self.run(
                "CREATE CONSTRAINT raganite_owner_id IF NOT EXISTS FOR (d:RaganiteOwner) REQUIRE (d.namespace, d.source, d.path) IS UNIQUE"
            )
            self.run(
                "CREATE CONSTRAINT raganite_edge_id IF NOT EXISTS FOR (e:RaganiteEdge) REQUIRE (e.namespace, e.raganite_id) IS UNIQUE"
            )
            self.run(
                "CREATE CONSTRAINT raganite_metadata_id IF NOT EXISTS FOR (m:RaganiteMetadata) REQUIRE m.namespace IS UNIQUE"
            )
            self.run(
                "MERGE (m:RaganiteMetadata {namespace:$namespace}) "
                "ON CREATE SET m.dimensions=$dimensions",
                {"namespace": self.namespace, "dimensions": self.dimensions},
            )
        rows = self._read(
            "MATCH (m:RaganiteMetadata {namespace:$namespace}) RETURN m.dimensions AS dimensions"
        )
        if not rows or rows[0]["dimensions"] != self.dimensions:
            raise ValueError("Stored namespace embedding dimensions do not match configuration")
        if self.read_only:
            return
        for name, label in (
            (self.config.vector_index, "Chunk"),
            (self.config.fact_vector_index, "__Fact__"),
        ):
            if self._lookup_index(name) != self._index(name):
                continue
            self.run(
                f"CREATE VECTOR INDEX {quote_identifier(self._index(name))} IF NOT EXISTS "
                f"FOR (n:{quote_identifier(label)}) ON (n.embedding) "
                "OPTIONS {indexConfig: {`vector.dimensions`:$dimensions, `vector.similarity_function`:'cosine'}}",
                {"dimensions": self.dimensions},
            )
        if self._lookup_index(self.config.fulltext_index) == self._index(
            self.config.fulltext_index
        ):
            self.run(
                f"CREATE FULLTEXT INDEX {quote_identifier(self._index(self.config.fulltext_index))} "
                "IF NOT EXISTS FOR (n:Chunk) ON EACH [n.text]",
            )
        self.run("CALL db.awaitIndexes(300)")
        missing = self._read(
            "MATCH (n:RaganiteNode {namespace:$namespace, kind:'entity'}) "
            "WHERE n.semantic_degree IS NULL RETURN 1 AS missing LIMIT 1",
        )
        if missing:
            with self._driver.session(database=self.config.database) as session:
                session.execute_write(lambda tx: self._refresh_degrees(tx))

    @staticmethod
    def _node(raw: dict[str, Any]) -> GraphNode:
        properties = json.loads(raw["properties_json"])
        if raw.get("semantic_degree") is not None:
            properties["semantic_degree"] = raw["semantic_degree"]
        return GraphNode(
            id=raw["raganite_id"],
            label=raw["label"],
            kind=raw["kind"],
            properties=properties,
            embedding=raw.get("embedding"),
        )

    def _refresh_degrees(self, tx, ids: Sequence[str] | None = None) -> None:
        """Store each entity's neighbor count so retrieval does not scan edges."""
        tx.run(
            "MATCH (n:RaganiteNode {namespace:$namespace, kind:'entity'}) "
            "WHERE $ids IS NULL OR n.raganite_id IN $ids "
            "OPTIONAL MATCH (n)-[r]-(other:RaganiteNode {namespace:$namespace, kind:'entity'}) "
            "WITH n, count(r) AS degree "
            "SET n.semantic_degree = toFloat(degree)",
            {"namespace": self.namespace, "ids": list(ids) if ids is not None else None},
        ).consume()

    def _cleanup(self, tx) -> None:
        params = {"namespace": self.namespace}
        tx.run(
            "MATCH (e:RaganiteEdge {namespace:$namespace}) WHERE NOT EXISTS { MATCH (:RaganiteOwner)-[:OWNS_EDGE]->(e) } "
            "OPTIONAL MATCH ()-[r]->() WHERE r.namespace=$namespace AND r.raganite_id=e.raganite_id "
            "DELETE r DETACH DELETE e",
            params,
        ).consume()
        tx.run(
            "MATCH (n:RaganiteNode {namespace:$namespace}) WHERE NOT EXISTS { MATCH (:RaganiteOwner)-[:OWNS_NODE]->(n) } "
            "DETACH DELETE n",
            params,
        ).consume()

    def replace_document(self, graph: DocumentGraph) -> None:
        self.require_writable()
        self.validate_graph(graph)
        params = {
            "namespace": self.namespace,
            "source": graph.source,
            "path": graph.path,
            "hash": graph.content_hash,
        }

        def replace(tx):
            tx.run(
                "MATCH (d:RaganiteOwner {namespace:$namespace, source:$source, path:$path}) DETACH DELETE d",
                params,
            ).consume()
            tx.run(
                "CREATE (:RaganiteOwner {namespace:$namespace, source:$source, path:$path, content_hash:$hash})",
                params,
            ).consume()
            for node in sorted(graph.nodes, key=lambda n: n.id):
                labels = ":".join(quote_identifier(label) for label in node.labels)
                # Keep arbitrary nested metadata losslessly as JSON. Expose text
                # and scalar fields natively for graph browsing and full-text.
                scalar = {
                    k: v
                    for k, v in node.properties.items()
                    if isinstance(v, (str, int, float, bool))
                }
                tx.run(
                    "MERGE (n:RaganiteNode {namespace:$namespace, raganite_id:$id}) "
                    f"SET n:{labels}, n += $scalar, n.namespace=$namespace, n.raganite_id=$id, "
                    "n.kind=$kind, n.label=$label, n.properties_json=$properties, n.embedding=$embedding "
                    "WITH n MATCH (d:RaganiteOwner {namespace:$namespace, source:$source, path:$path}) "
                    "MERGE (d)-[:OWNS_NODE]->(n)",
                    {
                        **params,
                        "id": node.id,
                        "kind": node.kind,
                        "label": node.label,
                        "scalar": scalar,
                        "properties": json.dumps(node.properties),
                        "embedding": node.embedding,
                    },
                ).consume()
            for edge in sorted(graph.edges, key=lambda e: e.id):
                tx.run(
                    "MERGE (e:RaganiteEdge {namespace:$namespace, raganite_id:$id}) "
                    "SET e.source=$start, e.target=$end, e.type=$type, e.properties_json=$properties "
                    "WITH e MATCH (d:RaganiteOwner {namespace:$namespace, source:$source, path:$path}) "
                    "MERGE (d)-[:OWNS_EDGE]->(e) "
                    "WITH e MATCH (a:RaganiteNode {namespace:$namespace, raganite_id:$start}), "
                    "(b:RaganiteNode {namespace:$namespace, raganite_id:$end}) "
                    f"MERGE (a)-[r:{quote_identifier(edge.type)} {{namespace:$namespace, raganite_id:$id}}]->(b) "
                    "SET r.properties_json=$properties",
                    {
                        **params,
                        "id": edge.id,
                        "start": edge.source,
                        "end": edge.target,
                        "type": edge.type,
                        "properties": json.dumps(edge.properties),
                    },
                ).consume()
            self._cleanup(tx)
            entity_ids = [node.id for node in graph.nodes if node.kind == "entity"]
            if entity_ids:
                self._refresh_degrees(tx, entity_ids)

        with self._driver.session(database=self.config.database) as session:
            session.execute_write(replace)

    def store_shared_edges(
        self, edges: Sequence[tuple[GraphEdge, Sequence[tuple[str, str]]]]
    ) -> None:
        """Write edges owned by every listed document. Each owner is a source and path."""
        self.require_writable()
        if not edges:
            return

        def write(tx):
            for edge, owners in edges:
                tx.run(
                    "MERGE (e:RaganiteEdge {namespace:$namespace, raganite_id:$id}) "
                    "SET e.source=$start, e.target=$end, e.type=$type, e.properties_json=$properties "
                    "WITH e MATCH (a:RaganiteNode {namespace:$namespace, raganite_id:$start}), "
                    "(b:RaganiteNode {namespace:$namespace, raganite_id:$end}) "
                    f"MERGE (a)-[r:{quote_identifier(edge.type)} {{namespace:$namespace, raganite_id:$id}}]->(b) "
                    "SET r.properties_json=$properties",
                    {
                        "namespace": self.namespace,
                        "id": edge.id,
                        "start": edge.source,
                        "end": edge.target,
                        "type": edge.type,
                        "properties": json.dumps(edge.properties),
                    },
                ).consume()
                for source, path in owners:
                    tx.run(
                        "MATCH (e:RaganiteEdge {namespace:$namespace, raganite_id:$id}) "
                        "MATCH (d:RaganiteOwner {namespace:$namespace, source:$source, path:$path}) "
                        "MERGE (d)-[:OWNS_EDGE]->(e)",
                        {
                            "namespace": self.namespace,
                            "id": edge.id,
                            "source": source,
                            "path": path,
                        },
                    ).consume()
            endpoints = [
                node_id for edge, _owners in edges for node_id in (edge.source, edge.target)
            ]
            self._refresh_degrees(tx, endpoints)

        with self._driver.session(database=self.config.database) as session:
            session.execute_write(write)

    def document_hash(self, source: str, path: str) -> str | None:
        if self.layout == "native":
            rows = self._read(
                "MATCH (:Source {name:$source})-[:HAS_FILE]->(f:FileHash {path:$path}) RETURN f.hash AS hash LIMIT 1",
                source=source,
                path=path,
            )
            return rows[0]["hash"] if rows else None
        rows = self._read(
            "MATCH (d:RaganiteOwner {namespace:$namespace, source:$source, path:$path}) RETURN d.content_hash AS hash",
            source=source,
            path=path,
        )
        return rows[0]["hash"] if rows else None

    def delete_documents(self, source: str | None = None, *, keep_paths: Sequence[str] = ()) -> int:
        self.require_writable()

        def delete(tx):
            row = tx.run(
                "MATCH (d:RaganiteOwner {namespace:$namespace}) "
                "WHERE ($source IS NULL OR d.source=$source) AND NOT d.path IN $paths "
                "WITH collect(d) AS docs FOREACH (d IN docs | DETACH DELETE d) RETURN size(docs) AS deleted",
                namespace=self.namespace,
                source=source,
                paths=list(keep_paths),
            ).single()
            self._cleanup(tx)
            self._refresh_degrees(tx)
            return row["deleted"] if row else 0

        with self._driver.session(database=self.config.database) as session:
            return session.execute_write(delete)

    def nodes(
        self,
        *,
        ids: Sequence[str] | None = None,
        term: str | None = None,
        kind: NodeKind | None = None,
        limit: int | None = None,
    ) -> list[GraphNode]:
        if self.layout == "native":
            return self._native_nodes(ids=ids, term=term, kind=kind, limit=limit)
        rows = self._read(
            "MATCH (n:RaganiteNode {namespace:$namespace}) WHERE ($ids IS NULL OR n.raganite_id IN $ids) "
            "AND ($kind IS NULL OR n.kind=$kind) RETURN properties(n) AS node ORDER BY n.raganite_id",
            ids=list(ids) if ids is not None else None,
            kind=kind,
        )
        nodes = [self._node(row["node"]) for row in rows]
        if term is not None:
            nodes = [
                n
                for n in nodes
                if any(
                    term.casefold() in str(v).casefold()
                    for v in n.properties.values()
                    if isinstance(v, (str, int, float, bool))
                )
            ]
        return nodes[:limit] if limit is not None else nodes

    def edges(self, *, node_ids: Sequence[str] | None = None) -> list[GraphEdge]:
        if self.layout == "native":
            return self._native_edges(node_ids=node_ids)
        rows = self._read(
            "MATCH (e:RaganiteEdge {namespace:$namespace}) "
            "WHERE $ids IS NULL OR e.source IN $ids OR e.target IN $ids "
            "RETURN properties(e) AS edge ORDER BY e.raganite_id",
            ids=list(node_ids) if node_ids is not None else None,
        )
        return [
            GraphEdge(
                id=(e := row["edge"])["raganite_id"],
                source=e["source"],
                target=e["target"],
                type=e["type"],
                properties=json.loads(e["properties_json"]),
            )
            for row in rows
        ]

    def vector_search(
        self,
        embedding: list[float],
        *,
        top_k: int = 5,
        source: str | None = None,
        kind: Literal["chunk", "fact"] = "chunk",
    ) -> list[SearchHit]:
        if self.layout == "native":
            return self._native_vector_search(embedding, top_k=top_k, source=source, kind=kind)
        if top_k < 1:
            raise ValueError("top_k must be positive")
        validate_vector(embedding, self.dimensions)
        scope = (
            "node.namespace=$namespace AND node.kind=$kind AND ($source IS NULL OR EXISTS { "
            "MATCH (d:RaganiteOwner {namespace:$namespace, source:$source})-[:OWNS_NODE]->(node) }) "
        )
        index = self.config.vector_index if kind == "chunk" else self.config.fact_vector_index
        params = {"embedding": embedding, "top_k": top_k, "source": source, "kind": kind}
        rows = self._read(
            "CALL db.index.vector.queryNodes($index, $candidates, $embedding) YIELD node, score "
            f"WHERE {scope} RETURN properties(node) AS node, score ORDER BY score DESC, node.raganite_id LIMIT $top_k",
            index=self._lookup_index(index),
            candidates=max(top_k * 20, top_k),
            **params,
        )
        if len(rows) < top_k:
            # Global index candidates can be consumed by other namespaces/sources.
            rows = self._read(
                f"MATCH (node:RaganiteNode) WHERE {scope} AND node.embedding IS NOT NULL "
                "WITH node, vector.similarity.cosine(node.embedding,$embedding) AS score "
                "RETURN properties(node) AS node, score ORDER BY score DESC, node.raganite_id LIMIT $top_k",
                **params,
            )
        return [SearchHit(node=self._node(row["node"]), score=row["score"]) for row in rows]

    def fulltext_search(
        self, query: str, *, top_k: int = 5, source: str | None = None
    ) -> list[SearchHit]:
        if self.layout == "native":
            return self._native_fulltext_search(query, top_k=top_k, source=source)
        if top_k < 1:
            raise ValueError("top_k must be positive")
        terms = re.findall(r"\w+", query, flags=re.UNICODE)
        if not terms:
            return []
        rows = self._read(
            "CALL db.index.fulltext.queryNodes($index,$query) YIELD node, score "
            "WHERE node.namespace=$namespace AND ($source IS NULL OR EXISTS { "
            "MATCH (:RaganiteOwner {namespace:$namespace,source:$source})-[:OWNS_NODE]->(node) }) "
            "RETURN properties(node) AS node, score ORDER BY score DESC, node.raganite_id LIMIT $top_k",
            index=self._lookup_index(self.config.fulltext_index),
            query=" OR ".join(f'"{t}"' for t in terms),
            top_k=top_k,
            source=source,
        )
        return [SearchHit(node=self._node(row["node"]), score=row["score"]) for row in rows]

    def close(self) -> None:
        if self._owns_driver:
            self._driver.close()

    def _ensure_native_indexes(self) -> None:
        rows = self._read(
            "SHOW INDEXES YIELD name, type, state, options "
            "WHERE name IN $names RETURN name, type, state, options",
            names=[self.config.vector_index, self.config.fulltext_index],
        )
        indexes = {row["name"]: row for row in rows}
        for name, kind in (
            (self.config.vector_index, "VECTOR"),
            (self.config.fulltext_index, "FULLTEXT"),
        ):
            if (
                name not in indexes
                or indexes[name]["type"] != kind
                or indexes[name]["state"] != "ONLINE"
            ):
                raise ValueError(f"Existing {kind} index {name!r} must be ONLINE")
        dimensions = (
            indexes[self.config.vector_index]
            .get("options", {})
            .get("indexConfig", {})
            .get("vector.dimensions")
        )
        if dimensions is not None and dimensions != self.dimensions:
            raise ValueError("Existing index dimensions do not match the embedding configuration")

    @staticmethod
    def _native_node(row: dict[str, Any]) -> GraphNode:
        labels = row["labels"]
        properties = dict(row["properties"])
        vector = properties.pop("embedding", None)
        kind: NodeKind = "entity"
        label = next((item for item in labels if not item.startswith("__")), "__Entity__")
        kinds: dict[str, NodeKind] = {
            "Chunk": "chunk",
            "Document": "document",
            "Source": "source",
            "Namespace": "namespace",
            "__Fact__": "fact",
        }
        for candidate, node_kind in kinds.items():
            if candidate in labels:
                label, kind = candidate, node_kind
                break
        if row.get("document_path") and not properties.get("path"):
            properties["path"] = row["document_path"]
        return GraphNode(
            id=row["id"], label=label, kind=kind, properties=properties, embedding=vector
        )

    def _native_nodes(
        self,
        *,
        ids: Sequence[str] | None,
        term: str | None,
        kind: NodeKind | None,
        limit: int | None,
    ) -> list[GraphNode]:
        rows = self._read(
            "MATCH (n) WHERE ($ids IS NULL OR elementId(n) IN $ids) "
            "RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS properties ORDER BY id",
            ids=list(ids) if ids is not None else None,
        )
        nodes = [self._native_node(row) for row in rows]
        if kind is not None:
            nodes = [node for node in nodes if node.kind == kind]
        if term is not None:
            nodes = [
                node
                for node in nodes
                if any(
                    term.casefold() in str(value).casefold()
                    for value in node.properties.values()
                    if isinstance(value, (str, int, float, bool))
                )
            ]
        return nodes[:limit] if limit is not None else nodes

    def _native_edges(self, *, node_ids: Sequence[str] | None) -> list[GraphEdge]:
        rows = self._read(
            "MATCH (a)-[r]->(b) WHERE $ids IS NULL OR elementId(a) IN $ids OR elementId(b) IN $ids "
            "RETURN elementId(r) AS id, elementId(a) AS source, elementId(b) AS target, "
            "type(r) AS type, properties(r) AS properties ORDER BY id",
            ids=list(node_ids) if node_ids is not None else None,
        )
        return [GraphEdge(**row) for row in rows]

    def _native_hits(self, rows: list[dict[str, Any]]) -> list[SearchHit]:
        return [SearchHit(node=self._native_node(row), score=row["score"]) for row in rows]

    def _native_vector_search(
        self,
        embedding: list[float],
        *,
        top_k: int,
        source: str | None,
        kind: Literal["chunk", "fact"],
    ) -> list[SearchHit]:
        if source is not None:
            raise ValueError(
                "Existing database adapter does not support source/namespace isolation"
            )
        if top_k < 1:
            raise ValueError("top_k must be positive")
        validate_vector(embedding, self.dimensions)
        rows = self._read(
            "CALL db.index.vector.queryNodes($index,$top_k,$embedding) YIELD node, score "
            "OPTIONAL MATCH (node)-[:FROM_DOCUMENT]-(document:Document) "
            "WITH node, score, head(collect(document.path)) AS document_path "
            "RETURN elementId(node) AS id, labels(node) AS labels, properties(node) AS properties, "
            "document_path, score ORDER BY score DESC",
            index=self.config.vector_index if kind == "chunk" else self.config.fact_vector_index,
            top_k=top_k,
            embedding=embedding,
        )
        return self._native_hits(rows)

    def _native_fulltext_search(
        self, query: str, *, top_k: int, source: str | None
    ) -> list[SearchHit]:
        if source is not None:
            raise ValueError(
                "Existing database adapter does not support source/namespace isolation"
            )
        if top_k < 1:
            raise ValueError("top_k must be positive")
        tokens = re.findall(r"\w+", query, flags=re.UNICODE)
        if not tokens:
            return []
        rows = self._read(
            "CALL db.index.fulltext.queryNodes($index,$query) YIELD node, score "
            "OPTIONAL MATCH (node)-[:FROM_DOCUMENT]-(document:Document) "
            "WITH node, score, head(collect(document.path)) AS document_path "
            "RETURN elementId(node) AS id, labels(node) AS labels, properties(node) AS properties, "
            "document_path, score ORDER BY score DESC LIMIT $top_k",
            index=self.config.fulltext_index,
            query=" OR ".join(f'"{token}"' for token in tokens),
            top_k=top_k,
        )
        return self._native_hits(rows)
