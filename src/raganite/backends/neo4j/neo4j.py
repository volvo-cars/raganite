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

"""Neo4j passage search and beam snapshots."""

from collections.abc import Sequence
from functools import cached_property

from ..base import validate_vector
from ..models import ChunkRecord, FactEdge, PassageHit
from ..snapshot import stored_chunk_records, stored_fact_edges
from .storage import _Neo4jStorage
from .traversal import TraversalSchema, discover, passage_columns, quote


class Neo4jBackend(_Neo4jStorage):
    """Fact-capable adapter for owned namespaces or an existing native graph."""

    _traversal_schema: TraversalSchema | None = None

    def _id(self, variable: str) -> str:
        return f"elementId({variable})" if self.native_graph else f"{variable}.raganite_id"

    def _scope(self, variable: str) -> str:
        return (
            "true"
            if self.native_graph
            else f"{variable}:RaganiteNode AND {variable}.namespace=$namespace"
        )

    def _schema(self) -> TraversalSchema:
        if self._traversal_schema is None:
            self._traversal_schema = discover(self)
        return self._traversal_schema

    @cached_property
    def _native_chunk_index(self) -> tuple[str, str]:
        return self._native_node_vector_schema(self.config.vector_index, "vector_index")

    @cached_property
    def _native_fact_index(self) -> tuple[str, str]:
        return self._native_node_vector_schema(self.config.fact_vector_index, "fact_vector_index")

    def _native_node_vector_schema(self, name: str, setting: str) -> tuple[str, str]:
        rows = self._read(
            "SHOW VECTOR INDEXES YIELD name, entityType, labelsOrTypes, properties "
            "WHERE name=$name AND entityType='NODE' RETURN labelsOrTypes, properties",
            name=self._lookup_index(name),
        )
        if rows:
            labels = rows[0]["labelsOrTypes"] or []
            properties = rows[0]["properties"] or []
            if len(labels) == len(properties) == 1:
                return labels[0], properties[0]
        raise ValueError(
            f"Native beam retrieval requires {setting}={name!r} to name "
            "a single-label, single-property node vector index"
        )

    def _name(self, variable: str) -> str:
        parts = [f"{variable}.name", f"{variable}.title", f"{variable}.label"]
        if self.native_graph:
            parts.append(f"{variable}.id")
        parts.append(self._id(variable))
        return "coalesce(" + ",".join(parts) + ")"

    def fact_edges(self) -> list[FactEdge]:
        subject = quote(self.topology.fact_subject)
        obj = quote(self.topology.fact_object)
        if not self.native_graph:
            return stored_fact_edges(
                self._read(
                    "MATCH (fact:RaganiteNode {namespace:$namespace, kind:'fact'}) "
                    "WHERE fact.embedding IS NOT NULL "
                    f"MATCH (fact)-[:{subject} {{namespace:$namespace}}]->"
                    "(s:RaganiteNode {namespace:$namespace, kind:'entity'}) "
                    f"MATCH (fact)-[:{obj} {{namespace:$namespace}}]->"
                    "(o:RaganiteNode {namespace:$namespace, kind:'entity'}) "
                    "RETURN fact.properties_json AS properties,fact.embedding AS embedding,"
                    "s.raganite_id AS subject_id,s.label AS subject_label,s.properties_json AS subject_properties,"
                    "o.raganite_id AS object_id,o.label AS object_label,o.properties_json AS object_properties"
                )
            )
        label, vector_property = self._native_fact_index
        rows = self._read(
            f"MATCH (fact:{quote(label)}) WHERE fact[$vector_property] IS NOT NULL "
            f"MATCH (fact)-[:{subject}]->(subject) "
            f"MATCH (fact)-[:{obj}]->(object) "
            f"RETURN {self._name('subject')} AS source,{self._name('object')} AS target,"
            "fact.text AS text,fact.relation_type AS relation_type,"
            "fact.relationship_id AS relationship_id,fact[$vector_property] AS embedding",
            vector_property=vector_property,
        )
        edges = []
        for row in rows:
            if not row["source"] or not row["target"] or not row["text"] or not row["embedding"]:
                continue
            edges.append(
                FactEdge(
                    source=row["source"],
                    target=row["target"],
                    text=row["text"],
                    relation_type=row["relation_type"] or "",
                    relationship_id=row["relationship_id"] or "",
                    embedding=tuple(row["embedding"]),
                )
            )
        return edges

    def chunk_records(self) -> list[ChunkRecord]:
        if self.native_graph:
            label, vector_property = self._native_chunk_index
            rows = self._read(
                f"MATCH (chunk:{quote(label)}) WHERE chunk[$vector_property] IS NOT NULL "
                f"RETURN {self._id('chunk')} AS chunk_id,"
                "coalesce(chunk.path, chunk.file_path, chunk.source, chunk.uri, '') AS path,"
                "coalesce(chunk.text, chunk.content, chunk.description, '') AS text,"
                "coalesce(chunk.title, chunk.name, '') AS title ORDER BY chunk_id",
                vector_property=vector_property,
            )
            return [
                ChunkRecord(
                    chunk_id=row["chunk_id"],
                    path=row["path"] or "",
                    text=row["text"] or "",
                    title=(row["title"] or "").strip(),
                )
                for row in rows
                if row["chunk_id"] and row["text"]
            ]
        rows = self._read(
            "MATCH (chunk:RaganiteNode {namespace:$namespace, kind:'chunk'}) "
            "OPTIONAL MATCH (owner:RaganiteOwner {namespace:$namespace})-[:OWNS_NODE]->(chunk) "
            "OPTIONAL MATCH (owner)-[:OWNS_NODE]->(doc:RaganiteNode {namespace:$namespace, kind:'document'}) "
            "RETURN chunk.raganite_id AS chunk_id,chunk.properties_json AS properties,"
            "owner.source AS owner_source,owner.path AS owner_path,"
            "doc.raganite_id AS doc_id,doc.properties_json AS document_properties"
        )
        links = self._read(
            "MATCH (owner:RaganiteOwner {namespace:$namespace})-[:OWNS_NODE]->"
            "(chunk:RaganiteNode {namespace:$namespace, kind:'chunk'}) "
            "MATCH (owner)-[:OWNS_NODE]->(entity:RaganiteNode {namespace:$namespace, kind:'entity'}) "
            "RETURN DISTINCT chunk.raganite_id AS chunk_id, entity.raganite_id AS entity_id,"
            "entity.label AS label, entity.properties_json AS properties"
        )
        return stored_chunk_records(rows, links)

    def chunk_embeddings(self, chunk_ids: Sequence[str]) -> dict[str, tuple[float, ...]]:
        ids = sorted(set(chunk_ids))
        if not ids:
            return {}
        if self.native_graph:
            label, vector_property = self._native_chunk_index
            chunk_id = self._id("chunk")
            rows = self._read(
                "UNWIND $ids AS id "
                f"MATCH (chunk:{quote(label)}) "
                f"WHERE {chunk_id}=id AND chunk[$vector_property] IS NOT NULL "
                f"RETURN {chunk_id} AS chunk_id, chunk[$vector_property] AS embedding",
                ids=ids,
                vector_property=vector_property,
            )
        else:
            rows = self._read(
                "UNWIND $ids AS id "
                "MATCH (chunk:RaganiteNode {namespace:$namespace, kind:'chunk', raganite_id:id}) "
                "WHERE chunk.embedding IS NOT NULL "
                "RETURN chunk.raganite_id AS chunk_id, chunk.embedding AS embedding",
                ids=ids,
            )
        return {
            row["chunk_id"]: tuple(row["embedding"])
            for row in sorted(rows, key=lambda row: row["chunk_id"])
        }

    def direct_passages(self, embedding: list[float], *, top_k: int) -> list[PassageHit]:
        if not self.native_graph:
            return super().direct_passages(embedding, top_k=top_k)
        validate_vector(embedding, self.dimensions)
        schema = self._schema()
        rows = self._read(
            "CALL db.index.vector.queryNodes($index,$top_k,$embedding) YIELD node,score "
            f"OPTIONAL MATCH (node)-[:{quote(schema.source_connector)}]-(document:{quote(schema.source_label)}) "
            f"RETURN {passage_columns(self, 'node', 'document')},score ORDER BY score DESC",
            index=self._index(self.config.vector_index),
            top_k=top_k,
            embedding=embedding,
        )
        return [PassageHit(**row) for row in rows]
