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

"""Neo4j fact search and schema-adaptive expansion; no APOC or Python graph scans."""

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from ..base import validate_vector
from ..models import FactHit, GraphExpansion, PassageHit, validate_expansion
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

    def _entity(self, variable: str) -> str:
        if self.native_graph:
            return f"{variable}:{quote(self._schema().entity_label)}"
        return f"{self._scope(variable)} AND {variable}.kind='entity'"

    def _name(self, variable: str) -> str:
        parts = [f"{variable}.name", f"{variable}.title", f"{variable}.label"]
        if self.native_graph:
            parts.append(f"{variable}.id")
        parts.append(self._id(variable))
        return "coalesce(" + ",".join(parts) + ")"

    def _degree(self, variable: str) -> str:
        # The live edge count takes relationship locks and deadlocks under concurrent reads.
        return f"coalesce(toFloat({variable}.semantic_degree), 0.0)"

    def prepare(self) -> None:
        self._schema()
        rows = self._read(
            "SHOW INDEXES YIELD name,type,state,options WHERE name=$name RETURN type,state,options",
            name=self._lookup_index(self.config.fact_vector_index),
        )
        if not rows or rows[0]["type"] != "VECTOR" or rows[0]["state"] != "ONLINE":
            raise ValueError(
                "Triple retrieval requires an ONLINE fact vector index; index facts first"
            )
        dimensions = rows[0].get("options", {}).get("indexConfig", {}).get("vector.dimensions")
        if dimensions is not None and dimensions != self.dimensions:
            raise ValueError("Fact index dimensions do not match the embedding configuration")

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

    def search_facts(self, embedding: list[float], *, top_k: int = 30) -> list[FactHit]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        validate_vector(embedding, self.dimensions)
        params: dict[str, Any] = {"embedding": embedding, "top_k": top_k}
        if self.native_graph:
            start = "CALL db.index.vector.queryNodes($index,$top_k,$embedding) YIELD node AS fact,score "
            params["index"] = self._index(self.config.fact_vector_index)
        else:
            hits = self.vector_search(embedding, top_k=top_k, kind="fact")
            params["hits"] = [{"id": h.node.id, "score": h.score} for h in hits]
            start = "UNWIND $hits AS hit MATCH (fact:RaganiteNode {namespace:$namespace,raganite_id:hit.id}) WITH fact,hit.score AS score "
        rows = self._read(
            start + "MATCH (fact)-[:FACT_SUBJECT]->(subject) MATCH (fact)-[:FACT_OBJECT]->(object) "
            f"WHERE {self._entity('subject')} AND {self._entity('object')} "
            f"RETURN {self._id('fact')} AS fact_id,fact.relationship_id AS relationship_id, "
            "fact.text AS text,fact.relation_type AS relation_type,score, "
            f"{self._id('subject')} AS subject_id,{self._name('subject')} AS subject, "
            f"{self._degree('subject')} AS subject_degree, "
            f"{self._id('object')} AS object_id,{self._name('object')} AS object, "
            f"{self._degree('object')} AS object_degree ORDER BY score DESC",
            **params,
        )
        return [FactHit(**row) for row in rows]

    def expand_facts(
        self,
        facts: Sequence[FactHit],
        *,
        depth: int = 2,
        limit: int = 5000,
        degree_penalty_weight: float = 0.25,
    ) -> list[GraphExpansion]:
        validate_expansion(depth, limit, degree_penalty_weight)
        if not facts:
            return []
        schema = self._schema()
        rows = self._read(
            f"UNWIND $facts AS selected MATCH (fact:__Fact__) WHERE {self._scope('fact')} AND {self._id('fact')}=selected.fact_id "
            f"MATCH (fact)-[:FACT_SUBJECT|FACT_OBJECT]->(seed:{quote(schema.entity_label)}) "
            f"MATCH path=(seed)-[*0..{depth}]-(related:{quote(schema.entity_label)}) "
            f"WHERE all(entity IN nodes(path) WHERE {self._entity('entity')}) "
            f"MATCH (expanded:{quote(schema.indexed_label)})-[:{quote(schema.entity_connector)}]-(related) "
            f"WHERE {self._scope('expanded')} "
            f"OPTIONAL MATCH (expanded)-[:{quote(schema.source_connector)}]-(document:{quote(schema.source_label)}) "
            f"WITH selected,seed,related,path,expanded,document,[entity IN nodes(path) | {self._degree('entity')}] AS degrees "
            "WITH *,selected.score/((1.0+length(path))*reduce(p=1.0,d IN degrees | p*(1.0+$weight*log(1.0+d)))) AS expansion_score "
            "ORDER BY expansion_score DESC LIMIT $limit "
            f"RETURN {passage_columns(self, 'expanded', 'document')},expansion_score, "
            "selected.relationship_id AS relationship_id,selected.text AS fact_text, "
            f"{self._name('seed')} AS seed_entity, "
            f"{self._name('related')} AS related_entity, "
            "degrees AS path_degrees,[r IN relationships(path) | type(r)] AS graph_path",
            facts=[asdict(f) for f in facts],
            weight=degree_penalty_weight,
            limit=limit,
        )
        return [GraphExpansion(**row) for row in rows]
