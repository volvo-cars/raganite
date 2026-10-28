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

"""SQLite fact retrieval and indexed recursive traversal."""

import json
import math
from collections.abc import Sequence
from dataclasses import asdict

from ..models import FactHit, GraphExpansion, validate_expansion
from .storage import _SQLiteStorage


def name(alias: str) -> str:
    return f"coalesce(json_extract({alias}.properties,'$.name'),json_extract({alias}.properties,'$.title'),json_extract({alias}.properties,'$.label'),{alias}.id)"


def degree(alias: str) -> str:
    return (
        f"coalesce(json_extract({alias}.properties,'$.semantic_degree'),(SELECT count(*) FROM ("
        f"SELECT id,target AS neighbor FROM edges WHERE namespace={alias}.namespace AND source={alias}.id "
        f"UNION SELECT id,source AS neighbor FROM edges WHERE namespace={alias}.namespace AND target={alias}.id"
        f") adjacency JOIN nodes other ON other.namespace={alias}.namespace AND other.id=adjacency.neighbor "
        "WHERE other.kind='entity'))"
    )


class SQLiteBackend(_SQLiteStorage):
    """Fact-capable SQLite backend with namespace-scoped recursive expansion."""

    def prepare(self) -> None:
        row = (
            self._reader()
            .execute(
                "SELECT 1 FROM nodes WHERE namespace=? AND kind='fact' AND embedding IS NOT NULL LIMIT 1",
                (self.namespace,),
            )
            .fetchone()
        )
        if row is None:
            raise ValueError(
                "Triple retrieval requires embedded facts; enable index_facts before indexing"
            )

    def search_facts(self, embedding: list[float], *, top_k: int = 30) -> list[FactHit]:
        hits = self.vector_search(embedding, top_k=top_k, kind="fact")
        payload = json.dumps([{"id": h.node.id, "score": h.score} for h in hits])
        rows = (
            self._reader()
            .execute(
                "SELECT f.id AS fact_id,json_extract(f.properties,'$.relationship_id') AS relationship_id, "
                "json_extract(f.properties,'$.text') AS text,json_extract(f.properties,'$.relation_type') AS relation_type, "
                "json_extract(hit.value,'$.score') AS score, "
                f"s.id AS subject_id,{name('s')} AS subject,{degree('s')} AS subject_degree, "
                f"o.id AS object_id,{name('o')} AS object,{degree('o')} AS object_degree "
                "FROM json_each(?) hit JOIN nodes f ON f.namespace=? AND f.id=json_extract(hit.value,'$.id') "
                f"JOIN edges es ON es.namespace=f.namespace AND es.source=f.id AND es.type='{self.topology.fact_subject}' "
                f"JOIN edges eo ON eo.namespace=f.namespace AND eo.source=f.id AND eo.type='{self.topology.fact_object}' "
                "JOIN nodes s ON s.namespace=f.namespace AND s.id=es.target AND s.kind='entity' "
                "JOIN nodes o ON o.namespace=f.namespace AND o.id=eo.target AND o.kind='entity' "
                "ORDER BY score DESC",
                (payload, self.namespace),
            )
            .fetchall()
        )
        return [FactHit(**dict(row)) for row in rows]

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
        # Namespace-scoped endpoint predicates use the adjacency indexes.
        # Edge IDs, not visited node IDs, enforce Neo4j relationship trails.
        query = f"""
        WITH RECURSIVE
        walk(relationship_id,fact_text,score,seed,current,depth,penalty,used,types,degrees) AS (
            SELECT json_extract(f.value,'$.relationship_id'),json_extract(f.value,'$.text'),
                   json_extract(f.value,'$.score'),s.id,s.id,0,
                   1.0+:weight*raganite_log1p({degree("s")}),json_array(),json_array(),json_array({degree("s")})
            FROM json_each(:facts) f
            JOIN edges link ON link.namespace=:ns AND link.source=json_extract(f.value,'$.fact_id')
                 AND link.type IN ('{self.topology.fact_subject}','{self.topology.fact_object}')
            JOIN nodes s ON s.namespace=:ns AND s.id=link.target AND s.kind='entity'
            UNION ALL
            SELECT w.relationship_id,w.fact_text,w.score,w.seed,n.id,w.depth+1,
                   w.penalty*(1.0+:weight*raganite_log1p({degree("n")})),
                   json_insert(w.used,'$[#]',e.id),json_insert(w.types,'$[#]',e.type),
                   json_insert(w.degrees,'$[#]',{degree("n")})
            FROM walk w
            JOIN edges e ON e.namespace=:ns AND (e.source=w.current OR e.target=w.current)
            JOIN nodes n ON n.namespace=:ns AND n.id=CASE WHEN e.source=w.current THEN e.target ELSE e.source END
                 AND n.kind='entity'
            WHERE w.depth<:depth AND NOT EXISTS (SELECT 1 FROM json_each(w.used) WHERE value=e.id)
        )
        SELECT c.id AS chunk_id,
               coalesce(json_extract(c.properties,'$.path'),json_extract(c.properties,'$.file_path'),
                        json_extract(c.properties,'$.source'),json_extract(c.properties,'$.uri'),'') AS source,
               coalesce(json_extract(c.properties,'$.text'),json_extract(c.properties,'$.content'),'') AS text,
               w.score/((1.0+w.depth)*w.penalty) AS expansion_score,
               w.relationship_id,w.fact_text,{name("seed")} AS seed_entity,{name("related")} AS related_entity,
               w.types AS graph_path,w.degrees AS path_degrees
        FROM walk w
        JOIN nodes seed ON seed.namespace=:ns AND seed.id=w.seed
        JOIN nodes related ON related.namespace=:ns AND related.id=w.current
        JOIN edges connector ON connector.namespace=:ns AND connector.type='{self.topology.chunk_entity}'
             AND (connector.source=w.current OR connector.target=w.current)
        JOIN nodes c ON c.namespace=:ns AND c.kind='chunk'
             AND c.id=CASE WHEN connector.source=w.current THEN connector.target ELSE connector.source END
        ORDER BY expansion_score DESC,chunk_id LIMIT :limit
        """
        connection = self._reader()
        connection.create_function("raganite_log1p", 1, math.log1p, deterministic=True)
        rows = connection.execute(
            query,
            {
                "ns": self.namespace,
                "facts": json.dumps([asdict(f) for f in facts]),
                "weight": degree_penalty_weight,
                "depth": depth,
                "limit": limit,
            },
        ).fetchall()
        return [
            GraphExpansion(
                **{
                    **dict(row),
                    "graph_path": json.loads(row["graph_path"]),
                    "path_degrees": json.loads(row["path_degrees"]),
                }
            )
            for row in rows
        ]
