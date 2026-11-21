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

"""SQLite snapshots for the beam walk."""

import json
from collections.abc import Sequence

from ..models import ChunkRecord, FactEdge
from ..snapshot import stored_chunk_records, stored_fact_edges
from .storage import _SQLiteStorage


class SQLiteBackend(_SQLiteStorage):
    """SQLite backend with namespace-scoped beam snapshots."""

    def fact_edges(self) -> list[FactEdge]:
        rows = self._rows(
            "SELECT f.properties,f.embedding,"
            "s.id AS subject_id,s.label AS subject_label,s.properties AS subject_properties,"
            "o.id AS object_id,o.label AS object_label,o.properties AS object_properties "
            "FROM nodes f "
            f"JOIN edges es ON es.namespace=f.namespace AND es.source=f.id AND es.type='{self.topology.fact_subject}' "
            "JOIN nodes s ON s.namespace=f.namespace AND s.id=es.target AND s.kind='entity' "
            f"JOIN edges eo ON eo.namespace=f.namespace AND eo.source=f.id AND eo.type='{self.topology.fact_object}' "
            "JOIN nodes o ON o.namespace=f.namespace AND o.id=eo.target AND o.kind='entity' "
            "WHERE f.namespace=? AND f.kind='fact' AND f.embedding IS NOT NULL",
            (self.namespace,),
        )
        return stored_fact_edges(dict(row, embedding=json.loads(row["embedding"])) for row in rows)

    def chunk_records(self) -> list[ChunkRecord]:
        rows = self._rows(
            "WITH docs AS ("
            "SELECT dn.source,dn.path,d.id,d.properties FROM nodes d "
            "JOIN document_nodes dn ON dn.namespace=d.namespace AND dn.node_id=d.id "
            "WHERE d.namespace=:ns AND d.kind='document') "
            "SELECT c.id AS chunk_id,c.properties,owner.source AS owner_source,"
            "owner.path AS owner_path,d.id AS doc_id,d.properties AS document_properties "
            "FROM nodes c "
            "LEFT JOIN document_nodes owner ON owner.namespace=c.namespace AND owner.node_id=c.id "
            "LEFT JOIN docs d ON d.source=owner.source AND d.path=owner.path "
            "WHERE c.namespace=:ns AND c.kind='chunk'",
            {"ns": self.namespace},
        )
        links = self._rows(
            "SELECT DISTINCT c.id AS chunk_id,e.id AS entity_id,e.label,e.properties "
            "FROM nodes c "
            "JOIN document_nodes co ON co.namespace=c.namespace AND co.node_id=c.id "
            "JOIN document_nodes eo ON eo.namespace=co.namespace AND eo.source=co.source "
            "AND eo.path=co.path "
            "JOIN nodes e ON e.namespace=eo.namespace AND e.id=eo.node_id AND e.kind='entity' "
            "WHERE c.namespace=? AND c.kind='chunk'",
            (self.namespace,),
        )
        return stored_chunk_records(map(dict, rows), map(dict, links))

    def chunk_embeddings(self, chunk_ids: Sequence[str]) -> dict[str, tuple[float, ...]]:
        if not chunk_ids:
            return {}
        rows = self._rows(
            "SELECT id,embedding FROM nodes WHERE namespace=? AND kind='chunk' "
            "AND embedding IS NOT NULL AND id IN (SELECT value FROM json_each(?)) ORDER BY id",
            (self.namespace, json.dumps(list(chunk_ids))),
        )
        return {row["id"]: tuple(json.loads(row["embedding"])) for row in rows}
