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

"""Discover Neo4j structural graph roles independently of the domain ontology."""

from dataclasses import dataclass


def quote(value: str) -> str:
    return "`" + value.replace("`", "``") + "`"


@dataclass(frozen=True)
class TraversalSchema:
    indexed_label: str
    entity_label: str
    entity_connector: str
    source_label: str
    source_connector: str


def discover(backend) -> TraversalSchema:
    rows = backend._read(
        "SHOW INDEXES YIELD name,type,labelsOrTypes WHERE name=$name AND type='VECTOR' RETURN labelsOrTypes",
        name=backend._lookup_index(backend.config.vector_index),
    )
    if not rows or len(rows[0]["labelsOrTypes"] or []) != 1:
        raise ValueError("A single-label chunk vector index is required")
    label = rows[0]["labelsOrTypes"][0]
    path_property = (
        "any(k IN keys(candidate) WHERE toLower(k) IN ['path','file_path','uri','url'] "
        "AND valueType(candidate[k]) STARTS WITH 'STRING')"
    )
    roles = []
    for provenance in (False, True):
        scope = backend._scope("candidate")
        if not backend.native_graph:
            scope += " AND candidate.kind=" + ("'document'" if provenance else "'entity'")
        rows = backend._read(
            f"MATCH (indexed:{quote(label)})-[connector]-(candidate) "
            f"WHERE {backend._scope('indexed')} AND {scope} "
            f"AND {'' if provenance else 'NOT '}{path_property} "
            "AND NOT $label IN labels(candidate) UNWIND labels(candidate) AS candidate_label "
            "RETURN type(connector) AS connector_type,candidate_label,count(*) AS occurrences "
            "ORDER BY occurrences DESC,connector_type,candidate_label LIMIT 1",
            label=label,
        )
        if not rows:
            raise ValueError(
                "Could not discover semantic/provenance connectors from the chunk index"
            )
        roles.append(rows[0])
    return TraversalSchema(
        label,
        roles[0]["candidate_label"],
        roles[0]["connector_type"],
        roles[1]["candidate_label"],
        roles[1]["connector_type"],
    )


def passage_columns(backend, node: str, document: str) -> str:
    sources = []
    for variable in (node, document):
        sources.append(
            f"head([k IN keys({variable}) WHERE toLower(k) IN ['path','file_path','source','uri'] "
            f"AND valueType({variable}[k]) STARTS WITH 'STRING' | {variable}[k]])"
        )
    return (
        f"{backend._id(node)} AS chunk_id, coalesce({','.join(sources)},'') AS source, "
        f"coalesce({node}.text,{node}.content,{node}.description,'') AS text"
    )
