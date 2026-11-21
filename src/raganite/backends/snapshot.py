"""Canonical records for Raganite-owned graphs, built from stored properties.

Native Neo4j layouts do not use this contract. Database ANN and full-text scores
are also outside it; snapshot embeddings come from storage, not an index.
"""

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from .models import ChunkRecord, FactEdge


def _string(properties: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = properties.get(key)
        if isinstance(value, str):
            return value
    return ""


def _title(properties: Mapping[str, Any]) -> str:
    for key in ("title", "name"):
        value = _string(properties, key).strip()
        if value:
            return value
    return ""


def _entity(properties_json: str, label: str, id_: str) -> str:
    properties = json.loads(properties_json)
    return (
        _string(properties, "name").strip() or _string(properties, "title").strip() or label or id_
    )


def stored_fact_edges(rows: Iterable[Mapping[str, Any]]) -> list[FactEdge]:
    facts = []
    for row in rows:
        properties = json.loads(row["properties"])
        text = _string(properties, "text")
        if text:
            facts.append(
                FactEdge(
                    source=_entity(
                        row["subject_properties"], row["subject_label"], row["subject_id"]
                    ),
                    target=_entity(row["object_properties"], row["object_label"], row["object_id"]),
                    text=text,
                    relation_type=_string(properties, "relation_type"),
                    relationship_id=_string(properties, "relationship_id"),
                    embedding=tuple(row["embedding"]),
                )
            )
    return sorted(
        facts,
        key=lambda fact: (
            fact.relationship_id,
            fact.source,
            fact.target,
            fact.relation_type,
            fact.text,
            fact.embedding,
        ),
    )


def stored_chunk_records(
    rows: Iterable[Mapping[str, Any]], links: Iterable[Mapping[str, Any]]
) -> list[ChunkRecord]:
    entities: dict[str, set[str]] = defaultdict(set)
    for link in links:
        entities[link["chunk_id"]].add(
            _entity(link["properties"], link["label"], link["entity_id"])
        )

    chunks: dict[str, tuple[str, str, str]] = {}
    # A chunk normally has one owner and one document. Resolve ambiguous
    # provenance by ID instead of letting either engine choose the first row.
    for row in sorted(
        rows,
        key=lambda row: (
            row["chunk_id"],
            row["doc_id"] or "",
            row["owner_source"] or "",
            row["owner_path"] or "",
        ),
    ):
        chunk_id = row["chunk_id"]
        if chunk_id not in chunks:
            properties = json.loads(row["properties"])
            chunks[chunk_id] = (
                _string(properties, "path", "file_path") or row["owner_path"] or "",
                _string(properties, "text", "content"),
                _title(properties),
            )
        path, text, title = chunks[chunk_id]
        if not title and row["document_properties"]:
            document = json.loads(row["document_properties"])
            metadata = document.get("metadata")
            title = _title(document) or (_title(metadata) if isinstance(metadata, dict) else "")
            chunks[chunk_id] = (path, text, title)
    return [
        ChunkRecord(id_, path, text, title, tuple(sorted(entities.get(id_, ()))))
        for id_, (path, text, title) in chunks.items()
    ]
