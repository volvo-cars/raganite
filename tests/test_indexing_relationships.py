"""A relationship whose endpoint was not extracted is dropped with a warning."""

import logging
from types import SimpleNamespace

from langchain_core.documents import Document
from neo4j_graphrag.components.types import Neo4jGraph, Neo4jNode, Neo4jRelationship

from raganite.backends.sqlite import SQLiteBackend
from raganite.config import RaganiteConfig, SQLiteConfig
from raganite.indexing.indexer import (
    KnowledgeGraphIndexer,
    _NamedExtractionTemplate,
    kept_relationships,
    passage_title,
)
from raganite.schema import default_schema


def test_dangling_relationship_is_dropped_and_warned(caplog):
    relationships = [
        SimpleNamespace(start_node_id="0", end_node_id="1", type="KNOWS", properties={}),
        SimpleNamespace(
            start_node_id="0", end_node_id="", type="LIVES_IN", properties={"since": "2024"}
        ),
    ]
    mapped = {"0": "person-ada", "1": "person-bea"}
    with caplog.at_level(logging.WARNING):
        kept = kept_relationships(relationships, mapped, path="conv-41/session_011.txt")
    assert [(edge.source, edge.type, edge.target) for edge in kept] == [
        ("person-ada", "KNOWS", "person-bea")
    ]
    assert "conv-41/session_011.txt" in caplog.text
    assert "LIVES_IN" in caplog.text


class _Embeddings:
    async def aembed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]


class _FilmExtractor:
    def __init__(self, *args, **kwargs):
        return None

    async def run(self, chunks, document_info, schema=None, examples="", **kwargs):
        nodes = [
            Neo4jNode(
                id=document_info.uid, label="Document", properties={"path": document_info.path}
            ),
            Neo4jNode(id="person", label="Person", properties={"name": "Chris Evans"}),
            Neo4jNode(
                id="film",
                label="Film",
                properties={"title": "The Newcomers", "release_year": 2000},
            ),
        ]
        nodes.extend(
            Neo4jNode(
                id=chunk.uid, label="Chunk", properties={"text": chunk.text, "index": chunk.index}
            )
            for chunk in chunks.chunks
        )
        return Neo4jGraph(
            nodes=nodes,
            relationships=[
                Neo4jRelationship(start_node_id="person", end_node_id="film", type="STARS_IN")
            ],
        )


def test_a_missing_name_uses_the_title_in_the_fact(monkeypatch):
    monkeypatch.setattr("raganite.indexing.indexer.LLMEntityRelationExtractor", _FilmExtractor)
    backend = SQLiteBackend(SQLiteConfig(path=":memory:"), namespace="demo", dimensions=2)
    indexer = KnowledgeGraphIndexer(
        backend,
        object(),
        _Embeddings(),
        RaganiteConfig(namespace="demo", embedding_dimensions=2),
    )
    report = indexer.index_documents(
        [Document(page_content="Chris Evans stars in The Newcomers.", metadata={"path": "film.txt"})],
        source="demo",
    )
    assert report.failed_files == []
    texts = [node.properties["text"] for node in backend.nodes(kind="fact")]
    film = next(node for node in backend.nodes(kind="entity") if node.label == "Film")

    assert "Chris Evans | stars in | The Newcomers" in texts
    assert film.properties["name"] == "The Newcomers"


def test_extraction_asks_for_a_name_and_the_fixed_schema_includes_one():
    prompt = _NamedExtractionTemplate().format(schema={}, examples="", text="A film.")
    assert "properties.name" in prompt
    assert "not the label" in prompt
    assert all(
        any(prop.name == "name" for prop in node.properties) for node in default_schema().node_types
    )


def test_a_document_title_is_kept_for_its_passages():
    assert passage_title({"path": "doc_00011.txt", "title": " Leslie West "}) == "Leslie West"
    assert passage_title({"name": "Harbor Authority"}) == "Harbor Authority"
    assert passage_title({"path": "notes.txt"}) == ""
    assert passage_title(None) == ""
