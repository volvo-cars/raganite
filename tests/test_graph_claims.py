"""Exact entity merge, unambiguous native discovery, and an unbuilt free schema."""

import pytest

from raganite.backends.neo4j.traversal import discover
from raganite.config import IndexingConfig
from raganite.indexing.indexer import extraction_schema, resolved_entity_id


def test_entity_merge_collapses_only_the_same_label_and_name():
    paris = resolved_entity_id("notes", "City", "Paris")
    assert resolved_entity_id("notes", "City", "Paris") == paris
    assert resolved_entity_id("notes", "City", "paris") != paris
    assert resolved_entity_id("notes", "City", "Paris ") != paris
    assert resolved_entity_id("notes", "Country", "Paris") != paris


def test_free_mode_does_not_build_the_code_schema():
    config = IndexingConfig()
    assert config.schema_mode == "free"
    assert config.graph_schema is None
    assert extraction_schema(config) is None


def test_fixed_mode_uses_the_code_schema_when_none_is_supplied():
    schema = extraction_schema(IndexingConfig(schema_mode="fixed"))
    labels = {node.label for node in schema.node_types}
    assert "Module" in labels
    assert "Class" in labels


class _Discovery:
    def __init__(self, responses):
        self.responses = list(responses)
        self.native_graph = True
        self.config = type("Config", (), {"vector_index": "chunks"})()

    def _lookup_index(self, name):
        return name

    def _scope(self, _variable):
        return "true"

    def _read(self, _query, **_params):
        return self.responses.pop(0)


def test_discovery_keeps_the_most_frequent_connector():
    schema = discover(
        _Discovery(
            [
                [{"labelsOrTypes": ["Chunk"]}],
                [
                    {"connector_type": "MENTIONS", "candidate_label": "Person", "occurrences": 5},
                    {"connector_type": "ABOUT", "candidate_label": "Topic", "occurrences": 2},
                ],
                [
                    {
                        "connector_type": "FROM_DOCUMENT",
                        "candidate_label": "Document",
                        "occurrences": 3,
                    }
                ],
            ]
        )
    )
    assert schema.indexed_label == "Chunk"
    assert schema.entity_label == "Person"
    assert schema.entity_connector == "MENTIONS"
    assert schema.source_label == "Document"
    assert schema.source_connector == "FROM_DOCUMENT"


def test_discovery_treats_repeated_labels_on_the_same_endpoints_as_one_connector():
    schema = discover(
        _Discovery(
            [
                [{"labelsOrTypes": ["RaganiteNode"]}],
                [
                    {
                        "connector_type": "FROM_CHUNK",
                        "candidate_label": "RaganiteNode",
                        "occurrences": 9,
                    },
                    {
                        "connector_type": "FROM_CHUNK",
                        "candidate_label": "__Entity__",
                        "occurrences": 9,
                    },
                    {
                        "connector_type": "FROM_CHUNK",
                        "candidate_label": "Person",
                        "occurrences": 4,
                    },
                ],
                [
                    {
                        "connector_type": "FROM_DOCUMENT",
                        "candidate_label": "RaganiteNode",
                        "occurrences": 3,
                    },
                    {
                        "connector_type": "FROM_DOCUMENT",
                        "candidate_label": "Document",
                        "occurrences": 3,
                    },
                ],
            ]
        )
    )
    assert schema.entity_label == "RaganiteNode"
    assert schema.entity_connector == "FROM_CHUNK"
    assert schema.source_label == "Document"
    assert schema.source_connector == "FROM_DOCUMENT"


def test_discovery_fails_when_two_domain_labels_share_one_connector():
    with pytest.raises(ValueError, match="Person") as raised:
        discover(
            _Discovery(
                [
                    [{"labelsOrTypes": ["Chunk"]}],
                    [
                        {
                            "connector_type": "MENTIONS",
                            "candidate_label": "Person",
                            "occurrences": 4,
                        },
                        {
                            "connector_type": "MENTIONS",
                            "candidate_label": "Topic",
                            "occurrences": 4,
                        },
                    ],
                ]
            )
        )
    assert "Topic" in str(raised.value)


def test_discovery_fails_when_two_connectors_share_the_top_count():
    with pytest.raises(ValueError, match="Person") as raised:
        discover(
            _Discovery(
                [
                    [{"labelsOrTypes": ["Chunk"]}],
                    [
                        {
                            "connector_type": "MENTIONS",
                            "candidate_label": "Person",
                            "occurrences": 4,
                        },
                        {"connector_type": "ABOUT", "candidate_label": "Topic", "occurrences": 4},
                    ],
                ]
            )
        )
    assert "Topic" in str(raised.value)
