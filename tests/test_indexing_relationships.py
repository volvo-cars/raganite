"""A relationship whose endpoint was not extracted is dropped with a warning."""

import logging
from types import SimpleNamespace

from raganite.indexing.indexer import kept_relationships


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
