"""Owned Neo4j stores entity degree and does not read properties it never writes."""

from raganite.backends.base import DocumentGraph, GraphEdge, GraphNode
from raganite.backends.models import FactHit
from raganite.backends.neo4j import Neo4jBackend
from raganite.backends.neo4j.storage import _Neo4jStorage
from raganite.backends.neo4j.traversal import TraversalSchema
from raganite.config import Neo4jConfig


class _Result:
    def consume(self):
        return None

    def data(self):
        return {}

    def single(self):
        return {"deleted": 0}

    def __iter__(self):
        return iter(())


class _Transaction:
    def __init__(self, queries: list[str]):
        self.queries = queries

    def run(self, cypher: str, parameters=None):
        self.queries.append(cypher)
        return _Result()


class _Session:
    def __init__(self, queries: list[str]):
        self.queries = queries

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute_write(self, work):
        return work(_Transaction(self.queries))

    def execute_read(self, work):
        return work(_Transaction(self.queries))


class _Driver:
    def __init__(self):
        self.queries: list[str] = []

    def session(self, **kwargs):
        return _Session(self.queries)

    def close(self):
        return None


def _backend() -> tuple[Neo4jBackend, _Driver]:
    driver = _Driver()
    backend = Neo4jBackend(
        Neo4jConfig(password="secret"), namespace="demo", dimensions=2, driver=driver
    )
    return backend, driver


def _graph() -> DocumentGraph:
    chunk = GraphNode(
        id="chunk", label="Chunk", kind="chunk", properties={"text": "Paris", "path": "a.md"}
    )
    paris = GraphNode(id="paris", label="City", kind="entity", properties={"name": "Paris"})
    france = GraphNode(id="france", label="Country", kind="entity", properties={"name": "France"})
    return DocumentGraph(
        source="docs",
        path="a.md",
        content_hash="h",
        nodes=[chunk, paris, france],
        edges=[
            GraphEdge(id="rel", source="paris", target="france", type="CAPITAL"),
            GraphEdge(id="mention", source="paris", target="chunk", type="FROM_CHUNK"),
        ],
    )


def test_fact_lookup_reads_the_stored_degree_and_skips_id():
    backend, driver = _backend()
    try:
        backend.search_facts([1.0, 0.0], top_k=1)
    finally:
        backend.close()
    query = next(item for item in driver.queries if "FACT_SUBJECT" in item)
    assert "subject.id" not in query
    assert "object.id" not in query
    assert "COUNT {" not in query
    assert "semantic_degree" in query


def test_expansion_reads_the_stored_degree_and_skips_id():
    backend, driver = _backend()
    backend._traversal_schema = TraversalSchema(
        "Chunk", "__Entity__", "FROM_CHUNK", "Document", "FROM_DOCUMENT"
    )
    fact = FactHit(
        "fact", "rel", "Paris", "CAPITAL", 1.0, "paris", "Paris", 1.0, "france", "France", 1.0
    )
    try:
        backend.expand_facts([fact], depth=1, limit=10)
    finally:
        backend.close()
    query = next(item for item in driver.queries if "expansion_score" in item)
    assert "seed.id" not in query
    assert "related.id" not in query
    assert ".body" not in query
    assert "COUNT {" not in query
    assert "semantic_degree" in query


def test_writing_a_document_stores_entity_degree_without_counting_chunks():
    backend, driver = _backend()
    try:
        backend.replace_document(_graph())
    finally:
        backend.close()
    query = next(item for item in driver.queries if "semantic_degree" in item)
    assert "kind:'entity'" in query or "kind: 'entity'" in query
    assert "SET n.semantic_degree" in query


def test_shared_edges_refresh_the_degree_of_their_endpoints():
    backend, driver = _backend()
    edge = GraphEdge(id="next", source="paris", target="france", type="NEXT")
    try:
        backend.store_shared_edges([(edge, (("docs", "a.md"), ("docs", "b.md")))])
    finally:
        backend.close()
    query = next(item for item in driver.queries if "semantic_degree" in item)
    assert "SET n.semantic_degree" in query


def test_opening_an_owned_graph_fills_missing_degrees():
    backend, driver = _backend()

    def read(cypher: str, **params):
        if "m.dimensions" in cypher:
            return [{"dimensions": 2}]
        if "semantic_degree IS NULL" in cypher:
            return [{"missing": 1}]
        return []

    backend._read = read
    try:
        backend.ensure_indexes()
    finally:
        backend.close()
    assert any("SET n.semantic_degree" in query for query in driver.queries)


def test_a_later_namespace_reuses_the_existing_vector_index():
    backend, _driver = _backend()
    calls: list[tuple[str, dict]] = []

    def read(cypher: str, **params):
        calls.append((cypher, params))
        if "SHOW INDEXES" in cypher:
            return [
                {
                    "name": "raganite_chunks_vector_existing",
                    "type": "VECTOR",
                    "labelsOrTypes": ["Chunk"],
                    "properties": ["embedding"],
                }
            ]
        return [
            {
                "node": {
                    "raganite_id": "chunk",
                    "label": "Chunk",
                    "kind": "chunk",
                    "properties_json": '{"text": "Paris"}',
                },
                "score": 1.0,
            }
        ]

    backend._read = read
    try:
        backend.vector_search([1.0, 0.0], top_k=1)
    finally:
        backend.close()
    queried = next(params["index"] for cypher, params in calls if "queryNodes" in cypher)
    assert queried == "raganite_chunks_vector_existing"


def test_a_read_node_keeps_the_stored_degree():
    node = _Neo4jStorage._node(
        {
            "raganite_id": "paris",
            "label": "City",
            "kind": "entity",
            "properties_json": '{"name": "Paris"}',
            "semantic_degree": 1.0,
        }
    )
    assert node.properties["semantic_degree"] == 1.0
