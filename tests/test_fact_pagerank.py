"""Fact-weighted PageRank ranks a place by its own file, along the closer sentence."""

import math
import time

import pytest
from langchain_core.embeddings import Embeddings

from raganite.backends.base import DocumentGraph, GraphEdge, GraphNode, cosine
from raganite.backends.models import ChunkRecord, FactEdge
from raganite.backends.sqlite import SQLiteBackend
from raganite.config import RetrievalConfig, SQLiteConfig
from raganite.retrieval.fact_walk.graph import FactWalkGraph
from raganite.retrieval.fact_walk.scoring import (
    closest_fact_seeds,
    home_scores,
)


def test_sparse_walk_without_a_known_positive_seed_returns_no_homes():
    graph = FactWalkGraph.from_edges(
        [FactEdge("Harbor", "Observatory", "The harbor overlooks it.", "R", "fact", (1.0,))],
        [ChunkRecord("observatory", "observatory.txt", "The observatory.")],
    )
    walk = graph.walk([1.0], tau=0.1, damping=0.85, seeds={"Unknown": 1.0, "Harbor": 0.0})
    assert (walk.seeds, walk.homes, walk.incoming) == (0, [], {})


def test_the_closer_fact_seeds_its_endpoints_and_a_common_name_shrinks():
    weights = closest_fact_seeds(
        [
            ("The Newcomers", "Chris Evans", 0.9),
            ("Film", "Movie", 0.8),
            ("Unrelated", "Other", 0.1),
        ],
        {"The Newcomers": 1, "Chris Evans": 1, "Film": 10, "Movie": 10},
        limit=2,
    )

    assert weights["The Newcomers"] == pytest.approx(0.9)
    assert weights["Chris Evans"] == pytest.approx(0.9)
    assert weights["Film"] == pytest.approx(0.08)
    assert "Unrelated" not in weights


def test_only_the_place_file_receives_its_mass():
    ranked = home_scores(
        {"Lighthouse Keepers": 0.4, "Harbor Authority": 0.1},
        [
            {"chunk_id": "keepers", "path": "lighthouse-keepers.txt", "text": "keepers"},
            {"chunk_id": "bell", "path": "bell-casters.txt", "text": "mentions the keepers"},
            {"chunk_id": "harbor", "path": "harbor-authority.txt", "text": "harbor"},
        ],
    )

    scores = {row["chunk_id"]: row["score"] for row in ranked}
    assert scores["keepers"] == 0.4
    assert scores["harbor"] == 0.1
    assert "bell" not in scores


def test_the_named_file_keeps_the_mass_when_another_passage_also_produced_it():
    ranked = home_scores(
        {"Lighthouse Keepers": 0.4},
        [
            {
                "chunk_id": "keepers",
                "path": "lighthouse-keepers.txt",
                "text": "keepers",
                "entities": ["Lighthouse Keepers"],
            },
            {
                "chunk_id": "bell",
                "path": "bell-casters.txt",
                "text": "mentions the keepers",
                "entities": ["Lighthouse Keepers"],
            },
        ],
    )

    assert [row["chunk_id"] for row in ranked] == ["keepers"]
    assert ranked[0]["score"] == 0.4


def test_a_title_keeps_the_mass_when_the_file_name_does_not():
    ranked = home_scores(
        {"Leslie West": 0.4, "Steve Perry": 0.2},
        [
            {
                "chunk_id": "west",
                "path": "doc_00011.txt",
                "title": "Leslie West",
                "text": "west",
                "entities": ["Leslie West"],
            },
            {
                "chunk_id": "perry",
                "path": "doc_00020.txt",
                "title": "Steve Perry (Oregon musician)",
                "text": "perry",
                "entities": ["Steve Perry", "Leslie West"],
            },
            {
                "chunk_id": "mention",
                "path": "doc_00099.txt",
                "title": "Climbing!",
                "text": "mentions Leslie West",
                "entities": ["Climbing"],
            },
        ],
    )

    scores = {row["chunk_id"]: row["score"] for row in ranked}
    assert scores["west"] == 0.4
    assert scores["perry"] == 0.2
    assert "mention" not in scores


def test_mass_is_split_across_the_passages_that_produced_the_entity():
    ranked = home_scores(
        {"Harbor": 0.4},
        [
            {
                "chunk_id": "a",
                "path": "doc_00001.txt",
                "title": "Notes",
                "text": "a",
                "entities": ["Harbor"],
            },
            {
                "chunk_id": "b",
                "path": "doc_00002.txt",
                "title": "Log",
                "text": "b",
                "entities": ["Harbor"],
            },
        ],
    )

    scores = {row["chunk_id"]: row["score"] for row in ranked}
    assert scores["a"] == pytest.approx(0.2)
    assert scores["b"] == pytest.approx(0.2)


class _Query(Embeddings):
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


def _node(node_id: str, kind: str, label: str, **properties) -> GraphNode:
    embedding = properties.pop("embedding", None)
    return GraphNode(id=node_id, label=label, kind=kind, properties=properties, embedding=embedding)


def test_every_question_and_worker_reuses_one_fact_graph():
    from raganite.retrieval import Retriever

    class CountingBackend(SQLiteBackend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fact_loads = 0
            self.chunk_loads = 0

        def fact_edges(self):
            self.fact_loads += 1
            time.sleep(0.3)
            return super().fact_edges()

        def chunk_records(self):
            self.chunk_loads += 1
            return super().chunk_records()

    backend = CountingBackend(SQLiteConfig(path=":memory:"), namespace="demo", dimensions=2)
    grain = _node("grain", "entity", "Organization", name="Grain Cooperative")
    harbor = _node("harbor", "entity", "Authority", name="Harbor Authority")
    fact = _node(
        "sends",
        "fact",
        "__Fact__",
        text="Harbor Authority | supplies | Grain Cooperative",
        relation_type="SUPPLIES",
        relationship_id="sends",
        embedding=[1.0, 0.0],
    )
    chunks = [
        _node(
            "c-grain",
            "chunk",
            "Chunk",
            path="grain-cooperative.txt",
            text="grain",
            embedding=[1.0, 0.0],
        ),
        _node(
            "c-harbor",
            "chunk",
            "Chunk",
            path="harbor-authority.txt",
            text="harbor",
            embedding=[1.0, 0.0],
        ),
    ]
    edges = [
        GraphEdge(id="s", source="sends", target="harbor", type="FACT_SUBJECT"),
        GraphEdge(id="o", source="sends", target="grain", type="FACT_OBJECT"),
    ]
    backend.replace_document(
        DocumentGraph(
            source="city",
            path="graph",
            content_hash="h",
            nodes=[grain, harbor, fact, *chunks],
            edges=edges,
        )
    )
    retriever = Retriever(
        backend,
        _Query(),
        None,
        RetrievalConfig(
            use_triples=True,
            use_fulltext=False,
            rerank_passages=False,
            workers=2,
            top_k=5,
        ),
    )
    try:
        first, second = retriever.retrieve_many(
            [
                "What does the Harbor Authority supply to the Grain Cooperative?",
                "Where does the Grain Cooperative get its supply from the Harbor Authority?",
            ]
        )
        again = retriever.retrieve("Grain Cooperative and Harbor Authority")
    finally:
        retriever.close()
        backend.close()

    assert backend.fact_loads == 1
    assert backend.chunk_loads == 1
    assert [passage.title for passage in first] == [passage.title for passage in second]
    assert {passage.title for passage in again} == {"grain-cooperative", "harbor-authority"}


def test_a_write_after_the_snapshot_is_visible_on_the_next_question():
    from raganite.retrieval import Retriever

    class CountingBackend(SQLiteBackend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fact_loads = 0

        def fact_edges(self):
            self.fact_loads += 1
            return super().fact_edges()

    backend = CountingBackend(SQLiteConfig(path=":memory:"), namespace="demo", dimensions=2)
    harbor = _node("harbor", "entity", "Authority", name="Harbor Authority")
    grain = _node("grain", "entity", "Organization", name="Grain Cooperative")
    fact = _node(
        "sends",
        "fact",
        "__Fact__",
        text="Harbor Authority | supplies | Grain Cooperative",
        relation_type="SUPPLIES",
        relationship_id="sends",
        embedding=[1.0, 0.0],
    )
    backend.replace_document(
        DocumentGraph(
            source="city",
            path="harbor",
            content_hash="h",
            nodes=[
                harbor,
                grain,
                fact,
                _node("c-harbor", "chunk", "Chunk", path="harbor-authority.txt", text="harbor"),
                _node("c-grain", "chunk", "Chunk", path="grain-cooperative.txt", text="grain"),
            ],
            edges=[
                GraphEdge(id="s", source="sends", target="harbor", type="FACT_SUBJECT"),
                GraphEdge(id="o", source="sends", target="grain", type="FACT_OBJECT"),
            ],
        )
    )
    retriever = Retriever(
        backend,
        _Query(),
        None,
        RetrievalConfig(
            use_triples=True,
            use_fulltext=False,
            rerank_passages=False,
            workers=1,
            top_k=5,
        ),
    )
    try:
        retriever.retrieve("Harbor Authority and Grain Cooperative")
        observatory = _node("observatory", "entity", "Observatory", name="Observatory")
        plants = _node(
            "plants",
            "fact",
            "__Fact__",
            text="Observatory | sets planting date for | Grain Cooperative",
            relation_type="SETS",
            relationship_id="plants",
            embedding=[1.0, 0.0],
        )
        backend.replace_document(
            DocumentGraph(
                source="city",
                path="observatory",
                content_hash="o",
                nodes=[
                    observatory,
                    grain,
                    plants,
                    _node(
                        "c-observatory",
                        "chunk",
                        "Chunk",
                        path="observatory.txt",
                        text="sky",
                        embedding=[1.0, 0.0],
                    ),
                ],
                edges=[
                    GraphEdge(id="s2", source="plants", target="observatory", type="FACT_SUBJECT"),
                    GraphEdge(id="o2", source="plants", target="grain", type="FACT_OBJECT"),
                ],
            )
        )
        passages = retriever.retrieve("What does the Observatory tell the Grain Cooperative?")
    finally:
        retriever.close()
        backend.close()

    assert backend.fact_loads == 2
    assert "observatory" in [passage.title for passage in passages]


def _reference_pagerank(weighted_edges, seeds, damping=0.85):
    """Small Python reference; exact graph storage and projection are tested separately."""
    nodes = set(seeds)
    outgoing = {}
    for source, target, weight in weighted_edges:
        nodes.update((source, target))
        outgoing.setdefault(source, []).append((target, weight))
    bias = {node: seeds.get(node, 0.0) / sum(seeds.values()) for node in nodes}
    mass = dict(bias)
    for _ in range(100):
        dangling = sum(mass[node] for node in nodes if node not in outgoing)
        updated = {node: value * (1 - damping + damping * dangling) for node, value in bias.items()}
        for source, links in outgoing.items():
            for target, weight in links:
                updated[target] += damping * mass[source] * weight
        if sum(abs(updated[node] - mass[node]) for node in nodes) < 1e-12:
            return updated
        mass = updated
    return mass


def test_matrix_walk_matches_cosine_softmax_and_pagerank():
    """The sparse walk keeps the same scores as the Python reference."""

    edges = [
        FactEdge(
            "Lighthouse Keepers",
            "Harbor Authority",
            "Lighthouse Keepers | reports to | Harbor Authority",
            "REPORTS_TO",
            "report",
            (0.0, 2.0),
        ),
        FactEdge(
            "Lighthouse Keepers",
            "Observatory",
            "Lighthouse Keepers | sends | Observatory",
            "SENDS",
            "sends",
            (3.0, 0.0),
        ),
        FactEdge(
            "Observatory",
            "Grain Cooperative",
            "Observatory | sets planting date for | Grain Cooperative",
            "SETS",
            "plants",
            (1.0, 1.0),
        ),
        FactEdge(
            "Harbor Authority",
            "Grain Cooperative",
            "Harbor Authority | notes | Grain Cooperative",
            "NOTES",
            "zero",
            (0.0, 0.0),
        ),
    ]
    chunks = [
        ChunkRecord("keepers", "lighthouse-keepers.txt", "keepers"),
        ChunkRecord("harbor", "harbor-authority.txt", "harbor"),
        ChunkRecord("observatory", "observatory.txt", "observatory"),
        ChunkRecord("grain", "grain-cooperative.txt", "grain"),
    ]
    embedding = [1.0, 0.0]
    grouped: dict[str, list[tuple[FactEdge, float]]] = {}
    for edge in edges:
        grouped.setdefault(edge.source, []).append((edge, cosine(edge.embedding, embedding)))
    seeds = {"Lighthouse Keepers": 1.0, "Grain Cooperative": 0.2}
    weighted = []
    carried = []
    for links in grouped.values():
        values = [math.exp(score / 0.1) for _, score in links]
        weights = [value / sum(values) for value in values]
        for (edge, _), weight in zip(links, weights, strict=True):
            weighted.append((edge.source, edge.target, weight))
            carried.append((edge, weight))
    mass = _reference_pagerank(weighted, seeds)
    incoming: dict[str, tuple[float, FactEdge]] = {}
    for edge, weight in carried:
        value = mass.get(edge.source, 0.0) * weight
        current = incoming.get(edge.target)
        if current is None or value > current[0]:
            incoming[edge.target] = (value, edge)
    expected = home_scores(
        mass,
        [{"chunk_id": chunk.chunk_id, "path": chunk.path, "text": chunk.text} for chunk in chunks],
    )

    walk = FactWalkGraph.from_edges(edges, chunks).walk(
        embedding, tau=0.1, damping=0.85, seeds=seeds
    )

    assert walk.seeds == len(seeds)
    assert [row["chunk_id"] for row in walk.homes] == [row["chunk_id"] for row in expected]
    assert [row["score"] for row in walk.homes] == pytest.approx([row["score"] for row in expected])
    for home in walk.homes:
        entity = str(home["entity"])
        assert (entity in walk.incoming) == (entity in incoming)
        if entity not in incoming:
            continue
        chosen = walk.incoming[entity]
        reference = incoming[entity]
        assert chosen[1].relationship_id == reference[1].relationship_id
        assert chosen[0] == pytest.approx(reference[0])


def test_the_sparse_walk_restarts_from_weighted_facts():
    edges = [
        FactEdge(
            "Grain Cooperative",
            "Harbor Authority",
            "Grain Cooperative | supplies | Harbor Authority",
            "SUPPLIES",
            "sends",
            (1.0, 0.0),
        )
    ]
    chunks = [
        ChunkRecord(
            "grain",
            "grain-cooperative.txt",
            "grain",
            entities=("Grain Cooperative",),
        ),
        ChunkRecord(
            "harbor",
            "harbor-authority.txt",
            "harbor",
            entities=("Harbor Authority",),
        ),
    ]
    graph = FactWalkGraph.from_edges(edges, chunks)
    weights = closest_fact_seeds(
        [(edges[0].source, edges[0].target, graph.fact_cosines([1.0, 0.0]).item())],
        {"Grain Cooperative": 1, "Harbor Authority": 9},
        limit=1,
    )
    walk = graph.walk([1.0, 0.0], tau=0.1, damping=0.85, seeds=weights)

    assert walk.seeds == 2
    # The sole outgoing edge goes to dangling Harbor. The restart bias
    # for Grain is 0.9, so its steady-state mass is 0.9 / (1 + 0.85 * 0.9).
    expected_grain_mass = 0.9 / (1 + 0.85 * 0.9)
    assert [row["chunk_id"] for row in walk.homes] == ["grain", "harbor"]
    assert [row["score"] for row in walk.homes] == pytest.approx(
        [expected_grain_mass, 1 - expected_grain_mass]
    )
