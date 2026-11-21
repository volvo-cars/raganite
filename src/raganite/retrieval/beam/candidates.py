"""Initial passage relevance and title-mention bridge candidates."""

import math
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from ..fact_walk.graph import FactWalkGraph
from ..fact_walk.scoring import FACT_SEED_COUNT, PASSAGE_SEED_WEIGHT, closest_fact_seeds

WORD = re.compile(r"[^\W_]+", re.UNICODE)
PARENTHETICAL = re.compile(r"\s*\([^)]*\)\s*$")
PUNCTUATION = str.maketrans("", "", string.punctuation)
MAX_CANDIDATES = 150
PAGE_FEATURE_NAMES = (
    "walk_log",
    "walk_rank",
    "cosine",
    "cosine_rank",
    "bm25",
    "bm25_rank",
    "fact",
    "fact_rank",
    "leader_cosine",
    "leftover",
)


def normalize_title(text: str) -> str:
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text.lower().translate(PUNCTUATION)).split())


def tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in WORD.finditer(text)]


def title_aliases(title: str) -> set[str]:
    """Full title and title without its final parenthetical qualifier."""
    return {
        alias
        for alias in (normalize_title(title), normalize_title(PARENTHETICAL.sub("", title)))
        if len(alias) >= 5
    }


def rank_units(values: np.ndarray) -> np.ndarray:
    """Best value scores 1 and the worst scores 0. Ties share the average."""
    count = len(values)
    if count < 2:
        return np.ones(count, dtype=np.float64)
    order = np.argsort(-values, kind="stable")
    units = np.zeros(count, dtype=np.float64)
    start = 0
    while start < count:
        end = start + 1
        while end < count and values[order[end]] == values[order[start]]:
            end += 1
        units[order[start:end]] = np.mean([1.0 - rank / (count - 1) for rank in range(start, end)])
        start = end
    return units


class BM25:
    """BM25 over passages, with postings for each term."""

    def __init__(self, documents: list[list[str]]) -> None:
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = np.asarray([len(words) or 1 for words in documents], dtype=np.float64)
        for index, words in enumerate(documents):
            for word, frequency in Counter(words).items():
                postings[word].append((index, frequency))
        average = float(lengths.mean()) if len(lengths) else 1.0
        denominator = 1.5 * (1 - 0.75 + 0.75 * lengths / average)
        self.count = len(documents)
        self.postings = {}
        for word, occurrences in postings.items():
            entries = np.asarray(occurrences, dtype=np.intp)
            indexes, frequencies = entries.T
            seen = len(indexes)
            weight = math.log(1.0 + (self.count - seen + 0.5) / (seen + 0.5))
            self.postings[word] = (
                indexes,
                weight * frequencies * (1.5 + 1) / (frequencies + denominator[indexes]),
            )

    def scores(self, query: str) -> np.ndarray:
        found = np.zeros(self.count, dtype=np.float64)
        for word in set(tokens(query)):
            if word in self.postings:
                indexes, values = self.postings[word]
                found[indexes] += values
        return found


def passage_seed_values(
    cosines: np.ndarray, *, top_k: int = 200, weight: float = PASSAGE_SEED_WEIGHT
) -> np.ndarray:
    """Positive top-k passage cosine, min-max scaled to ``weight``."""
    seeds = np.zeros(len(cosines), dtype=np.float64)
    positive = np.flatnonzero(cosines > 0)
    if not len(positive) or weight == 0:
        return seeds
    if len(positive) > top_k:
        positive = positive[np.argpartition(-cosines[positive], top_k - 1)[:top_k]]
    values = cosines[positive]
    low = float(values.min())
    span = float(values.max()) - low
    normalized = np.ones(len(values)) if span == 0 else (values - low) / span
    seeds[positive] = normalized * weight
    return seeds


def build_mention_graph(
    chunks: list[dict],
) -> tuple[list[dict[int, int]], list[dict[int, int]]]:
    """Index passage-to-title mentions and backlinks."""
    aliases = [title_aliases(str(chunk.get("title") or "")) for chunk in chunks]
    frequency = Counter(
        token for page_aliases in aliases for alias in page_aliases for token in set(alias.split())
    )
    anchored: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for page, page_aliases in enumerate(aliases):
        for alias in page_aliases:
            words = alias.split()
            anchor = min(words, key=lambda token: (frequency[token], token))
            anchored[anchor].append((alias, page, len(words)))
    forward: list[dict[int, int]] = [{} for _ in chunks]
    reverse: list[dict[int, int]] = [{} for _ in chunks]
    for source, chunk in enumerate(chunks):
        text = f" {normalize_title(str(chunk.get('text') or ''))} "
        for word in set(text.split()):
            for alias, target, width in anchored.get(word, ()):
                if target == source or f" {alias} " not in text:
                    continue
                forward[source][target] = max(forward[source].get(target, 0), width)
                reverse[target][source] = max(reverse[target].get(source, 0), width)
    return forward, reverse


def features_for(signals: np.ndarray, indexes: np.ndarray) -> np.ndarray:
    """Compute page features, with rank features relative to this candidate set."""
    walk, cosine, bm25, fact, leader, leftover = signals[indexes].T
    return np.column_stack(
        [
            np.log1p(walk),
            rank_units(walk),
            cosine,
            rank_units(cosine),
            bm25,
            rank_units(bm25),
            fact,
            rank_units(fact),
            leader,
            leftover,
        ]
    )


@dataclass
class BeamCandidates:
    indexes: tuple[int, ...]
    base_count: int
    page_features: np.ndarray
    bridge_scores: np.ndarray
    fallback: np.ndarray


class CandidateIndex:
    """Cached lexical, provenance and mention indexes."""

    def __init__(self, chunks: list[dict]) -> None:
        self.titles = [
            normalize_title(chunk["title"]) or f"\0{chunk['chunk_id']}" for chunk in chunks
        ]
        self.title_index = {title: index for index, title in enumerate(self.titles)}
        self.index_of = {chunk["chunk_id"]: index for index, chunk in enumerate(chunks)}
        words = [tokens(chunk.get("text") or "") for chunk in chunks]
        self.word_sets = [set(document) for document in words]
        self.bm25 = BM25(words)
        self.entities = [frozenset(chunk.get("entities") or ()) for chunk in chunks]
        self.forward, self.reverse = build_mention_graph(chunks)
        self.producer_counts: dict[str, int] = {}
        self.entity_chunks: dict[str, list[int]] = defaultdict(list)
        for index, names in enumerate(self.entities):
            for name in names:
                self.producer_counts[name] = self.producer_counts.get(name, 0) + 1
                pages = self.entity_chunks[name]
                if len(pages) < 4:
                    pages.append(index)

    def prepare(
        self,
        graph: FactWalkGraph,
        passages: np.ndarray,
        query: str,
        unit: np.ndarray,
        *,
        start_limit: int,
        bridge_limit: int,
        seed_limit: int = FACT_SEED_COUNT,
        seed_weight: float = PASSAGE_SEED_WEIGHT,
        tau: float = 0.1,
        damping: float = 0.85,
    ) -> BeamCandidates:
        fact_cosine = graph.fact_cosines(unit)
        seeds = closest_fact_seeds(
            (
                (link.source, link.target, float(score))
                for link, score in zip(graph.links, fact_cosine, strict=True)
            ),
            self.producer_counts,
            limit=seed_limit,
        )
        homes = graph.walk(
            unit, tau=tau, damping=damping, seeds=seeds, cosines=fact_cosine, evidence=False
        ).homes
        walk_score = {
            self.index_of[row["chunk_id"]]: float(row["score"])
            for row in homes
            if row["chunk_id"] in self.index_of
        }
        cosine = passages @ unit
        lexical = self.bm25.scores(query)
        fact_page = np.zeros(len(passages), dtype=np.float64)

        def fact_chunks(index: int) -> list[int]:
            link = graph.links[index]
            return self.entity_chunks.get(link.source, []) + self.entity_chunks.get(link.target, [])

        top_facts = (
            np.argpartition(-fact_cosine, 200)[:200]
            if len(fact_cosine) > 200
            else np.arange(len(fact_cosine))
        )
        for fact in top_facts:
            if fact_cosine[fact] > 0:
                for page in fact_chunks(int(fact)):
                    fact_page[page] = max(fact_page[page], fact_cosine[fact])
        # Preserve the checkpoint's initial relevance pools and tie order.
        chosen: dict[int, None] = {}
        for index in sorted(walk_score, key=lambda index: -walk_score[index])[:50]:
            chosen[index] = None
        for primary in (cosine, lexical):
            chosen.update(dict.fromkeys(np.argsort(-primary)[:50].tolist()))
        for fact in np.argsort(-fact_cosine)[:20]:
            chosen.update(dict.fromkeys(fact_chunks(int(fact))))
        initial = list(chosen)[:MAX_CANDIDATES]
        if not initial:
            raise ValueError("No passages available for beam retrieval")
        walk = np.asarray([walk_score.get(index, 0.0) for index in initial], dtype=np.float64)
        initial_leader = initial[int(np.argmax(walk))]
        query_words = set(tokens(query))
        residual = query_words - self.word_sets[initial_leader]
        direct = passage_seed_values(cosine, weight=seed_weight)
        # Title identity intentionally maps repeated titles to their last chunk unless
        # the caller selects distinct chunk titles.
        walk += [direct[self.title_index[self.titles[index]]] for index in initial]
        initial_signals = np.column_stack(
            (
                walk,
                cosine[initial],
                lexical[initial],
                fact_page[initial],
                passages[initial] @ passages[initial_leader],
                [
                    len(residual & self.word_sets[index]) / max(len(residual), 1)
                    for index in initial
                ],
            )
        )
        base: dict[int, np.ndarray] = {}
        for local in np.argsort(-walk, kind="stable"):
            index = self.title_index[self.titles[initial[local]]]
            base.setdefault(index, initial_signals[local])
            if len(base) == start_limit:
                break
        bridge_scores: dict[int, float] = {}
        if bridge_limit:
            for rank, source in enumerate(base):
                source_score = 1.0 - rank / max(len(base) - 1, 1)
                for neighbors, direction in ((self.forward, 1.0), (self.reverse, 0.5)):
                    for target, width in neighbors[source].items():
                        score = (
                            direction
                            + source_score
                            + 0.1 * min(width, 5)
                            + 0.2 * max(float(cosine[target]), 0.0)
                        )
                        bridge_scores[target] = max(bridge_scores.get(target, -math.inf), score)
        additions = sorted(
            (index for index in bridge_scores if index not in base),
            key=lambda index: -bridge_scores[index],
        )[:bridge_limit]
        leader = next(iter(base))
        uncovered = query_words - self.word_sets[leader]
        signals = np.asarray(
            [
                *base.values(),
                *(
                    base[self.title_index[self.titles[index]]]
                    if self.title_index[self.titles[index]] in base
                    else [
                        direct[index],
                        cosine[index],
                        0.0,
                        0.0,
                        float(passages[index] @ passages[leader]),
                        len(uncovered & self.word_sets[index]) / max(len(uncovered), 1),
                    ]
                    for index in additions
                ),
            ],
        )
        indexes = (*base, *additions)
        return BeamCandidates(
            indexes=indexes,
            base_count=len(base),
            page_features=features_for(signals, np.arange(len(indexes))),
            bridge_scores=np.asarray([bridge_scores.get(index, 0.0) for index in indexes]),
            fallback=rank_units(signals[:, 0]) + rank_units(signals[:, 1]),
        )
