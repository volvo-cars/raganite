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

"""Personalized PageRank whose steps follow the fact sentence closest to the question.

The walk starts from the closest fact sentences. Each fact lends its cosine to
the two names it connects, and a name that many passages produced keeps a smaller
share. Each retrieved passage also keeps a small score of its own. An entity's
mass then goes to a passage that produced that entity. When one of those passages
is the entity's own document, that passage takes the mass. A passage that did
not produce the entity does not receive it.
"""

import re
from collections.abc import Iterable, Mapping, Sequence
from heapq import nsmallest
from pathlib import Path
from typing import Any, TypedDict

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
FACT_SEED_COUNT = 5
PASSAGE_SEED_WEIGHT = 0.05


def _words(value: str) -> tuple[str, ...]:
    return tuple(word.casefold() for word in _WORD.findall(value))


def closest_fact_seeds(
    facts: Iterable[tuple[str, str, float]],
    producer_counts: Mapping[str, int],
    *,
    limit: int,
) -> dict[str, float]:
    """Restart weights from the closest facts.

    Each fact lends its cosine to its two endpoints. A name produced by several
    passages keeps only its share of that cosine. A fact that lists the same
    name again contributes to the average.
    """
    if limit < 1:
        raise ValueError("Fact seed count must be positive")
    chosen = nsmallest(limit, (fact for fact in facts if fact[2] > 0), key=lambda fact: -fact[2])
    totals: dict[str, float] = {}
    seen: dict[str, int] = {}
    for source, target, score in chosen:
        for name in (source, target):
            producers = producer_counts.get(name, 0)
            share = float(score) / producers if producers else float(score)
            totals[name] = totals.get(name, 0.0) + share
            seen[name] = seen.get(name, 0) + 1
    return {name: total / seen[name] for name, total in totals.items() if total > 0}


def _labels(chunk: Mapping[str, Any]) -> list[str]:
    labels = []
    stem = Path(str(chunk.get("path") or "")).stem
    if stem:
        labels.append(stem)
    title = str(chunk.get("title") or "").strip()
    if title:
        labels.append(title)
    return labels


def _contains(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    width = len(needle)
    if not width or width > len(haystack):
        return False
    return any(
        haystack[index : index + width] == needle for index in range(len(haystack) - width + 1)
    )


def _closeness(name: str, label: str) -> tuple[int, int] | None:
    """0 when the phrases are the same, otherwise how many extra words one has."""
    left, right = _words(name), _words(label)
    if not left or not right:
        return None
    if left == right:
        return (0, len(right))
    if _contains(left, right) or _contains(right, left):
        return (abs(len(left) - len(right)), len(right))
    return None


def _best_home(name: str, chunks: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    best: tuple[tuple[int, int, str], Mapping[str, Any]] | None = None
    for chunk in chunks:
        for label in _labels(chunk):
            gap = _closeness(name, label)
            if gap is None:
                continue
            rank = (*gap, str(chunk["chunk_id"]))
            if best is None or rank < best[0]:
                best = (rank, chunk)
    return None if best is None else best[1]


def _producers(
    chunks: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]] | None:
    """Entity name to the passages that produced it. None when no passage lists any."""
    produced: dict[str, list[Mapping[str, Any]]] = {}
    found = False
    for chunk in chunks:
        names = chunk.get("entities")
        if not names:
            continue
        found = True
        seen: set[str] = set()
        for entity in names:
            name = str(entity)
            if not name or name in seen:
                continue
            seen.add(name)
            bucket = produced.setdefault(name, [])
            if all(item["chunk_id"] != chunk["chunk_id"] for item in bucket):
                bucket.append(chunk)
    return produced if found else None


class HomeScore(TypedDict):
    chunk_id: str
    path: str
    text: str
    entity: str
    score: float


def home_scores(mass: Mapping[str, float], chunks: Sequence[Mapping[str, Any]]) -> list[HomeScore]:
    """Give each entity's mass to the passages that produced it.

    A filename or title that names the entity takes the whole mass. Otherwise the
    mass is split evenly across those passages. With no provenance, only a
    filename or title match receives the mass.
    """
    produced = _producers(chunks)
    ranked: list[HomeScore] = []
    for name, score in mass.items():
        if score <= 0:
            continue
        if produced is None:
            home = _best_home(name, chunks)
            recipients = [(home, score)] if home is not None else []
        else:
            candidates = produced.get(name, [])
            home = _best_home(name, candidates)
            if home is not None:
                recipients = [(home, score)]
            else:
                share = score / len(candidates) if candidates else 0.0
                recipients = [(chunk, share) for chunk in candidates]
        for chunk, share in recipients:
            ranked.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "path": chunk.get("path") or "",
                    "text": chunk.get("text") or "",
                    "entity": name,
                    "score": share,
                }
            )
    ranked.sort(key=lambda row: (-row["score"], row["path"], row["entity"]))
    combined: dict[str, HomeScore] = {}
    for row in ranked:
        current = combined.get(row["chunk_id"])
        if current is None:
            combined[row["chunk_id"]] = row.copy()
        else:
            current["score"] += row["score"]
    return sorted(combined.values(), key=lambda row: (-row["score"], row["path"]))
