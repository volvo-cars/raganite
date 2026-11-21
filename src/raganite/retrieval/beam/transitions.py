"""Pair-specific lexical and directed graph features for the beam walk."""

import math
import re
from collections import defaultdict
from collections.abc import Callable
from functools import cache

from .candidates import normalize_title, title_aliases, tokens

STOP_WORDS = frozenset(
    "a an and are as at be by did do does for from had has have how in is it of on or "
    "that the their this to was were what when where which who whom whose with".split()
)
NUMBER = re.compile(r"[0-9]+(?:st|nd|rd|th)?")
TRANSITION_FEATURES = (
    "residual_terms",
    "target_residual_coverage",
    "union_question_coverage",
    "numeric_union_coverage",
    "year_union_coverage",
    "target_residual_numeric_coverage",
    "target_is_bridge",
    "mention_full_forward",
    "mention_full_backward",
    "mention_forward_only",
    "mention_reverse_only",
    "mention_both",
    "fact_forward_count_log",
    "fact_backward_count_log",
    "fact_forward_only",
    "fact_reverse_only",
    "fact_both",
)


def _direction(prefix: str, forward: bool, backward: bool) -> dict[str, float]:
    return {
        f"{prefix}_forward_only": float(forward and not backward),
        f"{prefix}_reverse_only": float(backward and not forward),
        f"{prefix}_both": float(forward and backward),
    }


class TransitionIndex:
    """Index directed fact endpoints without using relationship labels."""

    def __init__(self, chunks: list[dict], links: list[dict]) -> None:
        self.words = [
            frozenset(tokens(f"{chunk.get('title') or ''} {chunk.get('text') or ''}"))
            for chunk in chunks
        ]
        self.text = [f" {normalize_title(chunk.get('text') or '')} " for chunk in chunks]
        self.titles = [normalize_title(chunk.get("title") or "") for chunk in chunks]
        self.aliases = [title_aliases(chunk.get("title") or "") for chunk in chunks]
        self.entities = [frozenset(chunk.get("entities") or ()) for chunk in chunks]
        outgoing: dict[str, set[str]] = defaultdict(set)
        for link in links:
            if link["source"] != link["target"]:
                outgoing[link["source"]].add(link["target"])
        self.outgoing = outgoing

    def for_question(
        self, text: str, candidates: tuple[int, ...], base_count: int
    ) -> Callable[[int, int], dict[str, float]]:
        question = frozenset(tokens(text)) - STOP_WORDS
        numbers = frozenset(word for word in question if NUMBER.fullmatch(word))
        years = frozenset(word for word in numbers if len(word) == 4 and word.isdecimal())
        covered = [question & self.words[page] for page in candidates]
        residual = [question - terms for terms in covered]

        @cache
        def edge(source: int, target: int) -> tuple[bool, bool, int]:
            left, right = candidates[source], candidates[target]
            body = self.text[left]
            full = self.titles[right]
            has_full = bool(full) and f" {full} " in body
            mention = has_full or any(f" {alias} " in body for alias in self.aliases[right])
            count = 0
            targets = self.entities[right]
            for entity in self.entities[left]:
                neighbors = self.outgoing.get(entity)
                if neighbors:
                    for endpoint in targets & neighbors:
                        # Shared-entity facts alone are not evidence of a bridge.
                        if entity in targets and endpoint in self.entities[left]:
                            continue
                        count += 1
            return mention, has_full, count

        def features(source: int, target: int) -> dict[str, float]:
            union = covered[source] | covered[target]
            uncovered = residual[source]
            new_terms = uncovered & covered[target]
            forward, full_forward, forward_count = edge(source, target)
            backward, full_backward, backward_count = edge(target, source)
            return {
                "residual_terms": float(len(uncovered)),
                "target_residual_coverage": len(new_terms) / max(len(uncovered), 1),
                "union_question_coverage": len(union) / max(len(question), 1),
                "numeric_union_coverage": len(numbers & union) / max(len(numbers), 1),
                "year_union_coverage": len(years & union) / max(len(years), 1),
                "target_residual_numeric_coverage": len(numbers & new_terms) / max(len(numbers), 1),
                "target_is_bridge": float(target >= base_count),
                "mention_full_forward": float(full_forward),
                "mention_full_backward": float(full_backward),
                **_direction("mention", forward, backward),
                "fact_forward_count_log": math.log1p(forward_count),
                "fact_backward_count_log": math.log1p(backward_count),
                **_direction("fact", bool(forward_count), bool(backward_count)),
            }

        return features
