"""Checkpoint loading and two-hop decoding."""

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from io import BytesIO

import numpy as np

from .candidates import PAGE_FEATURE_NAMES, BeamCandidates, CandidateIndex
from .transitions import TRANSITION_FEATURES, TransitionIndex

FEATURE_SCHEMA = "relation_agnostic_v1"
FEATURE_NAMES = (
    *(f"source_{name}" for name in PAGE_FEATURE_NAMES),
    *(f"target_{name}" for name in PAGE_FEATURE_NAMES if name != "leftover"),
    "passage_cosine",
    "entity_jaccard",
    "mention_forward",
    "mention_backward",
    "source_bridge",
    "target_bridge",
    "source_fallback",
    "target_fallback",
    *TRANSITION_FEATURES,
)


@dataclass(frozen=True)
class Checkpoint:
    weights: np.ndarray
    center: np.ndarray
    spread: np.ndarray


def load_checkpoint(data: bytes) -> Checkpoint:
    with np.load(BytesIO(data), allow_pickle=False) as model:
        if "feature_schema" not in model or model["feature_schema"].item() != FEATURE_SCHEMA:
            raise ValueError(
                "Retrain this beam checkpoint with the relation-agnostic feature schema"
            )
        if "feature_names" not in model or tuple(model["feature_names"]) != FEATURE_NAMES:
            raise ValueError("Checkpoint does not match the live transition feature schema")
        arrays = []
        for field in ("weights", "center", "spread"):
            if field not in model:
                raise ValueError(f"Beam checkpoint is missing {field}")
            values = np.asarray(model[field], dtype=np.float64)
            if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
                raise ValueError(f"Invalid beam checkpoint {field}")
            arrays.append(values)
        if (arrays[2] <= 0).any():
            raise ValueError("Beam checkpoint spread must be positive")
        for array in arrays:
            array.setflags(write=False)
        return Checkpoint(*arrays)


@lru_cache(maxsize=1)
def load_default_checkpoint() -> Checkpoint:
    return load_checkpoint(files("raganite.retrieval.beam").joinpath("weights.npz").read_bytes())


def select_passages(
    checkpoint: Checkpoint,
    candidates: BeamCandidates,
    index: CandidateIndex,
    transitions: TransitionIndex,
    passages: np.ndarray,
    query: str,
    *,
    width: int,
    limit: int,
) -> list[int]:
    pages = candidates.indexes
    limit = min(limit, len(pages))
    if limit == 1 and len(pages) == 1:
        return [pages[0]]
    raw = candidates.page_features
    page_width = len(PAGE_FEATURE_NAMES)
    scaled = (raw[: candidates.base_count] - checkpoint.center[:page_width]) / (
        checkpoint.spread[:page_width]
    )
    source_scores = scaled @ checkpoint.weights[:page_width]
    # Preserve candidate-order tie breaks when scoring only admitted sources.
    sources = sorted(map(int, np.argsort(-source_scores, kind="stable")[:width]))
    extract = transitions.for_question(query, pages, candidates.base_count)
    paths = []
    features = []
    for source in sources:
        left = pages[source]
        for target, right in enumerate(pages):
            if source == target:
                continue
            entities_left, entities_right = index.entities[left], index.entities[right]
            union = entities_left | entities_right
            paths.append((source, target))
            features.append(
                [
                    *raw[source],
                    *raw[target, :-1],
                    float(passages[left] @ passages[right]),
                    len(entities_left & entities_right) / len(union) if union else 0.0,
                    index.forward[left].get(right, 0) / 5.0,
                    index.forward[right].get(left, 0) / 5.0,
                    candidates.bridge_scores[source],
                    candidates.bridge_scores[target],
                    candidates.fallback[source],
                    candidates.fallback[target],
                    *extract(source, target).values(),
                ]
            )
    scores = ((np.asarray(features) - checkpoint.center) / checkpoint.spread) @ checkpoint.weights
    selected: dict[int, None] = {}
    for path in np.argsort(-scores, kind="stable"):
        for page in paths[path]:
            selected[page] = None
            if len(selected) == limit:
                return [pages[index] for index in selected]
    # Fallback candidates fill any unused result slots; they do not reorder
    # passages already admitted by a path.
    for page in np.argsort(-candidates.fallback, kind="stable"):
        selected[int(page)] = None
        if len(selected) == limit:
            break
    return [pages[index] for index in selected]
