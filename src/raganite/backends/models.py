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

"""Storage-neutral records for fact search and scored graph expansion."""

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PassageHit:
    chunk_id: str
    source: str
    text: str
    score: float


@dataclass(frozen=True)
class FactHit:
    fact_id: str
    relationship_id: str
    text: str
    relation_type: str
    score: float
    subject_id: str
    subject: str
    subject_degree: float
    object_id: str
    object: str
    object_degree: float


@dataclass(frozen=True)
class GraphExpansion:
    chunk_id: str
    source: str
    text: str
    expansion_score: float
    relationship_id: str
    fact_text: str
    seed_entity: str
    related_entity: str
    graph_path: list[str] = field(default_factory=list)
    path_degrees: list[float] = field(default_factory=list)


def validate_expansion(depth: int, limit: int, degree_penalty_weight: float) -> None:
    if not 0 <= depth <= 5 or limit < 1:
        raise ValueError("Expansion depth must be 0–5 and limit must be positive")
    if not math.isfinite(degree_penalty_weight) or degree_penalty_weight < 0:
        raise ValueError("Degree penalty weight must be finite and non-negative")
