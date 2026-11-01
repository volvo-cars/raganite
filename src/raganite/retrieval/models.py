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

"""Public passage results and their supporting graph evidence."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from raganite.clients.llm.usage import UsageReport


@dataclass(frozen=True)
class FactEvidence:
    fact: str
    seed_entity: str
    related_entity: str
    path: tuple[str, ...] = ()
    score: float = 0.0
    relationship_id: str = ""
    path_degrees: tuple[float, ...] = ()


@dataclass(frozen=True)
class Passage:
    source: str
    title: str
    text: str
    score: float
    evidence: tuple[FactEvidence, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Answer:
    """A question answered from the passages retrieved for it."""

    question: str
    answer: str
    passages: tuple[Passage, ...]
    usage: UsageReport


class RetrievalResult(list[Passage]):
    """Passages from one retrieval, with that run's timing and token totals.

    The result is still a list, so existing callers can index and iterate it.
    ``usage`` stays the retrieval run even when a benchmark run is wrapped around it.
    """

    usage: UsageReport

    def __init__(self, passages: Sequence[Passage], usage: UsageReport) -> None:
        super().__init__(passages)
        self.usage = usage
