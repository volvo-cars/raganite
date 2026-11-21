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

"""Storage-neutral records for passage search and beam snapshots."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PassageHit:
    chunk_id: str
    source: str
    text: str
    score: float


@dataclass(frozen=True)
class FactEdge:
    """One directed fact, with the embedding of its sentence."""

    source: str
    target: str
    text: str
    relation_type: str
    relationship_id: str
    embedding: tuple[float, ...]


@dataclass(frozen=True)
class ChunkRecord:
    """A stored chunk, its document title, and its owning documents' entity names."""

    chunk_id: str
    path: str
    text: str
    title: str = ""
    entities: tuple[str, ...] = ()
