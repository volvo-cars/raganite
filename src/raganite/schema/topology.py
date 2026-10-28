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

"""Structural graph roles shared by indexing and backend expansion."""

from dataclasses import dataclass


def _role(value: str) -> None:
    if not value.isidentifier() or not value.isupper():
        raise ValueError(f"Topology role {value!r} must be an uppercase identifier")


@dataclass(frozen=True)
class GraphTopology:
    """Edge types that connect a Namespace's structural nodes.

    Domain relations between Entities stay on the extraction schema. Adapters
    read these roles; they do not invent a second set.
    """

    chunk_entity: str = "FROM_CHUNK"
    chunk_document: str = "FROM_DOCUMENT"
    source_document: str = "HAS_DOCUMENT"
    document_namespace: str = "BELONGS_TO"
    fact_subject: str = "FACT_SUBJECT"
    fact_object: str = "FACT_OBJECT"
    entity_session: str = "IN_SESSION"
    next_session: str = "NEXT"

    def __post_init__(self) -> None:
        for value in (
            self.chunk_entity,
            self.chunk_document,
            self.source_document,
            self.document_namespace,
            self.fact_subject,
            self.fact_object,
            self.entity_session,
            self.next_session,
        ):
            _role(value)


OWNED = GraphTopology()
