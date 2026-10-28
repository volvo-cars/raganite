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

"""Configurable extraction schemas using neo4j-graphrag's native schema types."""

from neo4j_graphrag.components.schema import (
    GraphSchema,
    NodeType,
    Pattern,
    PropertyType,
    RelationshipType,
)

DEFAULT_ENTITIES = ("Module", "Class", "Function", "Fixture", "Test", "Helper", "API", "Concept")
DEFAULT_RELATIONS = ("IMPORTS", "DEFINES", "USES", "CALLS", "TESTS", "DEPENDS_ON")
DEFAULT_PATTERNS = (
    ("Module", "DEFINES", "Class"),
    ("Module", "DEFINES", "Function"),
    ("Module", "DEFINES", "Fixture"),
    ("Module", "IMPORTS", "Module"),
    ("Class", "USES", "Class"),
    ("Function", "CALLS", "Function"),
    ("Function", "USES", "Fixture"),
    ("Test", "TESTS", "Function"),
    ("Test", "USES", "Fixture"),
    ("Function", "DEPENDS_ON", "API"),
)


def default_schema() -> GraphSchema:
    """Return an independent copy of the sample's default code-oriented schema."""
    return GraphSchema(
        node_types=tuple(NodeType(label=label) for label in DEFAULT_ENTITIES),
        relationship_types=tuple(RelationshipType(label=label) for label in DEFAULT_RELATIONS),
        patterns=tuple(Pattern(source=s, relationship=r, target=t) for s, r, t in DEFAULT_PATTERNS),
    )


__all__ = [
    "DEFAULT_ENTITIES",
    "DEFAULT_PATTERNS",
    "DEFAULT_RELATIONS",
    "GraphSchema",
    "NodeType",
    "Pattern",
    "PropertyType",
    "RelationshipType",
    "default_schema",
]
