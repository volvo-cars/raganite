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

"""Configurable graph backends with a shared storage contract."""

from raganite.config import Neo4jConfig, RaganiteConfig

from .base import DocumentGraph, GraphBackend, GraphEdge, GraphNode, SearchHit
from .models import FactHit, GraphExpansion, PassageHit
from .neo4j import Neo4jBackend
from .sqlite import SQLiteBackend


def create_backend(config: RaganiteConfig | None = None) -> GraphBackend:
    config = config or RaganiteConfig()
    settings = config.backend
    backend: GraphBackend
    if isinstance(settings, Neo4jConfig):
        backend = Neo4jBackend(
            settings,
            namespace=config.namespace,
            dimensions=config.embedding_dimensions,
            layout=settings.layout,
        )
    else:
        backend = SQLiteBackend(
            settings, namespace=config.namespace, dimensions=config.embedding_dimensions
        )
    try:
        backend.ensure_indexes()
    except Exception:
        backend.close()
        raise
    return backend


__all__ = [
    "DocumentGraph",
    "GraphBackend",
    "GraphEdge",
    "GraphNode",
    "Neo4jBackend",
    "SQLiteBackend",
    "SearchHit",
    "create_backend",
    "FactHit",
    "GraphExpansion",
    "PassageHit",
]
