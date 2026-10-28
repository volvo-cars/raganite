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

"""Validated application settings; all backend-independent options live here."""

import os
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self

from raganite.schema import GraphSchema, default_schema


def cpu_workers() -> int:
    """Use the process CPU allocation when available, otherwise the machine count."""
    process_count = getattr(os, "process_cpu_count", os.cpu_count)
    return process_count() or 1


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ModelClientConfig(ConfigModel):
    """Provider selection and native LangChain constructor options for CLI clients."""

    provider: Literal["openai", "azure-openai"] = "openai"
    model: str = Field(min_length=1)
    api_version: str | None = None
    options: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_options(self) -> Self:
        if {"model", "model_name", "api_version", "openai_api_version"} & self.options.keys():
            raise ValueError("Set model and api_version directly, not in options")
        if self.api_version is not None and self.provider != "azure-openai":
            raise ValueError("api_version is only supported for azure-openai")
        return self


class SQLiteConfig(ConfigModel):
    kind: Literal["sqlite"] = "sqlite"
    path: str = ".raganite/graph.sqlite3"
    timeout: float = Field(default=30, gt=0)
    read_only: bool = False


class Neo4jConfig(ConfigModel):
    kind: Literal["neo4j"] = "neo4j"
    layout: Literal["owned", "native"] = "owned"
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: SecretStr | None = None
    database: str = "neo4j"
    read_only: bool = False
    vector_index: str = "raganite_chunks_vector"
    fulltext_index: str = "raganite_chunks_fulltext"
    fact_vector_index: str = "raganite_facts_vector"
    driver_options: dict = Field(default_factory=dict)


BackendConfig = Annotated[SQLiteConfig | Neo4jConfig, Field(discriminator="kind")]


class ChunkingConfig(ConfigModel):
    strategy: Literal["fixed", "recursive", "character"] = "fixed"
    chunk_size: int = Field(default=1500, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    options: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_overlap(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if {"chunk_size", "chunk_overlap"} & self.options.keys():
            raise ValueError("Set chunk_size and chunk_overlap directly, not in options")
        return self


class IndexingConfig(ConfigModel):
    workers: int = Field(default_factory=cpu_workers, ge=1)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    schema_mode: Literal["fixed", "free", "extracted"] = "fixed"
    graph_schema: GraphSchema | None = Field(default_factory=default_schema)
    perform_entity_resolution: bool = True
    on_error: Literal["RAISE", "IGNORE"] = "RAISE"
    extraction_prompt: str | None = None
    examples: str = ""
    index_facts: bool = True
    max_file_bytes: int = Field(default=1_000_000, gt=0)
    include_globs: tuple[str, ...] = (
        "*.py",
        "*.md",
        "*.rst",
        "*.txt",
        "*.json",
        "*.yaml",
        "*.yml",
        "*.toml",
        "*.ini",
        "*.cfg",
    )
    exclude_globs: tuple[str, ...] = (
        "*/.git/*",
        "*/.venv/*",
        "*/venv/*",
        "*/node_modules/*",
        "*/__pycache__/*",
        "*/.mypy_cache/*",
        "*/.pytest_cache/*",
        "*/dist/*",
        "*/build/*",
        "*.lock",
    )


class RetrievalConfig(ConfigModel):
    """Fact search, graph expansion, passage fusion, and optional LLM ranking."""

    workers: int = Field(default_factory=cpu_workers, ge=1)
    top_k: int = Field(default=200, ge=1)
    triple_top_k: int = Field(default=30, ge=1)
    filtered_triple_k: int = Field(default=8, ge=1)
    expansion_depth: int = Field(default=2, ge=0, le=5)
    expansion_limit: int = Field(default=5000, ge=1)
    degree_penalty_weight: float = Field(default=0.25, ge=0, allow_inf_nan=False)
    graph_weight: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    rrf_k: int = Field(default=60, ge=1)
    rerank_top_k: int = Field(default=20, ge=1)
    use_triples: bool = True
    filter_triples: bool = True
    rerank_passages: bool = True
    llm_options: dict = Field(
        default_factory=lambda: {
            "temperature": 0.0,
            "max_tokens": 256,
            "response_format": {"type": "json_object"},
        }
    )

    @model_validator(mode="after")
    def validate_fact_limit(self) -> Self:
        if self.filtered_triple_k > self.triple_top_k:
            raise ValueError("filtered_triple_k cannot exceed triple_top_k")
        return self


class RaganiteConfig(BaseSettings):
    """Settings from arguments or RAGANITE_ environment keys with __ nesting.

    For example, RAGANITE_INDEXING__WORKERS=4. No .env file is read implicitly.
    """

    model_config = SettingsConfigDict(
        env_prefix="RAGANITE_",
        env_nested_delimiter="__",
        extra="forbid",
    )
    backend: BackendConfig = Field(default_factory=SQLiteConfig)
    namespace: str = Field(default="default", min_length=1)
    embedding_dimensions: int = Field(default=1536, ge=1)
    chat: ModelClientConfig = Field(default_factory=lambda: ModelClientConfig(model="gpt-4o"))
    embeddings: ModelClientConfig = Field(
        default_factory=lambda: ModelClientConfig(model="text-embedding-3-small")
    )
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """Load JSON or TOML; nested settings use the same shape as Python config."""
        path = Path(path)
        if path.suffix == ".toml":
            import tomllib

            with path.open("rb") as stream:
                return cls.model_validate(tomllib.load(stream))
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
