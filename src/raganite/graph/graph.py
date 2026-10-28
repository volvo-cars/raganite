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

"""Application entry point for configurable graph indexing and retrieval."""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_text_splitters import TextSplitter

from raganite.backends import GraphBackend, create_backend
from raganite.clients.llm import EmbeddingClient, LLMClient
from raganite.config import RaganiteConfig
from raganite.indexing import KnowledgeGraphIndexer
from raganite.retrieval import Passage, Retriever


class KnowledgeGraph:
    def __init__(
        self,
        *,
        embeddings: EmbeddingClient | Embeddings,
        llm: LLMClient | BaseChatModel | None = None,
        config: RaganiteConfig | None = None,
        backend: GraphBackend | None = None,
        text_splitter: TextSplitter | None = None,
    ):
        if config is not None:
            self.config = config
        elif backend is not None:
            self.config = RaganiteConfig(
                namespace=backend.namespace,
                embedding_dimensions=backend.dimensions,
            )
        else:
            self.config = RaganiteConfig()
        if backend and (backend.namespace, backend.dimensions) != (
            self.config.namespace,
            self.config.embedding_dimensions,
        ):
            raise ValueError("Configured namespace/dimensions must match the injected backend")
        self._owns_backend = backend is None
        self.backend = backend if backend is not None else create_backend(self.config)
        self._llm = llm
        self._embeddings = embeddings
        self._retriever: Retriever | None = None
        self._indexer = (
            KnowledgeGraphIndexer(
                self.backend,
                llm,
                embeddings,
                self.config,
                text_splitter=text_splitter,
            )
            if llm is not None
            else None
        )

    @property
    def retriever(self) -> Retriever:
        """Initialize retrieval on demand so an empty graph can be indexed first."""
        if self._retriever is None:
            self._retriever = Retriever(
                self.backend,
                self._embeddings,
                self._llm,
                self.config.retrieval,
            )
        return self._retriever

    def retrieve(self, question: str, *, top_k: int | None = None) -> list[Passage]:
        """Return fact-expanded and reranked passages for a question."""
        return self.retriever.retrieve(question, top_k=top_k)

    @property
    def indexer(self) -> KnowledgeGraphIndexer:
        if self._indexer is None:
            raise ValueError("Supply an LLM when constructing KnowledgeGraph to enable indexing")
        return self._indexer

    def close(self) -> None:
        if self._retriever is not None:
            self._retriever.close()
        if self._owns_backend:
            self.backend.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
