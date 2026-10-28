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

"""Client contracts that preserve LangChain's native model interfaces."""

from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel, BaseLLM


class LLMClient(ABC):
    """Provide a chat model with the complete LangChain chat and Runnable API."""

    @property
    @abstractmethod
    def model(self) -> BaseChatModel:
        """Return the same configured model on every access, without wrapping calls."""


class EmbeddingClient(ABC):
    """Provide query and document embeddings, including their async operations."""

    @property
    @abstractmethod
    def model(self) -> Embeddings:
        """Return the same configured LangChain embedding model on every access."""


class CompletionClient(ABC):
    """Provide legacy text completions, distinct from message-based chat."""

    @property
    @abstractmethod
    def model(self) -> BaseLLM:
        """Return the native text-completion model, whose invoke result is a string."""
