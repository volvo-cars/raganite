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

"""OpenAI clients; all constructor options belong to LangChain."""

from typing import Any

from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings

from .base import CompletionClient, EmbeddingClient, LLMClient


class OpenAILLMClient(LLMClient):
    """Chat via ChatOpenAI, including Azure v1 when given its base_url.

    Pass LangChain constructor options unchanged. Use ``model`` for all calls,
    including streaming, tools, structured output, and Responses API operations.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._model = ChatOpenAI(**kwargs)

    @property
    def model(self) -> ChatOpenAI:
        return self._model


class OpenAIEmbeddingClient(EmbeddingClient):
    """OpenAI embeddings with native batching, token handling, and async methods."""

    def __init__(self, **kwargs: Any) -> None:
        self._model = OpenAIEmbeddings(**kwargs)

    @property
    def model(self) -> OpenAIEmbeddings:
        return self._model


class OpenAICompletionClient(CompletionClient):
    """Legacy text completions, not chat or Responses."""

    def __init__(self, **kwargs: Any) -> None:
        self._model = OpenAI(**kwargs)

    @property
    def model(self) -> OpenAI:
        return self._model
