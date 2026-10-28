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

"""Azure OpenAI clients for v1 and traditional versioned endpoints."""

from typing import Any

from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAI,
    AzureOpenAIEmbeddings,
    ChatOpenAI,
    OpenAIEmbeddings,
)

from .base import CompletionClient, EmbeddingClient, LLMClient


class AzureOpenAILLMClient(LLMClient):
    """Azure chat without changing LangChain's request or response behavior.

    For v1, pass ``base_url`` ending in ``/openai/v1/`` and ``model`` set to
    the deployment name. For the dated API, pass ``api_version`` along with
    AzureChatOpenAI options such as ``azure_endpoint`` and ``azure_deployment``.
    Credential resolution is left to the selected LangChain class.
    """

    def __init__(self, *, api_version: str | None = None, **kwargs: Any) -> None:
        self._model: ChatOpenAI | AzureChatOpenAI
        if api_version is None:
            self._model = ChatOpenAI(**kwargs)
        else:
            self._model = AzureChatOpenAI(api_version=api_version, **kwargs)

    @property
    def model(self) -> ChatOpenAI | AzureChatOpenAI:
        return self._model


class AzureOpenAIEmbeddingClient(EmbeddingClient):
    """Azure embeddings, with LangChain handling dimensions and batching.

    For v1, pass ``base_url`` ending in ``/openai/v1/`` and the deployment as
    ``model``. For the dated API, pass ``api_version`` and the native
    AzureOpenAIEmbeddings options (``azure_endpoint``, ``azure_deployment``, etc.).
    """

    def __init__(self, *, api_version: str | None = None, **kwargs: Any) -> None:
        self._model: OpenAIEmbeddings
        if api_version is None:
            self._model = OpenAIEmbeddings(**kwargs)
        else:
            self._model = AzureOpenAIEmbeddings(api_version=api_version, **kwargs)

    @property
    def model(self) -> OpenAIEmbeddings:
        return self._model


class AzureOpenAICompletionClient(CompletionClient):
    """Legacy Azure text completions via LangChain's dated AzureOpenAI class.

    Pass native AzureOpenAI options, including ``api_version`` (or configure
    its environment variable) and ``azure_deployment``. This is not chat.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._model = AzureOpenAI(**kwargs)

    @property
    def model(self) -> AzureOpenAI:
        return self._model
