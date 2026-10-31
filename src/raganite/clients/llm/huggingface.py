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

"""Hugging Face clients; constructor options belong to LangChain."""

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from .base import EmbeddingClient, LLMClient

try:
    import langchain_huggingface

    ChatHuggingFace: type[BaseChatModel] | None = langchain_huggingface.ChatHuggingFace
    HuggingFaceEmbeddings: type[Embeddings] | None = langchain_huggingface.HuggingFaceEmbeddings
    HuggingFaceEndpointEmbeddings: type[Embeddings] | None = (
        langchain_huggingface.HuggingFaceEndpointEmbeddings
    )
except ImportError:
    ChatHuggingFace = None
    HuggingFaceEmbeddings = None
    HuggingFaceEndpointEmbeddings = None

_EXTRA = "Hugging Face models need the optional extra: uv sync --extra huggingface"


def _require(implementation: Any) -> Any:
    if implementation is None:
        raise ImportError(_EXTRA)
    return implementation


class HuggingFaceEmbeddingClient(EmbeddingClient):
    """Local embeddings for any Sentence Transformers model on the Hub.

    Pass ``HuggingFaceEmbeddings`` options unchanged, including ``model_name``,
    ``model_kwargs``, ``encode_kwargs``, and ``query_encode_kwargs``. Query and
    document prompts belong in those encode dictionaries.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._model = _require(HuggingFaceEmbeddings)(**kwargs)

    @property
    def model(self) -> Embeddings:
        return self._model


class HuggingFaceEndpointEmbeddingClient(EmbeddingClient):
    """Hosted embeddings via Hugging Face Inference Providers or an endpoint.

    Pass ``HuggingFaceEndpointEmbeddings`` options unchanged, including
    ``model`` and ``provider``.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._model = _require(HuggingFaceEndpointEmbeddings)(**kwargs)

    @property
    def model(self) -> Embeddings:
        return self._model


class HuggingFaceLLMClient(LLMClient):
    """Chat models on the Hugging Face Hub.

    ``model_id`` is passed to ``ChatHuggingFace.from_model_id``. ``backend`` is
    ``pipeline`` for a local model, ``endpoint`` for a hosted Inference Provider,
    or ``text-gen`` for a text-generation-inference server. Remaining keyword
    arguments are LangChain's. Pass ``llm`` to supply a backend object directly.
    """

    def __init__(self, **kwargs: Any) -> None:
        chat = _require(ChatHuggingFace)
        if "llm" in kwargs:
            self._model = chat(**kwargs)
            return
        model_id = kwargs.pop("model_id", None)
        if not model_id:
            raise ValueError("Hugging Face chat requires model_id or llm")
        backend = kwargs.pop("backend", "pipeline")
        task = kwargs.pop("task", None)
        self._model = chat.from_model_id(model_id, task=task, backend=backend, **kwargs)

    @property
    def model(self) -> BaseChatModel:
        return self._model
