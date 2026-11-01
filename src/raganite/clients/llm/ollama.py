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

"""Ollama chat; constructor options belong to LangChain."""

from typing import Any

from langchain_core.language_models import BaseChatModel

from .base import LLMClient

try:
    import langchain_ollama

    ChatOllama: type[BaseChatModel] | None = langchain_ollama.ChatOllama
except ImportError:
    ChatOllama = None

_EXTRA = "Ollama chat needs the optional extra: uv sync --extra ollama"


class OllamaLLMClient(LLMClient):
    """Local chat via ChatOllama.

    Pass LangChain constructor options unchanged. ``model`` is the tag from
    ``ollama pull``. ``base_url`` defaults to the local Ollama server.
    """

    def __init__(self, **kwargs: Any) -> None:
        if ChatOllama is None:
            raise ImportError(_EXTRA)
        self._model = ChatOllama(**kwargs)

    @property
    def model(self) -> BaseChatModel:
        return self._model
