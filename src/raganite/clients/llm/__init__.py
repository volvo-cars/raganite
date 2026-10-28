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

"""LLM clients that expose native LangChain models through ``client.model``."""

from .azure_openai import (
    AzureOpenAICompletionClient,
    AzureOpenAIEmbeddingClient,
    AzureOpenAILLMClient,
)
from .base import CompletionClient, EmbeddingClient, LLMClient
from .openai import OpenAICompletionClient, OpenAIEmbeddingClient, OpenAILLMClient

__all__ = [
    "AzureOpenAICompletionClient",
    "AzureOpenAIEmbeddingClient",
    "AzureOpenAILLMClient",
    "CompletionClient",
    "EmbeddingClient",
    "LLMClient",
    "OpenAICompletionClient",
    "OpenAIEmbeddingClient",
    "OpenAILLMClient",
]
