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

"""Configuration-driven constructors; native LangChain options remain intact."""

import os
from typing import Any

from raganite.config import ModelClientConfig

from .azure_openai import AzureOpenAIEmbeddingClient, AzureOpenAILLMClient
from .base import EmbeddingClient, LLMClient
from .openai import OpenAIEmbeddingClient, OpenAILLMClient


def _options(config: ModelClientConfig) -> dict[str, Any]:
    options = {"model": config.model, **config.options}
    if config.provider != "azure-openai":
        return options
    if config.api_version is not None:
        options.setdefault("azure_deployment", config.model)
        return options
    # Azure v1 uses the ordinary OpenAI classes. Map Azure's native credential
    # names here rather than requiring users to duplicate OPENAI_* variables.
    if "base_url" not in options and "openai_api_base" not in options:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ValueError("Azure v1 needs AZURE_OPENAI_ENDPOINT or options.base_url")
        endpoint = endpoint.rstrip("/")
        options["base_url"] = endpoint + ("/" if endpoint.endswith("/openai/v1") else "/openai/v1/")
    if "api_key" not in options and "openai_api_key" not in options:
        key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_AD_TOKEN")
        if not key:
            raise ValueError("Azure v1 needs AZURE_OPENAI_API_KEY or AZURE_OPENAI_AD_TOKEN")
        options["api_key"] = key
    return options


def create_chat_client(config: ModelClientConfig) -> LLMClient:
    options = _options(config)
    if config.provider == "azure-openai":
        return AzureOpenAILLMClient(api_version=config.api_version, **options)
    return OpenAILLMClient(**options)


def create_embedding_client(config: ModelClientConfig, *, dimensions: int) -> EmbeddingClient:
    options = _options(config)
    # Do not impose the optional dimensions parameter on models that do not
    # support it. Configure it explicitly in native options when shortening
    # vectors; the backend validates every vector against the configured size.
    if "dimensions" in options and options["dimensions"] != dimensions:
        raise ValueError("Embedding options.dimensions must match embedding_dimensions")
    if config.provider == "azure-openai":
        return AzureOpenAIEmbeddingClient(api_version=config.api_version, **options)
    return OpenAIEmbeddingClient(**options)
