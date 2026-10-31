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
from .huggingface import (
    HuggingFaceEmbeddingClient,
    HuggingFaceEndpointEmbeddingClient,
    HuggingFaceLLMClient,
)
from .ollama import OllamaLLMClient
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
        if endpoint:
            endpoint = endpoint.rstrip("/")
            options["base_url"] = endpoint + (
                "/" if endpoint.endswith("/openai/v1") else "/openai/v1/"
            )
        else:
            # OPENAI_API_BASE is already the full base, including /openai/v1/ when used.
            base = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
            if not base:
                raise ValueError(
                    "Azure v1 needs AZURE_OPENAI_ENDPOINT, OPENAI_API_BASE, or options.base_url"
                )
            options["base_url"] = base if base.endswith("/") else base + "/"
    if "api_key" not in options and "openai_api_key" not in options:
        key = (
            os.getenv("AZURE_OPENAI_API_KEY")
            or os.getenv("AZURE_OPENAI_AD_TOKEN")
            or os.getenv("OPENAI_API_KEY")
        )
        if not key:
            raise ValueError(
                "Azure v1 needs AZURE_OPENAI_API_KEY, AZURE_OPENAI_AD_TOKEN, or OPENAI_API_KEY"
            )
        options["api_key"] = key
    return options


def create_chat_client(config: ModelClientConfig) -> LLMClient:
    if config.provider in {"huggingface", "huggingface-endpoint"}:
        options = dict(config.options)
        options.setdefault(
            "backend",
            "endpoint" if config.provider == "huggingface-endpoint" else "pipeline",
        )
        return HuggingFaceLLMClient(model_id=config.model, **options)
    options = _options(config)
    if config.provider == "ollama":
        return OllamaLLMClient(**options)
    if config.provider == "azure-openai":
        return AzureOpenAILLMClient(api_version=config.api_version, **options)
    return OpenAILLMClient(**options)


def create_embedding_client(config: ModelClientConfig, *, dimensions: int) -> EmbeddingClient:
    if config.provider == "ollama":
        raise ValueError("Ollama is a chat provider; set embeddings.provider separately")
    if config.provider == "huggingface":
        # Local Sentence Transformers models take model_name, not OpenAI's model.
        options = {"model_name": config.model, **config.options}
    elif config.provider == "huggingface-endpoint":
        options = {"model": config.model, **config.options}
    else:
        options = _options(config)
    # Do not impose the optional dimensions parameter on models that do not
    # support it. Configure it explicitly in native options when shortening
    # vectors; the backend validates every vector against the configured size.
    if "dimensions" in options and options["dimensions"] != dimensions:
        raise ValueError("Embedding options.dimensions must match embedding_dimensions")
    if config.provider == "huggingface":
        return HuggingFaceEmbeddingClient(**options)
    if config.provider == "huggingface-endpoint":
        return HuggingFaceEndpointEmbeddingClient(**options)
    if config.provider == "azure-openai":
        return AzureOpenAIEmbeddingClient(api_version=config.api_version, **options)
    return OpenAIEmbeddingClient(**options)
