"""Ollama chat forwards LangChain options and stays optional."""

from unittest.mock import Mock

import pytest

from raganite.clients.llm import LLMClient
from raganite.clients.llm.factory import create_chat_client, create_embedding_client
from raganite.clients.llm.ollama import OllamaLLMClient
from raganite.config import ModelClientConfig


def test_ollama_chat_forwards_native_options(monkeypatch):
    native = Mock()
    constructor = Mock(return_value=native)
    monkeypatch.setattr("raganite.clients.llm.ollama.ChatOllama", constructor)
    kwargs = {"model": "gpt-oss:20b", "temperature": 0, "base_url": "http://127.0.0.1:11434"}
    client = OllamaLLMClient(**kwargs)
    constructor.assert_called_once_with(**kwargs)
    assert isinstance(client, LLMClient)
    assert client.model is native
    assert client.model is client.model
    with pytest.raises(AttributeError):
        client.model = native


def test_ollama_chat_requires_the_optional_extra(monkeypatch):
    monkeypatch.setattr("raganite.clients.llm.ollama.ChatOllama", None)
    with pytest.raises(ImportError, match="ollama"):
        OllamaLLMClient(model="gpt-oss:20b")


def test_factory_builds_ollama_chat(monkeypatch):
    native = Mock()
    constructor = Mock(return_value=native)
    monkeypatch.setattr("raganite.clients.llm.ollama.ChatOllama", constructor)
    client = create_chat_client(
        ModelClientConfig(
            provider="ollama",
            model="llama3.2",
            options={"temperature": 0},
        )
    )
    constructor.assert_called_once_with(model="llama3.2", temperature=0)
    assert client.model is native


def test_factory_rejects_ollama_embeddings():
    with pytest.raises(ValueError, match="chat"):
        create_embedding_client(
            ModelClientConfig(provider="ollama", model="llama3.2"),
            dimensions=1024,
        )
