"""Hugging Face embedding clients forward LangChain options and stay optional."""

from unittest.mock import Mock

import pytest

from raganite.clients.llm import EmbeddingClient, LLMClient
from raganite.clients.llm.factory import create_chat_client, create_embedding_client
from raganite.clients.llm.huggingface import (
    HuggingFaceEmbeddingClient,
    HuggingFaceEndpointEmbeddingClient,
    HuggingFaceLLMClient,
)
from raganite.config import ModelClientConfig


def test_local_embeddings_forward_native_options(monkeypatch):
    native = Mock()
    constructor = Mock(return_value=native)
    monkeypatch.setattr(
        "raganite.clients.llm.huggingface.HuggingFaceEmbeddings",
        constructor,
    )
    kwargs = {
        "model_name": "BAAI/bge-large-en-v1.5",
        "model_kwargs": {"device": "cpu"},
        "encode_kwargs": {"normalize_embeddings": True},
        "query_encode_kwargs": {
            "prompt": "Represent this sentence for searching relevant passages: "
        },
    }
    client = HuggingFaceEmbeddingClient(**kwargs)
    constructor.assert_called_once_with(**kwargs)
    assert isinstance(client, EmbeddingClient)
    assert client.model is native
    assert client.model is client.model
    with pytest.raises(AttributeError):
        client.model = native


def test_endpoint_embeddings_forward_native_options(monkeypatch):
    native = Mock()
    constructor = Mock(return_value=native)
    monkeypatch.setattr(
        "raganite.clients.llm.huggingface.HuggingFaceEndpointEmbeddings",
        constructor,
    )
    kwargs = {"model": "BAAI/bge-large-en-v1.5", "provider": "hf-inference"}
    client = HuggingFaceEndpointEmbeddingClient(**kwargs)
    constructor.assert_called_once_with(**kwargs)
    assert isinstance(client, EmbeddingClient)
    assert client.model is native
    assert client.model is client.model


def test_local_embeddings_require_the_optional_extra(monkeypatch):
    monkeypatch.setattr("raganite.clients.llm.huggingface.HuggingFaceEmbeddings", None)
    with pytest.raises(ImportError, match="huggingface"):
        HuggingFaceEmbeddingClient(model_name="BAAI/bge-large-en-v1.5")


def test_factory_builds_local_huggingface_embeddings(monkeypatch):
    native = Mock()
    constructor = Mock(return_value=native)
    monkeypatch.setattr(
        "raganite.clients.llm.huggingface.HuggingFaceEmbeddings",
        constructor,
    )
    client = create_embedding_client(
        ModelClientConfig(
            provider="huggingface",
            model="BAAI/bge-large-en-v1.5",
            options={
                "encode_kwargs": {"normalize_embeddings": True},
                "query_encode_kwargs": {
                    "prompt": "Represent this sentence for searching relevant passages: "
                },
            },
        ),
        dimensions=1024,
    )
    constructor.assert_called_once_with(
        model_name="BAAI/bge-large-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs={"prompt": "Represent this sentence for searching relevant passages: "},
    )
    assert client.model is native


def test_factory_builds_huggingface_endpoint_embeddings(monkeypatch):
    native = Mock()
    constructor = Mock(return_value=native)
    monkeypatch.setattr(
        "raganite.clients.llm.huggingface.HuggingFaceEndpointEmbeddings",
        constructor,
    )
    client = create_embedding_client(
        ModelClientConfig(
            provider="huggingface-endpoint",
            model="BAAI/bge-large-en-v1.5",
            options={"provider": "hf-inference"},
        ),
        dimensions=1024,
    )
    constructor.assert_called_once_with(model="BAAI/bge-large-en-v1.5", provider="hf-inference")
    assert client.model is native


def test_chat_from_model_id_forwards_native_options(monkeypatch):
    native = Mock()
    chat = Mock()
    chat.from_model_id = Mock(return_value=native)
    monkeypatch.setattr("raganite.clients.llm.huggingface.ChatHuggingFace", chat)
    client = HuggingFaceLLMClient(
        model_id="HuggingFaceH4/zephyr-7b-beta",
        backend="pipeline",
        max_new_tokens=128,
    )
    chat.from_model_id.assert_called_once_with(
        "HuggingFaceH4/zephyr-7b-beta",
        task=None,
        backend="pipeline",
        max_new_tokens=128,
    )
    chat.assert_not_called()
    assert isinstance(client, LLMClient)
    assert client.model is native
    assert client.model is client.model


def test_chat_with_llm_uses_the_native_constructor(monkeypatch):
    native = Mock()
    llm = Mock()
    chat = Mock(return_value=native)
    chat.from_model_id = Mock()
    monkeypatch.setattr("raganite.clients.llm.huggingface.ChatHuggingFace", chat)
    client = HuggingFaceLLMClient(llm=llm, temperature=0)
    chat.assert_called_once_with(llm=llm, temperature=0)
    chat.from_model_id.assert_not_called()
    assert client.model is native


def test_chat_requires_model_id_or_llm(monkeypatch):
    monkeypatch.setattr("raganite.clients.llm.huggingface.ChatHuggingFace", Mock())
    with pytest.raises(ValueError, match="model_id"):
        HuggingFaceLLMClient(backend="endpoint")


def test_chat_requires_the_optional_extra(monkeypatch):
    monkeypatch.setattr("raganite.clients.llm.huggingface.ChatHuggingFace", None)
    with pytest.raises(ImportError, match="huggingface"):
        HuggingFaceLLMClient(model_id="org/model")


def test_factory_builds_local_and_hosted_huggingface_chat(monkeypatch):
    native = Mock()
    chat = Mock()
    chat.from_model_id = Mock(return_value=native)
    monkeypatch.setattr("raganite.clients.llm.huggingface.ChatHuggingFace", chat)
    local = create_chat_client(
        ModelClientConfig(
            provider="huggingface",
            model="HuggingFaceH4/zephyr-7b-beta",
            options={"max_new_tokens": 64},
        )
    )
    hosted = create_chat_client(
        ModelClientConfig(
            provider="huggingface-endpoint",
            model="deepseek-ai/DeepSeek-R1-0528",
            options={"provider": "hf-inference"},
        )
    )
    assert local.model is native
    assert hosted.model is native
    assert chat.from_model_id.call_args_list[0].args == ("HuggingFaceH4/zephyr-7b-beta",)
    assert chat.from_model_id.call_args_list[0].kwargs == {
        "task": None,
        "backend": "pipeline",
        "max_new_tokens": 64,
    }
    assert chat.from_model_id.call_args_list[1].args == ("deepseek-ai/DeepSeek-R1-0528",)
    assert chat.from_model_id.call_args_list[1].kwargs == {
        "task": None,
        "backend": "endpoint",
        "provider": "hf-inference",
    }


def test_endpoint_embeddings_require_the_optional_extra(monkeypatch):
    monkeypatch.setattr(
        "raganite.clients.llm.huggingface.HuggingFaceEndpointEmbeddings",
        None,
    )
    with pytest.raises(ImportError, match="huggingface"):
        HuggingFaceEndpointEmbeddingClient(model="BAAI/bge-large-en-v1.5")
