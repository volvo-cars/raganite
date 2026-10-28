"""Exercise the real LangChain implementation without calling model services."""

import asyncio
import inspect
import json
from typing import Any
from unittest.mock import Mock

import httpx
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.runnables import Runnable
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAI,
    AzureOpenAIEmbeddings,
    ChatOpenAI,
    OpenAI,
    OpenAIEmbeddings,
)
from pydantic import BaseModel

from raganite.clients.llm import (
    AzureOpenAICompletionClient,
    AzureOpenAIEmbeddingClient,
    AzureOpenAILLMClient,
    CompletionClient,
    EmbeddingClient,
    LLMClient,
    OpenAICompletionClient,
    OpenAIEmbeddingClient,
    OpenAILLMClient,
)

CHAT_CASES = [
    (OpenAILLMClient, {"base_url": "https://openai.test/v1/"}, ChatOpenAI),
    (
        AzureOpenAILLMClient,
        {"base_url": "https://azure.test/openai/v1/"},
        ChatOpenAI,
    ),
    (
        AzureOpenAILLMClient,
        {
            "azure_endpoint": "https://azure.test/",
            "azure_deployment": "chat-deployment",
            "api_version": "2024-10-21",
        },
        AzureChatOpenAI,
    ),
]

EMBEDDING_CASES = [
    (OpenAIEmbeddingClient, {"base_url": "https://openai.test/v1/"}, OpenAIEmbeddings),
    (
        AzureOpenAIEmbeddingClient,
        {"base_url": "https://azure.test/openai/v1/"},
        OpenAIEmbeddings,
    ),
    (
        AzureOpenAIEmbeddingClient,
        {
            "azure_endpoint": "https://azure.test/",
            "azure_deployment": "embedding-deployment",
            "api_version": "2024-10-21",
        },
        AzureOpenAIEmbeddings,
    ),
]

COMPLETION_CASES = [
    (OpenAICompletionClient, {"base_url": "https://openai.test/v1/"}, OpenAI),
    (
        AzureOpenAICompletionClient,
        {
            "azure_endpoint": "https://azure.test/",
            "azure_deployment": "completion-deployment",
            "api_version": "2024-10-21",
        },
        AzureOpenAI,
    ),
]


def respond(request):
    """Return realistic provider payloads for the native SDK's HTTP calls."""
    body = json.loads(request.content)
    if request.url.path.endswith("/embeddings"):
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": body["model"],
                "data": [
                    {"object": "embedding", "index": i, "embedding": [float(len(text)), 1.0]}
                    for i, text in enumerate(body["input"])
                ],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )
    if request.url.path.endswith("/responses"):
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": body["model"],
                "output": [
                    {
                        "id": "msg_test",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hello", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        )
    chat = request.url.path.endswith("/chat/completions")
    if body.get("stream"):
        chunks = [
            {
                "id": "chat_test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "Hello"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chat_test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": body["model"],
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n",
        )
    choice: dict[str, Any] = {"index": 0, "finish_reason": "stop", "logprobs": None}
    if chat:
        choice["message"] = {"role": "assistant", "content": '{"answer": 42}'}
    else:
        choice["text"] = "Hello"
    return httpx.Response(
        200,
        json={
            "id": "completion_test",
            "object": "chat.completion" if chat else "text_completion",
            "created": 1,
            "model": body["model"],
            "choices": [choice],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    # Do not consume developer credentials or endpoint overrides in offline tests.
    for key in (
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_API_VERSION",
        "OPENAI_PROXY",
        "OPENAI_ORG_ID",
        "OPENAI_ORGANIZATION",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_AD_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def http_options():
    requests = []

    def handler(request):
        requests.append(request)
        return respond(request)

    sync_client = httpx.Client(transport=httpx.MockTransport(handler))
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        yield (
            {
                "api_key": "offline-test-key",
                "max_retries": 0,
                "http_client": sync_client,
                "http_async_client": async_client,
            },
            requests,
        )
    finally:
        sync_client.close()
        asyncio.run(async_client.aclose())


@pytest.mark.parametrize("contract", [LLMClient, EmbeddingClient, CompletionClient])
def test_contract_is_abstract(contract):
    assert inspect.isabstract(contract)
    with pytest.raises(TypeError):
        contract()


def test_custom_provider_only_needs_model_property():
    native = Mock(spec=ChatOpenAI)

    class CustomClient(LLMClient):
        @property
        def model(self):
            return native

    assert CustomClient().model is native


@pytest.mark.parametrize(
    ("client_class", "options", "native_class"),
    CHAT_CASES + EMBEDDING_CASES + COMPLETION_CASES,
)
def test_constructor_options_are_forwarded_unchanged(
    monkeypatch, client_class: Any, options, native_class
):
    native = Mock()
    constructor = Mock(return_value=native)
    monkeypatch.setattr(f"{client_class.__module__}.{native_class.__name__}", constructor)
    token_provider = Mock(return_value="offline-token")
    kwargs = {**options, "api_key": token_provider, "timeout": 17, "max_retries": 3}
    client = client_class(**kwargs)
    constructor.assert_called_once_with(**kwargs)
    assert client.model is native
    assert client.model is client.model
    token_provider.assert_not_called()
    with pytest.raises(AttributeError):
        client.model = native


@pytest.mark.parametrize("client_class, options, native_class", CHAT_CASES)
def test_chat_keeps_native_calls_and_runnables(
    http_options, client_class: Any, options, native_class
):
    http, requests = http_options
    client = client_class(model="gpt-4o-mini", use_responses_api=False, **options, **http)
    assert isinstance(client, LLMClient)
    assert type(client.model) is native_class
    message = client.model.invoke([("user", "Hello")], response_format={"type": "json_object"})
    assert isinstance(message, AIMessage)
    assert message.content == '{"answer": 42}'
    assert message.usage_metadata["total_tokens"] == 2
    body = json.loads(requests[-1].content)
    assert body["response_format"] == {"type": "json_object"}
    assert "temperature" not in body
    assert len(client.model.batch(["One", "Two"])) == 2
    chunks = list(client.model.stream("Hello"))
    assert all(isinstance(chunk, AIMessageChunk) for chunk in chunks)
    assert "".join(chunk.content for chunk in chunks) == "Hello"

    class Answer(BaseModel):
        answer: int

    structured = client.model.with_structured_output(Answer, method="json_schema")
    assert structured.invoke("Answer") == Answer(answer=42)
    tool = {
        "type": "function",
        "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}},
    }
    bound = client.model.bind_tools([tool])
    assert isinstance(bound, Runnable)
    bound.invoke("Use a tool")
    assert json.loads(requests[-1].content)["tools"][0]["function"]["name"] == "lookup"
    assert isinstance(client.model.with_retry(), Runnable)
    if "api_version" in options:
        assert "/deployments/chat-deployment/" in requests[0].url.path
        assert requests[0].url.params["api-version"] == options["api_version"]
    else:
        assert requests[0].url.path == httpx.URL(options["base_url"]).path + "chat/completions"


@pytest.mark.asyncio
@pytest.mark.parametrize("client_class, options, native_class", CHAT_CASES)
async def test_async_chat(http_options, client_class: Any, options, native_class):
    http, _ = http_options
    model = client_class(model="gpt-4o-mini", use_responses_api=False, **options, **http).model
    assert isinstance(await model.ainvoke("Hello"), AIMessage)
    assert len(await model.abatch(["One", "Two"])) == 2
    chunks = [chunk async for chunk in model.astream("Hello")]
    assert "".join(chunk.content for chunk in chunks) == "Hello"


@pytest.mark.parametrize("client_class, options, native_class", CHAT_CASES[:2])
def test_explicit_responses_api(http_options, client_class: Any, options, native_class):
    http, requests = http_options
    model = client_class(
        model="chat-deployment",
        use_responses_api=True,
        output_version="responses/v1",
        **options,
        **http,
    ).model
    message = model.invoke("Hello")
    assert isinstance(message, AIMessage)
    assert isinstance(message.content, list)
    assert isinstance(message.content[0], dict)
    assert message.content[0]["text"] == "Hello"
    assert requests[-1].url.path.endswith("/responses")
    assert "api-version" not in requests[-1].url.params


@pytest.mark.parametrize("client_class, options, native_class", EMBEDDING_CASES)
def test_embeddings_keep_order_dimensions_and_batching(
    http_options, client_class: Any, options, native_class
):
    http, requests = http_options
    client = client_class(
        model="text-embedding-3-small",
        dimensions=2,
        chunk_size=2,
        check_embedding_ctx_length=False,
        **options,
        **http,
    )
    assert isinstance(client, EmbeddingClient)
    assert type(client.model) is native_class
    assert client.model.embed_query("abc") == [3.0, 1.0]
    assert client.model.embed_documents(["a", "bb", "ccc"]) == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    assert len(requests) == 3  # One query request and two native document batches.
    assert all(json.loads(request.content)["dimensions"] == 2 for request in requests)
    assert requests[-1].url.path.endswith("/embeddings")
    if "api_version" in options:
        assert "/deployments/embedding-deployment/" in requests[-1].url.path
        assert requests[-1].url.params["api-version"] == options["api_version"]
    else:
        assert "api-version" not in requests[-1].url.params


@pytest.mark.asyncio
@pytest.mark.parametrize("client_class, options, native_class", EMBEDDING_CASES)
async def test_async_embeddings(http_options, client_class: Any, options, native_class):
    http, _ = http_options
    model = client_class(
        model="text-embedding-3-small",
        check_embedding_ctx_length=False,
        **options,
        **http,
    ).model
    assert await model.aembed_query("abc") == [3.0, 1.0]
    assert await model.aembed_documents(["a", "bb"]) == [[1.0, 1.0], [2.0, 1.0]]


@pytest.mark.asyncio
@pytest.mark.parametrize("client_class, options, native_class", COMPLETION_CASES)
async def test_legacy_completions_remain_strings(
    http_options, client_class: Any, options, native_class
):
    http, requests = http_options
    client = client_class(model="gpt-3.5-turbo-instruct", **options, **http)
    assert isinstance(client, CompletionClient)
    assert type(client.model) is native_class
    assert client.model.invoke("Hello") == "Hello"
    assert await client.model.ainvoke("Hello") == "Hello"
    assert all(request.url.path.endswith("/completions") for request in requests)
    assert all("/chat/" not in request.url.path for request in requests)


def test_provider_validation_errors_are_not_hidden(monkeypatch):
    error = ValueError("Provider rejected configuration")
    monkeypatch.setattr(
        "raganite.clients.llm.openai.ChatOpenAI",
        Mock(side_effect=error),
    )
    with pytest.raises(ValueError) as caught:
        OpenAILLMClient()
    assert caught.value is error
