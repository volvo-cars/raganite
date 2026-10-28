# Model providers

Choose OpenAI or Azure OpenAI. Raganite keeps the LangChain object and exposes it as `client.model`. Calls, streaming, tools, and structured output are LangChain's.

You need two roles for a normal index-and-retrieve flow:

| Role | Contract | OpenAI | Azure |
| --- | --- | --- | --- |
| Chat | `LLMClient` | `OpenAILLMClient` | `AzureOpenAILLMClient` |
| Embeddings | `EmbeddingClient` | `OpenAIEmbeddingClient` | `AzureOpenAIEmbeddingClient` |

`CompletionClient` is the older prompt-to-string surface. Indexing and retrieval do not use it.

## OpenAI

```python
from raganite.clients.llm import OpenAIEmbeddingClient, OpenAILLMClient

chat = OpenAILLMClient(model="gpt-4o", use_responses_api=False)
embeddings = OpenAIEmbeddingClient(model="text-embedding-3-small")
```

LangChain reads `OPENAI_API_KEY` when you omit `api_key`. Keyword arguments are passed through to LangChain. Raganite does not set temperature, timeouts, or retry policy.

## Azure, current v1 endpoint

```python
from raganite.clients.llm import AzureOpenAIEmbeddingClient, AzureOpenAILLMClient

chat = AzureOpenAILLMClient(
    base_url="https://your-resource.openai.azure.com/openai/v1/",
    model="your-chat-deployment",
    api_key="<supply at runtime>",
    use_responses_api=False,
)
embeddings = AzureOpenAIEmbeddingClient(
    base_url="https://your-resource.openai.azure.com/openai/v1/",
    model="your-embedding-deployment",
    api_key="<supply at runtime>",
)
```

Omit `api_version` for this path. Pass `api_key` yourself, or let the LangChain class read the environment variables it already knows.

## Azure, dated API version

Pass `api_version` to select the older Azure classes:

```python
chat = AzureOpenAILLMClient(
    azure_endpoint="https://your-resource.openai.azure.com/",
    azure_deployment="your-chat-deployment",
    api_version="<version supported by your deployment>",
    use_responses_api=False,
)
```

Do not mix `base_url` from the v1 example with `api_version`.

## Dimensions

The embedding model's vector length must equal `RaganiteConfig.embedding_dimensions` and the stored index. The default is 1536, which matches `text-embedding-3-small` unless you request another size.

The README in the repository has more client examples, including streaming and batch embed calls.
