# Model providers

Choose OpenAI, Azure OpenAI, or Hugging Face. Chat and embeddings are selected independently. Raganite keeps the LangChain object and exposes it as `client.model`. Calls, streaming, tools, and structured output are LangChain's.

You need two roles for a normal index-and-retrieve flow:

| Role | Contract | OpenAI | Azure | Hugging Face |
| --- | --- | --- | --- | --- |
| Chat | `LLMClient` | `OpenAILLMClient` | `AzureOpenAILLMClient` | `HuggingFaceLLMClient` |
| Embeddings | `EmbeddingClient` | `OpenAIEmbeddingClient` | `AzureOpenAIEmbeddingClient` | `HuggingFaceEmbeddingClient`, `HuggingFaceEndpointEmbeddingClient` |

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

## Hugging Face

Install the optional extra before constructing these clients. It pulls in `sentence-transformers`, Transformers, and PyTorch:

```sh
uv sync --extra huggingface
```

Chat uses `ChatHuggingFace` from `langchain-huggingface`. LangChain moved that class out of `langchain-community`. A local model uses `backend="pipeline"`. A hosted Inference Provider uses `backend="endpoint"`. `backend="text-gen"` talks to a text-generation-inference server and imports that client from `langchain-community` when you select it.

```python
from raganite.clients.llm import HuggingFaceLLMClient

chat = HuggingFaceLLMClient(
    model_id="HuggingFaceH4/zephyr-7b-beta",
    backend="pipeline",
    max_new_tokens=512,
)
```

Hosted chat reads `HUGGINGFACEHUB_API_TOKEN`:

```python
chat = HuggingFaceLLMClient(
    model_id="deepseek-ai/DeepSeek-R1-0528",
    backend="endpoint",
    provider="hf-inference",
)
```

Pass `llm` when you already built a `HuggingFacePipeline` or `HuggingFaceEndpoint`. Other keyword arguments are LangChain's.

Local embedding models run in-process. The model id is `model_name`. Hosted embedding endpoints use `model`. Put query and document prompts in `encode_kwargs` and `query_encode_kwargs`.

```python
from raganite.clients.llm import HuggingFaceEmbeddingClient, OpenAILLMClient
from raganite.config import RaganiteConfig

chat = OpenAILLMClient(model="gpt-4o", use_responses_api=False)
embeddings = HuggingFaceEmbeddingClient(
    model_name="BAAI/bge-large-en-v1.5",
    encode_kwargs={"normalize_embeddings": True},
    query_encode_kwargs={
        "prompt": "Represent this sentence for searching relevant passages: "
    },
)
config = RaganiteConfig(namespace="bge-en", embedding_dimensions=1024)
```

`bge-large-en-v1.5` emits 1024-dimensional vectors. Set `embedding_dimensions` to that width and index into a new namespace. Read the model card for the prompt strings. Pass document prompts in `encode_kwargs` and query prompts in `query_encode_kwargs`. Models that need no prompt, such as `sentence-transformers/all-mpnet-base-v2`, omit both.

Hosted embeddings:

```python
from raganite.clients.llm import HuggingFaceEndpointEmbeddingClient

embeddings = HuggingFaceEndpointEmbeddingClient(
    model="BAAI/bge-large-en-v1.5",
    provider="hf-inference",
)
```

`HuggingFaceEndpointEmbeddings` reads `HUGGINGFACEHUB_API_TOKEN` when you omit the token argument.

`--provider huggingface` selects a local chat model and local embeddings together. `--provider huggingface-endpoint` selects hosted chat and hosted embeddings. To mix them, set the two providers separately:

```sh
RAGANITE_EMBEDDINGS__PROVIDER=huggingface
RAGANITE_EMBEDDINGS__MODEL=BAAI/bge-large-en-v1.5
RAGANITE_EMBEDDING_DIMENSIONS=1024
```

`huggingface-endpoint` is the hosted provider. Put endpoint options such as `provider` and prompts in `RAGANITE_EMBEDDINGS__OPTIONS`.

## Ollama

Ollama serves a local chat model. Install the extra, pull a model, and leave the Ollama server running. There is no API key. `model` is the same tag you pulled. Other keyword arguments, including `base_url` and `temperature`, are LangChain's.

```sh
uv sync --extra ollama
ollama pull llama3.2
```

```python
from raganite.clients.llm import OllamaLLMClient

chat = OllamaLLMClient(model="llama3.2", temperature=0)
```

`--provider` sets chat and embeddings together, and Ollama has no embedding client. Select it for chat only:

```sh
RAGANITE_CHAT__PROVIDER=ollama
RAGANITE_CHAT__MODEL=llama3.2
```

## Dimensions

The embedding model's vector length must equal `RaganiteConfig.embedding_dimensions` and the stored index. The default is 1536, which matches `text-embedding-3-small` unless you request another size.

The README in the repository has more client examples, including streaming and batch embed calls.
