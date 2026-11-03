# Raganite

Raganite indexes documents into a knowledge graph. `retrieve` returns the passages that help answer a question. `query` writes a short answer from those passages.

A passage is a piece of an indexed document, ranked for that question. The facts that supported a rank are on `evidence`.

## Quickstart

You need Python 3.11, 3.12, or 3.13, and [uv](https://docs.astral.sh/uv/).

```sh
git clone git@github.com:volvo-cars/raganite.git
cd raganite
uv sync
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. Do not commit `.env`. The CLI reads that file from the current working directory. The defaults use SQLite at `.raganite/graph.sqlite3`, the namespace `default`, and 1536-dimension embeddings.

```sh
mkdir -p documents
printf 'Paris is the capital of France.\n' > documents/paris.txt
uv run raganite index documents --source notes
uv run raganite retrieve "What is the capital of France?" --top-k 5 --format text
uv run raganite query "What is the capital of France?" --top-k 5 --format text
```

A directory index reads text files whose names match the include globs (`*.py`, `*.md`, `*.rst`, `*.txt`, and the other text patterns in [Indexing](docs/guides/indexing.md)). PDF, HTML, and Office files are left out. A default `query` embeds the question once, fuses chunk vectors with full-text hits and fact expansion, and calls the chat model once to write the answer. Fact filtering and passage reranking are off unless you turn them on. `--top-k` shortens the list after retrieval. SQLite scores every stored vector in the process.

`retrieve` prints each passage's rank, title, source, score, and text. `--format json` adds the supporting facts in `evidence`, plus metadata. `query` prints the answer, then those same passage lines. `--format json` on `query` also includes token and timing usage.

The commands are `init`, `config`, `index`, `retrieve`, `query`, and `clear`. The longer walkthrough is [Your first index](docs/getting-started/quickstart.md). Flags are in the [command line reference](docs/reference/cli.md).

## Python

The CLI loads `.env`. These clients read `OPENAI_API_KEY` from the process environment, so export the key before running the snippet.

```python
from raganite.clients.llm import OpenAIEmbeddingClient, OpenAILLMClient
from raganite.graph import KnowledgeGraph

embeddings = OpenAIEmbeddingClient(model="text-embedding-3-small")
chat = OpenAILLMClient(model="gpt-4o", use_responses_api=False)

with KnowledgeGraph(embeddings=embeddings, llm=chat) as graph:
    passages = graph.retrieve("What is the capital of France?", top_k=5)
    answered = graph.query("What is the capital of France?", top_k=5)

print(passages[0].title)
print(passages[0].text)
print(passages[0].evidence)
print(answered.answer)
```

`KnowledgeGraph` opens the same default SQLite namespace when you run it from the repository root. `passages.usage` and `answered.usage` record token and timing totals. Classes are in the [Python API](docs/reference/python-api.md).

## Stores

| | SQLite | Neo4j |
| --- | --- | --- |
| Best for | A graph on your machine | A graph you already run in Neo4j |
| Setup | Nothing extra after `uv sync` | A Bolt URI, user, and password |
| Default | Yes. File `.raganite/graph.sqlite3` | No. Opt in with config |

Start with SQLite. SQLite scores every stored vector in the process, so it fits a small corpus. Point the same `retrieve` and `query` calls at Neo4j when that scan is the wrong cost. There is no hosted Raganite service. Steps are in [Neo4j](docs/guides/neo4j.md).

## Model providers

OpenAI, Azure OpenAI, and Hugging Face can each supply the chat model and the embedding model. Ollama can supply the chat model. Install the optional extras with `uv sync --extra huggingface` or `uv sync --extra ollama`. Indexing accepts LangChain `Document` objects. [Model providers](docs/guides/providers.md) shows the clients.

## Documentation

| If you want to… | Read |
| --- | --- |
| Install | [Installation](docs/getting-started/installation.md) |
| Index, then ask | [Your first index](docs/getting-started/quickstart.md) |
| Index a folder from Python | [Indexing](docs/guides/indexing.md) |
| Retrieve passages or write an answer | [Search and retrieval](docs/guides/retrieval.md) |
| Change settings | [Configuration](docs/guides/configuration.md) |
| Choose a model provider | [Model providers](docs/guides/providers.md) |
| Separate graphs | [Namespaces](docs/guides/namespaces.md) |
| Store the graph in Neo4j | [Neo4j](docs/guides/neo4j.md) |
| Look up a class or a command | [Python API](docs/reference/python-api.md), [Command line](docs/reference/cli.md) |
| Read the scores in full | [Benchmarks](docs/guides/benchmarking.md) |

## Benchmarks

These are the latest benchmark runs. Both used gpt-4o and `text-embedding-3-large`. The HotpotQA comparison on the benchmarks page uses other models, so those rows are not a same-stack result. The full tables are in [Benchmarks](docs/guides/benchmarking.md).

| Benchmark | Score | What it counts |
| --- | --- | --- |
| HotpotQA exact match | 60.9% | 1,000 questions, short answer matches the gold text |
| HotpotQA answer F1 | 76.2% | Same questions, token overlap with the gold answer |
| HotpotQA recall at 5 | 95.7% | A supporting passage is in the top 5 |
| LoCoMo judge | 66.6% | 1,540 questions, adversarial left out, one pass |
| LoCoMo token F1 | 54.3% | Same 1,540 questions |
| LoCoMo single-hop judge | 79.5% | 841 questions |

To run the same checks, follow [Run the benchmarks](docs/guides/run-benchmarks.md).

## License

Apache 2.0. See [LICENSE](LICENSE).
