# Raganite

## Benchmarks

These are the latest benchmark runs.  The full tables are in [Benchmarks](docs/guides/benchmarking.md).

| Benchmark | Score | What it counts |
| --- | --- | --- |
| HotpotQA exact match | 60.9% | 1,000 questions, short answer matches the gold text |
| HotpotQA answer F1 | 76.2% | Same questions, token overlap with the gold answer |
| HotpotQA recall at 5 | 95.7% | A supporting passage is in the top 5 |
| LoCoMo judge | 66.6% | 1,540 questions, adversarial left out, one pass |
| LoCoMo token F1 | 54.3% | Same 1,540 questions |
| LoCoMo single-hop judge | 79.5% | 841 questions |

To run the same checks, follow [Run the benchmarks](docs/guides/run-benchmarks.md).



## Research Highlights

- **76.2% answer F1** on 1,000 HotpotQA questions. Exact match on that run is 60.9%.
- **95.7% recall at 5.** Recall at 1 is 47.3% because a question usually has two supporting passages.
- **79.5% single-hop judge** on LoCoMo. The overall judge, with adversarial questions left out, is 66.6%.

Read the comparisons on the [Benchmarks](docs/guides/benchmarking.md) page.

## Introduction

Raganite indexes documents into a knowledge graph, then returns the passages that help answer a question.

A passage is a piece of an indexed document, ranked for that question. Use the passages with your own model when you need a written answer.

## Key Features & Use Cases

**You can:**

- Index a folder, a file, or text from stdin.
- Search facts and chunks, then get a ranked list of passages.
- Separate graphs with namespaces.
- Switch the store from SQLite to Neo4j without changing the retrieve call.

**Use it when you:**

- Want passages from notes or a document set, not a chat answer from the library itself.
- Need the facts behind a passage, so you can show why it was returned.
- Want a local graph first, and Neo4j only if you outgrow the SQLite file.

## Quickstart Guide

You need Python 3.11, 3.12, or 3.13, and [uv](https://docs.astral.sh/uv/).

```sh
git clone git@github.com:volvo-cars/raganite.git
cd raganite
uv sync
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. Then index a folder and ask a question:

```sh
mkdir -p documents
printf 'Paris is the capital of France.\n' > documents/paris.txt
uv run raganite index documents --source notes
uv run raganite retrieve "What is the capital of France?" --top-k 5 --format text
```

You should see the passage text and the facts that supported it. The longer walkthrough is [Your first index](docs/getting-started/quickstart.md).

| | SQLite | Neo4j |
| --- | --- | --- |
| Best for | Trying the library on your machine | A graph you already run in Neo4j |
| Setup | Nothing extra after `uv sync` | A Bolt URI, user, and password |
| Default | Yes. File `.raganite/graph.sqlite3` | No. Opt in with config |

Start with SQLite. Move to Neo4j when you want that store. There is no hosted Raganite service.

## Library

```sh
uv sync
uv run raganite --help
```

You should see `init`, `config`, `index`, `retrieve`, and `clear`. Install details are in [Installation](docs/getting-started/installation.md).

## Neo4j

Run Neo4j yourself and point Raganite at it. The retrieve call stays the same. Steps are in [Neo4j](docs/guides/neo4j.md).

## CLI

```sh
uv run raganite index documents --source notes
uv run raganite retrieve "What is the capital of France?" --top-k 5 --format text
```

Flags are in the [command line reference](docs/reference/cli.md).


## Basic Usage

```python
from raganite.clients.llm import OpenAIEmbeddingClient, OpenAILLMClient
from raganite.graph import KnowledgeGraph

embeddings = OpenAIEmbeddingClient(model="text-embedding-3-small")
chat = OpenAILLMClient(model="gpt-4o", use_responses_api=False)

with KnowledgeGraph(embeddings=embeddings, llm=chat) as graph:
    passages = graph.retrieve("What is the capital of France?", top_k=5)

print(passages[0].title)
print(passages[0].text)
```

`passages[0].evidence` lists the facts used in the ranking. Classes are in the [Python API](docs/reference/python-api.md).

## Model providers

OpenAI and Azure OpenAI are the model providers. Indexing accepts LangChain `Document` objects. [Model providers](docs/guides/providers.md) shows both clients.

## Documentation & Support

| If you want to… | Read |
| --- | --- |
| Install | [Installation](docs/getting-started/installation.md) |
| Index, then ask | [Your first index](docs/getting-started/quickstart.md) |
| Change settings | [Configuration](docs/guides/configuration.md) |
| Choose OpenAI or Azure | [Model providers](docs/guides/providers.md) |
| Separate graphs | [Namespaces](docs/guides/namespaces.md) |
| Read the scores in full | [Benchmarks](docs/guides/benchmarking.md) |

## License

Apache 2.0. See [LICENSE](LICENSE).
