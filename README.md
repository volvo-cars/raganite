# Raganite

**Find the passages that answer a question, even when they’re in different files.**

Raganite is an open-source Python library and command-line tool for searching your documents. It indexes the original text alongside a graph of facts and relationships. A repair note might name a pump model, while a separate manual lists its replacement parts. The graph helps Raganite find both passages when you ask which part to order.

Use **`retrieve`** to get the original text and see where it came from. Use **`query`** to get a written answer based on those passages. By default, search uses an embedding model but does not call a chat model or run a separate reranker.

[Documentation](docs/index.md) · [Benchmark results](docs/explanation/benchmark-results.md) · [Compare with other systems](docs/explanation/benchmark-results.md#comparison-with-other-systems)

## Quickstart

You’ll need Python 3.11, 3.12, or 3.13, [uv](https://docs.astral.sh/uv/), and an OpenAI API key for this example. SQLite is the default, so you don’t need to start a database server.

```sh
git clone https://github.com/volvo-cars/raganite.git
cd raganite
uv sync
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env` and keep the file out of Git. Then add a text file, index it, and ask a question:

```sh
mkdir -p documents
printf 'The PX-200 pump uses replacement seal S-42.\n' > documents/manual.txt
uv run raganite index documents --source notes
uv run raganite retrieve "Which seal fits the PX-200 pump?" --top-k 5 --format text
uv run raganite query "Which seal fits the PX-200 pump?" --top-k 5 --format text
```

`retrieve` shows the passage and its source. `query` writes the answer; add `--format json` to include the retrieved passages and usage details. See [Your first index](docs/getting-started/quickstart.md) for the full walkthrough.

## Use it from Python

After indexing the example, run this from the repository root to open the same SQLite index. Export `OPENAI_API_KEY` in your shell first; these Python clients read it from the environment.

```python
from raganite.clients.llm import OpenAIEmbeddingClient, OpenAILLMClient
from raganite.graph import KnowledgeGraph

embeddings = OpenAIEmbeddingClient(model="text-embedding-3-small")
chat = OpenAILLMClient(model="gpt-4o", use_responses_api=False)
question = "Which seal fits the PX-200 pump?"

with KnowledgeGraph(embeddings=embeddings, llm=chat) as graph:
    for passage in graph.retrieve(question, top_k=5):
        print(passage.source, passage.text)
    print(graph.query(question, top_k=5).answer)
```

## Explore Raganite

| I want to... | Read |
| --- | --- |
| Index my own files | [Indexing](docs/guides/indexing.md) |
| Change the database or model provider | [Neo4j setup](docs/guides/neo4j.md), [Model providers](docs/guides/providers.md) |
| Configure search and answer generation | [Search and retrieval](docs/guides/retrieval.md), [Configuration](docs/guides/configuration.md) |
| See the results or run an evaluation | [Benchmark results](docs/explanation/benchmark-results.md), [Run the benchmarks](docs/explanation/benchmark-results.md#run-the-benchmarks-yourself) |
| Look up a class or command | [Python API](docs/reference/python-api.md), [CLI reference](docs/reference/cli.md) |

## License

[Apache 2.0](LICENSE)
