# Configuration

Set where the graph lives, which models to call, and how indexing and retrieval behave.

## Pick one place to set values

Highest priority wins:

1. Values you pass to `RaganiteConfig` or to a CLI flag.
2. Process environment variables.
3. `.env` in the current directory, for the CLI only.
4. A JSON or TOML file passed to `RaganiteConfig.from_file` or `raganite --config`.
5. Built-in defaults.

Library constructors do not read `.env` by themselves. The CLI loads `.env` from the working directory unless you pass `--no-env-file` or `--env-file`.

## Environment names

Prefix every variable with `RAGANITE_`. Nest fields with `__`.

```sh
RAGANITE_NAMESPACE=team-a
RAGANITE_EMBEDDING_DIMENSIONS=1536
RAGANITE_INDEXING__WORKERS=4
RAGANITE_BACKEND='{"kind":"sqlite","path":".raganite/graph.sqlite3"}'
```

See `.env.example` for the full list. Booleans are `true` or `false`. Leave a line commented to keep its default. Do not set a number to an empty string.

## From Python

```python
from raganite.config import RaganiteConfig, SQLiteConfig

config = RaganiteConfig(
    namespace="team-a",
    embedding_dimensions=1536,
    backend=SQLiteConfig(path=".raganite/team-a.sqlite3"),
)
config = RaganiteConfig.from_file("settings.toml")
```

`embedding_dimensions` must match the embedding model and the stored vectors. Changing it on an existing file raises `ValueError` when the store opens.

## The groups

| Group | What it controls |
| --- | --- |
| `backend` | SQLite path, or Neo4j connection and `layout` |
| `namespace` | Which graph you read and write. Default `default` |
| `embedding_dimensions` | Vector length. Default `1536` |
| `chat`, `embeddings` | Model name and provider used by the CLI |
| `indexing` | Chunk size, extraction, and whether facts are stored |
| `retrieval` | How many passages come back, and whether facts are filtered and reranked |

`raganite config` prints the resolved settings with secrets masked. It does not need a model key.

## Retrieval knobs you will actually change

| Field | Default | Effect |
| --- | --- | --- |
| `top_k` | `200` | How many passages `Retriever` keeps before a caller slices them |
| `use_triples` | `true` | Search facts and expand them, or use direct chunk search only |
| `filter_triples` | `true` | Ask the chat model to drop weak facts |
| `rerank_passages` | `true` | Ask the chat model to reorder the final passages |
| `expansion_depth` | `2` | How far fact expansion walks, from 0 to 5 |

The CLI `retrieve` command still defaults `--top-k` to `5`. That limit applies after ranking. It does not change `RetrievalConfig.top_k`.

Filtering or reranking requires a chat model. If either flag is on and no model was passed, constructing `Retriever` raises `ValueError`.
