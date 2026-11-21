# Configuration

Set where the graph lives, which models to call, and how indexing and retrieval behave.

## Pick one place to set values

Highest priority wins:

1. Values you pass to `RaganiteConfig` or to a CLI flag.
2. Process environment variables.
3. `.env` in the current directory, when using the CLI or `load_config()`.
4. A JSON or TOML settings file. Pass it to `RaganiteConfig.from_file` or `raganite --config`. If you omit that path, `load_config()` and the CLI use `raganite.json` in the current directory when that file exists, and otherwise `raganite.toml`.
5. Built-in defaults.

Library constructors do not read `.env` or those working-directory files by themselves. The CLI and `load_config()` load `.env` from the working directory if present, without searching parent directories. They apply the same rule to `raganite.json` and `raganite.toml`. Use `--env-file PATH` (or `load_config(env_file=Path(...))`) to select another existing dotenv file. An explicitly selected file that does not exist is an error. `RaganiteConfig.from_file` loads only the path you pass.

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
| `retrieval` | Search settings, passage limits, and optional passage reranking |

`raganite config` prints the resolved settings with secrets masked. It does not need a model key.

## Retrieval knobs you will actually change

| Field | Default | Effect |
| --- | --- | --- |
| `top_k` | `200` | How many passages `Retriever` keeps before a caller slices them |
| `use_triples` | `true` | Use the beam walk; set to `false` for direct chunk search |
| `use_fulltext` | `true` | Fuse full-text hits with vector hits during direct search |
| `fulltext_weight` | `1.0` | Weight of the full-text list in fusion |
| `rerank_passages` | `false` | Legacy setting that selects the LLM reranker when no other reranker is selected |
| `reranker` | `none` | `none` keeps the retrieval order. `llm` asks the chat model. `cross-encoder` uses LangChain's local cross-encoder. `compressor` uses the compressor you pass to `Retriever` |
| `reranker_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Model for `reranker=cross-encoder`. Install it with `uv sync --extra rerank` |
| `fact_temperature` / `fact_damping` | `0.1` / `0.85` | Outgoing fact softmax temperature and PageRank damping |
| `fact_seed_count` / `passage_seed_weight` | `5` / `0.05` | Number of restart facts and direct-passage weight |
| `beam_width` | `8` | Maximum number of first-hop states |
| `beam_start_candidates` / `beam_bridge_candidates` | `20` / `20` | Starting passages and maximum bridge additions |

The CLI `retrieve` command still defaults `--top-k` to `5`. That limit applies after ranking. It does not change `RetrievalConfig.top_k`.

The LLM reranker requires a chat model. The cross-encoder does not. To use a hosted reranking API, pass an object with `compress_documents` and set `reranker` to `compressor`.
