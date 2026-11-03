# Command line

```sh
raganite [options] COMMAND [options]
```

After `uv sync`, run commands with `uv run raganite`. The entry point is `raganite.cli:main`.

## Shared options

| Option | Meaning |
| --- | --- |
| `--env-file PATH` | Dotenv file. Default is `.env` in the current directory, if present |
| `--config PATH` | JSON or TOML settings |
| `--backend sqlite\|neo4j` | Store kind. Default `sqlite` |
| `--sqlite-path PATH` | SQLite file. Default `.raganite/graph.sqlite3` |
| `--neo4j-uri URI` | Bolt URI. Default `bolt://localhost:7687` |
| `--namespace NAME` | Graph namespace. Default `default` |
| `--dimensions N` | Embedding length. Default `1536`. Must match the model and the stored vectors |
| `--workers N` | Indexing and retrieval workers. Default is the available CPU count |
| `--provider openai\|azure-openai\|huggingface\|huggingface-endpoint` | Provider for both the chat and embedding clients. Default `openai`. `huggingface` is local. `huggingface-endpoint` is hosted. Set `RAGANITE_CHAT__PROVIDER` and `RAGANITE_EMBEDDINGS__PROVIDER` separately when the two roles differ |
| `--chat-model NAME` | Chat model, Azure deployment, Hugging Face model id, or Ollama tag. Default `gpt-4o`. Ollama chat is `RAGANITE_CHAT__PROVIDER=ollama`, because `--provider` also sets embeddings and Ollama has no embedding client |
| `--embedding-model NAME` | Embedding model, Azure deployment, or Hugging Face model id. Default `text-embedding-3-small` |
| `--api-version VERSION` | Dated Azure API version. Omit it to use the Azure v1 clients |
| `--format text\|json` | `text` for `retrieve` and `query` by default; `json` for the other commands |
| `-v`, `--verbose` | Repeat for more detail. One `-v` logs info. `-vv` logs debug output, including each caught failure. For `index`, normal output is the first line of the error with credentials removed. `-vv` puts the full error text on each failed file. |

CLI flags override environment variables, which override `.env`, which overrides the settings file, which overrides built-in defaults. If you omit `--config`, the command uses `raganite.json` in the current directory when that file exists, and otherwise `raganite.toml`. See [Configuration](../guides/configuration.md).

## `init`

Create or validate storage and indexes. Does not call a model.

```sh
raganite init
```

## `config`

Print resolved settings. Secrets are masked. Does not call a model.

```sh
raganite config
```

## `index`

```sh
raganite index [PATH] [--source NAME] [--document-path PATH] [--force] [--schema FILE] [--schema-mode fixed|free|extracted]
```

`PATH` defaults to the current directory. `-` reads stdin.

| Case | Behavior |
| --- | --- |
| Directory | Indexes selected files and deletes stored files from that source that were not selected |
| One file, or stdin | Does not delete other files in the source |
| Unchanged content | Skipped, unless `--force` |

`--source` defaults to the directory name, or `documents` for a file or stdin. `--schema-mode` defaults to `free`: the model names entities and relations from the text, and the code schema is not built. `fixed` uses `--schema` or the built-in code schema. `extracted` infers a schema from the input. Passing `--schema` selects `fixed` unless `--schema-mode` is set.

The command prints a JSON report, including `usage` for tokens and elapsed time. Each failed file is an object with `path` and `error`. `error` is the first line of the failure, with credentials removed. `-vv` puts the full error text there instead. A directory index keeps the extensions in [Indexing](../guides/indexing.md) and skips files larger than 1,000,000 bytes.

## `retrieve`

```sh
raganite retrieve QUESTION [--top-k N] [--format text|json]
```

Returns passages. Fact evidence is in the JSON output. It does not generate an answer. The store is opened read-only.

`--top-k` defaults to `5`. `--format text` prints rank, title, source, score, and text for each passage. When a fact filter or a passage rerank ran, that status is printed before the passages. `--format json` prints those passages as objects and adds `evidence` and metadata. The JSON list does not include the retrieval usage report. Read `RetrievalResult.usage` in Python for token and timing totals.

## `query`

```sh
raganite query QUESTION [--top-k N] [--format text|json]
```

Retrieves passages, then answers from those passages. The store is opened read-only. A chat model is required.

`--top-k` defaults to `5` and limits the passages given to the answer. `--format text` prints only the answer. `--format json` includes the passages, their `evidence`, and token and timing usage.

## `clear`

```sh
raganite clear [--source NAME] --yes
```

Deletes documents in the active namespace. `--source` limits the delete to one source. `--yes` is required. This command does not call a model.
