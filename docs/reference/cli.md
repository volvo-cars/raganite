# Command line

```sh
raganite [options] COMMAND [options]
```

After `uv sync`, run commands with `uv run raganite`. The entry point is `raganite.cli:main`.

## Shared options

| Option | Meaning |
| --- | --- |
| `--env-file PATH` | Dotenv file. Default is `.env` in the current directory, if present |
| `--no-env-file` | Do not load a dotenv file |
| `--config PATH` | JSON or TOML settings |
| `--backend sqlite\|neo4j` | Store kind. Default `sqlite` |
| `--sqlite-path PATH` | SQLite file. Default `.raganite/graph.sqlite3` |
| `--neo4j-uri URI` | Bolt URI. Default `bolt://localhost:7687` |
| `--namespace NAME` | Graph namespace. Default `default` |
| `--dimensions N` | Embedding length. Default `1536`. Must match the model and the stored vectors |
| `--workers N` | Indexing and retrieval workers. Default is the available CPU count |
| `--provider openai\|azure-openai` | Provider for the CLI chat and embedding clients. Default `openai` |
| `--chat-model NAME` | Chat model or Azure deployment. Default `gpt-4o` |
| `--embedding-model NAME` | Embedding model or Azure deployment. Default `text-embedding-3-small` |
| `--api-version VERSION` | Dated Azure API version. Omit it to use the Azure v1 clients |
| `--format text\|json` | `text` for `retrieve` by default; `json` for the other commands |

CLI flags override environment variables, which override `.env`, which overrides `--config`, which overrides built-in defaults.

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

`--source` defaults to the directory name, or `documents` for a file or stdin. `--schema-mode` defaults to `fixed`.

## `retrieve`

```sh
raganite retrieve QUESTION [--top-k N] [--format text|json]
```

Returns passages and fact evidence. It does not generate an answer. The store is opened read-only.

`--top-k` defaults to `5`. `--format text` is the human-readable default. `--format json` adds scores, evidence, and metadata.

## `clear`

```sh
raganite clear [--source NAME] --yes
```

Deletes documents in the active namespace. `--source` limits the delete to one source. `--yes` is required. This command does not call a model.
