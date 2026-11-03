# Indexing

Turn documents into chunks, entities, and facts in one namespace.

Indexing needs a chat model and an embedding model. Retrieval can run later without writing anything.

## Index a directory

```sh
uv run raganite index ./documents --source notes
```

The source name groups these files. A directory index also deletes stored files from that source when they are no longer selected. Indexing one file, or stdin, does not delete other files in the source.

Unchanged text is skipped. Pass `--force` to store it again.

```sh
printf 'Paris is the capital of France.\n' | uv run raganite index - --source notes --document-path paris.txt
```

The command prints a JSON report. `usage` has the run's token and timing totals. A failed file reports the first line of the error, with credentials removed. `-vv` includes the full error text. See the [command line reference](../reference/cli.md).

A directory index keeps a file when its name matches `IndexingConfig.include_globs`, it is no larger than `max_file_bytes` (default 1,000,000), and its path does not match `exclude_globs`. The default include patterns are `*.py`, `*.md`, `*.rst`, `*.txt`, `*.json`, `*.yaml`, `*.yml`, `*.toml`, `*.ini`, and `*.cfg`. PDF, HTML, and Office files are outside that set. The default exclude patterns are `*/.git/*`, `*/.venv/*`, `*/venv/*`, `*/node_modules/*`, `*/__pycache__/*`, `*/.mypy_cache/*`, `*/.pytest_cache/*`, `*/dist/*`, `*/build/*`, and `*.lock`. A single file or stdin is not filtered by these globs.

## Index from Python

```python
from langchain_core.documents import Document

from raganite.clients.llm import OpenAIEmbeddingClient, OpenAILLMClient
from raganite.graph import KnowledgeGraph

embeddings = OpenAIEmbeddingClient(model="text-embedding-3-small")
chat = OpenAILLMClient(model="gpt-4o", use_responses_api=False)

with KnowledgeGraph(embeddings=embeddings, llm=chat) as graph:
    report = graph.indexer.index_documents(
        [Document(page_content="Paris is the capital of France.", metadata={"path": "paris.txt"})],
        source="notes",
    )
    print(report.indexed_files, report.skipped_files, report.failed_files)
```

`index_text` is the same path for one string. `index_repository` walks a directory using `include_globs`, `exclude_globs`, and `max_file_bytes` on `IndexingConfig`.

Accessing `.indexer` without a chat model raises `ValueError`.

## What gets stored

Each document becomes chunks. The chat model names entities and the relations between them. That extraction is unconstrained (`schema_mode` is `free`), and free mode does not build the code schema. `fixed` limits extraction to `graph_schema` when you set one, and otherwise to the built-in code schema (`Module`, `Class`, `Function`, and the other labels in `default_schema`). `extracted` infers a schema from the document first. By default those extracted relations are also stored as facts (`indexing.index_facts` is `true`). Two entities become one when the label and the name are exactly the same. `paris` and `Paris` stay separate, and so does a name with a different space. Shared entities stay in the namespace until the last document that mentioned them is removed.

Chunk size defaults to 1500 characters with 200 characters of overlap. Overlap must be smaller than the chunk size.

## Ordered documents

Pass integer `order` and a non-empty string `time` on a LangChain document's metadata to link that document into a session sequence. Indexing adds a `Session` entity named `session {order}`. A `FROM_CHUNK` edge runs from the session to each of its chunks. An `IN_SESSION` edge runs from each other entity to the session. Sessions in the same source are linked in increasing `order` by `NEXT` edges. Each `NEXT` edge records the earlier time as `previous` and the later time as `next`. `IN_SESSION` and `NEXT` are entity-to-entity edges, so they are embedded as facts like any other such edge. A `NEXT` fact includes the `previous` and `next` times. A document that lacks either field is stored without a session.

The CLI path for a directory, a file, or stdin does not set `order` or `time`. Use `index_documents` when you have that metadata.

## Replace one source or the whole namespace

```python
graph.indexer.clear_source("notes")
graph.indexer.clear_graph()
```

`clear_graph` deletes documents in the active namespace only. The CLI equivalent is `raganite clear --source notes --yes`. `--yes` is required.
