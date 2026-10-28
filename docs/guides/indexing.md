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

`index_text` is the same path for one string. `index_repository` walks a directory using the include and exclude globs on `IndexingConfig`.

Accessing `.indexer` without a chat model raises `ValueError`.

## What gets stored

Each document becomes chunks. The chat model names entities and the relations between them. By default those relations are also stored as facts (`indexing.index_facts` is `true`). Shared entities stay in the namespace until the last document that mentioned them is removed.

Chunk size defaults to 1500 characters with 200 characters of overlap. Overlap must be smaller than the chunk size.

## Replace one source or the whole namespace

```python
graph.indexer.clear_source("notes")
graph.indexer.clear_graph()
```

`clear_graph` deletes documents in the active namespace only. The CLI equivalent is `raganite clear --source notes --yes`. `--yes` is required.
