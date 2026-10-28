# Python API

Public types for indexing and retrieval. Import them from the packages below.

## `raganite.graph`

### `KnowledgeGraph`

```python
KnowledgeGraph(
    *,
    embeddings,
    llm=None,
    config=None,
    backend=None,
    text_splitter=None,
)
```

| Argument | Required | Meaning |
| --- | --- | --- |
| `embeddings` | yes | `EmbeddingClient` or a LangChain `Embeddings` |
| `llm` | for indexing, and for default ranking | `LLMClient` or a LangChain chat model |
| `config` | no | `RaganiteConfig`. Default SQLite, namespace `default`, 1536 dimensions |
| `backend` | no | An open `GraphBackend`. Must match `config.namespace` and `config.embedding_dimensions` |
| `text_splitter` | no | LangChain splitter. Overrides the chunking strategy |

| Member | Returns | Errors |
| --- | --- | --- |
| `retrieve(question, *, top_k=None)` | `list[Passage]` | `ValueError` on a blank question or `top_k < 1` |
| `indexer` | `KnowledgeGraphIndexer` | `ValueError` if `llm` was omitted |
| `retriever` | `Retriever` | Built on first use. Default ranking also requires `llm` |
| `close()` | `None` | Closes retrieval workers and a backend this object created |

Use it as a context manager. An injected backend is left open.

## `raganite.retrieval`

### `Passage`

Frozen. Fields: `source`, `title`, `text`, `score`, `evidence`, `metadata`.

### `FactEvidence`

Frozen. Fields: `fact`, `seed_entity`, `related_entity`, `path`, `score`, `relationship_id`, `path_degrees`.

### `Retriever`

```python
Retriever(
    backend,
    embeddings,
    llm=None,
    config=None,
    *,
    title_by_source=None,
    triple_filter_prompt=FILTER_PROMPT,
    passage_rerank_prompt=RERANK_PROMPT,
)
```

| Method | Returns |
| --- | --- |
| `retrieve(query, *, top_k=None)` | `list[Passage]` |
| `retrieve_many(queries, *, top_k=None)` | `list[list[Passage]]` |
| `aretrieve(query, *, top_k=None)` | awaitable `list[Passage]` |
| `close()` | Stops the worker pool |

If `use_triples` is true, `Retriever` calls `backend.prepare()` at construction. If `filter_triples` or `rerank_passages` is true, `llm` is required.

A malformed ranking response does not raise. Retrieval keeps the unfiltered or unrranked order and records the fallback on passage metadata.

## `raganite.indexing`

### `KnowledgeGraphIndexer`

| Method | Meaning |
| --- | --- |
| `index_documents(documents, *, source="documents", force=False)` | Index LangChain `Document` values. Paths in one batch must be unique |
| `index_text(text, *, source, path="text", force=False)` | Index one string |
| `index_repository(repo_root, *, source_name=None, force=False)` | Index a directory. Removes files from that source that were not selected |
| `clear_source(source_name)` | Delete one source. Returns the number of documents removed |
| `clear_graph()` | Delete the active namespace |

All of them return `IndexingReport` except the clear methods, which return `int`.

### `IndexingReport`

`indexed_files`, `skipped_files`, `failed_files`, `deleted_files`, and `total`. Skipped and failed entries are `(path, reason)` pairs. Reasons include `empty` and `unchanged`.

## `raganite.config`

### `RaganiteConfig`

Fields: `backend`, `namespace`, `embedding_dimensions`, `chat`, `embeddings`, `indexing`, `retrieval`.

`from_file(path)` loads JSON or TOML. Environment variables use the prefix `RAGANITE_` and `__` for nesting. Extra fields are rejected.

`RetrievalConfig` defaults: `top_k=200`, `use_triples=True`, `filter_triples=True`, `rerank_passages=True`, `expansion_depth=2`. `filtered_triple_k` cannot exceed `triple_top_k`.

`IndexingConfig.index_facts` defaults to `True`. `ChunkingConfig` defaults to 1500 characters and 200 overlap.

`Neo4jConfig.layout` is `owned` or `native`. `SQLiteConfig.path` defaults to `.raganite/graph.sqlite3`.

## `raganite.backends`

### `GraphBackend`

The store contract. Callers normally use `KnowledgeGraph` and do not call these methods.

Writes: `ensure_indexes`, `replace_document`, `document_hash`, `delete_documents`.

Reads used by retrieval: `prepare`, `direct_passages`, `search_facts`, `expand_facts`, `search_chunks`, `search_fulltext`.

`close()` releases connections the backend owns.

### `SQLiteBackend` and `Neo4jBackend`

`create_backend(config)` builds one of them and calls `ensure_indexes`. `Neo4jBackend(..., layout="native")` is read-only.

`SQLiteBackend` is the in-process store used by tests and by the default config.

## `raganite.schema`

`GraphSchema` and `default_schema` describe what extraction may emit. `GraphTopology` names the structural edge types. The owned default is `FROM_CHUNK`, `FROM_DOCUMENT`, `HAS_DOCUMENT`, `BELONGS_TO`, `FACT_SUBJECT`, and `FACT_OBJECT`.

## `raganite.clients.llm`

`LLMClient`, `EmbeddingClient`, and `CompletionClient` each expose a LangChain model on `.model`.

| OpenAI | Azure |
| --- | --- |
| `OpenAILLMClient` | `AzureOpenAILLMClient` |
| `OpenAIEmbeddingClient` | `AzureOpenAIEmbeddingClient` |
| `OpenAICompletionClient` | `AzureOpenAICompletionClient` |

Constructor keyword arguments are LangChain's. See [Model providers](../guides/providers.md).
