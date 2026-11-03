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
| `retrieve(question, *, top_k=None)` | `RetrievalResult` | `ValueError` on a blank question or `top_k < 1` |
| `query(question, *, top_k=None)` | `Answer` | `ValueError` on a blank question, `top_k < 1`, or a missing LLM |
| `indexer` | `KnowledgeGraphIndexer` | `ValueError` if `llm` was omitted |
| `retriever` | `Retriever` | Built on first use. Default ranking also requires `llm` |
| `close()` | `None` | Closes retrieval workers and a backend this object created |

Use it as a context manager. An injected backend is left open.

## `raganite.retrieval`

### `Passage`

Frozen. Fields: `source`, `title`, `text`, `score`, `evidence`, `metadata`.

### `FactEvidence`

Frozen. Fields: `fact`, `seed_entity`, `related_entity`, `path`, `score`, `relationship_id`, `path_degrees`.

### `RetrievalResult`

A list of `Passage` values from one `retrieve` call. `usage` is a `UsageReport` for that run: `elapsed_seconds`, `input_tokens`, `output_tokens`, `total_tokens`, and `stages`. Ranking stages include `retrieval.filter` and `retrieval.rerank`. Each stage has `elapsed_seconds`, `model_seconds`, `calls`, and the same token totals.

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
| `retrieve(query, *, top_k=None)` | `RetrievalResult` |
| `retrieve_many(queries, *, top_k=None)` | `list[RetrievalResult]` |
| `aretrieve(query, *, top_k=None)` | awaitable `RetrievalResult` |
| `query(question, *, top_k=None)` | `Answer` |
| `aquery(question, *, top_k=None)` | awaitable `Answer` |
| `close()` | Stops the worker pool |

### `Answer`

Frozen. Fields: `question`, `answer`, `passages`, `usage`. `passages` are the retrieval result. `usage` includes the retrieval stages and `query.answer`.

The answer call uses `RetrievalConfig.llm_options` without `response_format`, so ranking can stay JSON while the answer stays plain text.

If `use_triples` is true, `Retriever` calls `backend.prepare()` at construction. If `filter_triples` or `rerank_passages` is true, `llm` is required.

A malformed ranking response does not raise. Retrieval keeps the unfiltered or unranked order and records the fallback on passage metadata.

## `raganite.indexing`

### `KnowledgeGraphIndexer`

| Method | Meaning |
| --- | --- |
| `index_documents(documents, *, source="documents", force=False)` | Index LangChain `Document` values. Paths in one batch must be unique |
| `index_documents_async(documents, *, source="documents", force=False)` | Awaitable form of `index_documents` |
| `index_text(text, *, source, path="text", force=False)` | Index one string |
| `index_text_async(text, *, source, path="text", force=False)` | Awaitable form of `index_text` |
| `index_repository(repo_root, *, source_name=None, force=False)` | Index a directory. Removes files from that source that were not selected |
| `index_repository_async(repo_root, *, source_name=None, force=False)` | Awaitable form of `index_repository` |
| `clear_source(source_name)` | Delete one source. Returns the number of documents removed |
| `clear_graph()` | Delete the active namespace |

The index methods return `IndexingReport`. The clear methods return `int`. The async methods are the same work with `await`.

### `IndexingReport`

`indexed_files`, `skipped_files`, `failed_files`, `deleted_files`, `usage`, and `total`. Skipped and failed entries are `(path, reason)` pairs. Reasons include `empty` and `unchanged`. `deleted_files` is a count of removed documents. `usage` is a `UsageReport` for the indexing run. The extraction stage is `indexing.extract`. `total` is the count of indexed, skipped, and failed files.

## `raganite.config`

### `RaganiteConfig`

Fields: `backend`, `namespace`, `embedding_dimensions`, `chat`, `embeddings`, `indexing`, `retrieval`.

`from_file(path)` loads JSON or TOML. Environment variables use the prefix `RAGANITE_` and `__` for nesting. Extra fields are rejected.

`RetrievalConfig` defaults: `top_k=200`, `use_triples=True`, `use_fulltext=True`, `fulltext_weight=1.0`, `filter_triples=False`, `rerank_passages=False`, `expansion_depth=2`. `filtered_triple_k` cannot exceed `triple_top_k`. Full-text hits are fused with chunk vectors and fact expansion, and a passage keeps the facts that led to it. Set the two chat flags true to filter facts and rerank passages. The command line names that step when it runs.

`IndexingConfig.index_facts` defaults to `True`. `IndexingConfig.schema_mode` defaults to `free`, and `graph_schema` defaults to `None`, so free mode does not build the code schema. `fixed` uses `graph_schema` when one is set, and otherwise `default_schema()`. Entity resolution merges an exact label and name. A different character stays a separate entity. `ChunkingConfig` defaults to 1500 characters and 200 overlap. `include_globs`, `exclude_globs`, and `max_file_bytes` (default 1,000,000) select files for `index_repository`. The patterns are listed in [Indexing](../guides/indexing.md).

`Neo4jConfig.layout` is `owned` or `native`. `SQLiteConfig.path` defaults to `.raganite/graph.sqlite3`.

## `raganite.backends`

### `GraphBackend`

The store contract. Callers normally use `KnowledgeGraph` and do not call these methods.

| Method | Meaning |
| --- | --- |
| `ensure_indexes()` | Create storage and indexes. Safe to call again |
| `replace_document(graph)` | Replace one document, its hash, and its graph memberships |
| `store_shared_edges(edges)` | Write edges owned by every listed document. Each owner is a source and a path |
| `delete_documents(source=None, *, keep_paths=())` | Delete documents in this namespace, except `keep_paths`. `source=None` covers every source |
| `document_hash(source, path)` | Hash of the last committed document, or `None` |
| `nodes(*, ids=None, term=None, kind=None, limit=None)` | Find nodes by ID or by a case-insensitive property substring |
| `edges(*, node_ids=None)` | Edges incident on the requested nodes, or every edge in the namespace |
| `vector_search(embedding, *, top_k=5, source=None, kind="chunk")` | Rank chunk or fact embeddings by cosine similarity |
| `fulltext_search(query, *, top_k=5, source=None)` | Rank chunks with the full-text index |
| `search_chunks(embedding, *, top_k=5, source=None)` | Chunk embedding search used by retrieval |
| `search_fulltext(query, *, top_k=5, source=None)` | Full-text search used by retrieval |
| `prepare()` | Check fact storage and cache traversal topology. Does not write |
| `direct_passages(embedding, *, top_k)` | Ranked chunk passages for an embedding |
| `lexical_passages(query, *, top_k)` | The same passage records, ranked by full text |
| `search_facts(embedding, *, top_k=30)` | Ranked facts with both endpoints and semantic degrees |
| `expand_facts(facts, *, depth=2, limit=5000, degree_penalty_weight=0.25)` | Rank entity paths from those facts back to chunks |
| `close()` | Release connections the backend owns |

### `SQLiteBackend` and `Neo4jBackend`

`create_backend(config)` builds one of them and calls `ensure_indexes`. `Neo4jBackend(..., layout="native")` is read-only.

`SQLiteBackend` is the in-process store used by tests and by the default config.

## `raganite.schema`

`GraphSchema` and `default_schema` describe what extraction may emit. `GraphTopology` names the structural edge types. The owned default is `FROM_CHUNK`, `FROM_DOCUMENT`, `HAS_DOCUMENT`, `BELONGS_TO`, `FACT_SUBJECT`, `FACT_OBJECT`, `IN_SESSION`, and `NEXT`. `IN_SESSION` and `NEXT` are written for documents that carry integer `order` and string `time` metadata, and those edges are embedded as facts. See [Indexing](../guides/indexing.md).

## `raganite.clients.llm`

`LLMClient`, `EmbeddingClient`, and `CompletionClient` each expose a LangChain model on `.model`.

| OpenAI | Azure | Hugging Face |
| --- | --- | --- |
| `OpenAILLMClient` | `AzureOpenAILLMClient` | `HuggingFaceLLMClient` |
| `OpenAIEmbeddingClient` | `AzureOpenAIEmbeddingClient` | `HuggingFaceEmbeddingClient`, `HuggingFaceEndpointEmbeddingClient` |
| `OpenAICompletionClient` | `AzureOpenAICompletionClient` | — |

The Hugging Face clients need `uv sync --extra huggingface`. `HuggingFaceLLMClient` wraps `ChatHuggingFace`. `backend="pipeline"` runs the chat model locally, and `backend="endpoint"` calls a hosted Inference Provider. `HuggingFaceEmbeddingClient` runs a Sentence Transformers model locally. `HuggingFaceEndpointEmbeddingClient` calls a hosted embedding endpoint.

`OllamaLLMClient` wraps `ChatOllama` and needs `uv sync --extra ollama`. It is chat only.

Constructor keyword arguments are LangChain's. See [Model providers](../guides/providers.md).
