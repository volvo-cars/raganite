# Changelog

Notable changes to Raganite are recorded here. Entries are grouped by change
type; the release date is added when a version is published.

## [Unreleased]

Prepared for the initial **0.1.0** release.

### Added

- Abstract chat, embedding, and legacy text-completion clients with OpenAI and
  Azure OpenAI implementations backed by LangChain. Native model functionality
  and provider options remain available through the underlying clients.
- Support for Azure OpenAI v1 and traditional, explicitly versioned endpoints.
- A shared abstract `GraphBackend` contract with fact-capable `SQLiteBackend`
  and `Neo4jBackend` implementations.
- Namespace-scoped document replacement, content-hash-based incremental indexing,
  and source-level document deletion.
- Neo4j support for Raganite-managed graphs and read-only access to existing
  native graphs, including schema-adaptive traversal discovery.
- Document, text, and repository indexing through `KnowledgeGraphIndexer`, with
  LangChain `Document` inputs and configurable fixed, recursive, or character
  chunking.
- Configurable extraction schemas: a default code-oriented schema, a supplied
  schema, unconstrained extraction, or a schema inferred from input.
- Fact-based retrieval combining fact-vector search, degree-penalized graph
  expansion, full-text search, and passage-level reciprocal rank fusion.
  Optional chat-model steps filter weak facts and rerank passages.
- Typed `Passage` and `FactEvidence` results containing source information,
  graph evidence, scores, and ranking metadata. Answer generation remains
  separate from retrieval.
- Synchronous, asynchronous, and batch retrieval entry points, with configurable
  worker counts and retrieval limits.
- A `KnowledgeGraph` facade for indexing and retrieval, with on-demand retriever
  initialization so a new graph can be indexed before facts are queried.
- Validated application, backend, indexing, retrieval, and model-client settings,
  configurable through JSON/TOML files and nested `RAGANITE_` environment keys.
- A Click CLI with `init`, `index`, `retrieve`, `config`, and confirmation-required
  `clear` commands. Indexing accepts a directory, a file, or standard input;
  retrieval opens the backend read-only.
- CLI dotenv loading, redacted configuration output, and explicit precedence:
  command options, process environment, dotenv, config file, then defaults.
- A comprehensive `.env.example` with defaults, allowed values, OpenAI and Azure
  credentials placeholders, backend settings, and separate HotpotQA harness
  settings.
- A HotpotQA evaluation harness for an existing knowledge base, with resumable
  successful results, recorded failures, retrieval and answer metrics, and
  matched-question baseline comparisons.
- Grounded queries through `KnowledgeGraph.query`, `Retriever.query`,
  `Retriever.aquery`, and the `raganite query` command. The answer is generated
  only from the retrieved passages and is returned with those passages.
- Token and timing usage on indexing reports, retrieval results, and query
  answers, including stage totals. `raganite query --format json` includes that
  usage, and HotpotQA and LoCoMo runs record it on each question.
- Optional Hugging Face chat and embedding clients, including hosted endpoints,
  and an Ollama chat client. Install them with `uv sync --extra huggingface` or
  `uv sync --extra ollama`.
- A choice of passage reranker. `none` keeps the fused order and `llm` asks the
  chat model. `cross-encoder` and a hosted rerank API both go through LangChain's
  `compress_documents` compressor. Install the local model with `uv sync --extra rerank`.
  `rerank_passages` still selects the chat model.
- Searchable facts for every entity-to-entity edge, including the `IN_SESSION`
  and `NEXT` edges written for documents that carry integer `order` and string
  `time` metadata. A `NEXT` fact includes the previous and next times, so
  session membership and sequence are available to fact search.

### Changed

- `raganite query --format text` prints only the answer. The passages remain in
  `--format json`.
- Fact search starts a walk from 10 facts instead of 30. Chunk, full-text, and
  fact search run at the same time. The HotpotQA benchmark still requests 30.
- Retrieval defaults to one embedding and no chat-model calls. Full-text hits
  are fused with the chunk-vector and fact lists, and a passage keeps the facts
  that led to it. Set `filter_triples` and `rerank_passages` to ask the chat
  model to drop weak facts and reorder passages.
- HotpotQA and LoCoMo take their chat and embedding models from
  `RAGANITE_CHAT__*` and `RAGANITE_EMBEDDINGS__*` when `AZURE_OPENAI_API_KEY`
  is unset.
- Indexing defaults to free extraction, so the model names entities and relations
  from the text. Pass `--schema`, or set `schema_mode` to `fixed`, to keep the
  built-in code schema. The previous default was `fixed`.
- `load_config()` and the CLI load `.env` from the working directory when that
  file is present, and they use `raganite.json` or `raganite.toml` there when no
  config path is given. An explicit dotenv path must exist. `--no-env-file` has
  been removed.
- Azure OpenAI v1 clients accept `AZURE_OPENAI_ENDPOINT` and
  `AZURE_OPENAI_API_KEY`, and also `OPENAI_API_KEY`, `OPENAI_API_BASE`, and
  `OPENAI_BASE_URL`, so those values do not have to be copied between names.

### Fixed

- SQLite searches that run together read through one connection when the file
  is not in WAL mode, so a parallel retrieval does not wait on its own write lock.
- Neo4j traversal discovery treats labels that only repeat the same endpoints
  as one connector. Owned entities carry `RaganiteNode` and `__Entity__`
  together, and owned documents carry `RaganiteNode` and `Document`. Two
  connector types, or two domain labels with the same count, still raise
  `ValueError`.


