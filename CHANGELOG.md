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
- Fact-based retrieval combining fact-vector search, LLM triple filtering,
  degree-penalized graph expansion, passage-level reciprocal rank fusion, and
  LLM passage reranking.
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

### Changed

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


