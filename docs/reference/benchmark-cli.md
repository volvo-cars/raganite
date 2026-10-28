# Benchmark command

`python -m benchmark` and `raganite-bench` run the same Click group. Commands run from the repository root. The finished scores are on [Benchmarks](../guides/benchmarking.md). The steps are in [Run the benchmarks](../guides/run-benchmarks.md).

The group reads `benchmark/.env`, then the repository `.env`. A variable that is already set stays as it is. When the Neo4j password, URI, username, or database is still the default below, and `NEO4J_PASSWORD`, `NEO4J_URI`, `NEO4J_USERNAME`, or `NEO4J_DATABASE` is set, the environment value is used. Names are listed in `benchmark/.env.example`.

`-v` repeats. `-vv` logs each caught failure.

## `hotpotqa`

Download HotpotQA, index it, and answer the first questions. Repeating a command resumes questions missing from `--output`.

| Option | Default | Meaning |
| --- | --- | --- |
| `--questions` | `200` | First N questions, from 1 to 1,000. The full corpus is still indexed |
| `--concurrency` | CPU count | Questions evaluated at once |
| `--index-workers` | CPU count | Documents indexed at once |
| `--mode` | `full` | `full` ranks facts and passages. `direct` skips facts. `triple-unfiltered` skips the fact filter and the passage rerank |
| `--backend` | `sqlite` | `sqlite` or `neo4j` |
| `--skip-index` | off | Reuse the selected graph |
| `--data-dir` | `benchmark/data` | Where the downloaded corpus is stored |
| `--sqlite-path` | `benchmark/data/hotpotqa.sqlite3` | SQLite file |
| `--neo4j-uri` | `bolt://localhost:7687` | Bolt URI |
| `--neo4j-username` | `neo4j` | Database user |
| `--neo4j-password` | `raganite-local` | Database password |
| `--neo4j-database` | `neo4j` | Database name |
| `--neo4j-layout` | `owned` | `owned` writes a Raganite graph. `native` reads an existing database |
| `--dimensions` | model width | Embedding width |
| `--triple-top-k` | `30` | Facts retrieved before filtering |
| `--filtered-triple-k` | `8` | Facts kept after filtering |
| `--expansion-depth` | `2` | Graph expansion depth, from 0 to 5 |
| `--expansion-limit` | `5000` | Expansion cap |
| `--degree-penalty-weight` | `0.25` | Degree penalty in the walk |
| `--graph-weight` | `1.0` | Weight of the graph list in fusion |
| `--rrf-k` | `60` | Reciprocal-rank fusion constant |
| `--rerank-top-k` | `20` | Passages sent to the reranker |
| `--top-k` | `200` | Passages kept for recall |
| `--qa-top-k` | `5` | Passages sent to the answer model |
| `--output` | `benchmark/results/hotpotqa.jsonl` | JSONL path |
| `--retrieval-only` | off | Score passages and skip answer generation |
| `--compare-only` | off | Rebuild the saved read-only comparison. Does not index or call a model |

Native layout also reads `AVESTA_VECTOR_INDEX`, `AVESTA_FULLTEXT_INDEX`, and `AVESTA_FACT_VECTOR_INDEX`.

## `locomo run`

Download LoCoMo, index each needed conversation, and answer its questions. Each conversation is its own namespace.

| Option | Default | Meaning |
| --- | --- | --- |
| `--questions` | `100` | First questions in conversation order |
| `--concurrency` | CPU count | Questions evaluated at once |
| `--top-k` | `20` | Passages retrieved per question |
| `--backend` | `sqlite` | `sqlite` or `neo4j` |
| `--skip-index` | off | Reuse each conversation graph |
| `--data-dir` | `benchmark/data` | Where the downloaded release is stored |
| `--sqlite-path` | `benchmark/data/locomo.sqlite3` | SQLite file |
| `--neo4j-uri` | `bolt://localhost:7687` | Bolt URI |
| `--neo4j-username` | `neo4j` | Database user |
| `--neo4j-password` | `raganite-local` | Database password |
| `--neo4j-database` | `neo4j` | Database name |
| `--neo4j-layout` | `owned` | `owned` writes a Raganite graph. `native` reads an existing database |
| `--dimensions` | model width | Embedding width |
| `--schema-mode` | `free` | `fixed` uses the built-in schema. `free` lets the model name relations from the text. `extracted` builds a schema from the session |
| `--llm-judge` | off | Also score each answer with the Mem0 LoCoMo judge. Token F1 is always recorded |
| `--output` | `benchmark/results/locomo.jsonl` | JSONL path |

## `locomo score`

Print category scores for questions already in a results file. Reads the file only.

| Option | Default | Meaning |
| --- | --- | --- |
| `--data` | required | Official `locomo10.json` |
| `--results` | required | JSONL written by `locomo run` |
| `--questions` | `100` | First questions in conversation order |
