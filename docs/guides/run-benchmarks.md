# Run the benchmarks

This guide shows how to repeat the HotpotQA or LoCoMo runs from the repository root. Each run writes a JSONL file of answers. The scores from the finished runs are on [Benchmarks](benchmarking.md).

You need a working install and model credentials. Put them in `benchmark/.env`. The command reads that file, then the repository `.env`. A variable that is already set stays as it is. See `benchmark/.env.example`.

When `AZURE_OPENAI_API_KEY` is set, chat and embeddings use the Azure deployments in that file. The published runs used gpt-4o and `text-embedding-3-large`. Without that key, the chat model is `OPENAI_CHAT_MODEL` (default `gpt-4o`) and the embedding model is `OPENAI_EMBEDDING_MODEL` (default `text-embedding-3-small`).

`python -m benchmark` and `raganite-bench` are the same command. Flags are listed in the [benchmark command reference](../reference/benchmark-cli.md).

## Run HotpotQA

The command downloads the 1,000 questions and 9,811 passages on the first run. Indexing covers the full corpus. `--questions` chooses how many of those questions to answer.

SQLite needs no extra service. This answers the full set:

```sh
python -m benchmark hotpotqa --questions 1000
```

The file `benchmark/results/hotpotqa.jsonl` gains one row per answered question. The same folder receives `hotpotqa.summary.json` and `hotpotqa.manifest.json`. Running the command again skips questions already in that file. A different mode, backend, or question count needs a new `--output` path.

### Use Neo4j

Start the database from the Compose file, then index into it:

```sh
docker compose -f benchmark/neo4j/docker-compose.yml up -d
python -m benchmark hotpotqa --backend neo4j --questions 1000
```

The Compose file listens on `bolt://localhost:7687` and uses the user `neo4j`. If `NEO4J_URI` or `NEO4J_PASSWORD` is set, those values replace the command defaults. Pass `--neo4j-uri` when you want a specific database.

### Read an existing graph

The HotpotQA scores on the benchmarks page come from a native read of a graph that was already built. Native layout reads the whole database and leaves index creation off. Set `AVESTA_VECTOR_INDEX`, `AVESTA_FULLTEXT_INDEX`, and `AVESTA_FACT_VECTOR_INDEX` to the index names in that database.

```sh
python -m benchmark hotpotqa \
  --questions 1000 \
  --mode full \
  --backend neo4j \
  --neo4j-layout native \
  --skip-index \
  --output benchmark/results/hotpotqa_neo4j_1000.jsonl
```

`--mode direct` searches chunks only. `--mode triple-unfiltered` uses facts and expansion without the chat-model filter or rerank. `--retrieval-only` scores passages and skips the written answer.

The default `--questions` is 200. That is a shorter answer pass over the same indexed corpus. Pass `--questions 2` when you want a two-question check after the corpus is indexed.

## Run LoCoMo

The command downloads the 10-conversation release, indexes each conversation that contains a selected question, and answers those questions. Each conversation is its own namespace. The full set is 1,986 questions. Add `--llm-judge` to record the Mem0 judge beside token F1.

```sh
python -m benchmark locomo run \
  --questions 1986 \
  --backend neo4j \
  --llm-judge \
  --output benchmark/results/locomo_neo4j_full.jsonl
```

Start Neo4j with the Compose file above before `--backend neo4j`. The published LoCoMo run used a second database at `bolt://localhost:7688`, separate from the HotpotQA graph. Pass that URI when port 7687 already holds another graph:

```sh
python -m benchmark locomo run \
  --questions 1986 \
  --backend neo4j \
  --neo4j-uri bolt://localhost:7688 \
  --llm-judge \
  --output benchmark/results/locomo_neo4j_full.jsonl
```

SQLite is the default backend. `--skip-index` reuses conversations already stored. `--schema-mode` defaults to `free`. The same `--output` file skips question ids that are already answered.

The default `--questions` is 100, in conversation order.

## Score a saved LoCoMo file

`locomo score` prints category scores for questions already in a JSONL file. The command only reads that file.

```sh
python -m benchmark locomo score \
  --data benchmark/data/locomo10.json \
  --results benchmark/results/locomo_neo4j_full.jsonl \
  --questions 1986
```
