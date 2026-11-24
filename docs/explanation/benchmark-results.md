# Benchmark results

These benchmarks use Raganite's two-hop beam retriever with bundled weights
and no passage reranker.

## Document question answering

Raganite retrieves passages with `text-embedding-3-large` and uses GPT-4o
to answer from the top five.

| Dataset | Recall@5 | All support@5 | Answer exact match | Answer token F1 |
| --- | ---: | ---: | ---: | ---: |
| Natural Questions | 63.87% | 33.10% | 39.44% | 55.42% |
| PopQA | 66.90% | 35.30% | 43.80% | 59.98% |
| 2WikiMultiHopQA | 92.75% | 80.70% | 59.20% | 69.09% |
| MuSiQue | 62.88% | 34.80% | 28.50% | 42.91% |
| HotpotQA (supporting titles) | 96.20% | 92.70% | 59.00% | 75.26% |

Recall@5 averages the fraction of annotated evidence found per question.
All support@5 is the percentage of questions whose top five include every
annotated item. Answer token F1 measures token overlap with the reference.

Each dataset has 1,000 questions; the Natural Questions answer scores exclude
one failed question.
Retrieval matches the exact passage (title and text), except on HotpotQA,
which matches supporting titles.

### Comparison with other systems

These tables compare Raganite with selected baselines from the HippoRAG 2
paper. The average gives each of the four datasets equal weight. HotpotQA is
left out because Raganite's result uses title matching and includes training
questions. These are published reference scores, not a head-to-head run.

**Retrieval: Recall@5 (%)** ([HippoRAG 2, Table 3](https://arxiv.org/html/2502.14802v2#S4.T3))

| System | NQ | PopQA | MuSiQue | 2Wiki | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Raganite** | **63.87** | **66.90** | **62.88** | **92.75** | **71.60** |
| HippoRAG 2 | 78.0 | 51.7 | 74.7 | 90.4 | 73.70 |
| NV-Embed-v2 | 75.4 | 51.0 | 69.7 | 76.5 | 68.15 |
| RAPTOR | 68.3 | 48.7 | 57.8 | 66.2 | 60.25 |
| HippoRAG (reproduced) | 44.4 | 53.8 | 53.2 | 90.4 | 60.45 |
| BM25 | 56.1 | 35.7 | 43.5 | 65.3 | 50.15 |

Raganite has higher recall than HippoRAG 2 on PopQA and 2Wiki, and lower
recall on Natural Questions (NQ) and MuSiQue.

**Answer quality: exact match / token F1 (%)** ([HippoRAG 2, Table 8](https://arxiv.org/html/2502.14802v2#A3.T8))

Raganite uses GPT-4o; the published rows below use Llama 3.3 70B Instruct.
These scores compare complete QA systems, not retrievers alone.

| System | NQ | PopQA | MuSiQue | 2Wiki | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Raganite** | **39.44 / 55.42** | **43.80 / 59.98** | **28.50 / 42.91** | **59.20 / 69.09** | **42.73 / 56.85** |
| HippoRAG 2 | 48.6 / 63.3 | 42.9 / 56.2 | 37.2 / 48.6 | 65.0 / 71.0 | 48.43 / 59.78 |
| NV-Embed-v2 | 47.3 / 61.9 | 42.9 / 55.7 | 34.7 / 45.7 | 57.5 / 61.5 | 45.60 / 56.20 |
| RAPTOR | 36.9 / 50.7 | 43.1 / 56.2 | 20.7 / 28.9 | 47.3 / 52.1 | 37.00 / 46.98 |
| Microsoft GraphRAG | 30.8 / 46.9 | 31.4 / 48.1 | 27.3 / 38.5 | 51.4 / 58.6 | 35.23 / 48.03 |
| LightRAG | 8.6 / 16.6 | 2.1 / 2.4 | 0.5 / 1.6 | 9.4 / 11.6 | 5.15 / 8.05 |
| BM25 | 44.7 / 59.0 | 39.1 / 49.9 | 20.3 / 28.8 | 47.9 / 51.2 | 38.00 / 47.23 |

## Conversation question answering

On **LoCoMo**, Raganite scored **75.26% judge accuracy** (1,159 of 1,540
questions) and **51.71% token-set F1**. The evaluation covered ten conversations
in a single pass, excluding adversarial questions. It used
`text-embedding-3-small`, GPT-4o mini, and the
[Mem0 evaluation prompts](https://arxiv.org/html/2504.19413v1#S4.T2), with up
to 60 retrieved conversation chunks per answer.

## Run the benchmarks yourself

From a repository checkout, run `uv sync --locked` and configure your
[model credentials](../guides/providers.md). The harness downloads pinned
datasets and writes per-question results and a summary to `benchmark/results/`.

### Document question answering

Set `dataset` to `hotpotqa`, `nq`, `popqa`, `2wiki`, or `musique`. Use a
dedicated Neo4j Compose project and volume for each dataset, one at a time
on port 7692.

```sh
dataset=2wiki
BENCHMARK_BOLT_PORT=7692 docker compose -p "raganite-qa-$dataset" \
  -f benchmark/neo4j/docker-compose.yml up -d --wait

RAGANITE_CHAT__MODEL=gpt-4o \
RAGANITE_EMBEDDINGS__MODEL=text-embedding-3-large \
uv run python -u -m benchmark qa --dataset "$dataset" \
  --backend neo4j --neo4j-uri bolt://127.0.0.1:7692 --dimensions 3072 \
  --questions 1000 --top-k 200 --qa-top-k 5 --concurrency 3 --index-workers 3 \
  --output "benchmark/results/${dataset}.title-text.jsonl"
```

Use `--retrieval-only` with a separate output path to skip answer generation.
The benchmark command scores exact passages for all five datasets, including
HotpotQA.

### LoCoMo

Use a separate Neo4j instance because the embedding dimension differs.
`--mem0` selects the non-adversarial questions and uses the pinned Mem0 prompts.