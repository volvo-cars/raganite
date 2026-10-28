# Benchmarks

Use this page to see how the finished HotpotQA and LoCoMo runs scored, and where to repeat them.

Both runs use gpt-4o and `text-embedding-3-large`. [Run the benchmarks](run-benchmarks.md) repeats them. Flag names are in the [benchmark command reference](../reference/benchmark-cli.md).

## HotpotQA

HotpotQA checks multi-hop questions. This run answers all 1,000 questions in the HippoRAG release. Retrieval is the full fact pipeline, and the answer uses the top 5 passages. Rows are in `benchmark/results/hotpotqa_neo4j_1000.jsonl`.

| Measure | Score |
| --- | --- |
| Exact match | 60.9% |
| Answer F1 | 76.2% |
| Answer precision | 76.8% |
| Answer recall | 80.5% |
| Both supporting titles in the top 5 | 91.9% |
| Recall at 1 | 47.3% |
| Recall at 2 | 89.5% |
| Recall at 5 | 95.7% |
| Recall at 10 | 96.2% |
| Recall at 20 | 96.5% |
| Recall at 50 | 98.6% |
| Recall at 100 | 98.9% |
| Recall at 200 | 99.2% |

Recall at 1 is lower because a HotpotQA question usually has two supporting passages.

### Published comparison

The other systems are from HippoRAG 2, Table 8 and the passage-recall table ([arXiv:2502.14802](https://arxiv.org/html/2502.14802v1)). That evaluation uses the same 1,000 questions, with Llama-3.3-70B-Instruct as the reader and NV-Embed-v2 as the retriever. Raganite’s row uses gpt-4o and `text-embedding-3-large`.

| System | Exact match | Answer F1 | Recall at 5 |
| --- | --- | --- | --- |
| Raganite | 60.9 | 76.2 | 95.7 |
| HippoRAG 2 | 62.7 | 75.5 | 96.3 |
| NV-Embed-v2 | 62.8 | 75.3 | 94.5 |
| RAPTOR | 56.8 | 69.5 | 86.9 |
| GraphRAG | 55.2 | 68.6 | — |
| HippoRAG | 52.6 | 63.5 | 77.3 |
| BM25 | 52.0 | 63.4 | 74.8 |
| Contriever | 51.3 | 62.3 | 75.3 |

GraphRAG and LightRAG do not report passage recall in that paper. LightRAG with the Llama reader is 2.0% exact match and 2.4% F1, so it is left out of the table. The same paper, with a GPT-4o-mini reader, reports HippoRAG 2 at 56.3% exact match and 71.1% F1, NV-Embed-v2 at 57.3% / 71.0%, GraphRAG at 51.4% / 67.6%, RAPTOR at 50.6% / 64.7%, HippoRAG at 46.3% / 60.0%, and LightRAG at 9.9% / 20.2%.

## LoCoMo

LoCoMo checks questions over long conversations. This run answers all 1,986 questions across 10 conversations. Each conversation is its own Neo4j namespace. The answer prompt asks for a short phrase. Token F1 is recorded for every question, and each answer was scored once with the Mem0 CORRECT/WRONG judge. Rows are in `benchmark/results/locomo_neo4j_full.jsonl`.

| Category | Questions | Token F1 |
| --- | --- | --- |
| Single-hop | 841 | 63.3% |
| Multi-hop | 282 | 45.0% |
| Temporal | 321 | 48.7% |
| Open-domain | 96 | 22.1% |
| Adversarial | 446 | 97.3% |

The adversarial prompt turns the question into a choice between the gold answer and "Not mentioned in the conversation", so that category scores higher. On the other 1,540 questions, token F1 is 54.3% and the judge is 66.6%. Across all 1,986 questions, token F1 is 64.0% and the judge is 73.5%.

### Published comparison

The Mem0 paper leaves adversarial questions out and reports a 10-run judge mean ([arXiv:2504.19413](https://arxiv.org/html/2504.19413v1)). Raganite’s judge is one pass.

| System | Single-hop | Multi-hop | Temporal | Open-domain |
| --- | --- | --- | --- | --- |
| Raganite | 79.5 | 58.5 | 43.6 | 54.2 |
| Mem0 | 72.93 | 67.13 | 55.51 | 51.15 |
| Mem0 graph | 75.71 | 65.71 | 58.13 | 47.19 |
| Zep | 76.60 | 61.70 | 49.31 | 41.35 |

| System | Overall judge |
| --- | --- |
| Full context | 72.90 |
| Mem0 graph | 68.44 |
| Mem0 | 66.88 |
| Raganite | 66.6 |
| Zep | 65.99 |
| Best chunk RAG | 60.97 |

Full context pastes the whole conversation into the model, about 26,031 tokens, and has no category split. The best chunk-RAG setting in that paper also has no category split.

## Run them yourself

Follow [Run the benchmarks](run-benchmarks.md).
