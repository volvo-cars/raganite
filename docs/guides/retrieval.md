# Search and retrieval

Get a ranked list of passages for one question. `embeddings` and `chat` are the clients from [Your first index](../getting-started/quickstart.md).

```python
with KnowledgeGraph(embeddings=embeddings, llm=chat) as graph:
    passages = graph.retrieve("What is the capital of France?", top_k=5)

for passage in passages:
    print(passage.score, passage.title)
    print(passage.text)
```

Each `Passage` has `source`, `title`, `text`, `score`, `evidence`, and `metadata`. The beam uses facts to select passages but does not attach the supporting graph path as structured `evidence`. `passages.usage` records elapsed time and token totals for that retrieval run.

`retrieve` embeds the question once. By default, the [two-hop beam walk](../explanation/question-to-passages.md) starts with candidates from directed fact PageRank and direct passage signals, adds candidates connected by title mentions, and scores directed paths. Raganite ships the learned weights and loads them automatically. No separate model file or chat model is needed for retrieval.

Passage reranking is off by default. To opt in, select `llm`, `cross-encoder` (install the `rerank` extra), or `compressor` (pass an object implementing `compress_documents` to the retriever). The legacy `rerank_passages=true` setting still selects the LLM reranker when no other reranker is selected. `retrieve` does not generate an answer.

`query` does that retrieval and then calls the chat model once to answer from the returned passages. With reranking off, that is the only chat call:

```python
answer = graph.query("What is the capital of France?", top_k=5)
print(answer.answer)
for passage in answer.passages:
    print(passage.title, passage.text)
```

The answer uses only those passages. `answer.usage` records retrieval and `query.answer` token totals. `await graph.retriever.aquery(question)` is the async form. A retriever without a chat model raises `ValueError`.

A blank question raises `ValueError`. `top_k` below 1 raises `ValueError`.

## Several questions

```python
batches = graph.retriever.retrieve_many(
    ["What is the capital of France?", "Who wrote the notes?"],
    top_k=5,
)
```

`await graph.retriever.aretrieve(question)` runs one question without blocking the event loop. Close the graph when you are finished so retrieval workers stop. The `with` block does that.

## Turn pieces off

Use this when you want direct chunk search instead of the beam walk. Neither mode needs a chat model unless you enable the LLM reranker:

```python
from raganite.config import RaganiteConfig, RetrievalConfig

config = RaganiteConfig(
    retrieval=RetrievalConfig(use_triples=False),
)
```

`use_triples=False` skips the beam and uses direct chunk search. `use_fulltext=False` skips the additional full-text list in direct search. `query` still needs a chat model.

## Titles

`KnowledgeGraph.retrieve` titles a passage from the stored path. Construct `Retriever` yourself when a path should display as a different title:

```python
from raganite.retrieval import Retriever

retriever = Retriever(
    graph.backend,
    embeddings,
    chat,
    graph.config.retrieval,
    title_by_source={"paris.txt": "Paris"},
)
passages = retriever.retrieve("What is the capital of France?")
retriever.close()
```

## Command line

```sh
uv run raganite retrieve "What is the capital of France?" --top-k 5 --format text
```

The command opens the store read-only. `--format text` prints each passage's rank, title, source, score, and text. `--format json` prints the same passages as objects and adds `evidence` and metadata. That JSON list does not include `passages.usage`.
