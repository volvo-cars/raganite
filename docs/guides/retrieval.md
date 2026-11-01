# Search and retrieval

Get a ranked list of passages for one question. `embeddings` and `chat` are the clients from [Your first index](../getting-started/quickstart.md).

```python
with KnowledgeGraph(embeddings=embeddings, llm=chat) as graph:
    passages = graph.retrieve("What is the capital of France?", top_k=5)

for passage in passages:
    print(passage.score, passage.title)
    print(passage.text)
    for fact in passage.evidence:
        print(fact.fact, fact.seed_entity, fact.related_entity)
```

Each `Passage` has `source`, `title`, `text`, `score`, `evidence`, and `metadata`. `evidence` is empty when the passage came only from a direct chunk match. `passages.usage` records elapsed time and token totals for that retrieval run.

`retrieve` embeds the question, finds matching chunks, and, by default, searches facts, expands them through the graph, fuses those hits, and asks the chat model to rerank. It does not generate an answer.

`query` does that retrieval and then asks the chat model to answer from the returned passages:

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

Use this when you want chunk search only, or when you have embeddings but no chat model:

```python
from raganite.config import RaganiteConfig, RetrievalConfig

config = RaganiteConfig(
    retrieval=RetrievalConfig(
        use_triples=False,
        filter_triples=False,
        rerank_passages=False,
    ),
)
```

`use_triples=False` skips fact search and expansion. The other two flags skip the chat-model steps. With both chat steps off, you can construct retrieval without a chat model.

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
