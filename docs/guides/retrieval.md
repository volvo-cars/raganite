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

Each `Passage` has `source`, `title`, `text`, `score`, and `evidence`. `evidence` is empty when the passage came only from a direct chunk match.

`retrieve` embeds the question, finds matching chunks, and, by default, searches facts, expands them through the graph, fuses those hits, and asks the chat model to rerank. It does not generate an answer.

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

The command opens the store read-only. `--format json` includes scores, evidence, and metadata.
