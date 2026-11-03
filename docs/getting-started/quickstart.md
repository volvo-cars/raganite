# Your first index

Index a folder of text files, then ask a question. You will get passages, not a written answer.

Finish [Installation](installation.md) first. Work from the repository root.

## 1. Add a model key

```sh
cp .env.example .env
```

Open `.env` and set `OPENAI_API_KEY`. Leave the other lines commented unless you need them. The defaults use SQLite at `.raganite/graph.sqlite3`, the namespace `default`, and 1536-dimension embeddings.

Do not commit `.env`.

## 2. Put something in a folder

```sh
mkdir -p documents
printf 'Paris is the capital of France.\n' > documents/paris.txt
```

## 3. Index the folder

```sh
uv run raganite index documents --source notes
```

Raganite creates the SQLite file, splits the text, extracts entities and facts, and stores them under the source name `notes`. A directory index reads text files that match the include globs. PDF, HTML, and Office files are left out. Unchanged files are skipped the next time you run the same command. SQLite scores every stored vector in the process. Use [Neo4j](../guides/neo4j.md) when that scan is the wrong cost.

## 4. Ask a question

```sh
uv run raganite retrieve "What is the capital of France?" --top-k 5 --format text
```

The command prints each passage's rank, title, source, score, and text. It does not print the supporting facts, and it does not write a final answer. Add `--format json` for those facts in `evidence`, plus metadata. `raganite query` uses the same index and writes an answer from the passages. See [Search and retrieval](../guides/retrieval.md).

## 5. Do the same from Python

```python
from raganite.clients.llm import OpenAIEmbeddingClient, OpenAILLMClient
from raganite.graph import KnowledgeGraph

embeddings = OpenAIEmbeddingClient(model="text-embedding-3-small")
chat = OpenAILLMClient(model="gpt-4o", use_responses_api=False)

with KnowledgeGraph(embeddings=embeddings, llm=chat) as graph:
    passages = graph.retrieve("What is the capital of France?", top_k=5)

print(passages[0].title)
print(passages[0].text)
```

`KnowledgeGraph` opens the same default SQLite namespace. `retrieve` returns `Passage` objects. `passages[0].evidence` lists the facts used in the ranking.

An empty or blank question raises `ValueError`.

## What you did

You built one namespace of a knowledge graph from files, then read passages back out of it. Indexing needs a chat model and an embedding model. Retrieval needs the embedding model, and the default ranking also needs the chat model to filter facts and rerank passages.
