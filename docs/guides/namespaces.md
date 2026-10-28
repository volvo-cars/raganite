# Namespaces

Keep one namespace's documents, entities, and facts out of another namespace.

A namespace is the name on `RaganiteConfig.namespace`. The default is `default`. SQLite and an owned Neo4j graph both isolate reads and writes by that name.

`embeddings` and `chat` are the clients from [Your first index](../getting-started/quickstart.md).

```python
from raganite.config import RaganiteConfig, SQLiteConfig
from raganite.graph import KnowledgeGraph

config = RaganiteConfig(
    namespace="team-a",
    backend=SQLiteConfig(path=".raganite/graph.sqlite3"),
)
with KnowledgeGraph(embeddings=embeddings, llm=chat, config=config) as graph:
    graph.indexer.index_text("Team A notes.", source="wiki", path="home.txt")
```

The same SQLite file can hold `team-a` and `team-b`. Retrieval in `team-a` does not return `team-b` passages.

`clear_graph` and `raganite clear --yes` delete the active namespace, not every namespace in the file.

## Match the store you inject

If you pass a `backend` into `KnowledgeGraph`, its namespace and embedding dimensions must equal the config. A mismatch raises `ValueError`.

## Native Neo4j

`layout="native"` reads one existing Neo4j database as a whole. It does not isolate by namespace. Use it for a dedicated database, not for multi-tenant data. See [Neo4j](neo4j.md).
