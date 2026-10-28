# Neo4j

Use Neo4j when the graph should live in Neo4j instead of a SQLite file. You still call `KnowledgeGraph` the same way.

## Owned layout

Owned is the default. Raganite creates its own nodes and indexes in the database and isolates them by namespace. `embeddings` and `chat` are the clients from [Your first index](../getting-started/quickstart.md).

```python
from pydantic import SecretStr

from raganite.config import Neo4jConfig, RaganiteConfig
from raganite.graph import KnowledgeGraph

config = RaganiteConfig(
    namespace="team-a",
    embedding_dimensions=1536,
    backend=Neo4jConfig(
        uri="bolt://localhost:7687",
        username="neo4j",
        password=SecretStr("<supply at runtime>"),
        database="neo4j",
    ),
)

with KnowledgeGraph(embeddings=embeddings, llm=chat, config=config) as graph:
    graph.indexer.index_text("Paris is the capital of France.", source="notes", path="paris.txt")
    passages = graph.retrieve("What is the capital of France?", top_k=5)
```

Set `read_only=True` on `Neo4jConfig` when this process must not write. Indexing then raises `PermissionError`.

## Native layout

Native reads a database that already has chunks, documents, and facts. Raganite does not create indexes or write the graph. The whole database is in scope. There is no namespace filter.

```python
from raganite.backends import Neo4jBackend
from raganite.config import Neo4jConfig
from raganite.retrieval import Retriever

backend = Neo4jBackend(
    Neo4jConfig(
        uri="bolt://localhost:7687",
        username="neo4j",
        password=SecretStr("<supply at runtime>"),
        database="existing",
        vector_index="chunks_vector",
        fulltext_index="chunks_fulltext",
        fact_vector_index="facts_vector",
    ),
    dimensions=1536,
    layout="native",
)
```

`layout="native"` forces read-only, even if the config says otherwise. Writes raise `PermissionError`. Use a dedicated database.

Pass an injected `driver` if you already have a Neo4j driver. Raganite closes a driver it created. It leaves an injected driver open.

## CLI

```sh
uv run raganite --backend neo4j --neo4j-uri bolt://localhost:7687 index ./documents
```

Password and database can come from `.env`. See `.env.example`.
