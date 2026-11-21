# Neo4j

Use Neo4j when the graph should live in Neo4j instead of a SQLite file. Both backends use the same beam search over graph records loaded into memory. For direct vector search, SQLite scores stored vectors in process, while Neo4j uses a vector index. You still call `KnowledgeGraph` the same way.

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

Owned Neo4j and SQLite use the same indexer and retrieval code. Beam search
loads the same kinds of records from either store: fact endpoints and
vectors, chunk IDs and text, and the provenance connecting them. Their native
vector and full-text indexes use different search implementations, so a
direct-search result or score is not guaranteed to match across stores.

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

Retrieval discovers the chunk label from the vector index, then the semantic connector and the document connector by how often each pair appears. The most frequent pair is kept. Labels that only repeat the same endpoints, such as `RaganiteNode` with `__Entity__` or `Document`, stay one connector. If two connector types share that count, or two domain labels do, retrieval raises `ValueError` and names both.

Pass an injected `driver` if you already have a Neo4j driver. Raganite closes a driver it created. It leaves an injected driver open.

## CLI

```sh
uv run raganite --backend neo4j --neo4j-uri bolt://localhost:7687 index ./documents
```

Password and database can come from `.env`. See `.env.example`.
