# How a question becomes passages

This page explains the words Raganite uses. The steps for indexing and asking are in the guides.

## The graph

A **knowledge graph** is the connected record of what you indexed: documents, the chunks cut from them, the entities named in those chunks, and the facts that relate those entities.

A **namespace** is one knowledge graph. Two namespaces in the same SQLite file do not share passages.

A **source** is the named origin of a set of documents, such as one repository or one notes folder.

A **document** is one indexed text. A **chunk** is a contiguous piece of that document, small enough to embed. A **passage** is chunk text returned for a question, with a score. Passage is the read side of a chunk. The library avoids calling that returned text a hit or a result.

An **entity** is a named thing in a chunk, such as a city or a person. A **fact** is a statement that one entity stands in a relation to another. "Paris is the capital of France" can be stored as a fact linking Paris and France, and as the chunk text you might show a reader.

## Topology and domain relations

**Topology** is the structural connections: a source has documents, a document has chunks, a chunk mentions entities, and a fact points at its two entities. Those connections use stable names (`HAS_DOCUMENT`, `FROM_CHUNK`, `FACT_SUBJECT`, and so on).

Domain relations, such as `CAPITAL_OF`, are not topology. They come from extraction and can differ by schema. Adapters store both. They do not invent a second set of structural names.

## Why retrieval looks at facts

A chunk search finds text that is similar to the question. A fact search finds a statement, then walks from its entities back to chunks. Fusing both lists is why one passage can carry `evidence`: the facts that pulled it up. The chat model can drop weak facts and reorder the list. That reorder is ranking, not an answer.

## Two stores, one contract

SQLite and Neo4j both satisfy the same store contract: write a document, search chunks, search facts, expand facts. Owned Neo4j behaves like SQLite, including namespaces. Native Neo4j is a read-only view of someone else's database, so namespace isolation does not apply. The caller still uses `retrieve` and receives `Passage` objects.
