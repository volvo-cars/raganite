# Future work

This list is the library work that should happen next. Nothing here has a release date. Behavior that the guides already document is not repeated.

The order is the order of value to someone calling the library. A new provider or a new protocol waits until the defaults, the documented limits, and the default store match what the code does.

## Scale the store the tutorial uses

SQLite is the default because it needs no server. Its vector search reads every embedding in the namespace.

- Name the corpus size at which that scan is the wrong default.
- Add indexed vector search for the SQLite file, returning the same scores and the same passage objects.
- Keep Neo4j as the store that already uses vector indexes.

Indexing imports Neo4j's extraction library even when the graph is a SQLite file. Splitting that dependency, so a SQLite install can omit the Neo4j packages, is larger than the vector index and comes after it.

## Later, if callers ask

These can wait. They do not change retrieval cost, lexical misses, or the default store.

- An MCP server that exposes indexing and retrieval to compatible assistants.
- A Llama client for a runtime that the Hugging Face and Ollama extras do not already cover.
- Parsers for PDF and Office files, after the text-only limit is stated on the first page.

## Out of scope

Raganite is a library. A hosted service, accounts, and billing are outside this list. Questions that need community summaries of a whole corpus are a different retrieval design, and they are outside this list too.
