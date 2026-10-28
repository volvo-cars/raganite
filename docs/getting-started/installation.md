# Installation

This lesson installs Raganite from the repository so the later lesson can index a folder.

You need Python 3.11, 3.12, or 3.13, and [uv](https://docs.astral.sh/uv/).

## 1. Get the repository

```sh
git clone git@github.com:volvo-cars/raganite.git
cd raganite
```

## 2. Install the environment

```sh
uv sync
```

uv uses `uv.lock` when it is compatible with the project settings and configured index, and refreshes it when needed. Review any resulting lockfile changes. To intentionally upgrade compatible dependencies later, run `uv lock --upgrade` and then `uv sync` again.

## 3. Check that the command exists

```sh
uv run raganite --help
```

You should see the commands `init`, `config`, `index`, `retrieve`, and `clear`.

## What you have

The install includes the library, the `raganite` command, SQLite storage, and the Neo4j client libraries. SQLite needs no extra server. Neo4j is optional and is covered in [Neo4j](../guides/neo4j.md).

Model calls need an OpenAI or Azure OpenAI key. The next lesson sets that up. `init`, `config`, and `clear` do not call a model.

## See also

- [Your first index](quickstart.md) continues from here.
- [Model providers](../guides/providers.md) shows Azure as well as OpenAI.
