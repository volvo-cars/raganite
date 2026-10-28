#
# Copyright 2026 Volvo Car Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Command-line access to configurable graph indexing and fact retrieval."""

import json
import logging
import sys
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import click


def _common_options(command):
    """Share Click options without replacing environment/file values with defaults."""
    options = [
        click.option(
            "--env-file",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            help="Dotenv file",
            show_default=".env in the current directory, if present",
        ),
        click.option("--no-env-file", is_flag=True, default=None, help="Skip dotenv loading"),
        click.option(
            "--config",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            help="JSON or TOML settings file",
        ),
        click.option("--backend", type=click.Choice(["sqlite", "neo4j"]), show_default="sqlite"),
        click.option(
            "--sqlite-path", help="SQLite database path", show_default=".raganite/graph.sqlite3"
        ),
        click.option(
            "--neo4j-uri", help="Neo4j connection URI", show_default="bolt://localhost:7687"
        ),
        click.option("--namespace", show_default="default", help="Graph namespace"),
        click.option(
            "--dimensions",
            type=click.IntRange(min=1),
            show_default="1536",
            help="Stored embedding vector size; must match your model",
        ),
        click.option(
            "--workers",
            type=click.IntRange(min=1),
            show_default="available CPU count",
            help="Indexing and retrieval workers",
        ),
        click.option(
            "--provider",
            type=click.Choice(["openai", "azure-openai"]),
            show_default="openai",
            help="Provider for both clients",
        ),
        click.option(
            "--chat-model", show_default="gpt-4o", help="Chat model ID or Azure deployment name"
        ),
        click.option(
            "--embedding-model",
            show_default="text-embedding-3-small",
            help="Embedding model ID or Azure deployment name",
        ),
        click.option(
            "--api-version",
            help="Dated Azure API version for both clients",
            show_default="unset; Azure v1",
        ),
        click.option(
            "--format",
            "output_format",
            type=click.Choice(["json", "text"]),
            show_default="text for retrieve; json otherwise",
            help="Output format",
        ),
        click.option(
            "-v",
            "--verbose",
            count=True,
            help="Repeat for more detail. -vv logs each caught failure.",
        ),
    ]
    for option in reversed(options):
        command = option(command)
    return command


try:
    _VERSION = version("raganite")
except PackageNotFoundError:
    _VERSION = "uninstalled"


@click.group(
    name="raganite",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(version=_VERSION)
@_common_options
@click.pass_context
def main(ctx: click.Context, **options: Any) -> None:
    """Index documents and retrieve fact-grounded passages.

    Start with an OPENAI_API_KEY in .env: SQLite, the default schema, model
    choices, chunking, and CPU-based worker counts are configured automatically.
    Indexing creates storage as needed; a separate init command is optional.

    Common options work before or after the command. Explicit options override
    environment variables, then dotenv, config-file values, and built-in defaults.
    Use 'raganite config' to inspect effective settings without calling a model.
    """
    ctx.obj = {key: value for key, value in options.items() if value is not None}


def _overrides(args: SimpleNamespace) -> dict[str, Any]:
    values = vars(args)
    overrides: dict[str, Any] = {}
    for argument, field in (("namespace", "namespace"), ("dimensions", "embedding_dimensions")):
        if argument in values:
            overrides[field] = values[argument]
    backend = {
        field: values[arg]
        for arg, field in (
            ("backend", "kind"),
            ("sqlite_path", "path"),
            ("neo4j_uri", "uri"),
        )
        if arg in values
    }
    if args.command == "retrieve":
        backend["read_only"] = True
    if backend:
        overrides["backend"] = backend
    if "workers" in values:
        overrides["indexing"] = {"workers": args.workers}
        overrides["retrieval"] = {"workers": args.workers}
    for section, model_argument in (("chat", "chat_model"), ("embeddings", "embedding_model")):
        options = {
            field: values[arg]
            for arg, field in (
                ("provider", "provider"),
                (model_argument, "model"),
                ("api_version", "api_version"),
            )
            if arg in values
        }
        if options:
            overrides[section] = options
    if args.command == "index":
        indexing = overrides.setdefault("indexing", {})
        if args.schema is not None:
            indexing["graph_schema"] = json.loads(args.schema.read_text(encoding="utf-8"))
        if args.schema_mode is not None:
            indexing["schema_mode"] = args.schema_mode
    return overrides


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            name = key.lower()
            sensitive = any(
                part in name
                for part in (
                    "password",
                    "secret",
                    "credential",
                    "authorization",
                    "api_key",
                    "ad_token",
                )
            )
            sensitive = sensitive or name in {
                "headers",
                "default_headers",
                "default_query",
                "token",
                "access_token",
            }
            result[key] = "********" if sensitive and item is not None else _redact(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if isinstance(value, str) and "://" in value and "@" in value:
        return "********"  # A connection URL may contain a username/password.
    return value


def _emit(payload: Any, output_format: str) -> None:
    if output_format == "text" and isinstance(payload, list):
        for rank, passage in enumerate(payload, 1):
            print(f"[{rank}] {passage['title']} ({passage['source']}) score={passage['score']:.6f}")
            print(passage["text"])
            print()
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))


def _execute(args: SimpleNamespace, config: Any) -> int:
    output_format = getattr(args, "format", "json")
    if args.command == "config":
        _emit(_redact(config.model_dump(mode="json")), output_format)
        return 0

    from raganite.backends import create_backend

    if args.command in {"init", "clear"}:
        with create_backend(config) as backend:
            if args.command == "init":
                result = {
                    "backend": config.backend.kind,
                    "namespace": backend.namespace,
                    "status": "ready",
                }
            else:
                result = {
                    "namespace": backend.namespace,
                    "source": args.source,
                    "deleted_documents": backend.delete_documents(args.source),
                }
        _emit(result, output_format)
        return 0

    from raganite.clients.llm.factory import create_chat_client, create_embedding_client
    from raganite.graph import KnowledgeGraph

    needs_llm = (
        args.command == "index"
        or config.retrieval.rerank_passages
        or (config.retrieval.use_triples and config.retrieval.filter_triples)
    )
    embeddings = create_embedding_client(config.embeddings, dimensions=config.embedding_dimensions)
    llm = create_chat_client(config.chat) if needs_llm else None
    with KnowledgeGraph(config=config, embeddings=embeddings, llm=llm) as graph:
        if args.command == "retrieve":
            passages = graph.retrieve(args.question, top_k=args.top_k)
            _emit([asdict(passage) for passage in passages], output_format)
            return 0
        if args.path != "-" and Path(args.path).is_dir():
            report = graph.indexer.index_repository(
                args.path, source_name=args.source, force=args.force
            )
        else:
            path = Path(args.path)
            text = sys.stdin.read() if args.path == "-" else path.read_text(encoding="utf-8")
            report = graph.indexer.index_text(
                text,
                source=args.source or "documents",
                path=args.document_path or ("stdin" if args.path == "-" else path.name),
                force=args.force,
            )
    payload = asdict(report)
    # Provider error messages can include request payloads or credentials.
    if getattr(args, "verbose", 0) >= 2:
        payload["failed_files"] = [
            {"path": path, "error": reason} for path, reason in report.failed_files
        ]
    else:
        payload["failed_files"] = [
            {"path": path, "error": "indexing_failed"} for path, _ in report.failed_files
        ]
    payload["total"] = report.total
    _emit(payload, output_format)
    return 1 if report.failed_files else 0


def _run(ctx: click.Context, command: str, **options: Any) -> None:
    from raganite.log import configure_logging

    parent = ctx.find_root().obj or {}
    values = {**parent, **{key: value for key, value in options.items() if value is not None}}
    values["verbose"] = max(parent.get("verbose") or 0, options.get("verbose") or 0)
    configure_logging(values["verbose"])
    values["format"] = values.pop("output_format", "text" if command == "retrieve" else "json")
    if command == "index":
        for key in ("source", "document_path", "schema", "schema_mode"):
            values.setdefault(key, None)
    elif command == "clear":
        values.setdefault("source", None)
    args = SimpleNamespace(command=command, **values)
    if getattr(args, "no_env_file", False) and getattr(args, "env_file", None) is not None:
        raise click.UsageError("--env-file and --no-env-file cannot be combined", ctx)
    if args.command == "clear" and not args.yes:
        raise click.UsageError(
            "clear requires --yes; without --source it deletes the entire configured namespace", ctx
        )
    if (
        args.command == "index"
        and getattr(args, "document_path", None)
        and args.path != "-"
        and Path(args.path).is_dir()
    ):
        raise click.UsageError("--document-path applies only to a single file or stdin", ctx)
    if args.command == "retrieve" and not args.question.strip():
        raise click.BadParameter("question must not be empty", param_hint="QUESTION", ctx=ctx)
    try:
        from raganite.config.loading import load_config

        env_file = getattr(args, "env_file", None)
        if not getattr(args, "no_env_file", False) and env_file is None and Path(".env").is_file():
            env_file = Path(".env")
        config = load_config(
            getattr(args, "config", None), env_file=env_file, overrides=_overrides(args)
        )
        exit_code = _execute(args, config)
    except KeyboardInterrupt:
        click.echo("Cancelled.", err=True)
        exit_code = 130
    except Exception as exc:
        # Do not print SDK responses, environment values, or validation inputs.
        from pydantic import ValidationError

        if isinstance(exc, ValidationError):
            fields = ", ".join(".".join(map(str, error["loc"])) for error in exc.errors())
            message = (
                f"Invalid configuration fields: {fields}. See raganite config and .env.example."
            )
            exit_code = 2
        else:
            status = getattr(exc, "status_code", None)
            message = f"Operation failed ({type(exc).__name__})" + (
                f", HTTP {status}" if isinstance(status, int) else ""
            )
            if values.get("verbose", 0) >= 2:
                logging.getLogger("raganite.cli").debug("%s", message, exc_info=exc)
                message = f"{message}: {exc}"
            message += ". Check input paths, backend access, and provider settings in .env.example."
            exit_code = 1
        click.echo(message, err=True)
    ctx.exit(exit_code)


@main.command("init")
@_common_options
@click.pass_context
def initialize(ctx: click.Context, **options: Any) -> None:
    """Create or validate backend storage and indexes. No API key needed."""
    _run(ctx, "init", **options)


@main.command("config")
@_common_options
@click.pass_context
def show_config(ctx: click.Context, **options: Any) -> None:
    """Show resolved settings with secrets masked. No API key needed."""
    _run(ctx, "config", **options)


@main.command("index")
@click.argument("path", default=".", type=click.Path(exists=True, allow_dash=True))
@click.option(
    "--source", help="Source name", show_default="directory name, or documents for files/stdin"
)
@click.option(
    "--document-path", help="Stored path for a file/stdin", show_default="filename or stdin"
)
@click.option("--force", is_flag=True, help="Reindex unchanged content")
@click.option(
    "--schema",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Extraction GraphSchema as JSON",
    show_default="built-in code schema",
)
@click.option(
    "--schema-mode", type=click.Choice(["fixed", "free", "extracted"]), show_default="fixed"
)
@_common_options
@click.pass_context
def index(ctx: click.Context, **options: Any) -> None:
    """Index PATH (default: current directory), a file, or stdin (-).

    A directory synchronizes its source, removing previously indexed files that
    are no longer present or selected. Files and stdin do not prune their source.
    Unchanged content is skipped automatically unless --force is supplied.
    """
    _run(ctx, "index", **options)


@main.command("retrieve")
@click.argument("question")
@click.option(
    "--top-k",
    type=click.IntRange(min=1),
    default=5,
    show_default=True,
    help="Maximum returned passages",
)
@_common_options
@click.pass_context
def retrieve(ctx: click.Context, **options: Any) -> None:
    """Retrieve passages and fact evidence for QUESTION; no answer generation.

    Uses existing indexed data in read-only mode. Defaults to five passages in
    human-readable text; --format json includes scores, evidence, and metadata.
    """
    _run(ctx, "retrieve", **options)


@main.command("clear")
@click.option("--source", help="Delete only this source; omit to clear the namespace")
@click.option("--yes", is_flag=True, help="Required confirmation for destructive deletion")
@_common_options
@click.pass_context
def clear(ctx: click.Context, **options: Any) -> None:
    """Delete documents in the active namespace. Requires --yes."""
    _run(ctx, "clear", **options)


if __name__ == "__main__":
    main()
