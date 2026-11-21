"""Benchmark commands: ``python -m benchmark``."""

import json
from pathlib import Path
from typing import Any

import click

from benchmark.locomo import score_locomo
from benchmark.store import prepare_connection
from raganite.config import cpu_workers
from raganite.log import configure_logging


def _verbose_option(command):
    return click.option(
        "-v",
        "--verbose",
        count=True,
        help="Repeat for more detail. -v logs each stage. -vv also logs debug detail.",
    )(command)


def _use_verbosity(ctx: click.Context, verbose: int) -> None:
    parent = ctx.find_root().obj or {}
    configure_logging(max(parent.get("verbose") or 0, verbose))


def run_hotpotqa(options: dict[str, Any]) -> None:
    """Compare saved results on matching questions."""
    from benchmark.metrics import compare
    from benchmark.results import read_rows, save

    baseline_path = options.get("baseline")
    if baseline_path is None:
        raise click.UsageError("--compare-only requires --baseline")
    completed = read_rows(Path(options["output"]))
    queries_path = Path(options["data_dir"]) / "hotpotqa.json"
    queries = json.loads(queries_path.read_text(encoding="utf-8"))[: options["questions"]]
    candidate = [completed[query["_id"]] for query in queries if query["_id"] in completed]
    if not candidate:
        raise click.ClickException("No completed results to compare")
    baseline = list(read_rows(Path(baseline_path)).values())
    result = compare(candidate, baseline)
    save(Path(options["output"]).with_suffix(".comparison.json"), result)
    click.echo(json.dumps(result, indent=2))


@click.group()
@_verbose_option
@click.pass_context
def main(ctx: click.Context, verbose: int) -> None:
    """Run HotpotQA and LoCoMo evaluations and print their scores."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    configure_logging(verbose)


@main.command("hotpotqa")
@_verbose_option
@click.option(
    "--questions",
    type=click.IntRange(1, 1000),
    default=200,
    show_default=True,
    help="First N questions. The full corpus is indexed either way.",
)
@click.option(
    "--concurrency",
    type=click.IntRange(min=1),
    default=cpu_workers(),
    show_default="available CPU count",
    help="Questions evaluated at once.",
)
@click.option(
    "--index-workers",
    type=click.IntRange(min=1),
    default=cpu_workers(),
    show_default="available CPU count",
    help="Documents indexed at once.",
)
@click.option(
    "--backend", type=click.Choice(["sqlite", "neo4j"]), default="sqlite", show_default=True
)
@click.option(
    "--skip-index", is_flag=True, help="Reuse the selected graph instead of indexing the corpus."
)
@click.option("--namespace", default="hotpotqa", show_default=True, help="Owned graph namespace.")
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path),
    default=Path("benchmark/data"),
    show_default=True,
    help="Where the downloaded corpus is stored.",
)
@click.option(
    "--sqlite-path",
    type=click.Path(path_type=Path),
    default=Path("benchmark/data/hotpotqa.sqlite3"),
    show_default=True,
)
@click.option("--neo4j-uri", envvar="NEO4J_URI", default="bolt://127.0.0.1:7690", show_default=True)
@click.option("--neo4j-username", default="neo4j", show_default=True)
@click.option("--neo4j-password", default="raganite-local", show_default=True)
@click.option("--neo4j-database", default="neo4j", show_default=True)
@click.option(
    "--dimensions",
    type=click.IntRange(min=1),
    default=None,
    help="Embedding width. Default follows the configured model.",
)
@click.option(
    "--top-k",
    type=click.IntRange(min=1),
    default=200,
    show_default=True,
    help="Passages kept for recall.",
)
@click.option(
    "--qa-top-k",
    type=click.IntRange(min=1),
    default=5,
    show_default=True,
    help="Passages sent to the answer model.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("benchmark/results/hotpotqa.jsonl"),
    show_default=True,
)
@click.option("--retrieval-only", is_flag=True, help="Score passages and skip answer generation.")
@click.option(
    "--baseline",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Baseline JSONL for --compare-only.",
)
@click.option(
    "--compare-only",
    is_flag=True,
    help="Rebuild the saved read-only comparison. Does not index or call a model.",
)
@click.pass_context
def hotpotqa(ctx: click.Context, **options: Any) -> None:
    """Download HotpotQA, index it, and answer the first questions.

    SQLite is the default graph. Neo4j uses the local database from
    ``benchmark/neo4j/docker-compose.yml``. ``--skip-index`` reuses that graph.
    Repeating a command resumes questions missing from --output.
    """
    _use_verbosity(ctx, options["verbose"])
    if options.get("compare_only"):
        run_hotpotqa(options)
        return
    from benchmark.hotpotqa_flow import run_hotpotqa_benchmark

    run_hotpotqa_benchmark(prepare_connection(options))


@main.group("locomo")
def locomo() -> None:
    """Download and run LoCoMo, or score a results file."""


@locomo.command("score")
@click.option(
    "--data",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Official locomo10.json.",
)
@click.option(
    "--results",
    "results_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="JSONL written by the LoCoMo harness.",
)
@click.option(
    "--questions",
    type=click.IntRange(min=1),
    default=100,
    show_default=True,
    help="First questions in conversation order.",
)
@click.option("--mem0", is_flag=True, help="Score the non-adversarial Mem0 question set.")
def locomo_score(data: Path, results_path: Path, questions: int, mem0: bool) -> None:
    """Print category scores for the first LoCoMo questions already in a results file."""
    click.echo(score_locomo(data, results_path, questions, mem0=mem0))


@locomo.command("run")
@_verbose_option
@click.option(
    "--questions",
    type=click.IntRange(min=1),
    default=100,
    show_default=True,
    help="First questions in conversation order.",
)
@click.option(
    "--concurrency",
    type=click.IntRange(min=1),
    default=cpu_workers(),
    show_default="available CPU count",
)
@click.option("--top-k", type=click.IntRange(min=1), default=20, show_default=True)
@click.option(
    "--qa-top-k", type=click.IntRange(min=1), default=5, show_default=True,
    help="Maximum number of retrieved chunks sent to the answer model.",
)
@click.option(
    "--mem0", is_flag=True, help="Use the pinned Mem0 prompts and skip adversarial questions."
)
@click.option(
    "--answer-template",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Use a Mem0-format answer template instead of the original LoCoMo prompt.",
)
@click.option(
    "--mem0-judge-template",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Use the Mem0 evaluation protocol; --questions counts non-adversarial questions.",
)
@click.option(
    "--backend", type=click.Choice(["sqlite", "neo4j"]), default="sqlite", show_default=True
)
@click.option(
    "--skip-index", is_flag=True, help="Reuse each conversation graph instead of indexing it."
)
@click.option("--namespace-prefix", default="", help="Prefix the owned conversation namespaces.")
@click.option(
    "--data-dir", type=click.Path(path_type=Path), default=Path("benchmark/data"), show_default=True
)
@click.option(
    "--sqlite-path",
    type=click.Path(path_type=Path),
    default=Path("benchmark/data/locomo.sqlite3"),
    show_default=True,
)
@click.option("--neo4j-uri", envvar="NEO4J_URI", default="bolt://127.0.0.1:7690", show_default=True)
@click.option("--neo4j-username", default="neo4j", show_default=True)
@click.option("--neo4j-password", default="raganite-local", show_default=True)
@click.option("--neo4j-database", default="neo4j", show_default=True)
@click.option("--dimensions", type=click.IntRange(min=1), default=None)
@click.option(
    "--schema-mode",
    type=click.Choice(["fixed", "free", "extracted"]),
    default="free",
    show_default=True,
    help="fixed uses the built-in schema. free lets the model name relations from the text. extracted builds a schema from the session.",
)
@click.option(
    "--llm-judge",
    is_flag=True,
    help="Also score each answer with the Mem0 LoCoMo judge. Token F1 is always recorded.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("benchmark/results/locomo.jsonl"),
    show_default=True,
)
@click.pass_context
def locomo_run(ctx: click.Context, **options: Any) -> None:
    """Download LoCoMo, index each needed conversation, and answer its questions.

    Each conversation is its own namespace, so retrieval stays inside that conversation.
    Extraction is free unless --schema-mode says otherwise.
    SQLite is the default. Neo4j uses ``benchmark/neo4j/docker-compose.yml``.
    """
    from benchmark.locomo_flow import run_locomo_benchmark

    _use_verbosity(ctx, options["verbose"])
    run_locomo_benchmark(prepare_connection(options))
