"""Benchmark commands: ``python -m benchmark``."""

import json
from pathlib import Path
from typing import Any

import click

from benchmark.hipporag import DATASETS, normalize_questions
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


def _locomo_options(command):
    for option in (
        click.option(
            "--mem0", is_flag=True, help="Use pinned Mem0 prompts and skip adversarial questions."
        ),
        click.option(
            "--answer-template",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            help="Use a Mem0-format answer template instead of the original LoCoMo prompt.",
        ),
        click.option(
            "--mem0-judge-template",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            help="Use a custom Mem0 judge; --questions counts non-adversarial questions.",
        ),
        click.option("--namespace-prefix", default="", help="Prefix LoCoMo conversation namespaces."),
        click.option(
            "--schema-mode",
            type=click.Choice(["fixed", "free", "extracted"]),
            default="free",
            show_default=True,
            help="LoCoMo extraction schema: built-in, free relations, or inferred from each session.",
        ),
        click.option(
            "--llm-judge",
            is_flag=True,
            help="Also score each LoCoMo answer with the Mem0 judge. Token F1 is always recorded.",
        ),
    ):
        command = option(command)
    return command


def run_hotpotqa(options: dict[str, Any]) -> None:
    """Compare saved results on matching questions."""
    from benchmark.metrics import compare
    from benchmark.results import read_rows, save

    baseline_path = options.get("baseline")
    if baseline_path is None:
        raise click.UsageError("--compare-only requires --baseline")
    completed = read_rows(Path(options["output"]))
    dataset = options["dataset"]
    stem, _ = DATASETS[dataset]
    queries_path = options["data_dir"] / f"{stem}.json"
    queries = normalize_questions(dataset, json.loads(queries_path.read_text(encoding="utf-8")))
    queries = queries[: options["questions"]]
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
    """Run document and conversation QA evaluations."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    configure_logging(verbose)


@main.command("qa")
@_verbose_option
@_locomo_options
@click.option(
    "--dataset", type=click.Choice([*DATASETS, "locomo"]), default="hotpotqa", show_default=True
)
@click.option(
    "--questions",
    type=click.IntRange(min=1),
    help="First N questions. Defaults to 200 for document QA and 100 for LoCoMo.",
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
@click.option("--namespace", help="Owned graph namespace. Defaults to <dataset>_title_text.")
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
    help="SQLite path. Defaults to benchmark/data/<dataset>.sqlite3.",
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
    help="Passages retrieved. Defaults to 200 for document QA and 20 for LoCoMo.",
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
    help="Result path. Defaults to benchmark/results/<dataset>.jsonl.",
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
def qa(ctx: click.Context, **options: Any) -> None:
    """Index and evaluate a document or conversation QA dataset.

    Document QA indexes the whole corpus. LoCoMo indexes each selected
    conversation separately. SQLite is the default; --skip-index reuses a graph.
    Repeating a command resumes questions missing from --output.
    """
    _use_verbosity(ctx, options["verbose"])
    dataset = options["dataset"]
    options["sqlite_path"] = options["sqlite_path"] or options["data_dir"] / f"{dataset}.sqlite3"
    options["output"] = options["output"] or Path(f"benchmark/results/{dataset}.jsonl")
    if dataset == "locomo":
        if any(options[name] for name in ("retrieval_only", "compare_only", "baseline", "namespace")):
            raise click.UsageError(
                "--retrieval-only, --compare-only, --baseline, and --namespace are not supported "
                "for LoCoMo; use --namespace-prefix for conversation namespaces."
            )
        options["questions"] = options["questions"] or 100
        options["top_k"] = options["top_k"] or 20
        _run_locomo(options)
        return

    locomo_only = (
        "mem0", "answer_template", "mem0_judge_template", "namespace_prefix", "schema_mode", "llm_judge"
    )
    if any(
        ctx.get_parameter_source(name) == click.core.ParameterSource.COMMANDLINE
        for name in locomo_only
    ):
        raise click.UsageError("LoCoMo-specific options require --dataset locomo.")
    options["questions"] = options["questions"] or 200
    if options["questions"] > 1000:
        raise click.BadParameter(
            "document QA datasets have at most 1000 questions", param_hint="--questions"
        )
    options["top_k"] = options["top_k"] or 200
    options["namespace"] = options["namespace"] or f"{dataset}_title_text"
    if options.get("compare_only"):
        run_hotpotqa(options)
        return
    from benchmark.hotpotqa_flow import run_hotpotqa_benchmark

    run_hotpotqa_benchmark(prepare_connection(options))


main.add_command(qa, "hotpotqa")


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
@_locomo_options
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
    "--backend", type=click.Choice(["sqlite", "neo4j"]), default="sqlite", show_default=True
)
@click.option(
    "--skip-index", is_flag=True, help="Reuse each conversation graph instead of indexing it."
)
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
    _use_verbosity(ctx, options["verbose"])
    _run_locomo(options)


def _run_locomo(options: dict[str, Any]) -> None:
    from benchmark.locomo_flow import run_locomo_benchmark

    run_locomo_benchmark(prepare_connection(options))
