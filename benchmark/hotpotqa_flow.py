"""Download the HotpotQA release, index it, then score the first questions."""

import asyncio
import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

from langchain_core.documents import Document
from openai import BadRequestError

from benchmark.fetch import HOTPOT_CORPUS_URL, HOTPOT_QUERIES_URL, ensure_download
from benchmark.metrics import aggregate, answer_metrics, benchmark_usage_report, retrieval_metrics
from benchmark.results import passages_from_rows, read_rows, save
from benchmark.store import maybe_index, store_config

logger = logging.getLogger(__name__)

PROMPTS = Path(__file__).resolve().parent / "prompts"


def corpus_documents(corpus: list[dict]) -> tuple[list[Document], dict[str, str]]:
    documents, titles = [], {}
    for passage in corpus:
        name = f"doc_{int(passage['idx']):05d}.txt"
        titles[name] = passage["title"]
        documents.append(
            Document(
                page_content=passage["text"],
                metadata={"path": name, "title": passage["title"]},
            )
        )
    return documents, titles


def load_hotpotqa(data_dir: Path) -> tuple[list[dict], list[dict]]:
    queries_path = ensure_download(data_dir / "hotpotqa.json", HOTPOT_QUERIES_URL)
    corpus_path = ensure_download(data_dir / "hotpotqa_corpus.json", HOTPOT_CORPUS_URL)
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    if len(queries) != 1000 or len(corpus) != 9811:
        raise ValueError("Expected 1,000 HotpotQA questions and 9,811 passages")
    return queries, corpus


def load_benchmark_env() -> None:
    from dotenv import load_dotenv

    benchmark_dir = Path(__file__).resolve().parent
    load_dotenv(benchmark_dir / ".env", override=False)
    load_dotenv(benchmark_dir.parent / ".env", override=False)


def _clients(dimensions: int, *, answer_max_tokens: int = 64):
    from raganite.clients.llm.factory import create_chat_client, create_embedding_client
    from raganite.config.loading import load_config

    settings = load_config()
    ranking = create_chat_client(settings.chat)
    embeddings = create_embedding_client(settings.embeddings, dimensions=dimensions)
    answer = ranking
    if settings.chat.provider in {"openai", "azure-openai"}:
        answer = create_chat_client(
            settings.chat.model_copy(
                update={"options": {**settings.chat.options, "temperature": 0, "max_tokens": answer_max_tokens}}
            )
        )
    return embeddings, ranking, answer.model


def _rate_limited(exc: BaseException) -> bool:
    return (
        type(exc).__name__ in {"OpenAIRateLimitError", "RateLimitError"}
        or getattr(exc, "status_code", None) == 429
    )


class RateLimitPause:
    """Hold every question for 30 seconds after any rate limit."""

    def __init__(self, seconds: float = 30):
        self.seconds = seconds
        self._until = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        while True:
            remaining = self._until - asyncio.get_running_loop().time()
            if remaining <= 0:
                return
            await asyncio.sleep(remaining)

    async def trigger(self, question_id: str) -> None:
        loop = asyncio.get_running_loop()
        async with self._lock:
            until = loop.time() + self.seconds
            if until <= self._until:
                return
            self._until = until
            print(f"{question_id}: rate limited, pausing {self.seconds:.0f}s", flush=True)


async def evaluate_hotpotqa(
    queries,
    retriever,
    title_map,
    *,
    answer_model,
    system: str,
    top_k: int,
    qa_top_k: int,
    output: Path,
    concurrency: int,
    indexing_usage: dict | None = None,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = read_rows(output) if output.exists() else {}
    logger.info(
        "Answering %s questions with concurrency %s (%s already stored)",
        len(queries),
        concurrency,
        len(completed),
    )
    failures, lock = [], asyncio.Lock()
    semaphore = asyncio.Semaphore(concurrency)
    pause = RateLimitPause()

    async def evaluate(query):
        if query["_id"] in completed:
            return
        async with semaphore:
            for attempt in range(8):
                await pause.wait()
                try:
                    await _score_hotpot_question(
                        query,
                        retriever,
                        title_map,
                        answer_model=answer_model,
                        system=system,
                        top_k=top_k,
                        qa_top_k=qa_top_k,
                        output=output,
                        lock=lock,
                        completed=completed,
                        total=len(queries),
                    )
                    return
                except Exception as exc:
                    logger.info("%s failed: %s", query["_id"], type(exc).__name__)
                    logger.debug("%s failed", query["_id"], exc_info=True)
                    if _rate_limited(exc) and attempt < 7:
                        await pause.trigger(query["_id"])
                        continue
                    failures.append({"id": query["_id"], "error": type(exc).__name__})
                    print(f"{query['_id']}: failed ({type(exc).__name__})", flush=True)
                    return

    await asyncio.gather(*(evaluate(query) for query in queries))
    summary = aggregate([completed[query["_id"]] for query in queries if query["_id"] in completed])
    usage = benchmark_usage_report(indexing_usage, summary.get("usage"))
    summary["usage_report"] = usage
    summary.update({"requested_questions": len(queries), "failures": failures})
    save(output.with_suffix(".summary.json"), summary)
    save(output.with_suffix(".usage.json"), usage)
    print(json.dumps(summary, indent=2))
    if failures:
        raise RuntimeError("Some questions failed; completed results can be resumed")
    return summary


async def _score_hotpot_question(
    query,
    retriever,
    title_map,
    *,
    answer_model,
    system,
    top_k,
    qa_top_k,
    output,
    lock,
    completed,
    total,
):
    from raganite.clients.llm.usage import track_run, usage_stage

    with track_run("benchmark") as usage:
        logger.info("%s: retrieving", query["_id"])
        rows = [
            {
                "source": passage.source,
                "text": passage.text,
                "score": passage.score,
                "node_id": passage.source,
            }
            for passage in await retriever.aretrieve(query["question"])
        ]
        passages = passages_from_rows(rows, title_map, top_k)
        logger.info("%s: retrieved %s passages", query["_id"], len(passages))
        gold_titles = sorted({title for title, _ in query["supporting_facts"]})
        answers = query["answer"] if isinstance(query["answer"], list) else [query["answer"]]
        item = {
            "id": query["_id"],
            "question": query["question"],
            "gold_answers": answers,
            "gold_titles": gold_titles,
            "retrieved": passages,
            "retrieval_metrics": retrieval_metrics([p["title"] for p in passages], gold_titles),
        }
        if answer_model is not None:
            logger.info("%s: answering from the top %s passages", query["_id"], qa_top_k)
            context = "\n\n".join(
                f"[{i}] {p['title']}\n{p['text']}" for i, p in enumerate(passages[:qa_top_k], 1)
            )
            try:
                with usage_stage("benchmark.answer"):
                    response = await answer_model.ainvoke(
                        [
                            ("system", system),
                            (
                                "user",
                                f"Context:\n{context}\n\nQuestion: {query['question']}\n\nAnswer:",
                            ),
                        ]
                    )
                item["answer"] = response.text.strip()
            except BadRequestError as exc:
                if not isinstance(exc.body, dict) or exc.body.get("code") != "content_filter":
                    raise
                item["answer"] = ""
                item["answer_error"] = "azure_content_filter"
            item["answer_metrics"] = answer_metrics(item["answer"], answers)
    item["usage"] = asdict(usage)
    async with lock:
        with output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        completed[item["id"]] = item
        print(f"[{len(completed)}/{total}] {query['_id']}: complete", flush=True)


def run_hotpotqa_benchmark(options) -> None:
    """Download, index, and evaluate. ``--skip-index`` reuses the selected store."""
    from raganite.config import ChunkingConfig, IndexingConfig, RetrievalConfig
    from raganite.config.loading import load_config
    from raganite.graph import KnowledgeGraph
    from raganite.retrieval import Retriever

    load_benchmark_env()
    namespace = options.get("namespace", "hotpotqa")
    if options["qa_top_k"] > options["top_k"]:
        raise ValueError("--qa-top-k must be no larger than --top-k")
    logger.info(
        "HotpotQA: %s questions, backend %s, index workers %s, concurrency %s",
        options["questions"],
        options["backend"],
        options["index_workers"],
        options["concurrency"],
    )
    data_dir = Path(options["data_dir"])
    queries, corpus = load_hotpotqa(data_dir)
    selected = queries[: options["questions"]]
    documents, title_map = corpus_documents(corpus)
    dimensions = options["dimensions"]
    dimensions = dimensions or load_config().embedding_dimensions
    retrieval = RetrievalConfig(
        workers=options["concurrency"],
        top_k=options["top_k"],
    )
    config = store_config(
        options["backend"],
        sqlite_path=Path(options["sqlite_path"]),
        neo4j_uri=options["neo4j_uri"],
        username=options["neo4j_username"],
        password=options["neo4j_password"],
        database=options["neo4j_database"],
        dimensions=dimensions,
        namespace=namespace,
        retrieval=retrieval,
    )
    config = config.model_copy(
        update={
            "indexing": IndexingConfig(
                workers=options["index_workers"],
                chunking=ChunkingConfig(chunk_size=100_000, chunk_overlap=0),
            )
        }
    )
    output = Path(options["output"])
    system = (PROMPTS / "answer_hotpot_question.txt").read_text(encoding="utf-8")
    settings = load_config()
    protocol = {
        "backend": options["backend"],
        "questions": options["questions"],
        "top_k": options["top_k"],
        "qa_top_k": options["qa_top_k"],
        "dimensions": dimensions,
        "namespace": namespace,
        "retrieval_only": options["retrieval_only"],
        "chat_provider": settings.chat.provider,
        "chat_model": settings.chat.model,
        "embedding_provider": settings.embeddings.provider,
        "embedding_model": settings.embeddings.model,
        "answer_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
    }
    manifest = output.with_suffix(".manifest.json")
    if output.exists() and (
        not manifest.exists()
        or json.loads(manifest.read_text(encoding="utf-8")).get("protocol") != protocol
    ):
        raise ValueError("Existing output uses another protocol; select a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        save(manifest, {"protocol": protocol})
    if options["skip_index"]:
        config.backend.read_only = True
    embeddings, ranking, answer = _clients(dimensions)
    if options["retrieval_only"]:
        answer = None
    graph = KnowledgeGraph(embeddings=embeddings, llm=ranking, config=config)
    retriever = None
    try:
        report = maybe_index(
            graph.indexer, documents, source="hotpotqa", skip_index=options["skip_index"]
        )
        indexing_usage = None
        if report is not None:
            indexing_usage = asdict(report.usage)
            print(
                f"Indexed {len(report.indexed_files)} passages, "
                f"skipped {len(report.skipped_files)}, failed {len(report.failed_files)}, "
                f"{report.usage.total_tokens} tokens in {report.usage.elapsed_seconds:.1f}s",
                flush=True,
            )
            save(Path(options["output"]).with_suffix(".usage.json"), {"indexing": indexing_usage})
            if report.failed_files:
                raise RuntimeError("Some passages failed to index; rerun before scoring")
        retriever = Retriever(graph.backend, embeddings, config=retrieval, title_by_source=title_map)
        asyncio.run(
            evaluate_hotpotqa(
                selected,
                retriever,
                title_map,
                answer_model=answer,
                system=system,
                top_k=options["top_k"],
                qa_top_k=options["qa_top_k"],
                output=output,
                concurrency=options["concurrency"],
                indexing_usage=indexing_usage,
            )
        )
    finally:
        if retriever is not None:
            retriever.close()
        graph.close()
