"""Download the HotpotQA release, index it, then score the first questions."""

import asyncio
import json
import logging
import os
from pathlib import Path

from langchain_core.documents import Document
from openai import BadRequestError

from benchmark.fetch import HOTPOT_CORPUS_URL, HOTPOT_QUERIES_URL, ensure_download
from benchmark.metrics import aggregate, answer_metrics, retrieval_metrics
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
    load_benchmark_env()
    if os.getenv("AZURE_OPENAI_API_KEY"):
        import httpx

        from raganite.clients.llm import AzureOpenAIEmbeddingClient, AzureOpenAILLMClient

        timeout = httpx.Timeout(
            float(os.getenv("AZURE_OPENAI_READ_TIMEOUT", "300")),
            connect=float(os.getenv("AZURE_OPENAI_CONNECT_TIMEOUT", "30")),
        )
        verify = os.getenv("AZURE_OPENAI_DISABLE_SSL_VERIFY", "false").lower() not in {
            "1",
            "true",
            "yes",
        }
        sync_http = httpx.Client(timeout=timeout, verify=verify)
        async_http = httpx.AsyncClient(timeout=timeout, verify=verify)
        common = {
            "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
            "api_key": os.environ["AZURE_OPENAI_API_KEY"],
            "http_client": sync_http,
            "http_async_client": async_http,
            "timeout": timeout,
            "max_retries": 4,
        }
        embeddings = AzureOpenAIEmbeddingClient(
            api_version=os.environ["AZURE_API_EMBEDDING_API_VERSION"],
            model=os.environ["AZURE_EMBEDDING_DEPLOYMENT"],
            dimensions=dimensions,
            check_embedding_ctx_length=False,
            **common,
        )
        ranking = AzureOpenAILLMClient(
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            use_responses_api=False,
            **common,
        )
        answer = AzureOpenAILLMClient(
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            temperature=0,
            max_tokens=answer_max_tokens,
            use_responses_api=False,
            **common,
        ).model
        return embeddings, ranking, answer, (sync_http, async_http)
    from raganite.clients.llm.factory import create_chat_client, create_embedding_client
    from raganite.config import ModelClientConfig

    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    ranking = create_chat_client(ModelClientConfig(model=chat_model))
    embeddings = create_embedding_client(
        ModelClientConfig(model=embedding_model), dimensions=dimensions
    )
    return embeddings, ranking, ranking.model, ()


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
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = read_rows(output) if output.exists() else {}
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
                    logger.debug("%s failed: %s", query["_id"], exc, exc_info=True)
                    if _rate_limited(exc) and attempt < 7:
                        await pause.trigger(query["_id"])
                        continue
                    failures.append({"id": query["_id"], "error": type(exc).__name__})
                    print(f"{query['_id']}: failed ({type(exc).__name__})", flush=True)
                    return

    await asyncio.gather(*(evaluate(query) for query in queries))
    summary = aggregate([completed[query["_id"]] for query in queries if query["_id"] in completed])
    summary.update({"requested_questions": len(queries), "failures": failures})
    save(output.with_suffix(".summary.json"), summary)
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
        context = "\n\n".join(
            f"[{i}] {p['title']}\n{p['text']}" for i, p in enumerate(passages[:qa_top_k], 1)
        )
        try:
            response = await answer_model.ainvoke(
                [
                    ("system", system),
                    ("user", f"Context:\n{context}\n\nQuestion: {query['question']}\n\nAnswer:"),
                ]
            )
            item["answer"] = response.text.strip()
        except BadRequestError as exc:
            if not isinstance(exc.body, dict) or exc.body.get("code") != "content_filter":
                raise
            item["answer"] = ""
            item["answer_error"] = "azure_content_filter"
        item["answer_metrics"] = answer_metrics(item["answer"], answers)
    async with lock:
        with output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        completed[item["id"]] = item
        print(f"[{len(completed)}/{total}] {query['_id']}: complete", flush=True)


def run_hotpotqa_benchmark(options) -> None:
    """Download, index, and evaluate. ``--skip-index`` reuses the selected store."""
    from raganite.config import ChunkingConfig, IndexingConfig, RetrievalConfig
    from raganite.graph import KnowledgeGraph
    from raganite.retrieval import Retriever

    load_benchmark_env()
    data_dir = Path(options["data_dir"])
    queries, corpus = load_hotpotqa(data_dir)
    selected = queries[: options["questions"]]
    documents, title_map = corpus_documents(corpus)
    dimensions = options["dimensions"]
    if dimensions is None:
        dimensions = int(
            os.getenv("AVESTA_EMBEDDING_DIM") or os.getenv("RAGANITE_EMBEDDING_DIMENSIONS") or 1536
        )
    retrieval = RetrievalConfig(
        workers=options["concurrency"],
        top_k=options["top_k"],
        triple_top_k=options["triple_top_k"],
        filtered_triple_k=options["filtered_triple_k"],
        expansion_depth=options["expansion_depth"],
        expansion_limit=options["expansion_limit"],
        degree_penalty_weight=options["degree_penalty_weight"],
        graph_weight=options["graph_weight"],
        rrf_k=options["rrf_k"],
        rerank_top_k=options["rerank_top_k"],
        use_triples=options["mode"] != "direct",
        filter_triples=options["mode"] == "full",
        rerank_passages=options["mode"] == "full",
    )
    config = store_config(
        options["backend"],
        sqlite_path=Path(options["sqlite_path"]),
        neo4j_uri=options["neo4j_uri"],
        username=options["neo4j_username"],
        password=options["neo4j_password"],
        database=options["neo4j_database"],
        dimensions=dimensions,
        namespace="hotpotqa",
        retrieval=retrieval,
        layout=options.get("neo4j_layout", "owned"),
        vector_index=options.get("vector_index"),
        fulltext_index=options.get("fulltext_index"),
        fact_vector_index=options.get("fact_vector_index"),
    )
    config = config.model_copy(
        update={
            "indexing": IndexingConfig(
                workers=options["index_workers"],
                chunking=ChunkingConfig(chunk_size=100_000, chunk_overlap=0),
            )
        }
    )
    embeddings, ranking, answer, http_clients = _clients(dimensions)
    if options["retrieval_only"]:
        answer = None
    graph = KnowledgeGraph(embeddings=embeddings, llm=ranking, config=config)
    retriever = None
    try:
        report = maybe_index(
            graph.indexer, documents, source="hotpotqa", skip_index=options["skip_index"]
        )
        if report is not None:
            print(
                f"Indexed {len(report.indexed_files)} passages, "
                f"skipped {len(report.skipped_files)}, failed {len(report.failed_files)}",
                flush=True,
            )
        retriever = Retriever(
            graph.backend,
            embeddings,
            None if options["mode"] != "full" else ranking,
            retrieval,
            title_by_source=title_map,
            triple_filter_prompt=(PROMPTS / "filter_hotpot_triples.txt").read_text(
                encoding="utf-8"
            ),
            passage_rerank_prompt=(PROMPTS / "rerank_hotpot_passages.txt").read_text(
                encoding="utf-8"
            ),
        )
        system = (PROMPTS / "answer_hotpot_question.txt").read_text(encoding="utf-8")
        output = Path(options["output"])
        protocol = {
            "backend": options["backend"],
            "mode": options["mode"],
            "questions": options["questions"],
            "top_k": options["top_k"],
            "qa_top_k": options["qa_top_k"],
            "dimensions": dimensions,
            "namespace": "hotpotqa",
        }
        manifest = output.with_suffix(".manifest.json")
        if (
            output.exists()
            and manifest.exists()
            and json.loads(manifest.read_text(encoding="utf-8")).get("protocol") != protocol
        ):
            raise ValueError("Existing output uses another protocol; select a new output path")
        if not output.exists():
            save(manifest, {"protocol": protocol})
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
            )
        )
    finally:
        if retriever is not None:
            retriever.close()
        graph.close()
        for client in http_clients:
            closer = getattr(client, "close", None)
            if closer is None:
                continue
            try:
                closer()
            except RuntimeError:
                pass
