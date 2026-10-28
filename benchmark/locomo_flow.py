"""Download LoCoMo, index one namespace per conversation, then score questions."""

import asyncio
import json
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path

from benchmark.fetch import LOCOMO_URL, ensure_download
from benchmark.hotpotqa_flow import RateLimitPause, _clients, _rate_limited, load_benchmark_env
from benchmark.locomo import selected_samples, session_documents
from benchmark.locomo_prompt import judge_label, judge_prompt, paper_prompt
from benchmark.store import maybe_index, store_config

logger = logging.getLogger(__name__)

CATEGORY_NAMES = {
    1: "multi_hop",
    2: "single_hop",
    3: "temporal",
    4: "open_domain",
    5: "adversarial",
}


def load_locomo(data_dir: Path) -> list[dict]:
    path = ensure_download(data_dir / "locomo10.json", LOCOMO_URL)
    data = json.loads(path.read_text(encoding="utf-8"))
    if len(data) != 10:
        raise ValueError("Expected the 10-conversation LoCoMo release")
    return data


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _f1(prediction: str, gold: str) -> float:
    predicted, expected = Counter(_tokens(prediction)), Counter(_tokens(gold))
    overlap = sum((predicted & expected).values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(sum(predicted.values()), 1)
    recall = overlap / max(sum(expected.values()), 1)
    return 2 * precision * recall / (precision + recall)


def _qa_f1(prediction: str, gold: str, category: int) -> float:
    if category == 5:
        lowered = prediction.lower()
        return float("no information available" in lowered or "not mentioned" in lowered)
    expected = gold.split(";", 1)[0] if category == 3 else gold
    parts = (
        [part.strip() for part in expected.split(",") if part.strip()]
        if category == 1
        else [expected]
    )
    return max(_f1(prediction, part) for part in parts)


def _evidence_recall(passages: list[dict], evidence: list[str], cutoff: int) -> float:
    if not evidence:
        return 0.0
    text = "\n".join(passage["text"] for passage in passages[:cutoff])
    return sum(dialog_id in text for dialog_id in evidence) / len(evidence)


async def _evaluate(
    sample,
    pairs,
    retriever,
    *,
    answer_model,
    output,
    lock,
    done: set[str],
    concurrency: int,
    llm_judge: bool,
):
    sample_id = str(sample["sample_id"])
    semaphore = asyncio.Semaphore(concurrency)
    pause = RateLimitPause()

    async def evaluate(index, question):
        question_id = f"{sample_id}:{index}"
        if question_id in done:
            return
        async with semaphore:
            if question_id in done:
                return
            for attempt in range(8):
                await pause.wait()
                try:
                    await _answer_one(
                        sample_id,
                        index,
                        question,
                        retriever,
                        answer_model=answer_model,
                        output=output,
                        lock=lock,
                        done=done,
                        llm_judge=llm_judge,
                    )
                    return
                except Exception as exc:
                    logger.debug("%s failed: %s", question_id, exc, exc_info=True)
                    if _rate_limited(exc) and attempt < 7:
                        await pause.trigger(question_id)
                        continue
                    raise

    await asyncio.gather(*(evaluate(index, question) for index, question in pairs))


async def _answer_one(
    sample_id,
    index,
    question,
    retriever,
    *,
    answer_model,
    output,
    lock,
    done: set[str],
    llm_judge: bool,
):
    question_id = f"{sample_id}:{index}"
    try:
        passages = [
            {"source": item.source, "title": item.title, "text": item.text, "score": item.score}
            for item in await retriever.aretrieve(question["question"])
        ]
        gold = str(question.get("answer") or "")
        category = int(question["category"])
        context = "\n\n".join(item["text"] for item in passages[:5])
        response = await answer_model.ainvoke([("user", paper_prompt(context, question))])
        prediction = response.text.strip()
        evidence = [str(item) for item in question.get("evidence") or []]
        row = {
            "id": question_id,
            "sample_id": sample_id,
            "question": question["question"],
            "category": category,
            "category_name": CATEGORY_NAMES.get(category, str(category)),
            "gold": gold,
            "prediction": prediction,
            "qa_f1": _qa_f1(prediction, gold, category),
            "exact_match": float(_tokens(prediction) == _tokens(gold)),
            "evidence_recall_at_5": _evidence_recall(passages, evidence, 5),
            "evidence_recall_at_10": _evidence_recall(passages, evidence, 10),
        }
        if llm_judge:
            judge_gold = "Not mentioned in the conversation" if category == 5 else gold
            if category == 3:
                judge_gold = judge_gold.split(";", 1)[0].strip()
            verdict = await answer_model.ainvoke(
                [("user", judge_prompt(str(question["question"]), judge_gold, prediction))],
            )
            row["llm_judge"] = judge_label(verdict.text)
        async with lock:
            if question_id in done:
                return
            with output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            done.add(question_id)
            print(f"[{len(done)}] {question_id}: complete", flush=True)
    except Exception as exc:
        if not _rate_limited(exc):
            print(f"{question_id}: failed ({type(exc).__name__})", flush=True)
        raise


def _close_http(http_clients) -> None:
    for client in http_clients:
        closer = getattr(client, "close", None)
        if closer is None:
            continue
        try:
            closer()
        except RuntimeError:
            pass


def run_locomo_benchmark(options) -> None:
    """Download LoCoMo, index each needed conversation, then answer its questions."""
    load_benchmark_env()
    data = load_locomo(Path(options["data_dir"]))
    available = sum(len(sample["qa"]) for sample in data)
    if options["questions"] > available:
        raise ValueError(
            f"Dataset has {available} questions, which is below {options['questions']}"
        )
    chosen = selected_samples(data, options["questions"])
    dimensions = options["dimensions"] or int(
        os.getenv("AVESTA_EMBEDDING_DIM") or os.getenv("RAGANITE_EMBEDDING_DIMENSIONS") or 1536
    )
    _run_locomo_conversations(options, chosen, dimensions)


def _run_locomo_conversations(options, chosen, dimensions) -> None:
    from raganite.config import RetrievalConfig
    from raganite.graph import KnowledgeGraph
    from raganite.retrieval import Retriever

    retrieval = RetrievalConfig(
        workers=options["concurrency"], top_k=options["top_k"], rerank_top_k=20
    )
    output = Path(options["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    for sample, pairs in chosen:
        namespace = str(sample["sample_id"])
        documents = session_documents(sample)
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
            layout=options.get("neo4j_layout", "owned"),
            vector_index=options.get("vector_index"),
            fulltext_index=options.get("fulltext_index"),
            fact_vector_index=options.get("fact_vector_index"),
        )
        config.indexing.workers = options["concurrency"]
        config.indexing.schema_mode = options.get("schema_mode", "free")
        # Indexing and answering each open their own event loop. A shared async
        # HTTP client is bound to the first loop and fails on the second.
        embeddings, ranking, answer, http_clients = _clients(dimensions, answer_max_tokens=256)
        graph = KnowledgeGraph(embeddings=embeddings, llm=ranking, config=config)
        retriever = None
        try:
            if not options["skip_index"]:
                print(f"Indexing {namespace}: {len(documents)} sessions", flush=True)
            report = None
            for attempt in range(4):
                report = maybe_index(
                    graph.indexer,
                    documents,
                    source=namespace,
                    skip_index=options["skip_index"],
                )
                if report is None or not report.failed_files:
                    break
                for path, reason in report.failed_files:
                    logger.debug("%s failed: %s", path, reason)
                print(
                    f"{namespace}: {len(report.failed_files)} sessions failed, pausing 30s "
                    f"(attempt {attempt + 1}/4)",
                    flush=True,
                )
                time.sleep(30)
            if report is not None:
                print(
                    f"Indexed {namespace}: {len(report.indexed_files)} sessions, "
                    f"{len(report.skipped_files)} unchanged",
                    flush=True,
                )
                if report.failed_files:
                    failed = "; ".join(f"{path}: {reason}" for path, reason in report.failed_files)
                    raise RuntimeError(
                        f"Indexing {namespace} failed for {len(report.failed_files)} sessions: {failed}"
                    )
            _close_http(http_clients)
            embeddings, ranking, answer, http_clients = _clients(dimensions, answer_max_tokens=256)
            retriever = Retriever(graph.backend, embeddings, ranking, retrieval)
            asyncio.run(
                _evaluate(
                    sample,
                    pairs,
                    retriever,
                    answer_model=answer,
                    output=output,
                    lock=asyncio.Lock(),
                    done=done,
                    concurrency=options["concurrency"],
                    llm_judge=bool(options.get("llm_judge")),
                )
            )
        finally:
            if retriever is not None:
                retriever.close()
            graph.close()
            _close_http(http_clients)
