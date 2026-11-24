"""Download LoCoMo, index one namespace per conversation, then score questions."""

import asyncio
import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from benchmark.fetch import LOCOMO_URL, ensure_download
from benchmark.hotpotqa_flow import RateLimitPause, _clients, _rate_limited, load_benchmark_env
from benchmark.locomo import selected_samples, session_documents
from benchmark.locomo_mem0 import (
    Mem0Evaluation,
    load_mem0_prompts,
    mem0_f1,
    non_adversarial_samples,
    summarize_runs,
)
from benchmark.locomo_prompt import judge_label, judge_prompt, memory_prompt, paper_prompt
from benchmark.metrics import benchmark_usage_report
from benchmark.results import read_rows, save
from benchmark.store import store_config

logger = logging.getLogger(__name__)

CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
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
    expected = gold.split(";", 1)[0] if category == 2 else gold
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
    qa_top_k: int = 5,
    answer_template: str | None = None,
    mem0: Mem0Evaluation | None = None,
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
                        qa_top_k=qa_top_k,
                        answer_template=answer_template,
                        mem0=mem0,
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
    qa_top_k: int = 5,
    answer_template: str | None = None,
    mem0: Mem0Evaluation | None = None,
):
    from raganite.clients.llm.usage import track_run, usage_stage

    question_id = f"{sample_id}:{index}"
    try:
        with track_run("benchmark") as usage:
            passages = [
                {
                    "source": item.source,
                    "title": item.title,
                    "text": item.text,
                    "score": item.score,
                    "metadata": item.metadata,
                }
                for item in await retriever.aretrieve(question["question"])
            ]
            gold = str(question.get("answer") or "")
            category = int(question["category"])
            context_passages = passages[:qa_top_k]
            chunks = [item["text"] for item in context_passages]
            messages = [
                ("system", memory_prompt(answer_template, chunks, question["question"]))
                if answer_template is not None
                else ("user", paper_prompt("\n\n".join(chunks), question))
            ]
            with usage_stage("benchmark.answer"):
                response = await answer_model.ainvoke(
                    messages, **({"temperature": 0.0} if mem0 else {})
                )
            prediction = response.text if mem0 else response.text.strip()
            evidence = [str(item) for item in question.get("evidence") or []]
            row = {
                "id": question_id,
                "sample_id": sample_id,
                "question": question["question"],
                "category": category,
                "category_name": CATEGORY_NAMES.get(category, str(category)),
                "gold": gold,
                "prediction": prediction,
                "qa_f1": mem0_f1(prediction, gold) if mem0 else _qa_f1(prediction, gold, category),
                "exact_match": (
                    float(prediction.strip().lower() == gold.strip().lower())
                    if mem0 else float(_tokens(prediction) == _tokens(gold))
                ),
                "evidence_recall_at_5": _evidence_recall(passages, evidence, 5),
                "evidence_recall_at_10": _evidence_recall(passages, evidence, 10),
                "evidence_recall_at_20": _evidence_recall(passages, evidence, 20),
                "evidence_recall_in_answer_context": _evidence_recall(context_passages, evidence, qa_top_k),
                "answer_context_chunks": len(context_passages),
                "beam_start_candidate_count": int(
                    (passages[0]["metadata"] or {}).get("beam_start_candidate_count", 0)
                ) if passages else 0,
                "beam_bridge_candidate_count": int(
                    (passages[0]["metadata"] or {}).get("beam_bridge_candidate_count", 0)
                ) if passages else 0,
                "beam_candidate_count": int(
                    (passages[0]["metadata"] or {}).get("beam_candidate_count", 0)
                ) if passages else 0,
            }
            if answer_template is not None:
                row["answer_template_sha256"] = hashlib.sha256(answer_template.encode()).hexdigest()
            if mem0:
                row["mem0_judge_sha256"] = mem0.fingerprint
                with usage_stage("benchmark.judge"):
                    row["llm_judge"] = await mem0.judge(
                        answer_model, str(question["question"]), gold, prediction
                    )
            elif llm_judge:
                judge_gold = "Not mentioned in the conversation" if category == 5 else gold
                if category == 2:
                    judge_gold = judge_gold.split(";", 1)[0].strip()
                with usage_stage("benchmark.judge"):
                    messages = [
                        ("user", judge_prompt(str(question["question"]), judge_gold, prediction))
                    ]
                    for attempt in range(3):
                        options = {"response_format": {"type": "json_object"}} if attempt else {}
                        verdict = await answer_model.ainvoke(messages, **options)
                        try:
                            row["llm_judge"] = judge_label(verdict.text)
                            break
                        except (ValueError, KeyError, TypeError):
                            if attempt == 2:
                                raise
                            logger.warning("%s: malformed judge reply, retrying", question_id)
        row["usage"] = asdict(usage)
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


def _save_locomo_usage(output: Path, indexing_usage: list[dict]) -> None:
    """Write indexing usage, question usage, and their sum beside the answers."""
    from raganite.clients.llm.usage import combine_usage

    question_usage = [
        row["usage"] for row in read_rows(output).values() if isinstance(row.get("usage"), dict)
    ]
    if not indexing_usage and not question_usage:
        return
    save(
        output.with_suffix(".usage.json"),
        benchmark_usage_report(
            combine_usage(indexing_usage) if indexing_usage else None,
            combine_usage(question_usage) if question_usage else None,
        ),
    )


async def _index_sessions(indexer, documents, namespace, indexing_usage):
    remaining = documents
    for attempt in range(1, 5):
        report = await indexer.index_documents_async(remaining, source=namespace)
        indexing_usage.append(asdict(report.usage))
        print(
            f"Indexed {namespace}: {len(report.indexed_files)} sessions, "
            f"{len(report.skipped_files)} unchanged, "
            f"{report.usage.total_tokens} tokens in {report.usage.elapsed_seconds:.1f}s",
            flush=True,
        )
        failed = dict(report.failed_files)
        if not failed or attempt == 4:
            return report
        logger.debug("Indexing failures in %s: %s", namespace, failed)
        print(
            f"{namespace}: retrying {len(failed)} failed sessions in 30s "
            f"(attempt {attempt + 1}/4)",
            flush=True,
        )
        remaining = [doc for doc in remaining if doc.metadata["path"] in failed]
        await asyncio.sleep(30)


def run_locomo_benchmark(options) -> None:
    """Download LoCoMo, index each needed conversation, then answer its questions."""
    load_benchmark_env()
    if not 1 <= options.get("qa_top_k", 5) <= options["top_k"]:
        raise ValueError("--qa-top-k must be between 1 and --top-k")
    data = load_locomo(Path(options["data_dir"]))
    available = sum(len(sample["qa"]) for sample in data)
    if options["questions"] > available:
        raise ValueError(
            f"Dataset has {available} questions, which is below {options['questions']}"
        )
    if options.get("mem0") and (options.get("answer_template") or options.get("mem0_judge_template")):
        raise ValueError("--mem0 cannot be combined with custom templates")
    if options.get("mem0") or options.get("mem0_judge_template"):
        if not options.get("mem0") and not options.get("answer_template"):
            raise ValueError("--mem0-judge-template requires --answer-template")
        chosen = non_adversarial_samples(data, options["questions"])
    else:
        chosen = selected_samples(data, options["questions"])
    from raganite.config.loading import load_config

    dimensions = options["dimensions"] or load_config().embedding_dimensions
    _run_locomo_conversations(options, chosen, dimensions)


def _run_locomo_conversations(options, chosen, dimensions) -> None:
    asyncio.run(_run_locomo_conversations_async(options, chosen, dimensions))


async def _run_locomo_conversations_async(options, chosen, dimensions) -> None:
    from raganite.config import RetrievalConfig
    from raganite.graph import KnowledgeGraph

    qa_top_k = options.get("qa_top_k", 5)
    if options.get("mem0"):
        answer_template, judge_template = load_mem0_prompts(Path(options["data_dir"]))
    else:
        template_path = options.get("answer_template")
        answer_template = template_path.read_text(encoding="utf-8") if template_path else None
        judge_path = options.get("mem0_judge_template")
        judge_template = judge_path.read_text(encoding="utf-8") if judge_path else None
    mem0 = Mem0Evaluation(judge_template) if judge_template is not None else None
    template_hash = (
        hashlib.sha256(answer_template.encode()).hexdigest() if answer_template is not None else None
    )
    if answer_template is not None:
        memory_prompt(answer_template, [], "")
    retrieval = RetrievalConfig(
        workers=options["concurrency"],
        top_k=options["top_k"],
        beam_start_candidates=max(RetrievalConfig().beam_start_candidates, qa_top_k),
    )
    output = Path(options["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    indexing_usage: list[dict] = []
    done: set[str] = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("answer_template_sha256") != template_hash:
                    raise ValueError(f"{output} uses a different answer prompt; choose a new output file")
                if row.get("mem0_judge_sha256") != (mem0.fingerprint if mem0 else None):
                    raise ValueError(f"{output} uses a different judge protocol; choose a new output file")
                done.add(row["id"])
    for sample, pairs in chosen:
        namespace = f"{options.get('namespace_prefix', '')}{sample['sample_id']}"
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
        )
        config.indexing.workers = options.get("index_workers", options["concurrency"])
        config.indexing.schema_mode = options.get("schema_mode", "free")
        if options["skip_index"]:
            config.backend.read_only = True
        embeddings, ranking, answer = _clients(dimensions, answer_max_tokens=256)
        graph = KnowledgeGraph(embeddings=embeddings, llm=ranking, config=config)
        try:
            report = None
            if not options["skip_index"]:
                print(f"Indexing {namespace}: {len(documents)} sessions", flush=True)
                report = await _index_sessions(graph.indexer, documents, namespace, indexing_usage)
                if report.failed_files:
                    failed = "; ".join(f"{path}: {reason}" for path, reason in report.failed_files)
                    raise RuntimeError(
                        f"Indexing {namespace} failed for {len(report.failed_files)} sessions: {failed}"
                    )
            await _evaluate(
                sample,
                pairs,
                graph.retriever,
                answer_model=answer,
                output=output,
                lock=asyncio.Lock(),
                done=done,
                concurrency=options["concurrency"],
                llm_judge=bool(options.get("llm_judge")),
                qa_top_k=qa_top_k,
                answer_template=answer_template,
                mem0=mem0,
            )
        finally:
            graph.close()
    _save_locomo_usage(output, indexing_usage)
    if mem0 is not None:
        expected_ids = {f"{sample['sample_id']}:{index}" for sample, pairs in chosen for index, _ in pairs}
        summary = summarize_runs([list(read_rows(output).values())], expected_ids)
        save(output.with_suffix(".summary.json"), summary)
        print(json.dumps(summary, indent=2))
