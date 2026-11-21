"""Score a LoCoMo JSONL, and turn conversations into indexable documents."""

import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

from langchain_core.documents import Document


def question_ids(data: list[dict], limit: int) -> list[str]:
    selected: list[str] = []
    for sample in data:
        sample_id = str(sample["sample_id"])
        for index, _question in enumerate(sample["qa"]):
            selected.append(f"{sample_id}:{index}")
            if len(selected) == limit:
                return selected
    return selected


def _metrics(rows: list[dict]) -> dict[str, float | int]:
    result = {
        "questions": len(rows),
        "qa_f1": mean(row["qa_f1"] for row in rows),
        "exact_match": mean(row["exact_match"] for row in rows),
        "evidence_recall_at_5": mean(row["evidence_recall_at_5"] for row in rows),
        "evidence_recall_at_10": mean(row["evidence_recall_at_10"] for row in rows),
    }
    for metric in (
        "evidence_recall_at_20",
        "evidence_recall_in_answer_context",
        "answer_context_chunks",
        "beam_start_candidate_count",
        "beam_bridge_candidate_count",
        "beam_candidate_count",
    ):
        if all(metric in row for row in rows):
            result[metric] = mean(row[metric] for row in rows)
    judged = [row["llm_judge"] for row in rows if row.get("llm_judge") is not None]
    if judged:
        result.update(llm_judge=mean(judged), judged_questions=len(judged))
    return result


def session_documents(sample: dict) -> list[Document]:
    """One document per session, with dialogue ids kept in the text."""
    conversation = sample["conversation"]
    keys = sorted(
        (
            key
            for key, value in conversation.items()
            if re.fullmatch(r"session_\d+", key) and isinstance(value, list)
        ),
        key=lambda key: int(key.removeprefix("session_")),
    )
    sample_id = str(sample["sample_id"])
    documents = []
    for key in keys:
        number = int(key.removeprefix("session_"))
        date = str(conversation.get(f"{key}_date_time", ""))
        lines = [f"Conversation: {sample_id}", f"Session: {number}", f"Date: {date}", ""]
        for turn in conversation[key]:
            dialog_id = str(turn.get("dia_id", ""))
            speaker = str(turn.get("speaker", "Unknown"))
            text = str(turn.get("text") or turn.get("content") or "").strip()
            caption = str(turn.get("blip_caption") or "").strip()
            if caption:
                text = f"{text} [Image caption: {caption}]".strip()
            when = f" ({date})" if date else ""
            lines.append(f"[{dialog_id}] {speaker}{when}: {text}")
        documents.append(
            Document(
                page_content="\n".join(lines),
                metadata={
                    "path": f"{sample_id}/session_{number:03d}.txt",
                    "title": f"{sample_id} session {number}",
                    "order": number,
                    "time": date,
                },
            )
        )
    return documents


def selected_samples(data: list[dict], questions: int) -> list[tuple[dict, list[tuple[int, dict]]]]:
    """Conversations that contain the first ``questions`` items, in dataset order."""
    remaining = questions
    chosen = []
    for sample in data:
        if remaining <= 0:
            break
        pairs = list(enumerate(sample["qa"]))[:remaining]
        remaining -= len(pairs)
        chosen.append((sample, pairs))
    return chosen


def score_locomo(
    data_path: Path, results_path: Path, questions: int, *, mem0: bool = False
) -> str:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    if mem0:
        from benchmark.locomo_mem0 import non_adversarial_samples

        selected = [
            f"{sample['sample_id']}:{index}"
            for sample, pairs in non_adversarial_samples(data, questions)
            for index, _ in pairs
        ]
    else:
        selected = question_ids(data, questions)
    if len(selected) < questions:
        raise ValueError(f"Dataset has {len(selected)} questions, which is below {questions}")
    found: dict[str, dict] = {}
    with results_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in selected:
                found[row["id"]] = row
    missing = [item for item in selected if item not in found]
    if missing:
        raise ValueError(f"{len(missing)} selected questions are missing from {results_path.name}")
    rows = [found[item] for item in selected]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["category_name"]].append(row)
    report = {
        "questions": len(rows),
        "first_id": selected[0],
        "last_id": selected[-1],
        "overall": _metrics(rows),
        "by_category": {name: _metrics(items) for name, items in sorted(grouped.items())},
    }
    non_adversarial = [row for row in rows if row["category_name"] != "adversarial"]
    if non_adversarial:
        report["non_adversarial"] = _metrics(non_adversarial)
    return json.dumps(report, indent=2)
