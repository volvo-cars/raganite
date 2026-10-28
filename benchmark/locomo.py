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
    return {
        "questions": len(rows),
        "qa_f1": mean(row["qa_f1"] for row in rows),
        "exact_match": mean(row["exact_match"] for row in rows),
        "evidence_recall_at_5": mean(row["evidence_recall_at_5"] for row in rows),
        "evidence_recall_at_10": mean(row["evidence_recall_at_10"] for row in rows),
    }


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


def score_locomo(data_path: Path, results_path: Path, questions: int) -> str:
    data = json.loads(data_path.read_text(encoding="utf-8"))
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
    return json.dumps(report, indent=2)
