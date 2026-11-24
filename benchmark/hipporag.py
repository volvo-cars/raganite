"""Normalize the pinned HippoRAG 2 QA datasets for the shared evaluator."""

import json
from pathlib import Path

from benchmark.fetch import HOTPOT_RELEASE, ensure_download

DATASETS = {
    "hotpotqa": ("hotpotqa", 9811),
    "nq": ("nq_rear", 9633),
    "popqa": ("popqa", 8676),
    "musique": ("musique", 11656),
    "2wiki": ("2wikimultihopqa", 6119),
}


def passage_text(title: str, text: str) -> str:
    return f"{title}\n{text}"


def normalize_questions(dataset: str, rows: list[dict]) -> list[dict]:
    normalized = []
    for index, row in enumerate(rows):
        if dataset in {"hotpotqa", "2wiki"}:
            key = row["_id"]
            answer = row["answer"]
            answers = [answer] if isinstance(answer, str) else answer
            titles = {title for title, _ in row["supporting_facts"]}
            separator = "" if dataset == "hotpotqa" else " "
            gold = {
                passage_text(title, separator.join(sentences))
                for title, sentences in row["context"]
                if title in titles
            }
        else:
            if dataset == "nq":
                key = f"nq:{index}"
                answers = row["reference"]
                passages = row["contexts"]
            else:
                key = str(row["id"])
                answers = (
                    json.loads(row["possible_answers"])
                    if dataset == "popqa"
                    else [row["answer"], *row["answer_aliases"]]
                )
                passages = row["paragraphs"]
            titles = {p["title"] for p in passages if p["is_supporting"]}
            gold = {
                passage_text(p["title"], p["text"] if "text" in p else p["paragraph_text"])
                for p in passages
                if p["is_supporting"]
            }
        if not answers or not all(isinstance(a, str) for a in answers) or not gold:
            raise ValueError(f"Missing answer or supporting passages for {dataset} question {key}")
        normalized.append(
            {
                "_id": key,
                "question": row["question"],
                "answer": answers,
                "gold_titles": sorted(titles),
                "gold_passages": sorted(gold),
            }
        )
    if len({row["_id"] for row in normalized}) != len(normalized):
        raise ValueError(f"Duplicate question IDs in {dataset}")
    return normalized


def load_dataset(dataset: str, data_dir: Path) -> tuple[list[dict], list[dict]]:
    stem, passage_count = DATASETS[dataset]

    def read(filename: str) -> list[dict]:
        path = ensure_download(data_dir / filename, f"{HOTPOT_RELEASE}/{filename}")
        return json.loads(path.read_text(encoding="utf-8"))

    queries, corpus = read(f"{stem}.json"), read(f"{stem}_corpus.json")
    if len(queries) != 1000 or len(corpus) != passage_count:
        raise ValueError(f"Expected 1,000 questions and {passage_count:,} passages for {dataset}")
    return normalize_questions(dataset, queries), corpus
