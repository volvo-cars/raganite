"""HotpotQA answer, supporting-passage, and matched-question metrics."""

import re
import string
from collections import Counter
from statistics import mean


def normalize(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def _answer_score(prediction: str, answer: str) -> dict[str, float]:
    prediction, answer = normalize(prediction), normalize(answer)
    exact_match = float(prediction == answer)
    special = {"yes", "no", "noanswer"}
    predicted, expected = prediction.split(), answer.split()
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if (prediction in special or answer in special) and prediction != answer:
        overlap = 0
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"exact_match": exact_match, "f1": f1, "precision": precision, "recall": recall}


def answer_metrics(prediction: str, answers: list[str]) -> dict[str, float]:
    scores = [_answer_score(prediction, answer) for answer in answers]
    return {
        key: max((score[key] for score in scores), default=0.0)
        for key in ("exact_match", "f1", "precision", "recall")
    }


def retrieval_metrics(
    titles: list[str],
    gold_titles: list[str],
    cutoffs: tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100, 200),
) -> dict[str, float]:
    gold = {normalize(title) for title in gold_titles}
    metrics = {}
    for cutoff in cutoffs:
        found = {normalize(title) for title in titles[:cutoff]}
        metrics[f"recall@{cutoff}"] = len(gold & found) / len(gold) if gold else 0.0
        metrics[f"all_support@{cutoff}"] = float(bool(gold) and gold <= found)
    return metrics


def _average(rows: list[dict], field: str) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row.get(field, {})})
    return {key: mean(row.get(field, {}).get(key, 0.0) for row in rows) for key in keys}


def aggregate(rows: list[dict]) -> dict:
    answered = [row for row in rows if "answer" in row]
    return {
        "questions": len(rows),
        "answered_questions": len(answered),
        "qa": _average(answered, "answer_metrics"),
        "retrieval": _average(rows, "retrieval_metrics"),
    }


def _by_id(rows: list[dict]) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        if row["id"] in indexed:
            raise ValueError("Duplicate question ID in results")
        indexed[row["id"]] = row
    return indexed


def compare(candidate: list[dict], baseline: list[dict]) -> dict:
    _by_id(candidate)
    indexed = _by_id(baseline)
    matched = []
    for row in candidate:
        original = indexed.get(row["id"])
        if original is None:
            raise ValueError("Baseline is missing a candidate question")
        if any(row[key] != original[key] for key in ("question", "gold_answers", "gold_titles")):
            raise ValueError("Question or ground truth differs from baseline")
        matched.append(original)
    result, reference = aggregate(candidate), aggregate(matched)
    return {
        "questions_compared": len(candidate),
        "same_question_ids_and_ground_truth": True,
        "candidate": result,
        "baseline_matched_questions": reference,
        "delta": {
            group: {key: value - reference[group][key] for key, value in result[group].items()}
            for group in ("qa", "retrieval")
        },
        "identical_answer_count": sum(
            "answer" in row and "answer" in original and row["answer"] == original["answer"]
            for row, original in zip(candidate, matched, strict=True)
        ),
        "identical_top5_titles_count": sum(
            [passage["title"] for passage in row["retrieved"][:5]]
            == [passage["title"] for passage in original["retrieved"][:5]]
            for row, original in zip(candidate, matched, strict=True)
        ),
    }
