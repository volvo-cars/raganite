"""Mem0 paper evaluation protocol for Raganite LoCoMo answers."""

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

from benchmark.fetch import ensure_download
from benchmark.locomo import selected_samples

MEM0_EVALUATION = (
    "https://raw.githubusercontent.com/mem0ai/mem0/"
    "d85fcda0378e3e2ba8772705b075049484bd8fa8/evaluation"
)


def load_mem0_prompts(data_dir: Path) -> tuple[str, str]:
    """Read literal prompt constants from pinned upstream files without executing them."""
    def prompt(filename: str, source: str, name: str) -> str:
        path = ensure_download(data_dir / filename, f"{MEM0_EVALUATION}/{source}")
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in node.targets
            ):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    return node.value.value
        raise ValueError(f"Could not read literal {name} from {path}")

    return (
        prompt("mem0-prompts.py", "prompts.py", "ANSWER_PROMPT"),
        prompt("mem0-judge.py", "metrics/llm_judge.py", "ACCURACY_PROMPT"),
    )


def non_adversarial_samples(data: list[dict], questions: int):
    remaining = questions
    chosen = []
    for sample, pairs in selected_samples(data, sum(len(item["qa"]) for item in data)):
        included = [(index, qa) for index, qa in pairs if int(qa["category"]) != 5][:remaining]
        if included:
            chosen.append((sample, included))
        remaining -= len(included)
        if not remaining:
            return chosen
    raise ValueError(f"Dataset has fewer than {questions} non-adversarial questions")


def mem0_f1(prediction: str, gold: str) -> float:
    def tokens(text: str) -> set[str]:
        for mark in ".,!?":
            text = text.replace(mark, " ")
        return set(text.lower().split())

    predicted, expected = tokens(prediction), tokens(gold)
    if not predicted or not expected:
        return 0.0
    return 2 * len(predicted & expected) / (len(predicted) + len(expected))


@dataclass(frozen=True)
class Mem0Evaluation:
    prompt: str

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.prompt.encode()).hexdigest()

    async def judge(self, model, question: str, gold: str, prediction: str) -> int:
        prompt = self.prompt.format(
            question=question, gold_answer=gold, generated_answer=prediction
        )
        response = await model.ainvoke(
            [("user", prompt)], temperature=0.0, response_format={"type": "json_object"}
        )
        return int(json.loads(response.text)["label"] == "CORRECT")


def summarize_runs(runs: list[list[dict]], expected_ids: set[str]) -> dict:
    if not runs or not expected_ids:
        raise ValueError("Runs and expected question IDs are required")
    for rows in runs:
        if len(rows) != len(expected_ids) or {row["id"] for row in rows} != expected_ids:
            raise ValueError("Each run must contain every selected question exactly once")
    per_run = [
        {
            group: {
                key: mean(row[key] for row in rows if group == "overall"
                          or row["category_name"] == group)
                for key in ("llm_judge", "qa_f1")
            }
            for group in ("overall", *sorted({row["category_name"] for row in rows}))
        }
        for rows in runs
    ]
    return {
        "questions_per_run": len(expected_ids),
        "runs": per_run,
        "summary": {
            group: {
                metric: {
                    "mean": mean(values),
                    "stddev": pstdev(values),
                }
                for metric in per_run[0][group]
                for values in [[run[group][metric] for run in per_run]]
            }
            for group in per_run[0]
        },
    }
