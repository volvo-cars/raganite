import json
from pathlib import Path

from benchmark.cli import main
from benchmark.locomo import score_locomo
from click.testing import CliRunner


def test_benchmark_help_lists_both_datasets():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "hotpotqa" in result.output
    assert "locomo" in result.output


def test_hotpotqa_help_lists_modes():
    result = CliRunner().invoke(main, ["hotpotqa", "--help"])
    assert result.exit_code == 0
    assert "triple-unfiltered" in result.output
    assert "--compare-only" in result.output


def test_locomo_score_uses_conversation_order(tmp_path: Path):
    data = tmp_path / "locomo.json"
    results = tmp_path / "results.jsonl"
    data.write_text(
        json.dumps(
            [
                {"sample_id": "conv-a", "qa": [{}, {}]},
                {"sample_id": "conv-b", "qa": [{}]},
            ]
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "id": "conv-b:0",
            "category_name": "temporal",
            "qa_f1": 0.0,
            "exact_match": 0.0,
            "evidence_recall_at_5": 0.0,
            "evidence_recall_at_10": 0.0,
        },
        {
            "id": "conv-a:0",
            "category_name": "multi_hop",
            "qa_f1": 1.0,
            "exact_match": 1.0,
            "evidence_recall_at_5": 1.0,
            "evidence_recall_at_10": 1.0,
        },
        {
            "id": "conv-a:1",
            "category_name": "multi_hop",
            "qa_f1": 0.0,
            "exact_match": 0.0,
            "evidence_recall_at_5": 0.0,
            "evidence_recall_at_10": 0.0,
        },
    ]
    results.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = json.loads(score_locomo(data, results, 2))
    assert report["questions"] == 2
    assert report["first_id"] == "conv-a:0"
    assert report["last_id"] == "conv-a:1"
    assert report["overall"]["qa_f1"] == 0.5
    assert "temporal" not in report["by_category"]
