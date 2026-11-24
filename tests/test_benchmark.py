import pytest
from benchmark.cli import main
from benchmark.hipporag import normalize_questions
from benchmark.hotpotqa_flow import corpus_documents
from benchmark.locomo_mem0 import (
    load_mem0_prompts,
    mem0_f1,
    non_adversarial_samples,
    summarize_runs,
)
from benchmark.metrics import passage_retrieval_metrics
from click.testing import CliRunner


@pytest.mark.parametrize(
    ("dataset", "fields", "expected_id"),
    [
        (
            "nq",
            {
                "reference": ["A", "Alias"],
                "contexts": [
                    {"title": "Gold", "text": "First Second", "is_supporting": True},
                    {"title": "Other", "text": "Wrong", "is_supporting": False},
                ],
            },
            "nq:0",
        ),
        (
            "popqa",
            {
                "id": 12,
                "possible_answers": '["A", "Alias"]',
                "paragraphs": [
                    {"title": "Gold", "text": "First Second", "is_supporting": True},
                ],
            },
            "12",
        ),
        (
            "musique",
            {
                "id": "m",
                "answer": "A",
                "answer_aliases": ["Alias"],
                "paragraphs": [
                    {"title": "Gold", "paragraph_text": "First Second", "is_supporting": True},
                ],
            },
            "m",
        ),
        (
            "2wiki",
            {
                "_id": "w",
                "answer": ["A", "Alias"],
                "supporting_facts": [["Gold", 1], ["Gold", 2]],
                "context": [["Gold", ["First", "Second"]]],
            },
            "w",
        ),
        (
            "hotpotqa",
            {
                "_id": "h",
                "answer": ["A", "Alias"],
                "supporting_facts": [["Gold", 1]],
                "context": [["Gold", ["First ", "Second"]]],
            },
            "h",
        ),
    ],
)
def test_hipporag_schema_adapters(dataset, fields, expected_id):
    [row] = normalize_questions(dataset, [{"question": "Which?", **fields}])
    assert row == {
        "_id": expected_id,
        "question": "Which?",
        "answer": ["A", "Alias"],
        "gold_titles": ["Gold"],
        "gold_passages": ["Gold\nFirst Second"],
    }


def test_hipporag_corpus_retains_passages_with_the_same_title():
    docs, titles = corpus_documents(
        [
            {"title": "Shared", "text": "First"},
            {"title": "Shared", "text": "Second"},
        ]
    )
    assert [doc.page_content for doc in docs] == ["Shared\nFirst", "Shared\nSecond"]
    assert titles == {"doc_00000.txt": "Shared", "doc_00001.txt": "Shared"}


def test_passage_recall_does_not_credit_the_wrong_passage_with_same_title():
    assert passage_retrieval_metrics(
        ["Shared\nWrong", "Shared\nRight"], ["Shared\nRight"], cutoffs=(1, 2)
    ) == {"recall@1": 0.0, "all_support@1": 0.0, "recall@2": 1.0, "all_support@2": 1.0}


@pytest.mark.parametrize("dataset", ["hotpotqa", "musique"])
def test_qa_uses_dataset_specific_default_paths(monkeypatch, dataset):
    options = {}
    monkeypatch.setattr("benchmark.hotpotqa_flow.run_hotpotqa_benchmark", options.update)
    result = CliRunner().invoke(main, ["qa", "--dataset", dataset, "--questions", "1"])
    assert result.exit_code == 0, result.output
    assert options["namespace"] == f"{dataset}_title_text"
    assert options["output"].name == f"{dataset}.jsonl"
    assert options["sqlite_path"].name == f"{dataset}.sqlite3"


def test_qa_runs_full_mem0_locomo(monkeypatch, tmp_path):
    options = {}
    monkeypatch.setattr("benchmark.locomo_flow.run_locomo_benchmark", options.update)
    result = CliRunner().invoke(
        main,
        [
            "qa", "--dataset", "locomo", "--mem0", "--questions", "1540",
            "--top-k", "60", "--qa-top-k", "60", "--data-dir", str(tmp_path),
            "--index-workers", "3", "--concurrency", "5",
        ],
    )
    assert result.exit_code == 0, result.output
    assert options["mem0"] is True
    assert options["questions"] == 1540
    assert options["sqlite_path"] == tmp_path / "locomo.sqlite3"
    assert options["output"].name == "locomo.jsonl"
    assert options["index_workers"] == 3
    assert options["concurrency"] == 5


@pytest.mark.parametrize("command", [["qa", "--dataset", "locomo"], ["locomo", "run"]])
def test_locomo_cli_keeps_protocol_defaults(monkeypatch, command):
    options = {}
    monkeypatch.setattr("benchmark.locomo_flow.run_locomo_benchmark", options.update)
    result = CliRunner().invoke(main, command)
    assert result.exit_code == 0, result.output
    assert options["questions"] == 100
    assert options["top_k"] == 20
    assert options["schema_mode"] == "free"


@pytest.mark.parametrize(
    ("dataset", "args", "message"),
    [
        ("hotpotqa", ["--questions", "1001"], "at most 1000"),
        ("popqa", ["--mem0"], "require --dataset locomo"),
        ("locomo", ["--retrieval-only"], "not supported for LoCoMo"),
    ],
)
def test_qa_rejects_unsupported_dataset_options(dataset, args, message):
    result = CliRunner().invoke(main, ["qa", "--dataset", dataset, *args])
    assert result.exit_code == 2
    assert message in result.output


def test_mem0_reads_literal_prompts_without_executing_source(tmp_path):
    (tmp_path / "mem0-prompts.py").write_text(
        'raise RuntimeError("must not run")\nANSWER_PROMPT = "\\n{question}\\n"\n'
    )
    (tmp_path / "mem0-judge.py").write_text('ACCURACY_PROMPT = "{gold_answer}"\n')
    assert load_mem0_prompts(tmp_path) == ("\n{question}\n", "{gold_answer}")


def test_mem0_rejects_nonliteral_prompts(tmp_path):
    (tmp_path / "mem0-prompts.py").write_text('ANSWER_PROMPT = str("not a literal")\n')
    with pytest.raises(ValueError, match="literal ANSWER_PROMPT"):
        load_mem0_prompts(tmp_path)


def test_mem0_limit_preserves_original_question_ids():
    data = [
        {"sample_id": "first", "qa": [{"category": 5}, {"category": 1}]},
        {"sample_id": "second", "qa": [{"category": 2}, {"category": 3}]},
    ]
    selected = non_adversarial_samples(data, 2)
    assert [(s["sample_id"], [i for i, _ in pairs]) for s, pairs in selected] == [
        ("first", [1]),
        ("second", [0]),
    ]


def test_mem0_summary_uses_set_f1_and_requires_complete_results():
    assert mem0_f1("Paris, Paris!", "paris") == 1
    row = {"id": "sample:1", "category_name": "Single hop", "llm_judge": 1, "qa_f1": 0.5}
    summary = summarize_runs([[row]], {"sample:1"})
    assert summary["summary"]["overall"]["llm_judge"] == {"mean": 1, "stddev": 0}
    assert summary["summary"]["Single hop"]["qa_f1"]["mean"] == 0.5
    with pytest.raises(ValueError, match="exactly once"):
        summarize_runs([[row]], {"sample:1", "sample:2"})
