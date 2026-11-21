import pytest
from benchmark.locomo_mem0 import (
    load_mem0_prompts,
    mem0_f1,
    non_adversarial_samples,
    summarize_runs,
)


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
