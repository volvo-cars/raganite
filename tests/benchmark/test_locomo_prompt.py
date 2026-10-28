"""LoCoMo asks the paper's question, and the LLM judge is optional."""

from benchmark.cli import main
from benchmark.locomo_prompt import asked_question, judge_label, paper_prompt
from click.testing import CliRunner


def test_the_paper_prompt_asks_for_a_short_phrase():
    question = {"question": "Where did she move?", "answer": "the city", "category": 1}
    prompt = paper_prompt("DATE: 8 May, 2023\nCaroline said she moved.", question)
    assert prompt.startswith("DATE: 8 May, 2023\nCaroline said she moved.")
    assert "write an answer in the form of a short phrase" in prompt
    assert "Answer with exact words from the context whenever possible." in prompt
    assert "Question: Where did she move? Short answer:" in prompt


def test_category_two_asks_for_the_conversation_date():
    question = {"question": "When did Caroline go?", "answer": "7 May 2023", "category": 2}
    assert asked_question(question).endswith(
        "Use DATE of CONVERSATION to answer with an approximate date."
    )


def test_adversarial_questions_offer_the_official_choice():
    question = {"question": "What is her dog's name?", "answer": "Luna", "category": 5}
    prompt = paper_prompt("context", question)
    assert "Based on the above context, answer the following question." in prompt
    assert "Select the correct answer:" in prompt
    assert "Not mentioned in the conversation" in prompt
    assert "Luna" in prompt


def test_the_judge_reads_a_correct_or_wrong_label():
    assert judge_label('{"label": "CORRECT"}') == 1
    assert judge_label('{"label": "WRONG"}') == 0


def test_the_cli_can_turn_the_judge_on():
    result = CliRunner().invoke(main, ["locomo", "run", "--help"])
    assert result.exit_code == 0
    assert "--llm-judge" in result.output
