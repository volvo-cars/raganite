"""LoCoMo question wording from the published evaluation harness, and its judge label."""

import hashlib
import json
import re

# snap-research/locomo task_eval/gpt_utils.py. The leading newline is part of the published string.
QA_PROMPT = """
Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {} Short answer:
"""

QA_PROMPT_CAT_5 = """
Based on the above context, answer the following question.

Question: {} Short answer:
"""

REFUSAL = "Not mentioned in the conversation"

# Mem0's LoCoMo judge. It accepts a different wording when it is the same fact or the same date.
JUDGE_PROMPT = """Your task is to label an answer to a question as "CORRECT" or "WRONG". You will be given the following data: (1) a question (posed by one user to another user), (2) a 'gold' (ground truth) answer, (3) a generated answer which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations. The gold answer will usually be a concise and short answer that includes the referenced topic, for example:

Question: Do you remember what I got the last time I went to Hawaii?

Gold answer: A shell necklace

The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like 'last Tuesday' or 'next month'), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., 'May 7th' vs '7 May'), consider it CORRECT if it's the same date.

Now it's time for the real question:

Question: {question}

Gold answer: {gold}

Generated answer: {prediction}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label"."""


def asked_question(question: dict) -> str:
    """Question text the published harness sends, including its category suffixes."""
    text = str(question["question"])
    category = int(question["category"])
    if category == 2:
        return text + " Use DATE of CONVERSATION to answer with an approximate date."
    if category == 5:
        gold = str(question.get("answer") or "")
        refusal_first = int(hashlib.sha256(text.encode()).hexdigest(), 16) % 2 == 0
        options = (REFUSAL, gold) if refusal_first else (gold, REFUSAL)
        return text + f" Select the correct answer: (a) {options[0]} (b) {options[1]}. "
    return text


def paper_prompt(context: str, question: dict) -> str:
    """One user message: retrieved context, then the published instruction."""
    template = QA_PROMPT_CAT_5 if int(question["category"]) == 5 else QA_PROMPT
    return context + "\n\n" + template.replace("{}", asked_question(question), 1)


def judge_prompt(question: str, gold: str, prediction: str) -> str:
    return (
        JUDGE_PROMPT.replace("{question}", question)
        .replace("{gold}", gold)
        .replace("{prediction}", prediction)
    )


def judge_label(text: str) -> int:
    """1 when the judge says CORRECT, 0 when it says WRONG."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    payload = json.loads(match.group(0) if match else text)
    label = str(payload["label"]).strip().upper()
    if label not in {"CORRECT", "WRONG"}:
        raise ValueError(f"Judge label must be CORRECT or WRONG, got {label!r}")
    return int(label == "CORRECT")
