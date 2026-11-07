from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Adapt these three things to your own task:
#   1. TASK_FIELDS: the JSONL fields that make up one input item;
#   2. TASK_SYSTEM: the instruction given to the model;
#   3. the labels, which live in the config file under task.labels.
# Every record also needs an "id" and a "label" field.
#
# Project task: extractive QA on English TyDi QA GoldP. The "label" is the
# first gold answer span; "answers" holds all gold answers for scoring.
# ---------------------------------------------------------------------------

TASK_FIELDS = ("context", "question")

TASK_SYSTEM = (
    "Answer the question using the context. "
    "The answer is a short span copied exactly from the context. "
    "End with 'Answer: <answer>'."
)

KNOWLEDGE_SYSTEM = (
    "Answer the multiple-choice question. Choose exactly A, B, C, or D. "
    "End with 'Answer: <letter>'."
)


def field_title(field: str) -> str:
    return field.replace("_", " ").capitalize()


def task_example(record: dict[str, Any], include_answer: bool) -> str:
    text = "\n".join(f"{field_title(field)}: {record[field]}" for field in TASK_FIELDS)
    if include_answer:
        text += f"\nAnswer: {record['label']}"
    return text


def task_prompt(record: dict[str, Any], demos: list[dict[str, Any]]) -> str:
    sections = ["Examples:"]
    for index, demo in enumerate(demos, start=1):
        sections.append(f"Example {index}\n{task_example(demo, include_answer=True)}")
    if not demos:
        sections = []
    sections.append("Answer this question:\n" + task_example(record, include_answer=False))
    sections.append("Answer:")
    return "\n\n".join(sections)


def knowledge_prompt(record: dict[str, Any]) -> str:
    choices = "\n".join(
        f"{letter}. {text}" for letter, text in zip("ABCD", record["choices"])
    )
    return f"Question: {record['question']}\n{choices}\nAnswer:"


def extract_task_label(text: str, labels: list[str] | None = None) -> str | None:
    """Return the answer span from the model output, or None if it is empty.

    Kept under its original name so every condition uses the same extractor.
    `labels` is unused for QA. Drops any <think> block, then takes the text
    after the first "Answer:" (or the first non-empty line if there is none),
    up to the end of that line, without surrounding quotes or a final period.
    """
    del labels
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    explicit = re.search(r"(?i)answer\s*:\s*(.*)", text)
    if explicit:
        candidate = explicit.group(1)
    else:
        lines = [line for line in text.splitlines() if line.strip()]
        candidate = lines[0] if lines else ""
    candidate = candidate.splitlines()[0] if candidate else ""
    candidate = candidate.strip().rstrip(".").strip("\"'*` ").rstrip(".").strip()
    return candidate or None


def extract_choice(text: str) -> str | None:
    explicit = re.search(r"(?i)(?:answer|choice)\s*:\s*([A-D])\b", text)
    if explicit:
        return explicit.group(1).upper()
    bare = re.search(r"(?i)\b([A-D])\b", text.strip())
    return bare.group(1).upper() if bare else None


def messages_for_sft(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": TASK_SYSTEM},
        {"role": "user", "content": task_prompt(record, [])},
        {"role": "assistant", "content": f"Answer: {record['label']}"},
    ]
