from __future__ import annotations

import re

ACTION_VERBS = [
    "complete",
    "prepare",
    "study",
    "call",
    "send",
    "review",
    "attend",
    "upload",
    "finish",
    "submit",
    "create",
    "write",
    "read",
    "learn",
    "set up",
    "plan",
    "fix",
    "update",
    "check",
    "email",
    "organize",
    "schedule",
    "book",
    "buy",
    "pay",
    "practice",
    "research",
    "build",
    "design",
    "print",
    "join",
    "watch",
    "download",
    "install",
]


def _action_start_pattern() -> str:
    alternatives = sorted((re.escape(verb) for verb in ACTION_VERBS), key=len, reverse=True)
    return r"(?:" + "|".join(alternatives) + r")\b"


def normalize_transcript(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \n\t.,;:!?\"")


def extract_tasks(text: str) -> list[str]:
    normalized = normalize_transcript(text)
    if not normalized:
        return []

    tasks: list[str] = []
    seen: set[str] = set()
    action_start = _action_start_pattern()
    split_pattern = rf"\s+(?:and\s+then|then|also|plus|and)\s+(?={action_start})"

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    for sentence in sentences:
        for segment in re.split(split_pattern, sentence, flags=re.IGNORECASE):
            candidate = segment.strip(" ,;.!?")
            candidate = re.sub(
                r"^(?:(?:and\s+then|then|also|plus|and)\s+)+",
                "",
                candidate,
                flags=re.IGNORECASE,
            )
            if not candidate:
                continue

            if any(
                re.search(rf"\b{re.escape(verb)}\b", candidate, flags=re.IGNORECASE)
                for verb in ACTION_VERBS
            ):
                normalized_candidate = re.sub(r"\s+", " ", candidate).casefold()
                if normalized_candidate not in seen:
                    tasks.append(candidate)
                    seen.add(normalized_candidate)

    if not tasks:
        # Graceful fallback: allow a single actionable sentence without conjunction splitting.
        candidate = normalized.strip(" ,;.!?")
        if any(
            re.search(rf"\b{re.escape(verb)}\b", candidate, flags=re.IGNORECASE)
            for verb in ACTION_VERBS
        ):
            tasks.append(candidate)

    return tasks
