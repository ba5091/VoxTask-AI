from __future__ import annotations

import re

DEADLINE_PATTERNS = [
    r"\bby\s+next\s+(?:week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\bnext\s+(?:week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\bthis\s+(?:evening|afternoon|morning|weekend)\b",
    r"\bin\s+\d+\s+days?\b",
    r"\btomorrow\b",
    r"\btonight\b",
    r"\btoday\b",
    r"\bby\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\blater\b",
    r"\beventually\b",
    r"\bsomeday\b",
    r"\bwithin\s+\d+\s+days?\b",
]


def normalize_deadline_label(value: str) -> str:
    cleaned = value.strip().strip(" ,.!?")
    if not cleaned:
        return "Not specified"

    cleaned = re.sub(r"^(?:by|on|before|at)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.title()


def extract_deadline(task: str) -> tuple[str, str]:
    if task is None:
        return "", "Not specified"

    text = str(task).strip()
    if not text:
        return "", "Not specified"

    for pattern in DEADLINE_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            deadline_text = match.group(0)
            cleaned_task = re.sub(pattern, "", text, flags=re.IGNORECASE)
            cleaned_task = re.sub(r"\s+", " ", cleaned_task).strip(" ,.!?")
            normalized_deadline = normalize_deadline_label(deadline_text)
            return cleaned_task, normalized_deadline

    return re.sub(r"\s+", " ", text).strip(" ,.!?"), "Not specified"
