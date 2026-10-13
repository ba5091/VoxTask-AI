from __future__ import annotations

import re

HIGH_PRIORITY_WORDS = ["urgent", "urgently", "asap", "immediately", "critical", "due today"]
MEDIUM_PRIORITY_WORDS = [
    "soon",
    "this week",
    "by tomorrow",
    "tomorrow",
    "today",
    "tonight",
    "this evening",
    "in 2 days",
    "in 3 days",
    "by friday",
    "friday",
]
LOW_PRIORITY_WORDS = ["later", "eventually", "when possible"]


def classify_priority(task: str, deadline: str) -> str:
    text = f"{task or ''} {deadline or ''}".lower()

    if any(re.search(rf"\b{re.escape(word)}\b", text) for word in HIGH_PRIORITY_WORDS):
        return "High"

    if any(re.search(rf"\b{re.escape(word)}\b", text) for word in MEDIUM_PRIORITY_WORDS):
        return "Medium"

    if any(re.search(rf"\b{re.escape(word)}\b", text) for word in LOW_PRIORITY_WORDS):
        return "Low"

    if str(deadline).lower() not in {"not specified", "none", ""}:
        return "Medium"

    return "Medium"
