from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def compute_confusion_matrix(y_true: Iterable[str], y_pred: Iterable[str], labels: list[str] | None = None):
    target_labels = labels or ["High", "Medium", "Low"]
    return confusion_matrix(list(y_true), list(y_pred), labels=target_labels)


def compute_classification_report(y_true: Iterable[str], y_pred: Iterable[str], labels: list[str] | None = None):
    target_labels = labels or ["High", "Medium", "Low"]
    return classification_report(list(y_true), list(y_pred), labels=target_labels, output_dict=True)


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.strip().split()
    hyp_tokens = hypothesis.strip().split()

    if not ref_tokens and not hyp_tokens:
        return 0.0
    if not ref_tokens or not hyp_tokens:
        return 1.0

    rows = len(ref_tokens) + 1
    cols = len(hyp_tokens) + 1
    dp = np.zeros((rows, cols), dtype=np.int32)

    for i in range(rows):
        dp[i, 0] = i
    for j in range(cols):
        dp[0, j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )

    return float(dp[-1, -1] / max(len(ref_tokens), 1))
