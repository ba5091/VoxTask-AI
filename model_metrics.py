from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from typing import Callable


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return " ".join(text.split())


def edit_distance(reference: str, hypothesis: str) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, start=1):
        current = [i]
        for j, right in enumerate(hypothesis, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize_text(reference).split(), normalize_text(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, left in enumerate(ref, start=1):
        current = [i]
        for j, right in enumerate(hyp, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1] / len(ref)


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


@dataclass
class SampleMetrics:
    model: str
    audio_path: str
    reference: str
    hypothesis: str
    wer: float
    cer: float
    audio_seconds: float
    inference_seconds: float
    real_time_factor: float
    task_precision: float
    task_recall: float
    task_f1: float


def task_extraction_scores(reference: str, hypothesis: str) -> tuple[float, float, float]:
    from task_extraction import extract_tasks

    def canonical(text: str) -> set[str]:
        return {normalize_text(task) for task in extract_tasks(text) if normalize_text(task)}

    expected, predicted = canonical(reference), canonical(hypothesis)
    if not expected and not predicted:
        return 1.0, 1.0, 1.0
    if not predicted or not expected:
        return (0.0 if predicted else 1.0), (0.0 if expected else 1.0), 0.0
    true_positive = len(expected & predicted)
    precision = true_positive / len(predicted)
    recall = true_positive / len(expected)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def measure_inference(model_name: str, audio_path: str, reference: str, duration: float, transcribe: Callable[[], str]) -> SampleMetrics:
    started = time.perf_counter()
    hypothesis = transcribe()
    elapsed = time.perf_counter() - started
    precision, recall, f1 = task_extraction_scores(reference, hypothesis)
    return SampleMetrics(model_name, audio_path, reference, hypothesis, word_error_rate(reference, hypothesis), character_error_rate(reference, hypothesis), duration, elapsed, elapsed / duration if duration > 0 else 0.0, precision, recall, f1)


def summarize_samples(rows: list[SampleMetrics]) -> dict:
    if not rows:
        return {"samples": 0, "mean_wer": None, "mean_cer": None, "mean_inference_seconds": None, "mean_real_time_factor": None, "mean_task_precision": None, "mean_task_recall": None, "mean_task_f1": None}
    return {
        "samples": len(rows),
        "mean_wer": sum(row.wer for row in rows) / len(rows),
        "mean_cer": sum(row.cer for row in rows) / len(rows),
        "mean_inference_seconds": sum(row.inference_seconds for row in rows) / len(rows),
        "mean_real_time_factor": sum(row.real_time_factor for row in rows) / len(rows),
        "mean_task_precision": sum(row.task_precision for row in rows) / len(rows),
        "mean_task_recall": sum(row.task_recall for row in rows) / len(rows),
        "mean_task_f1": sum(row.task_f1 for row in rows) / len(rows),
    }
