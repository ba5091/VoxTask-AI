from __future__ import annotations

import csv
from pathlib import Path


def load_manifest(manifest_path: str | Path) -> list[dict[str, str]]:
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"audio_path", "reference_text"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError("Manifest must contain columns: audio_path, reference_text")
        rows = []
        for line, row in enumerate(reader, start=2):
            audio = (row.get("audio_path") or "").strip()
            reference = (row.get("reference_text") or "").strip()
            if not audio or not reference:
                raise ValueError(f"Manifest row {line} has an empty audio_path or reference_text")
            audio_path = Path(audio)
            if not audio_path.is_absolute():
                audio_path = path.parent / audio_path
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio file from manifest row {line} not found: {audio_path}")
            rows.append({"audio_path": str(audio_path.resolve()), "reference_text": reference})
    if not rows:
        raise ValueError("Manifest contains no samples")
    return rows


def load_librispeech(split: str = "train.100", limit: int | None = 100):
    """Load a reproducible subset of LibriSpeech through Hugging Face Datasets."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the optional training dependencies with: pip install -r requirements-asr.txt") from exc
    dataset = load_dataset("librispeech_asr", "clean", split=split)
    if limit is not None:
        dataset = dataset.select(range(min(int(limit), len(dataset))))
    return dataset
