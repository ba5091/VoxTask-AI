from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from audio_processing import validate_audio_file, load_whisper_model, transcribe_audio
from dataset_loader import load_manifest
from model_metrics import SampleMetrics, measure_inference, summarize_samples

WAV2VEC2_MODEL = "facebook/wav2vec2-base-960h"


class Wav2Vec2Transcriber:
    def __init__(self, model_name: str = WAV2VEC2_MODEL, device: str | None = None):
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        except ImportError as exc:
            raise RuntimeError("Wav2Vec2 requires the optional dependencies in requirements-asr.txt") from exc
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.model_name = model_name

    def transcribe(self, samples):
        inputs = self.processor(samples, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(self.device)
        with torch.inference_mode():
            logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(predicted_ids)[0].strip()


def run_comparison(manifest_path: str, output_dir: str = "outputs/asr_comparison", whisper_model: str = "base", wav2vec2_model: str = WAV2VEC2_MODEL, include_whisper: bool = True, include_wav2vec2: bool = True) -> dict:
    records = load_manifest(manifest_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    engines = []
    if include_whisper:
        engines.append((f"whisper-{whisper_model}", "whisper", None))
    if include_wav2vec2:
        engines.append((f"wav2vec2:{wav2vec2_model}", "wav2vec2", None))
    if not engines:
        raise ValueError("Enable at least one ASR model")
    results: list[SampleMetrics] = []
    model_metadata = {}
    for name, kind, _ in engines:
        if kind == "whisper":
            model = load_whisper_model(whisper_model)
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            transcriber = lambda samples, model=model: transcribe_audio(samples, model=model, model_name=whisper_model)
            model_metadata[name] = {"model_id": f"whisper-{whisper_model}", "device": str(next(model.parameters()).device), "parameter_count": parameter_count}
        else:
            adapter = Wav2Vec2Transcriber(wav2vec2_model)
            parameter_count = sum(parameter.numel() for parameter in adapter.model.parameters())
            transcriber = adapter.transcribe
            model_metadata[name] = {"model_id": wav2vec2_model, "device": adapter.device, "parameter_count": parameter_count}
        for item in records:
            validated = validate_audio_file(item["audio_path"])
            metrics = measure_inference(name, item["audio_path"], item["reference_text"], validated.duration, lambda s=validated.audio_samples, fn=transcriber: fn(s))
            results.append(metrics)
            print(f"{name}: {Path(item['audio_path']).name} WER={metrics.wer:.3f} CER={metrics.cer:.3f} time={metrics.inference_seconds:.2f}s")
    detailed_path = output / "sample_metrics.csv"
    with detailed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].__dict__.keys()))
        writer.writeheader()
        writer.writerows([row.__dict__ for row in results])
    summaries = {name: summarize_samples([r for r in results if r.model == name]) for name, _, _ in engines}
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    metadata_path = output / "model_metadata.json"
    metadata_path.write_text(json.dumps(model_metadata, indent=2), encoding="utf-8")
    _write_markdown_report(output / "report.md", summaries, detailed_path.name)
    return {"summary": summaries, "sample_metrics": str(detailed_path), "summary_json": str(summary_path), "model_metadata": str(metadata_path), "report": str(output / "report.md")}


def _write_markdown_report(path: Path, summaries: dict, details_filename: str):
    lines = ["# ASR Comparison Report", "", "Results are calculated from the supplied labeled manifest. No result is inferred or pre-filled.", "", "| Model | Samples | Mean WER ↓ | Mean CER ↓ | Mean inference (s) ↓ | Mean RTF ↓ | Task precision ↑ | Task recall ↑ | Task F1 ↑ |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model, metrics in summaries.items():
        def fmt(value):
            return "—" if value is None else f"{value:.4f}"
        lines.append(f"| {model} | {metrics['samples']} | {fmt(metrics['mean_wer'])} | {fmt(metrics['mean_cer'])} | {fmt(metrics['mean_inference_seconds'])} | {fmt(metrics['mean_real_time_factor'])} | {fmt(metrics.get('mean_task_precision'))} | {fmt(metrics.get('mean_task_recall'))} | {fmt(metrics.get('mean_task_f1'))} |")
    lines.extend(["", "WER = word error rate; CER = character error rate; RTF = inference seconds divided by audio duration. Lower is better for WER, CER, and RTF; higher is better for task precision, recall, and F1.", "Task scores compare tasks extracted from the human reference transcript against tasks extracted from each model transcript using Voice2Task’s current rule-based extractor.", "", f"Per-sample output: `{details_filename}`", "Model IDs, devices, and parameter counts are in `model_metadata.json`.", "", "Use the same held-out samples and reference transcripts for all models. Keep training data separate from evaluation data."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Compare Whisper and Wav2Vec2 on a labeled audio manifest.")
    parser.add_argument("--manifest", required=True, help="CSV with audio_path,reference_text columns")
    parser.add_argument("--output-dir", default="outputs/asr_comparison")
    parser.add_argument("--whisper-model", default="base", choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--wav2vec2-model", default=WAV2VEC2_MODEL)
    parser.add_argument("--whisper-only", action="store_true")
    parser.add_argument("--wav2vec2-only", action="store_true")
    args = parser.parse_args()
    result = run_comparison(args.manifest, args.output_dir, args.whisper_model, args.wav2vec2_model, not args.wav2vec2_only, not args.whisper_only)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
