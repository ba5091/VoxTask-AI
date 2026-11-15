from __future__ import annotations

import argparse
import json
import inspect
from pathlib import Path

import numpy as np

from dataset_loader import load_librispeech

BASE_MODEL = "facebook/wav2vec2-base-960h"


def train_wav2vec2(output_dir: str = "models/wav2vec2-finetuned", train_limit: int = 200, eval_limit: int = 40, epochs: float = 1.0, batch_size: int = 2, gradient_accumulation_steps: int = 4, learning_rate: float = 1e-5, model_name: str = BASE_MODEL, fp16: bool = False):
    try:
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("Install requirements-asr.txt before training.") from exc
    if train_limit < 1 or eval_limit < 1:
        raise ValueError("train_limit and eval_limit must be positive")
    if fp16 and not torch.cuda.is_available():
        raise ValueError("fp16 was requested but CUDA is not available")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    train_data = load_librispeech("train.100", train_limit)
    eval_data = load_librispeech("validation", eval_limit)

    def prepare(batch):
        audio = batch["audio"]
        values = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        with processor.as_target_processor():
            labels = processor(text=batch["text"].lower()).input_ids
        return {"input_values": values, "labels": labels}

    train_data = train_data.map(prepare, remove_columns=train_data.column_names)
    eval_data = eval_data.map(prepare, remove_columns=eval_data.column_names)

    class CTCDataCollator:
        def __init__(self, processor):
            self.processor = processor
        def __call__(self, features):
            inputs = [{"input_values": f["input_values"]} for f in features]
            labels = [{"input_ids": f["labels"]} for f in features]
            batch = self.processor.pad(inputs, padding=True, return_tensors="pt")
            with self.processor.as_target_processor():
                labels_batch = self.processor.pad(labels, padding=True, return_tensors="pt")
            batch["labels"] = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            return batch

    def compute_metrics(pred):
        from model_metrics import word_error_rate, character_error_rate
        logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        pred_ids = np.argmax(logits, axis=-1)
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        hypotheses = processor.batch_decode(pred_ids)
        references = processor.batch_decode(label_ids, group_tokens=False)
        return {"wer": float(np.mean([word_error_rate(r, h) for r, h in zip(references, hypotheses)])), "cer": float(np.mean([character_error_rate(r, h) for r, h in zip(references, hypotheses)]))}

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    evaluation_key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments).parameters else "evaluation_strategy"
    training_args = {
        "output_dir": str(output / "checkpoints"),
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "num_train_epochs": epochs,
        evaluation_key: "epoch",
        "save_strategy": "epoch",
        "logging_steps": 10,
        "fp16": fp16,
        "gradient_checkpointing": torch.cuda.is_available(),
        "report_to": [],
        "save_total_limit": 2,
        "load_best_model_at_end": False,
    }
    args = TrainingArguments(**training_args)
    trainer = Trainer(model=model, args=args, train_dataset=train_data, eval_dataset=eval_data, data_collator=CTCDataCollator(processor), tokenizer=processor.feature_extractor, compute_metrics=compute_metrics)
    train_result = trainer.train()
    eval_result = trainer.evaluate()
    trainer.save_model(str(output))
    processor.save_pretrained(str(output))
    metadata = {
        "base_model": model_name,
        "output_dir": str(output),
        "train_samples": len(train_data),
        "eval_samples": len(eval_data),
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_result,
        "note": "This validation split is for training feedback. Use a separate held-out test set for final model comparison."
    }
    (output / "training_metadata.json").write_text(json.dumps(metadata, indent=2, default=float), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Wav2Vec2 CTC locally on a small LibriSpeech subset.")
    parser.add_argument("--output-dir", default="models/wav2vec2-finetuned")
    parser.add_argument("--train-limit", type=int, default=200)
    parser.add_argument("--eval-limit", type=int, default=40)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--model-name", default=BASE_MODEL)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    result = train_wav2vec2(**vars(args))
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
