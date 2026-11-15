# Voice2Task + ASR Comparison Research Extension

This project preserves the local Voice2Task Gradio productivity app and adds a reproducible research workflow for comparing Whisper with Wav2Vec2 and a locally fine-tuned Wav2Vec2 checkpoint.

## Models and experimental design

- **Baseline:** the app's existing OpenAI Whisper model (`tiny`, `base`, `small`, `medium`, or `large` where supported by the local Whisper installation).
- **Second model:** `facebook/wav2vec2-base-960h`, loaded lazily by `asr_comparison.py`.
- **Fine-tuned model:** a checkpoint produced by `training_pipeline.py`. Pass its local directory as `--wav2vec2-model` to compare it with the other models.

The comparison script uses the same manifest and reference text for every enabled model. It writes per-sample WER/CER and timing metrics, a JSON summary, and a Markdown report. **No performance values are pre-filled or claimed.**

## Windows / VS Code setup

Use Python 3.10 or 3.11 for the broadest compatibility with the audio and ML dependencies. Create and activate a virtual environment in the project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The original Voice2Task app uses PyAV for decoding and Whisper for transcription. Run the existing app with:

```powershell
python app.py
```

It normally opens at `http://127.0.0.1:7860/`. The first use of a model can download weights.

### Install ASR research / fine-tuning dependencies

```powershell
python -m pip install -r requirements-asr.txt
```

PyTorch should be installed for the hardware you actually have. If you have an NVIDIA GPU, install a CUDA-compatible PyTorch build using the official PyTorch selector before installing the remaining requirements. CPU-only fine-tuning is supported by the script but can be extremely slow. Start with the small subset below before attempting a larger run.

## Prepare a labeled evaluation manifest

Copy `asr_manifest_template.csv` to a new file, for example `data/evaluation_manifest.csv`, and add one row per audio recording. `audio_path` can be absolute or relative to the CSV; `reference_text` must be the human-verified transcript. The loader checks that files exist, but decoding and transcription are performed by the comparison runner.

```csv
audio_path,reference_text
Recording.m4a,Call Maya tomorrow and submit the report
Recording2.wav,Review the project and email the team
```

The text above is an illustrative format only, not a transcript for your uploaded recordings. Do not use synthetic/example references as ground truth. For an academic evaluation, keep the test set fixed and separate from all fine-tuning data; use the same held-out recordings for all models. Include varied speakers, accents, noise levels, and command types if available.

## Run the baseline comparison

```powershell
python asr_comparison.py --manifest data/evaluation_manifest.csv --output-dir outputs/asr_comparison
```

Useful options:

```powershell
python asr_comparison.py --manifest data/evaluation_manifest.csv --whisper-model tiny
python asr_comparison.py --manifest data/evaluation_manifest.csv --wav2vec2-only
python asr_comparison.py --manifest data/evaluation_manifest.csv --wav2vec2-model models/wav2vec2-finetuned
```

Outputs:

- `outputs/asr_comparison/sample_metrics.csv` — one row per model and recording
- `outputs/asr_comparison/summary.json` — aggregate metrics
- `outputs/asr_comparison/report.md` — readable comparison table

Metrics: **WER** (word error rate), **CER** (character error rate), inference seconds, and **RTF** (inference time divided by audio duration). Lower is better for WER, CER, and RTF. Inference time depends on hardware, warm-up, model downloads, and cache state; record machine details when reporting results.

## Fine-tune Wav2Vec2 locally

The training script downloads the LibriSpeech `clean` configuration from Hugging Face Datasets and uses a small, configurable subset. The default is intentionally a smoke-scale experiment, not a claim of state-of-the-art performance. The pretrained base checkpoint is `facebook/wav2vec2-base-960h`.

```powershell
python training_pipeline.py --train-limit 200 --eval-limit 40 --epochs 1 --batch-size 2 --gradient-accumulation-steps 4
```

For a larger run, increase the subset and epochs only after the small run completes and you understand your available RAM/VRAM and disk space. The model and processor are saved under `models/wav2vec2-finetuned` with `training_metadata.json`. Hugging Face dataset downloads and model weights require internet access on first use; inference after weights are cached can be local. This pipeline is not a zero-download setup.

Then evaluate the fine-tuned checkpoint on the **separate held-out manifest**:

```powershell
python asr_comparison.py --manifest data/evaluation_manifest.csv --wav2vec2-model models/wav2vec2-finetuned
```

Do not use the training/validation examples as the final test set. For a fair academic report, record the exact dataset split, sample counts, model/checkpoint IDs, library versions, hardware, random seed, and all metrics. The current training script records core run settings and trainer metrics; repeat runs if you need variance estimates.

## Existing Voice2Task app

`app.py`, audio validation, rule-based task extraction, deadline labeling, priority classification, task management, charts, and the original tests are retained. The ASR experiment is intentionally a separate command-line workflow so that missing optional Transformers dependencies or model downloads do not break the existing Whisper app.

Run tests:

```powershell
python -m pytest -q
```

Some tests require the main requirements to be installed. ASR-specific tests for metrics and manifest validation do not need to download a model.

## Interpretation and limitations

- Wav2Vec2 Base 960h is an English ASR checkpoint. It may not be appropriate for multilingual speech or all accents.
- A general-purpose ASR benchmark and a task-command benchmark answer different questions. For Voice2Task, also manually label whether each reference command's tasks, deadlines, and priorities were recovered correctly. WER alone does not measure task extraction quality.
- The included fine-tuning workflow demonstrates a reproducible path but does not automatically guarantee a better model. A small subset may underfit and will not represent all command-style speech.
- The current project does not claim a result until you run the evaluation on a labeled, held-out set.
- The original rule-based task extractor is unchanged. Its own limitations (fixed action vocabulary and rule-based deadline/priority handling) still apply.
