# Voice2Task

Voice2Task is a local-first productivity dashboard that transcribes short recordings with OpenAI Whisper and extracts reviewable tasks using transparent, rule-based NLP.

## Current application workflow

1. Upload WAV, MP3, M4A, or FLAC audio (up to 10 minutes) and select a Whisper model.
2. Transcribe locally. The original Whisper transcript is shown separately from the editable copy.
3. Correct the editable transcript, extract task candidates, and review or edit those candidates.
4. Accept candidates into the task manager. Exact duplicate descriptions are skipped.
5. Search, filter, sort, directly edit, manually add, update, complete, or delete tasks.
6. View dashboard KPIs, analytics charts, and a deadline-label timeline; export current tasks as CSV or XLSX.

The light/dark toggle updates the full page palette without restarting the app and stores the preference in browser local storage.

## Features and implementation

- **Audio:** PyAV decodes and resamples supported formats in-process, so installing a separate FFmpeg executable is not required. Whisper runs on CPU or CUDA when available. The first model use can require a model download.
- **Task extraction:** imperative/action-word rules split on conjunctions only where another supported action starts, and deduplicate repeated task text. Rule-based extraction is not full language understanding; inspect the editable transcript and candidate list.
- **Deadlines:** the extractor retains relative labels such as `Tomorrow`, `Next Monday`, `In 2 Days`, `Later`, and `Someday`. Labels are not resolved to calendar dates; due-today and overdue calculations are therefore intentionally omitted.
- **Priority:** deterministic rules classify High, Medium, or Low. High urgency takes precedence over medium and low signals.
- **Task data:** accepted tasks are persisted locally in `outputs/tasks.json` using atomic replacement. Transcript and review candidates are session-only Gradio state. This is a single-user/local app; concurrent writes from multiple users are not coordinated.
- **Exports:** CSV and XLSX exports reflect the current task list. Empty task lists show a helpful message rather than creating misleading downloads.
- **Evaluation:** `evaluation.py` includes classification metrics and WER helpers. No benchmark or model quality claim is made without a verified, labeled dataset.

## Project layout

- `app.py` — Gradio UI, review workflow, task callbacks, local persistence.
- `audio_processing.py` — PyAV decoding, validation, Whisper loading and transcription.
- `task_extraction.py`, `deadline_extraction.py`, `priority_classification.py` — rule-based NLP.
- `task_management.py` — task schema, IDs, CRUD helpers, filters, sorting, exports.
- `visualizations.py` — empty-safe Matplotlib dashboard and analytics.
- `evaluation.py` — confusion matrix, classification report, and WER utility.
- `test_*.py` — isolated pytest coverage.
- `outputs/` — task JSON and generated exports.

The original `Voice2Task.ipynb` remains unmodified.

## Windows setup (PowerShell)

From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

No external FFmpeg installation is needed. For a CPU-only installation of PyTorch, use the official PyTorch CPU install instructions and then install the remaining project requirements.

Run tests and start the app:

```powershell
python -m pytest -q
python app.py
```

Open the local URL printed by Gradio (normally `http://127.0.0.1:7860/`). Keep the terminal open while using the application.

## Manual verification checklist

- Switch light → dark → light and confirm page text, forms, tables, and cards remain readable.
- Upload a supported audio file, transcribe, compare the original and editable transcript, change the editable text, and extract without rerunning Whisper.
- Edit a candidate before accepting; accept it and verify IDs, KPIs, charts, timeline, and exports update.
- In Manage Tasks, try search, each filter, sorting, direct table edits, manual add, focused edit, completion status, and delete.
- Export both CSV and XLSX and check that task IDs, descriptions, deadlines, priorities, and statuses match the current list.
- Restart the app and confirm accepted tasks reload from `outputs/tasks.json`.
- Try silence, corrupt audio, an empty transcript, and a transcript without a recognized action verb.

## Known limitations

- The deterministic action vocabulary will miss some requests, inflections, implicit instructions, and complex sentence structures.
- Relative deadline text is not converted into dates, so calendar dates, overdue metrics, and due-today metrics are not shown.
- The JSON store is intended for a local, single-user application rather than concurrent multi-user writes.
- Theme preference is saved per browser through local storage; task data is local to this machine.
- Model download, inference time, and transcription quality depend on the selected model and local CPU/GPU resources.
- Speaker identification, voice commands, historical completion trends, and external LLM integrations are not implemented.
