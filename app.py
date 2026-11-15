from __future__ import annotations

import html
import json
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from audio_processing import load_whisper_model, transcribe_audio, validate_audio_file
from deadline_extraction import extract_deadline
from priority_classification import classify_priority
from task_extraction import extract_tasks, normalize_transcript
from task_management import TaskManager, TaskRecord
from visualizations import create_dashboard

TASK_COLUMNS = [
    "Task ID",
    "Task Description",
    "Deadline",
    "Priority",
    "Completion Status",
]
PRIORITIES = ["High", "Medium", "Low"]
STATUSES = ["Pending", "Completed"]
DATA_PATH = Path(__file__).resolve().parent / "outputs" / "tasks.json"

CSS = """
:root {
  --page: #f7f8fc; --surface: #ffffff; --surface-soft: #f0f1ff;
  --text: #0f172a; --muted: #64748b; --border: #e2e8f0;
  --primary: #4f46e5; --accent: #0891b2; --shadow: 0 8px 28px rgba(40, 48, 90, .08);
}
:root[data-v2t-theme="dark"] {
  --page: #111827; --surface: #1f2937; --surface-soft: #262f48;
  --text: #f9fafb; --muted: #cbd5e1; --border: #374151;
  --primary: #818cf8; --accent: #22d3ee; --shadow: 0 8px 28px rgba(0, 0, 0, .24);
  color-scheme: dark;
}
body, .gradio-container { background: var(--page) !important; color: var(--text) !important; }
.gradio-container { max-width: 1440px !important; }
.block, .form, .wrap, .tabs, .tabitem, .table-wrap {
  background-color: var(--surface) !important; color: var(--text) !important;
  border-color: var(--border) !important;
}
.v2t-header, .v2t-card, .v2t-kpi {
  background: var(--surface); border: 1px solid var(--border); border-radius: 18px;
  box-shadow: var(--shadow); color: var(--text);
}
.v2t-header { padding: 24px 28px; display:flex; align-items:center; justify-content:space-between; gap:20px; }
.v2t-brand { display:flex; align-items:center; gap:14px; }
.v2t-logo { width:48px; height:48px; display:grid; place-items:center; border-radius:15px;
  background:linear-gradient(135deg,#4f46e5,#0891b2); color:white; font-size:24px; }
.v2t-title { font-size:25px; font-weight:750; margin:0; color:var(--text); }
.v2t-subtitle { margin:4px 0 0; color:var(--muted); }
.v2t-badge { display:inline-block; margin-top:8px; padding:4px 10px; border-radius:999px;
  color:var(--primary); background:var(--surface-soft); font-size:12px; font-weight:700; }
.v2t-theme-button { border:1px solid var(--border); background:var(--surface); color:var(--text);
  padding:10px 14px; border-radius:12px; cursor:pointer; font-weight:650; }
.v2t-theme-button button { border:0; background:transparent; color:var(--text); cursor:pointer; font-weight:650; }
.v2t-theme-button:hover { border-color:var(--primary); transform:translateY(-1px); }
.v2t-kpis { display:grid; grid-template-columns:repeat(5,minmax(130px,1fr)); gap:14px; margin:18px 0; }
.v2t-kpi { padding:17px 18px; transition:transform .16s ease, box-shadow .16s ease; }
.v2t-kpi:hover { transform:translateY(-2px); box-shadow:0 12px 30px rgba(40,48,90,.13); }
.v2t-kpi-label { color:var(--muted); font-size:13px; }
.v2t-kpi-value { color:var(--text); font-size:30px; font-weight:760; margin-top:6px; }
.v2t-kpi-icon { float:right; font-size:21px; }
.v2t-note { padding:12px 15px; border-radius:12px; background:var(--surface-soft); color:var(--text); }
.v2t-empty { padding:22px; text-align:center; color:var(--muted); }
.v2t-status { color:var(--muted); font-weight:600; }
button.primary { background:var(--primary) !important; }
input, textarea, select {
  color:var(--text) !important; background:var(--surface) !important;
  border-color:var(--border) !important;
}
@media(max-width:800px) {
  .v2t-kpis { grid-template-columns:repeat(2,minmax(130px,1fr)); }
  .v2t-header { align-items:flex-start; padding:18px; }
}
"""

THEME_JS = """
() => {
  const root = document.documentElement;
  const saved = localStorage.getItem('voice2task-theme') || 'light';
  root.setAttribute('data-v2t-theme', saved);
  const button = document.querySelector('.v2t-theme-button button') || document.querySelector('.v2t-theme-button');
  if (button) button.textContent = saved === 'dark' ? '☀️ Light mode' : '🌙 Dark mode';
  return [];
}
"""

THEME_TOGGLE_JS = """
() => {
  const root = document.documentElement;
  const next = root.getAttribute('data-v2t-theme') === 'dark' ? 'light' : 'dark';
  root.setAttribute('data-v2t-theme', next);
  localStorage.setItem('voice2task-theme', next);
  const button = document.querySelector('.v2t-theme-button button') || document.querySelector('.v2t-theme-button');
  if (button) button.textContent = next === 'dark' ? '☀️ Light mode' : '🌙 Dark mode';
  return [];
}
"""


def _rows_to_manager(rows: list[dict] | None) -> TaskManager:
    records = []
    for row in rows or []:
        records.append(
            TaskRecord(
                task_id=str(row.get("Task ID", "")),
                description=str(row.get("Task Description", "")),
                deadline=str(row.get("Deadline", "Not specified")),
                priority=str(row.get("Priority", "Medium")),
                completion_status=str(row.get("Completion Status", "Pending")),
            )
        )
    return TaskManager(records)


def _load_persisted_tasks() -> list[dict]:
    if not DATA_PATH.exists():
        return []
    try:
        with DATA_PATH.open("r", encoding="utf-8") as file:
            stored = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not load saved tasks from {DATA_PATH}: {exc}") from exc
    if not isinstance(stored, list):
        raise RuntimeError(f"Saved task data in {DATA_PATH} must be a JSON list.")
    return stored


def _persist_tasks(rows: list[dict]) -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=DATA_PATH.parent,
            suffix=".tmp",
            delete=False,
        ) as file:
            temp_path = Path(file.name)
            json.dump(rows, file, ensure_ascii=False, indent=2)
        temp_path.replace(DATA_PATH)
    except OSError as exc:
        if temp_path and temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Could not save tasks to {DATA_PATH}: {exc}") from exc


def _dataframe(rows: list[dict] | None) -> pd.DataFrame:
    return _rows_to_manager(rows).to_dataframe()


def _metrics_html(rows: list[dict] | None) -> str:
    df = _dataframe(rows)
    total = len(df)
    values = [
        ("🗂️", "Total Tasks", total, "#4f46e5"),
        ("⏳", "Pending Tasks", int((df["Completion Status"] == "Pending").sum()), "#0891b2"),
        ("✅", "Completed Tasks", int((df["Completion Status"] == "Completed").sum()), "#059669"),
        ("🔥", "High Priority", int((df["Priority"] == "High").sum()), "#dc2626"),
        ("📅", "No Deadline", int((df["Deadline"] == "Not specified").sum()), "#d97706"),
    ]
    cards = "".join(
        f'<div class="v2t-kpi" style="border-top:3px solid {color}">'
        f'<span class="v2t-kpi-icon">{icon}</span>'
        f'<div class="v2t-kpi-label">{label}</div>'
        f'<div class="v2t-kpi-value">{value}</div></div>'
        for icon, label, value, color in values
    )
    return f'<div class="v2t-kpis">{cards}</div>'


def _filtered_rows(
    rows: list[dict] | None,
    search: str = "",
    priority: str = "All",
    status: str = "All",
    sort_by: str = "Task ID",
) -> list[dict]:
    manager = _rows_to_manager(rows)
    query = (search or "").strip().casefold()
    filtered = [
        task for task in manager.tasks
        if (not query or query in task.description.casefold())
        and (priority == "All" or task.priority == priority)
        and (status == "All" or task.completion_status == status)
    ]
    if sort_by == "Deadline":
        filtered.sort(key=lambda task: (task.deadline == "Not specified", task.deadline.casefold()))
    elif sort_by == "Priority":
        order = {"High": 0, "Medium": 1, "Low": 2}
        filtered.sort(key=lambda task: (order.get(task.priority, 99), task.description.casefold()))
    else:
        filtered.sort(key=lambda task: task.task_id)
    return [task.to_dict() for task in filtered]


def _calendar_html(rows: list[dict] | None) -> str:
    df = _dataframe(rows)
    if df.empty:
        return '<div class="v2t-card v2t-empty">No tasks yet. Accepted tasks will appear here.</div>'
    dated = df[df["Deadline"] != "Not specified"]
    if dated.empty:
        return '<div class="v2t-card v2t-empty">No task deadlines are set yet.</div>'
    items = []
    for _, task in dated.iterrows():
        label = html.escape(str(task["Deadline"]))
        description = html.escape(str(task["Task Description"]))
        items.append(
            f'<div class="v2t-card" style="padding:14px 18px;margin:10px 0">'
            f'<strong>{label}</strong> · {description} '
            f'<span style="color:var(--muted)">({html.escape(str(task["Completion Status"]))})</span></div>'
        )
    return (
        '<div class="v2t-note">Relative deadline labels are shown as recorded; '
        'they are not converted to calendar dates.</div>' + "".join(items)
    )


def _refresh_outputs(rows: list[dict] | None, search="", priority="All", status="All", sort_by="Task ID"):
    all_rows = rows or []
    visible = _filtered_rows(all_rows, search, priority, status, sort_by)
    manager = _rows_to_manager(all_rows)
    df = manager.to_dataframe()
    dashboard = create_dashboard(df)
    analytics = create_dashboard(df)
    return (
        pd.DataFrame(visible, columns=TASK_COLUMNS),
        f"Showing {len(visible)} of {len(all_rows)} tasks",
        _metrics_html(all_rows),
        dashboard,
        analytics,
        _calendar_html(all_rows),
        _calendar_html(all_rows),
    )


def transcribe_audio_callback(audio_file, model_name: str):
    if audio_file is None:
        return "", "", "", "Upload an audio file first."
    try:
        path = audio_file.name if hasattr(audio_file, "name") else str(audio_file)
        validated = validate_audio_file(path)
        model = load_whisper_model(model_name=model_name)
        transcript = transcribe_audio(
            validated.audio_samples, model=model, model_name=model_name
        )
        if not transcript:
            return "", "", "", "No speech was recognized. Try a clearer recording or edit the transcript."
        return transcript, transcript, transcript, f"Transcription ready · {validated.duration:.1f} seconds"
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        return "", "", "", f"Transcription error: {exc}"


def extract_tasks_callback(transcript: str):
    normalized = normalize_transcript(transcript or "")
    if not normalized:
        return pd.DataFrame(columns=TASK_COLUMNS), [], "Enter or transcribe text before extracting tasks."

    extracted = extract_tasks(normalized)
    if not extracted:
        return (
            pd.DataFrame(columns=TASK_COLUMNS),
            [],
            "No tasks were detected. This rule-based extractor may need an explicit action verb; review or edit the transcript.",
        )

    manager = TaskManager()
    for text in extracted:
        description, deadline = extract_deadline(text)
        if not description:
            continue
        manager.add_task(
            description=description,
            deadline=deadline,
            priority=classify_priority(description, deadline),
        )
    candidates = manager.to_dataframe()
    return candidates, candidates.to_dict(orient="records"), f"Review {len(candidates)} candidate task(s), then accept them."


def candidate_rows_callback(candidate_df):
    if candidate_df is None:
        return []
    rows = candidate_df.to_dict(orient="records")
    for row in rows:
        if not str(row.get("Task Description", "")).strip():
            raise gr.Error("Every candidate needs a task description before it can be accepted.")
        if row.get("Priority") not in PRIORITIES:
            raise gr.Error("Candidate priority must be High, Medium, or Low.")
    return rows


def accept_tasks_callback(candidates: list[dict] | None, current_rows: list[dict] | None):
    rows = list(current_rows or [])
    manager = _rows_to_manager(rows)
    known = {task.description.strip().casefold() for task in manager.tasks}
    added = 0
    for candidate in candidates or []:
        description = str(candidate.get("Task Description", "")).strip()
        if not description or description.casefold() in known:
            continue
        try:
            manager.add_task(
                description=description,
                deadline=str(candidate.get("Deadline", "Not specified")),
                priority=str(candidate.get("Priority", "Medium")),
                completion_status="Pending",
            )
        except ValueError as exc:
            return rows, pd.DataFrame(columns=TASK_COLUMNS), f"Candidate task was not accepted: {exc}"
        known.add(description.casefold())
        added += 1
    saved = manager.to_dataframe().to_dict(orient="records")
    try:
        _persist_tasks(saved)
    except RuntimeError as exc:
        return (
            rows,
            pd.DataFrame(columns=TASK_COLUMNS),
            f"Tasks were not accepted because persistence failed: {exc}",
        )
    return saved, pd.DataFrame(columns=TASK_COLUMNS), f"Accepted {added} new task(s); duplicates were skipped."


def add_task_callback(description, deadline, priority, rows, search, filter_priority, filter_status, sort_by):
    try:
        manager = _rows_to_manager(rows)
        manager.add_task(description, deadline or "Not specified", priority or "Medium")
        updated = manager.to_dataframe().to_dict(orient="records")
        _persist_tasks(updated)
    except (ValueError, RuntimeError) as exc:
        outputs = _refresh_outputs(rows, search, filter_priority, filter_status, sort_by)
        return (*outputs, rows or [], f"Could not add task: {exc}")
    outputs = _refresh_outputs(updated, search, filter_priority, filter_status, sort_by)
    return (*outputs, updated, "Task added.")


def update_task_callback(task_id, description, deadline, priority, status, rows, search, filter_priority, filter_status, sort_by):
    manager = _rows_to_manager(rows)
    try:
        record = manager.update_task(
            task_id or "",
            description=str(description or "").strip(),
            deadline=deadline or "Not specified",
            priority=priority or "Medium",
            completion_status=status or "Pending",
        )
    except ValueError as exc:
        outputs = _refresh_outputs(rows, search, filter_priority, filter_status, sort_by)
        return (*outputs, rows or [], f"Could not update task: {exc}")
    if record is None or not record.description:
        outputs = _refresh_outputs(rows, search, filter_priority, filter_status, sort_by)
        return (*outputs, rows or [], "Select an existing task and provide a description.")
    updated = manager.to_dataframe().to_dict(orient="records")
    try:
        _persist_tasks(updated)
    except RuntimeError as exc:
        outputs = _refresh_outputs(rows, search, filter_priority, filter_status, sort_by)
        return (*outputs, rows or [], f"Could not save task changes: {exc}")
    outputs = _refresh_outputs(updated, search, filter_priority, filter_status, sort_by)
    return (*outputs, updated, f"Updated {task_id}.")


def delete_task_callback(task_id, rows, search, filter_priority, filter_status, sort_by):
    manager = _rows_to_manager(rows)
    if not task_id or not manager.delete_task(task_id):
        outputs = _refresh_outputs(rows, search, filter_priority, filter_status, sort_by)
        return (*outputs, rows or [], "Choose a task ID to delete.")
    updated = manager.to_dataframe().to_dict(orient="records")
    try:
        _persist_tasks(updated)
    except RuntimeError as exc:
        outputs = _refresh_outputs(rows, search, filter_priority, filter_status, sort_by)
        return (*outputs, rows or [], f"Could not delete task because saving failed: {exc}")
    outputs = _refresh_outputs(updated, search, filter_priority, filter_status, sort_by)
    return (*outputs, updated, f"Deleted {task_id}.")


def sync_table_callback(edited_df, rows, search, filter_priority, filter_status, sort_by):
    if edited_df is None:
        edited_records = []
    else:
        edited_records = edited_df.to_dict(orient="records")
    by_id = {str(row.get("Task ID")): row for row in edited_records}
    manager = _rows_to_manager(rows)
    for task in manager.tasks:
        edited = by_id.get(task.task_id)
        if edited is None:
            continue
        description = str(edited.get("Task Description", "")).strip()
        priority = str(edited.get("Priority", "Medium") or "Medium")
        completion_status = str(edited.get("Completion Status", "Pending") or "Pending")
        if not description or priority not in PRIORITIES or completion_status not in STATUSES:
            outputs = _refresh_outputs(rows, search, filter_priority, filter_status, sort_by)
            return (*outputs, rows or [], "Edit rejected: descriptions, priorities, or statuses were invalid.")
        task.description = description
        task.deadline = str(edited.get("Deadline", "Not specified") or "Not specified")
        task.priority = priority
        task.completion_status = completion_status
    updated = manager.to_dataframe().to_dict(orient="records")
    try:
        _persist_tasks(updated)
    except RuntimeError as exc:
        outputs = _refresh_outputs(rows, search, filter_priority, filter_status, sort_by)
        return (*outputs, rows or [], f"Edits could not be saved: {exc}")
    outputs = _refresh_outputs(updated, search, filter_priority, filter_status, sort_by)
    return (*outputs, updated, "Task edits saved.")


def export_callback(rows):
    if not rows:
        return None, None, "There are no tasks to export."
    manager = _rows_to_manager(rows)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        csv_path = manager.export_csv(DATA_PATH.parent / "voice2task_tasks.csv")
        xlsx_path = manager.export_excel(DATA_PATH.parent / "voice2task_tasks.xlsx")
    except (OSError, ValueError) as exc:
        return None, None, f"Export failed: {exc}"
    return csv_path, xlsx_path, f"Exported {len(rows)} task(s) to CSV and Excel."


def load_task_options(rows):
    ids = [str(row.get("Task ID")) for row in rows or []]
    return gr.update(choices=ids, value=ids[0] if ids else None)


INITIAL_ROWS = _load_persisted_tasks()

with gr.Blocks(title="Voice2Task — AI Productivity", theme=gr.themes.Soft(), css=CSS) as app:
    task_state = gr.State(value=INITIAL_ROWS)
    candidate_state = gr.State(value=[])
    original_transcript = gr.State(value="")

    gr.HTML(
        """
        <header class="v2t-header">
          <div class="v2t-brand">
            <div class="v2t-logo" aria-hidden="true">🎙️</div>
            <div>
              <h1 class="v2t-title">Voice2Task</h1>
              <p class="v2t-subtitle">Turn your voice into organized, actionable tasks.</p>
              <span class="v2t-badge">✦ AI-powered · Local-first</span>
            </div>
          </div>
        </header>
        """
    )
    theme_toggle = gr.Button("🌙 Dark mode", elem_classes=["v2t-theme-button"])
    theme_toggle.click(fn=None, js=THEME_TOGGLE_JS)

    metrics = gr.HTML(_metrics_html(INITIAL_ROWS), elem_classes=["v2t-metrics"])
    with gr.Tabs():
        with gr.Tab("Dashboard", id="dashboard"):
            gr.Markdown("## Your productivity at a glance")
            dashboard_chart = gr.Plot(label="Task overview")
            calendar_view = gr.HTML(_calendar_html(INITIAL_ROWS))

        with gr.Tab("Process Audio", id="audio"):
            gr.Markdown("## Capture and review\nSupported audio: WAV, MP3, M4A, FLAC · up to 10 minutes.")
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(label="Recording", type="filepath")
                    model_dropdown = gr.Dropdown(
                        choices=["tiny", "base", "small", "medium"],
                        value="base",
                        label="Whisper model",
                        info="Larger models may be more accurate but use more memory and time.",
                    )
                    transcribe_button = gr.Button("Transcribe audio", variant="primary")
                    reset_audio_button = gr.Button("Process another recording")
                    audio_status = gr.Markdown("Ready when you are.", elem_classes=["v2t-status"])
                with gr.Column(scale=2):
                    gr.Markdown("### Review transcript\nCorrect recognition errors here before extracting tasks.")
                    original_transcript_box = gr.Textbox(
                        label="Original Whisper transcript",
                        lines=4,
                        interactive=False,
                    )
                    transcript_box = gr.Textbox(label="Editable transcript", lines=8, max_lines=18)
                    with gr.Row():
                        extract_button = gr.Button("Extract tasks", variant="primary")
                        clear_transcript_button = gr.Button("Clear transcript")
                    candidate_table = gr.Dataframe(
                        headers=TASK_COLUMNS,
                        datatype=["str", "str", "str", "str", "str"],
                        interactive=True,
                        row_count=(0, "dynamic"),
                        label="Review candidates before accepting",
                    )
                    accept_button = gr.Button("Accept reviewed tasks", variant="primary")
                    extraction_status = gr.Markdown("")

        with gr.Tab("Manage Tasks", id="tasks"):
            gr.Markdown("## Manage your task list\nEdit cells directly, or select a task for a focused edit.")
            with gr.Row():
                search_box = gr.Textbox(label="Search description", placeholder="Search tasks…")
                priority_filter = gr.Dropdown(["All", *PRIORITIES], value="All", label="Priority")
                status_filter = gr.Dropdown(["All", *STATUSES], value="All", label="Status")
                sort_dropdown = gr.Dropdown(["Task ID", "Deadline", "Priority"], value="Task ID", label="Sort by")
            with gr.Row():
                result_count = gr.Markdown("")
                reset_filters_button = gr.Button("Reset filters")
            tasks_table = gr.Dataframe(
                value=_dataframe(INITIAL_ROWS),
                headers=TASK_COLUMNS,
                datatype=["str", "str", "str", "str", "str"],
                interactive=True,
                row_count=(0, "dynamic"),
                label="Tasks (direct cell edits are saved)",
            )
            with gr.Accordion("Add a task manually", open=False):
                with gr.Row():
                    new_description = gr.Textbox(label="Description")
                    new_deadline = gr.Textbox(label="Deadline", value="Not specified")
                    new_priority = gr.Dropdown(PRIORITIES, value="Medium", label="Priority")
                add_button = gr.Button("Add task")
            with gr.Accordion("Edit or delete a task", open=False):
                selected_id = gr.Dropdown(label="Task ID", choices=[])
                with gr.Row():
                    edit_description = gr.Textbox(label="Description")
                    edit_deadline = gr.Textbox(label="Deadline")
                with gr.Row():
                    edit_priority = gr.Dropdown(PRIORITIES, value="Medium", label="Priority")
                    edit_status = gr.Dropdown(STATUSES, value="Pending", label="Completion status")
                with gr.Row():
                    update_button = gr.Button("Save task changes")
                    delete_button = gr.Button("Delete selected task", variant="stop")
                gr.Markdown("Deletion is immediate. Check the selected Task ID before confirming.")
            task_status = gr.Markdown("")
            export_button = gr.Button("Export current tasks")
            with gr.Row():
                csv_file = gr.File(label="CSV download")
                excel_file = gr.File(label="Excel download")
            export_status = gr.Markdown("")

        with gr.Tab("Analytics", id="analytics"):
            gr.Markdown("## Task analytics\nCharts describe the current task data; no historical trends are inferred.")
            analytics_chart = gr.Plot(label="Priority and completion")

        with gr.Tab("Calendar", id="calendar"):
            gr.Markdown("## Deadline timeline\nRelative labels remain labels until a reliable date resolver is configured.")
            calendar_tab_view = gr.HTML(_calendar_html(INITIAL_ROWS))

        with gr.Tab("About / Help", id="about"):
            gr.Markdown(
                """
                ## How Voice2Task works
                1. Upload a recording and transcribe it with local Whisper.
                2. Review and correct the transcript; the original recognition is retained separately during the session.
                3. Extract candidates using transparent rule-based action-word matching.
                4. Review and accept candidates into the task list.

                **Privacy and storage:** Whisper runs locally. Accepted tasks are saved as JSON in `outputs/tasks.json`;
                session transcript and candidate state is held in Gradio memory. CSV and XLSX exports contain current tasks.

                **Limitations:** This rule-based NLP does not understand all grammatical forms or implicit requests.
                Deadline labels are not automatically mapped to dates; consequently, the calendar view does not claim
                overdue or due-today calculations. Whisper may download its model on first use.
                """
            )

    outputs = [
        tasks_table,
        result_count,
        metrics,
        dashboard_chart,
        analytics_chart,
        calendar_view,
        calendar_tab_view,
    ]
    audio_outputs = [transcript_box, original_transcript_box, original_transcript, audio_status]
    refresh_filter_inputs = [task_state, search_box, priority_filter, status_filter, sort_dropdown]

    transcribe_button.click(
        fn=transcribe_audio_callback,
        inputs=[audio_input, model_dropdown],
        outputs=audio_outputs,
    )
    reset_audio_button.click(
        fn=lambda: (None, "", "", "", pd.DataFrame(columns=TASK_COLUMNS), [], "Ready when you are."),
        outputs=[
            audio_input,
            transcript_box,
            original_transcript_box,
            original_transcript,
            candidate_table,
            candidate_state,
            audio_status,
        ],
    )
    extract_button.click(
        fn=extract_tasks_callback,
        inputs=[transcript_box],
        outputs=[candidate_table, candidate_state, extraction_status],
    )
    candidate_table.change(
        fn=candidate_rows_callback,
        inputs=[candidate_table],
        outputs=[candidate_state],
    )
    clear_transcript_button.click(
        fn=lambda: ("", "", "", pd.DataFrame(columns=TASK_COLUMNS), [], "Transcript and candidates cleared."),
        outputs=[
            transcript_box,
            original_transcript_box,
            original_transcript,
            candidate_table,
            candidate_state,
            extraction_status,
        ],
    )
    accept_button.click(
        fn=accept_tasks_callback,
        inputs=[candidate_state, task_state],
        outputs=[task_state, candidate_table, extraction_status],
    ).then(
        fn=_refresh_outputs,
        inputs=refresh_filter_inputs,
        outputs=outputs,
    ).then(
        fn=load_task_options,
        inputs=[task_state],
        outputs=[selected_id],
    )

    filters = [search_box, priority_filter, status_filter, sort_dropdown]
    for component in filters:
        component.change(
            fn=_refresh_outputs,
            inputs=refresh_filter_inputs,
            outputs=outputs,
        )
    reset_filters_button.click(
        fn=lambda: ("", "All", "All", "Task ID"),
        outputs=filters,
    ).then(fn=_refresh_outputs, inputs=refresh_filter_inputs, outputs=outputs)

    tasks_table.change(
        fn=sync_table_callback,
        inputs=[tasks_table, *refresh_filter_inputs],
        outputs=[*outputs, task_state, task_status],
    )

    common_inputs = [task_state, *filters]
    common_outputs = [*outputs, task_state, task_status]
    add_button.click(
        fn=add_task_callback,
        inputs=[new_description, new_deadline, new_priority, *common_inputs],
        outputs=common_outputs,
    ).then(fn=load_task_options, inputs=[task_state], outputs=[selected_id])
    selected_id.change(
        fn=lambda task_id, rows: next(
            (
                (
                    row.get("Task Description", ""),
                    row.get("Deadline", "Not specified"),
                    row.get("Priority", "Medium"),
                    row.get("Completion Status", "Pending"),
                )
                for row in rows or []
                if row.get("Task ID") == task_id
            ),
            ("", "Not specified", "Medium", "Pending"),
        ),
        inputs=[selected_id, task_state],
        outputs=[edit_description, edit_deadline, edit_priority, edit_status],
    )
    update_button.click(
        fn=update_task_callback,
        inputs=[selected_id, edit_description, edit_deadline, edit_priority, edit_status, *common_inputs],
        outputs=common_outputs,
    )
    delete_button.click(
        fn=delete_task_callback,
        inputs=[selected_id, *common_inputs],
        outputs=common_outputs,
    ).then(fn=load_task_options, inputs=[task_state], outputs=[selected_id])
    export_button.click(fn=export_callback, inputs=[task_state], outputs=[csv_file, excel_file, export_status])

    app.load(fn=None, js=THEME_JS)
    app.load(fn=_refresh_outputs, inputs=[task_state], outputs=outputs).then(
        fn=load_task_options,
        inputs=[task_state],
        outputs=[selected_id],
    )


if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
