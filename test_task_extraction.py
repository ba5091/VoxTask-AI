from task_extraction import extract_tasks


def test_extract_tasks_splits_multiple_actions():
    transcript = "Complete my assignment tomorrow and prepare SQL questions by Friday."
    tasks = extract_tasks(transcript)
    assert len(tasks) == 2
    assert tasks[0].lower().startswith("complete")
    assert tasks[1].lower().startswith("prepare")


def test_extract_tasks_ignores_non_actionable_text():
    transcript = "The meeting finished and everyone left the room."
    assert extract_tasks(transcript) == []


def test_extract_tasks_does_not_split_a_description_at_and():
    assert extract_tasks("Prepare the agenda and meeting notes.") == [
        "Prepare the agenda and meeting notes"
    ]


def test_extract_tasks_splits_only_when_another_action_starts():
    tasks = extract_tasks("Read the research paper and then prepare a summary.")
    assert tasks == ["Read the research paper", "prepare a summary"]


def test_extract_tasks_deduplicates_case_insensitively():
    assert extract_tasks("Read the report. Also read the report.") == [
        "Read the report"
    ]
