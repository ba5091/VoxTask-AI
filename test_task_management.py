import pandas as pd

from task_management import TaskManager


def test_task_manager_generates_unique_ids():
    manager = TaskManager()
    manager.add_task("Finish draft")
    manager.add_task("Send email")

    task_ids = [task.task_id for task in manager.tasks]
    assert task_ids == ["T001", "T002"]


def test_task_manager_export_dataframe():
    manager = TaskManager()
    manager.add_task("Prepare notes", deadline="Tomorrow", priority="Medium")
    df = manager.to_dataframe()

    assert list(df.columns) == ["Task ID", "Task Description", "Deadline", "Priority", "Completion Status"]
    assert df.iloc[0]["Task Description"] == "Prepare notes"


def test_ids_remain_unique_after_deleting_a_task():
    manager = TaskManager()
    manager.add_task("First")
    manager.add_task("Second")
    manager.add_task("Third")
    manager.delete_task("T002")

    added = manager.add_task("Fourth")
    assert added.task_id == "T004"
    assert len({task.task_id for task in manager.tasks}) == 3


def test_task_manager_rejects_duplicate_task_ids():
    manager = TaskManager()
    manager.add_task("First", task_id="T001")

    try:
        manager.add_task("Duplicate", task_id="T001")
    except ValueError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("Duplicate task IDs must be rejected.")


def test_task_manager_search_and_filter():
    manager = TaskManager()
    manager.add_task("Prepare notes", priority="High", completion_status="Pending")
    manager.add_task("Read report", priority="Low", completion_status="Completed")

    assert [task.description for task in manager.search("NOTES")] == ["Prepare notes"]
    assert [task.description for task in manager.filter_tasks(priority="low")] == ["Read report"]
    assert [task.description for task in manager.filter_tasks(status="completed")] == ["Read report"]


def test_task_manager_crud_updates_status_and_deletes():
    manager = TaskManager()
    task = manager.add_task("Draft summary")

    updated = manager.update_task(task.task_id, description="Write summary", completion_status="Completed")
    assert updated is not None
    assert updated.description == "Write summary"
    assert manager.delete_task(task.task_id)
    assert manager.to_dataframe().empty


def test_task_manager_exports_csv_and_excel(tmp_path):
    manager = TaskManager()
    manager.add_task("Send notes", deadline="Tomorrow", priority="High")

    csv_path = manager.export_csv(tmp_path / "tasks.csv")
    excel_path = manager.export_excel(tmp_path / "tasks.xlsx")
    csv_frame = pd.read_csv(csv_path)
    excel_frame = pd.read_excel(excel_path)

    assert csv_frame.iloc[0]["Task Description"] == "Send notes"
    assert excel_frame.iloc[0]["Task ID"] == "T001"
