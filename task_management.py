from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


@dataclass
class TaskRecord:
    task_id: str
    description: str
    deadline: str = "Not specified"
    priority: str = "Medium"
    completion_status: str = "Pending"
    source: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "Task ID": self.task_id,
            "Task Description": self.description,
            "Deadline": self.deadline,
            "Priority": self.priority,
            "Completion Status": self.completion_status,
        }


class TaskManager:
    def __init__(self, tasks: Iterable[TaskRecord] | None = None):
        self.tasks: list[TaskRecord] = []
        if tasks:
            for item in tasks:
                self.add_task(
                    description=item.description,
                    deadline=item.deadline,
                    priority=item.priority,
                    completion_status=item.completion_status,
                    source=item.source,
                    task_id=item.task_id,
                )

    def add_task(
        self,
        description: str,
        deadline: str = "Not specified",
        priority: str = "Medium",
        completion_status: str = "Pending",
        source: str = "",
        task_id: str | None = None,
    ) -> TaskRecord:
        cleaned_description = str(description or "").strip()
        if not cleaned_description:
            raise ValueError("Task description cannot be empty.")

        if task_id is None:
            task_id = self._next_task_id()
        elif any(task.task_id == task_id for task in self.tasks):
            raise ValueError(f"Task ID {task_id!r} already exists.")
        if priority not in {"High", "Medium", "Low"}:
            raise ValueError("Priority must be High, Medium, or Low.")
        if completion_status not in {"Pending", "Completed"}:
            raise ValueError("Completion status must be Pending or Completed.")

        record = TaskRecord(
            task_id=task_id,
            description=cleaned_description,
            deadline=str(deadline or "Not specified"),
            priority=str(priority or "Medium"),
            completion_status=str(completion_status or "Pending"),
            source=str(source or ""),
        )
        self.tasks.append(record)
        return record

    def _next_task_id(self) -> str:
        existing_ids = {
            int(match.group(1))
            for task in self.tasks
            if (match := re.fullmatch(r"T(\d+)", task.task_id))
        }
        next_index = max(existing_ids, default=0) + 1
        return f"T{next_index:03d}"

    def delete_task(self, task_id: str) -> bool:
        initial_count = len(self.tasks)
        self.tasks = [task for task in self.tasks if task.task_id != task_id]
        return len(self.tasks) != initial_count

    def update_task(self, task_id: str, **updates: Any) -> TaskRecord | None:
        for task in self.tasks:
            if task.task_id == task_id:
                allowed = {
                    "description",
                    "deadline",
                    "priority",
                    "completion_status",
                    "source",
                }
                unknown = set(updates) - allowed
                if unknown:
                    raise ValueError(f"Unsupported task fields: {', '.join(sorted(unknown))}")
                if "description" in updates and not str(updates["description"] or "").strip():
                    raise ValueError("Task description cannot be empty.")
                if "priority" in updates and updates["priority"] not in {"High", "Medium", "Low"}:
                    raise ValueError("Priority must be High, Medium, or Low.")
                if "completion_status" in updates and updates["completion_status"] not in {"Pending", "Completed"}:
                    raise ValueError("Completion status must be Pending or Completed.")
                for key, value in updates.items():
                    setattr(task, key, value)
                return task
        return None

    def search(self, query: str) -> list[TaskRecord]:
        query_text = (query or "").lower()
        if not query_text:
            return list(self.tasks)

        return [
            task for task in self.tasks if query_text in task.description.lower()
        ]

    def filter_tasks(self, *, priority: str | None = None, status: str | None = None) -> list[TaskRecord]:
        filtered = self.tasks
        if priority:
            filtered = [task for task in filtered if task.priority.lower() == priority.lower()]
        if status:
            filtered = [task for task in filtered if task.completion_status.lower() == status.lower()]
        return filtered

    def sort_tasks(self, by: str = "task_id") -> list[TaskRecord]:
        field_name = by.lower()
        if field_name == "deadline":
            return sorted(self.tasks, key=lambda task: (task.deadline == "Not specified", task.deadline.lower()))
        if field_name == "priority":
            priority_order = {"High": 0, "Medium": 1, "Low": 2}
            return sorted(self.tasks, key=lambda task: (priority_order.get(task.priority, 99), task.description.lower()))
        return sorted(self.tasks, key=lambda task: task.task_id.lower())

    def to_dataframe(self) -> pd.DataFrame:
        if not self.tasks:
            columns = ["Task ID", "Task Description", "Deadline", "Priority", "Completion Status"]
            return pd.DataFrame(columns=columns)

        records = [task.to_dict() for task in self.tasks]
        df = pd.DataFrame(records)
        if "Task ID" not in df.columns:
            df.insert(0, "Task ID", [task.task_id for task in self.tasks])
        return df[["Task ID", "Task Description", "Deadline", "Priority", "Completion Status"]]

    def export_csv(self, path: str | Path) -> str:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(destination, index=False)
        return str(destination)

    def export_excel(self, path: str | Path) -> str:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_excel(destination, index=False)
        return str(destination)

    @property
    def is_empty(self) -> bool:
        return len(self.tasks) == 0
