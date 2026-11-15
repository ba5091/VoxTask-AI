from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def create_priority_chart(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))

    if df.empty:
        ax.text(0.5, 0.5, "No tasks", ha="center", va="center")
        ax.axis("off")
        return fig

    counts = df["Priority"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
    colors = ["#d9534f", "#f0ad4e", "#5cb85c"]
    bars = ax.bar(counts.index, counts.values, color=colors)
    ax.set_title("Priority Distribution")
    ax.set_ylabel("Count")

    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.05, str(value), ha="center", va="bottom")

    fig.tight_layout()
    return fig


def create_dashboard(df: pd.DataFrame):
    if df.empty:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for ax in axes.flat:
            ax.text(0.5, 0.5, "No tasks available", ha="center", va="center")
            ax.axis("off")
        fig.tight_layout()
        return fig

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    priority_counts = df["Priority"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
    axes[0, 0].bar(priority_counts.index, priority_counts.values, color=["#d9534f", "#f0ad4e", "#5cb85c"])
    axes[0, 0].set_title("Priority distribution")
    axes[0, 0].set_ylabel("Tasks")

    status_counts = df["Completion Status"].value_counts().reindex(["Pending", "Completed"], fill_value=0)
    axes[0, 1].pie(status_counts.values, labels=status_counts.index, autopct="%1.0f%%", startangle=90)
    axes[0, 1].set_title("Completion status")

    deadline_counts = (
        df["Deadline"]
        .fillna("Not specified")
        .replace("", "Not specified")
        .value_counts()
        .sort_values(ascending=False)
        .head(6)
    )
    axes[1, 0].barh(deadline_counts.index[::-1], deadline_counts.values[::-1], color="#0891b2")
    axes[1, 0].set_title("Deadline labels")
    axes[1, 0].set_xlabel("Tasks")

    comparison = (
        df.groupby(["Priority", "Completion Status"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=["High", "Medium", "Low"], fill_value=0)
        .reindex(columns=["Pending", "Completed"], fill_value=0)
    )
    comparison.plot(
        kind="bar",
        ax=axes[1, 1],
        color=["#818cf8", "#34d399"],
        rot=0,
    )
    axes[1, 1].set_title("Priority by status")
    axes[1, 1].set_ylabel("Tasks")
    axes[1, 1].legend(title="Status")

    fig.tight_layout()
    return fig
