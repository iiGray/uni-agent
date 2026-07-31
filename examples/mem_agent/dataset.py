"""Dataset adapter for the MemAgent training recipe."""

from __future__ import annotations

from typing import Any

from verl.utils.dataset.rl_dataset import RLHFDataset


def context_to_text(context: Any) -> str:
    """Normalize common long-context dataset shapes into plain text."""

    if context is None:
        return ""
    if isinstance(context, str):
        return context
    if isinstance(context, dict):
        return "\n".join(f"{key}: {context_to_text(value)}" for key, value in context.items())
    if isinstance(context, list | tuple):
        return "\n".join(part for value in context if (part := context_to_text(value)))
    return str(context)


def build_task_config(row: dict[str, Any]) -> dict[str, Any]:
    """Build the sample-wise Task Config consumed by ``run_task``."""

    existing_tools_kwargs = row.get("tools_kwargs")
    if not isinstance(existing_tools_kwargs, dict):
        existing_tools_kwargs = {}
    existing_task = existing_tools_kwargs.get("task")
    task = dict(existing_task) if isinstance(existing_task, dict) else {}
    task.setdefault("name", "hotpotqa")

    metadata = dict(task.get("metadata") or {})
    context = row.get("context")
    if context is None and isinstance(row.get("extra_info"), dict):
        context = row["extra_info"].get("context")
    metadata["context"] = context_to_text(context)

    if row.get("reward_model") is not None:
        metadata["reward_model"] = row["reward_model"]
    elif row.get("ground_truth") is not None:
        metadata["reward_model"] = {"ground_truth": row["ground_truth"]}

    task["metadata"] = metadata
    return task


class MemAgentDataset(RLHFDataset):
    """Pack long context and answers into the standard Task runner payload."""

    def __getitem__(self, item):
        row = super().__getitem__(item)
        tools_kwargs = dict(row.get("tools_kwargs") or {})
        tools_kwargs["task"] = build_task_config(row)
        row["tools_kwargs"] = tools_kwargs
        return row
