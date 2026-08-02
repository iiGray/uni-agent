"""Dataset adapter for the MemAgent training recipe."""

from __future__ import annotations

from typing import Any

from verl.utils.dataset.rl_dataset import RLHFDataset

DEFAULT_CONTEXT_CHUNK_SIZE = 5_000


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


def split_context_into_token_chunks(context: Any, *, tokenizer, chunk_size: int) -> list[str]:
    """Tokenize a long context once in the Dataset and return decoded text chunks."""

    if chunk_size <= 0:
        raise ValueError(f"context_chunk_size must be positive, got {chunk_size}")
    context_ids = tokenizer.encode(context_to_text(context), add_special_tokens=False)
    return [
        tokenizer.decode(context_ids[offset : offset + chunk_size], skip_special_tokens=True)
        for offset in range(0, len(context_ids), chunk_size)
    ]


def build_task_config(row: dict[str, Any], *, tokenizer, chunk_size: int) -> dict[str, Any]:
    """Build the sample-wise Task Config consumed by ``run_task``."""

    existing_tools_kwargs = row.get("tools_kwargs")
    if not isinstance(existing_tools_kwargs, dict):
        existing_tools_kwargs = {}
    existing_task = existing_tools_kwargs.get("task")
    task = dict(existing_task) if isinstance(existing_task, dict) else {}
    task.setdefault("name", "hotpotqa")

    metadata = dict(task.get("metadata") or {})
    metadata.pop("context", None)
    context = row.get("context")
    if context is None and isinstance(row.get("extra_info"), dict):
        context = row["extra_info"].get("context")
    metadata["chunks"] = split_context_into_token_chunks(context, tokenizer=tokenizer, chunk_size=chunk_size)

    if row.get("reward_model") is not None:
        metadata["reward_model"] = row["reward_model"]
    elif row.get("ground_truth") is not None:
        metadata["reward_model"] = {"ground_truth": row["ground_truth"]}

    task["metadata"] = metadata
    return task


class MemAgentDataset(RLHFDataset):
    """Token-chunk long context and pack it into the standard Task runner payload."""

    def __getitem__(self, item):
        row = super().__getitem__(item)
        chunk_size = int(self.config.get("context_chunk_size", DEFAULT_CONTEXT_CHUNK_SIZE))
        tools_kwargs = dict(row.get("tools_kwargs") or {})
        tools_kwargs["task"] = build_task_config(row, tokenizer=self.tokenizer, chunk_size=chunk_size)
        row["tools_kwargs"] = tools_kwargs

        # The task payload now owns the decoded chunks; do not send the original
        # long context through the rollout path a second time.
        row.pop("context", None)
        if isinstance(row.get("extra_info"), dict) and "context" in row["extra_info"]:
            row["extra_info"] = dict(row["extra_info"])
            row["extra_info"].pop("context", None)
        return row
