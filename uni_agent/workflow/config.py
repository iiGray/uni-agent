"""Pydantic configuration models for the workflow module."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

class WorkflowConfig(BaseModel):
    """Top-level workflow configuration.

    Instead of defining rigid steps and routers, this config allows
    you to specify a custom `Workflow` class to instantiate and provides
    arbitrary extra arguments for it.
    """
    
    workflow_class: str = Field(
        default="uni_agent.workflow.workflow.WorkflowBase",
        description="The python path to your custom WorkflowBase subclass."
    )

    tools: list[dict[str, str]] = Field(
        default_factory=list,
        description=(
            "Union of all tool names used across all steps.  These are installed once "
            "before the first step.  Each item is a dict with a ``name`` key, e.g. "
            "``{\"name\": \"search\"}``."
        ),
    )
    tool_parser: str = Field(
        default="qwen3_coder",
        description="Tool-call parser name (see :class:`ToolsManagerConfig`).",
    )


