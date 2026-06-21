"""Workflow module — programmatic multi-step agent orchestration.

The workflow module lets you build custom multi-step agents (like MemAgent or RememR1)
by subclassing :class:`~uni_agent.workflow.workflow.WorkflowBase`. This gives you
complete turn-by-turn control over the context and history.

Key components
--------------
- :class:`WorkflowConfig` — Pydantic model for YAML-based configuration.
- :class:`WorkflowBase` — Abstract base class for programmatic workflows.
- :class:`WorkflowResult` — The resulting trajectory of a workflow execution.
- :class:`UniAgentWorkflowLoop` — verl :class:`~verl.experimental.agent_loop.agent_loop.AgentLoopBase`
  subclass for RL training.
- :func:`split_workflow_output` — split a merged :class:`AgentLoopOutput` into per-step
  outputs for MemAgent-style per-step GRPO advantage computation.
- :func:`pad_per_step_outputs` — pad per-step output lists to uniform length across rollouts.
"""

from uni_agent.workflow.config import WorkflowConfig
from uni_agent.workflow.workflow import AgentWorkflowBase, AgentWorkflowResult

__all__ = [
    # config
    "WorkflowConfig",
    # core
    "AgentWorkflowBase",
    "AgentWorkflowResult",
]

# Lazy imports for symbols that depend on verl (avoids ImportError at
# package-load time when verl is not installed).


def __getattr__(name: str):
    if name == "AgentWorkflow":
        from .workflow import AgentWorkflowBase
    if name in ("split_workflow_output", "pad_per_step_outputs"):
        from uni_agent.workflow import agent_loop as _mod
        return getattr(_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

