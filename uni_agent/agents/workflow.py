"""Programmable agents with explicit context management."""

from __future__ import annotations

import copy
import logging
import time
import uuid
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from uni_agent.tools import Toolbox

from .base import Agent, AgentConfig, AgentResult
from .react.model import OpenAICompatibleChatModel

if TYPE_CHECKING:
    from uni_agent.sandbox import Sandbox

logger = logging.getLogger(__name__)

__all__ = [
    "AgentWorkflowResult",
    "Workflow",
    "WorkflowConfig",
    "WorkflowStepOutput",
    "WorkflowTurnOutput",
]

_FINISH_TOOLS = {"finish", "submit"}


class WorkflowConfig(AgentConfig):
    """Configuration shared by context-managed workflow agents."""

    name: str = "workflow"
    tools: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Host-side tools available to every workflow step.",
    )
    max_steps: int = Field(default=50, description="Maximum model calls across all context segments.")
    action_timeout: float | None = Field(
        default=None,
        description="Per-tool-call timeout; None defers to the tool's own timeout.",
    )


class WorkflowTurnOutput(BaseModel):
    """One model turn inside a context segment."""

    step_idx: int
    response: str = ""
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    done: bool = False
    exit_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


class WorkflowStepOutput(BaseModel):
    """One independently materialized context segment."""

    prompt_messages: list[dict[str, Any]] = Field(default_factory=list)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    steps: list[WorkflowTurnOutput] = Field(default_factory=list)
    reward: float = 0.0
    execution_time: float = 0.0

    def set_prompt_messages(self, messages: list[dict[str, Any]]) -> None:
        self.prompt_messages = copy.deepcopy(messages)

    def set_messages(self, messages: list[dict[str, Any]]) -> None:
        self.messages = copy.deepcopy(messages)

    def set_execution_time(self, execution_time: float) -> None:
        self.execution_time = execution_time

    def add_step(self, step_output: WorkflowTurnOutput) -> None:
        self.steps.append(step_output)

    def set_reward(self, reward: float) -> None:
        self.reward = reward

    def get_reward(self) -> float:
        return self.reward


class AgentWorkflowResult(BaseModel):
    """Aggregate result of a context-managed agent execution."""

    run_id: str
    execution_time: float = 0.0
    trajectory: list[WorkflowStepOutput] = Field(default_factory=list)
    final_state: WorkflowStepOutput = Field(default_factory=WorkflowStepOutput)
    total_steps: int = 0

    def set_reward(self, reward: float) -> None:
        for step in self.trajectory:
            step.set_reward(reward)


class Workflow(Agent):
    """Base agent for custom context-management policies.

    Subclasses implement :meth:`workflow` and call :meth:`update_context` whenever
    they want to finalize the current trajectory segment and start a new one.
    Calls to :meth:`step` generate against the current context. The gateway sees
    each context as a separate chain, so all segments can participate in training.
    """

    config_model = WorkflowConfig

    def __init__(self, config: WorkflowConfig | None = None) -> None:
        super().__init__(config)
        self._model: OpenAICompatibleChatModel | None = None
        self._toolbox: Toolbox | None = None
        self._trajectory: list[WorkflowStepOutput] = []
        self._current_workflow_step: WorkflowStepOutput | None = None
        self._global_step_idx = 0
        self._step_idx = 0
        self._total_completion_tokens = 0
        self._interaction_start = 0.0
        self.messages: list[dict[str, Any]] = []

    async def run(
        self,
        *,
        sandbox: Sandbox,
        messages: list[dict[str, Any]],
        raw_data: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Initialize shared runtime state and execute the subclass workflow."""

        cfg: WorkflowConfig = self.config  # type: ignore[assignment]
        if cfg.model.base_url is None:
            raise ValueError(f"{cfg.name}: config.model.base_url is not set (the endpoint the policy calls)")

        self._trajectory = []
        self._current_workflow_step = None
        self._global_step_idx = 0
        self._step_idx = 0
        self._total_completion_tokens = 0
        self.messages = []

        self._toolbox = Toolbox.from_specs(cfg.tools, sandbox=sandbox)
        self._model = OpenAICompatibleChatModel(
            base_url=cfg.model.base_url,
            api_key=cfg.model.api_key,
            model_name=cfg.model.model_name,
            sampling_params=self._default_sampling_params(),
            tools_schemas=self._toolbox.schemas(),
        )

        workflow_input = dict(raw_data or {})
        workflow_input.setdefault("prompt", copy.deepcopy(messages))
        started_at = time.perf_counter()
        try:
            async with self._toolbox.entered(retry=3, timeout=60):
                await self.workflow(workflow_input)
                self._collect_workflow_step()
        finally:
            await self._model.aclose()

        result = AgentWorkflowResult(
            run_id=str(uuid.uuid4()),
            execution_time=time.perf_counter() - started_at,
            trajectory=self._trajectory,
            final_state=self._trajectory[-1] if self._trajectory else WorkflowStepOutput(),
            total_steps=self._global_step_idx,
        )
        final_response = ""
        if result.final_state.steps:
            final_response = result.final_state.steps[-1].response
        transcript = [message for segment in result.trajectory for message in segment.messages]
        return AgentResult(
            output={"response": final_response, "workflow_result": result},
            transcript=transcript,
            info={
                "execution_time": result.execution_time,
                "total_steps": result.total_steps,
                "num_contexts": len(result.trajectory),
            },
        )

    @abstractmethod
    async def workflow(self, raw_data: dict[str, Any]) -> None:
        """Implement custom control flow using ``update_context`` and ``step``."""

    async def update_context(
        self,
        messages: list[dict[str, Any]],
        *,
        insert_skill: bool = True,
    ) -> None:
        """Finalize the current segment and continue from a newly built context.

        ``insert_skill`` is retained for source compatibility. Skills are now
        represented by regular tools/configuration in the agent layer.
        """

        del insert_skill
        self._collect_workflow_step()
        self._track_workflow_step()
        assert self._current_workflow_step is not None
        self._current_workflow_step.set_prompt_messages(messages)
        self.messages = copy.deepcopy(messages)
        for message in self.messages:
            logger.info("%s PROMPT:\n%s", str(message.get("role", "")).upper(), message.get("content", ""))

    async def step(self, sampling_params: dict[str, Any] | None = None) -> WorkflowTurnOutput:
        """Run one model call, optionally dispatching returned tool calls."""

        if self._current_workflow_step is None:
            raise RuntimeError("Please call `update_context` before calling the first `step`")
        if self._model is None or self._toolbox is None:
            raise RuntimeError("Workflow runtime is not initialized; call `run` through a Task")

        cfg: WorkflowConfig = self.config  # type: ignore[assignment]
        if self._global_step_idx >= cfg.max_steps:
            raise RuntimeError(f"Workflow reached max_steps={cfg.max_steps}")

        self._global_step_idx += 1
        self._step_idx += 1
        params = self._sampling_params_for_step(sampling_params)
        content, tool_calls, generation_info = await self._model.query(
            self.messages,
            sampling_params=params,
        )
        self._total_completion_tokens += generation_info["completion_tokens"]

        assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        self.messages.append(assistant_message)

        output = WorkflowTurnOutput(
            step_idx=self._step_idx,
            response=content,
            prompt_tokens=generation_info["prompt_tokens"],
            completion_tokens=generation_info["completion_tokens"],
        )

        saw_finish = False
        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            name = str(function.get("name", ""))
            result = await self._toolbox.call(
                name,
                function.get("arguments"),
                timeout=cfg.action_timeout,
            )
            output.tool_results.append({"name": name, "status": result.status, "text": result.text})
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "name": name,
                    "content": result.to_observation(),
                }
            )
            saw_finish = saw_finish or name in _FINISH_TOOLS

        output.done = not tool_calls or saw_finish
        output.exit_reason = "finished" if output.done else "completed"
        self._current_workflow_step.add_step(output)
        return output

    def get_global_step_idx(self) -> int:
        """Return the number of model calls across all context segments."""

        return self._global_step_idx

    def get_current_step_idx(self) -> int:
        """Return the number of model calls in the current context segment."""

        return self._step_idx

    def get_current_workflow_step(self) -> WorkflowStepOutput:
        """Return the segment currently being constructed."""

        if self._current_workflow_step is None:
            raise RuntimeError("No context is active; call `update_context` first")
        return self._current_workflow_step

    def _track_workflow_step(self) -> None:
        self._current_workflow_step = WorkflowStepOutput()
        self._step_idx = 0
        self._interaction_start = time.perf_counter()

    def _collect_workflow_step(self) -> None:
        if self._current_workflow_step is None or not self._current_workflow_step.steps:
            return
        self._current_workflow_step.set_messages(self.messages)
        self._current_workflow_step.set_execution_time(time.perf_counter() - self._interaction_start)
        self._trajectory.append(self._current_workflow_step)
        self._current_workflow_step = None

    def _default_sampling_params(self) -> dict[str, Any]:
        cfg: WorkflowConfig = self.config  # type: ignore[assignment]
        return {
            "temperature": cfg.model.temperature,
            "top_p": cfg.model.top_p,
            "top_k": cfg.model.top_k,
        }

    def _sampling_params_for_step(self, overrides: dict[str, Any] | None) -> dict[str, Any]:
        cfg: WorkflowConfig = self.config  # type: ignore[assignment]
        params = self._default_sampling_params()
        params.update(overrides or {})

        max_tokens = params.get("max_tokens", cfg.model.max_tokens_per_turn)
        if cfg.model.max_total_tokens is not None:
            remaining = cfg.model.max_total_tokens - self._total_completion_tokens
            if remaining <= 0:
                raise RuntimeError(f"Workflow reached max_total_tokens={cfg.model.max_total_tokens}")
            max_tokens = min(max_tokens, remaining) if max_tokens is not None else remaining
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        return params
