"""Composable context management for agents."""

from __future__ import annotations

import copy
import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from uni_agent.agents.base import AgentConfig, AgentResult
from uni_agent.agents.react.model import OpenAICompatibleChatModel
from uni_agent.tools import Toolbox

if TYPE_CHECKING:
    from uni_agent.sandbox import Sandbox

logger = logging.getLogger(__name__)

__all__ = [
    "ContextManager",
    "ContextManagerConfig",
    "ContextManagerResult",
    "ContextStepOutput",
    "ContextTurnOutput",
]

_FINISH_TOOLS = {"finish", "submit"}


class ContextManagerConfig(AgentConfig):
    """Configuration shared by agents that explicitly manage their context."""

    tools: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Host-side tools available to every context step.",
    )
    max_steps: int = Field(default=50, gt=0, description="Maximum model calls across all context segments.")
    action_timeout: float | None = Field(
        default=None,
        description="Per-tool-call timeout; None defers to the tool's own timeout.",
    )


class ContextTurnOutput(BaseModel):
    """One model turn inside a context segment."""

    step_idx: int
    response: str = ""
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    done: bool = False
    exit_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ContextStepOutput(BaseModel):
    """One independently materialized context segment."""

    prompt_messages: list[dict[str, Any]] = Field(default_factory=list)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    steps: list[ContextTurnOutput] = Field(default_factory=list)
    reward: float = 0.0
    execution_time: float = 0.0

    def set_prompt_messages(self, messages: list[dict[str, Any]]) -> None:
        self.prompt_messages = copy.deepcopy(messages)

    def set_messages(self, messages: list[dict[str, Any]]) -> None:
        self.messages = copy.deepcopy(messages)

    def set_execution_time(self, execution_time: float) -> None:
        self.execution_time = execution_time

    def add_step(self, step_output: ContextTurnOutput) -> None:
        self.steps.append(step_output)

    def set_reward(self, reward: float) -> None:
        self.reward = reward

    def get_reward(self) -> float:
        return self.reward


class ContextManagerResult(BaseModel):
    """Aggregate result of an explicitly context-managed agent execution."""

    run_id: str
    execution_time: float = 0.0
    trajectory: list[ContextStepOutput] = Field(default_factory=list)
    final_state: ContextStepOutput = Field(default_factory=ContextStepOutput)
    total_steps: int = 0

    def set_reward(self, reward: float) -> None:
        """Assign a workflow-level reward to every context segment."""

        for step in self.trajectory:
            step.set_reward(reward)


class ContextManager(ABC):
    """Mixin that gives an :class:`Agent` explicit context-management methods.

    Put this mixin before ``Agent`` in the base-class list so its cooperative
    initializer reaches ``Agent.__init__``::

        class MyAgent(ContextManager, Agent):
            async def context_management(self, raw_data):
                await self.update_context(...)
                await self.step()

    Each :meth:`update_context` call finalizes the current trajectory segment.
    The gateway recognizes the next unrelated message prefix as a new chain,
    allowing all context segments to participate in training.
    """

    def __init__(self, config: ContextManagerConfig | None = None) -> None:
        super().__init__(config)  # type: ignore[misc]
        self._model: OpenAICompatibleChatModel | None = None
        self._toolbox: Toolbox | None = None
        self._trajectory: list[ContextStepOutput] = []
        self._current_context_step: ContextStepOutput | None = None
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
        """Initialize shared runtime state and execute the agent's policy."""

        cfg = self._context_config
        if cfg.model.base_url is None:
            raise ValueError(f"{cfg.name}: config.model.base_url is not set (the endpoint the policy calls)")

        self._trajectory = []
        self._current_context_step = None
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

        context_input = dict(raw_data or {})
        context_input.setdefault("prompt", copy.deepcopy(messages))
        started_at = time.perf_counter()
        try:
            async with self._toolbox.entered(retry=3, timeout=60):
                await self.context_management(context_input)
                self._collect_context_step()
        finally:
            await self._model.aclose()

        result = ContextManagerResult(
            run_id=str(uuid.uuid4()),
            execution_time=time.perf_counter() - started_at,
            trajectory=self._trajectory,
            final_state=self._trajectory[-1] if self._trajectory else ContextStepOutput(),
            total_steps=self._global_step_idx,
        )
        final_response = result.final_state.steps[-1].response if result.final_state.steps else ""
        transcript = [message for segment in result.trajectory for message in segment.messages]
        return AgentResult(
            output={"response": final_response, "context_manager_result": result},
            transcript=transcript,
            info={
                "execution_time": result.execution_time,
                "total_steps": result.total_steps,
                "num_contexts": len(result.trajectory),
            },
        )

    @abstractmethod
    async def context_management(self, raw_data: dict[str, Any]) -> None:
        """Implement the policy using :meth:`update_context` and :meth:`step`."""

    async def update_context(self, messages: list[dict[str, Any]]) -> None:
        """Finalize the current segment and continue from a newly built context."""

        self._collect_context_step()
        self._track_context_step()
        assert self._current_context_step is not None
        self._current_context_step.set_prompt_messages(messages)
        self.messages = copy.deepcopy(messages)
        for message in self.messages:
            logger.info("%s PROMPT:\n%s", str(message.get("role", "")).upper(), message.get("content", ""))

    async def step(self, sampling_params: dict[str, Any] | None = None) -> ContextTurnOutput:
        """Run one model call, optionally dispatching returned tool calls."""

        if self._current_context_step is None:
            raise RuntimeError("Please call update_context() before calling the first step()")
        if self._model is None or self._toolbox is None:
            raise RuntimeError("ContextManager runtime is not initialized; call run() through a Task")

        cfg = self._context_config
        if self._global_step_idx >= cfg.max_steps:
            raise RuntimeError(f"ContextManager reached max_steps={cfg.max_steps}")

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

        output = ContextTurnOutput(
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
        self._current_context_step.add_step(output)
        return output

    def get_global_step_idx(self) -> int:
        """Return the number of model calls across all context segments."""

        return self._global_step_idx

    def get_current_step_idx(self) -> int:
        """Return the number of model calls in the current context segment."""

        return self._step_idx

    def get_current_context_step(self) -> ContextStepOutput:
        """Return the segment currently being constructed."""

        if self._current_context_step is None:
            raise RuntimeError("No context is active; call update_context() first")
        return self._current_context_step

    @property
    def _context_config(self) -> ContextManagerConfig:
        config = getattr(self, "config", None)
        if not isinstance(config, ContextManagerConfig):
            raise TypeError("ContextManager requires an Agent config derived from ContextManagerConfig")
        return config

    def _track_context_step(self) -> None:
        self._current_context_step = ContextStepOutput()
        self._step_idx = 0
        self._interaction_start = time.perf_counter()

    def _collect_context_step(self) -> None:
        if self._current_context_step is None or not self._current_context_step.steps:
            return
        self._current_context_step.set_messages(self.messages)
        self._current_context_step.set_execution_time(time.perf_counter() - self._interaction_start)
        self._trajectory.append(self._current_context_step)
        self._current_context_step = None

    def _default_sampling_params(self) -> dict[str, Any]:
        return self._context_config.model.sampling_params()

    def _sampling_params_for_step(self, overrides: dict[str, Any] | None) -> dict[str, Any]:
        cfg = self._context_config
        params = self._default_sampling_params()
        params.update(overrides or {})

        max_tokens = params.get("max_tokens", cfg.model.max_tokens_per_turn)
        if cfg.model.max_total_tokens is not None:
            remaining = cfg.model.max_total_tokens - self._total_completion_tokens
            if remaining <= 0:
                raise RuntimeError(f"ContextManager reached max_total_tokens={cfg.model.max_total_tokens}")
            max_tokens = min(max_tokens, remaining) if max_tokens is not None else remaining
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        return params
