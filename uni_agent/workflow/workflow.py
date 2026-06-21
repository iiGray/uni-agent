"""Workflow orchestration using programmatic step-by-step control.

This module provides the :class:`WorkflowBase` class which users can subclass
to define custom programmatic workflows, seamlessly integrating with
:class:`~uni_agent.interaction.AgentInteraction`.
"""

from __future__ import annotations

import time
import uuid, importlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any
from transformers import PreTrainedTokenizerBase
from pydantic import BaseModel, Field

from uni_agent.async_logging import get_logger
from uni_agent.skills.manager import SkillsManager
from uni_agent.utils import auto_await, simple_timer

from uni_agent.interaction.env import ActionIncorrectSyntaxError, ActionTimeoutError, AgentEnv, TerminalNotAliveError
from uni_agent.interaction.model import AgentChatModel, MaxTokenExceededError

from uni_agent.interaction.tool_parser import FunctionCallFormatError
from uni_agent.interaction.tool_schemas import OpenAIFunctionToolCall

from uni_agent.interaction.interaction import StepOutput, ToolStatus, ToolResult, fast_deepcopy, AgentInteraction

if TYPE_CHECKING:
    from uni_agent.interaction.env import AgentEnv
    from uni_agent.interaction.model import AgentChatModel, OpenAICompatibleChatModel

    from uni_agent.skills import SkillsManager
    from uni_agent.interaction.tools_manager import ToolsManager


class WorkflowStepOutput(BaseModel):
    prompt_messages: list[dict[str, str]] = Field(default_factory = list)
    messages: list[dict[str, str]] = Field(default_factory = list)
    rollout_cache: dict[str, Any] = Field(default_factory = dict)
    steps: list[StepOutput] = Field(default_factory = list)
    _reward: float = 0.0
    _execution_time: float = 0.0

    def set_prompt_messages(self, messages):
        self.prompt_messages = [m.copy() for m in messages]
    
    def set_messages(self, messages):
        self.messages = messages

    def set_rollout_cache(self, rollout_cache):
        self.rollout_cache = rollout_cache

    def set_execution_time(self, time: float):
        self._execution_time = time

    def add_step(self, step_output: StepOutput):
        self.steps.append(step_output)

    def set_reward(self, reward: float):
        self._reward = reward

    def get_reward(self) -> float:
        return self._reward

    def to_interaction_result(self) -> dict:
        return dict(
            trajectory = self.steps,
            rollout_cache = self.rollout_cache,
            execution_time = self._execution_time,
            messages = self.messages,
            metrics = dict(),
            reward_score = self.get_reward()
        )


class AgentWorkflowResult(BaseModel):
    """Aggregate result of a multi-step workflow."""

    run_id: str
    execution_time: float = 0.0
    trajectory: list[WorkflowStepOutput] = Field(default_factory=list)
    rollout_caches: list[Any] = Field(default_factory=list)
    final_state: WorkflowStepOutput = Field(default_factory = WorkflowStepOutput)
    total_steps: int = 0

    def set_reward(self, reward: float):
        for s in self.trajectory:
            s.set_reward(reward)


class AgentWorkflowBase(ABC):
    """Abstract base class for programmatic agentic workflows.

    Users subclass this to define custom workflows like MemAgent or RememR1.
    The `run` method is where you instantiate `AgentInteraction` and manually
    loop over `await interaction.step()`, inserting your custom evaluation and
    context-switching logic.
    """

    def __init__(
        self,
        config_dict: dict,
        run_id: str,
        env: AgentEnv,
        model: AgentChatModel | OpenAICompatibleChatModel,
        tokenizer: PreTrainedTokenizerBase,
        tools_manager: ToolsManager,
        raw_data: dict,
        action_timeout: int = 60,
        timeout_budget: int = 3,
        max_turns: int = 50,
        skills_manager: SkillsManager | None = None,
        chat_mode: bool = False,
        ** kwargs
    ):
        self.config_dict = config_dict
        self.run_id = run_id
        self.env = env
        self.model = model
        self.tokenizer= tokenizer
        self.tools_manager = tools_manager
        self.skills_manager = skills_manager
        self.raw_data = raw_data
        '''The initial prompt messages'''

        self.action_timeout = action_timeout
        self.timeout_budget = timeout_budget
        self.max_turns = max_turns
        self.chat_mode = chat_mode
        self.logger = get_logger("workflow", run_id)



    @classmethod
    def cls_from_name(cls, workflow_class: str) -> type[AgentWorkflowBase]:
        module_path, class_name = workflow_class.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def inject_skills_manifest(self, messages: list[dict[str, str]]) -> None:
        """Append the skills manifest to the first system message.

        The manifest lists each discovered skill (name + description +
        path to its SKILL.md) so the model knows what is available and
        how to load it on demand. Skill *bodies* are not in the prompt --
        they live as real files on disk (read lazily, progressive
        disclosure).

        Call this exactly once, after ``AgentEnv.install_skills`` has
        populated ``runtime_paths``. The method is **not** idempotent --
        calling it twice will append the manifest twice. The single
        in-tree caller (``UniAgentLoop.run``) already enforces this.
        """
        if self.skills_manager is None:
            return
        manifest = self.skills_manager.build_manifest()
        if not manifest:
            return

        block = "\n\n" + manifest
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content") or ""
                msg["content"] = content + block
                return
        messages.insert(0, {"role": "system", "content": manifest})

    async def step(self, sampling_params: dict = dict()) -> StepOutput:
        assert self._has_prepared_cache, (
            "Please call `update_context` before calling the first `step`"
        )
        self._global_step_idx += 1
        self._step_idx += 1
        kwargs = dict(sampling_params = sampling_params) if sampling_params is not None else dict()
        step_output = AgentInteraction.step(self, self._step_idx, **kwargs)
        self.get_current_workflow_step().add_step(step_output)
        
        return step_output
    
    def get_global_step_idx(self) -> int:
        '''The steps that has been executed across all interactions'''
        return self._global_step_idx

    def get_current_step_idx(self) -> int:
        '''steps in a single interaction'''
        return self._step_idx

    def _track_workflow_step(self):
        self._current_workflow_step = WorkflowStepOutput()
        self._step_idx = 0
        self._has_prepared_cache = False

        self._interaction_start = time.perf_counter()
    
    def _collect_workflow_step(self):
        if not (self._current_workflow_step.steps): return

        step = self.get_current_workflow_step()
        step.set_messages(self.messages)
        step.set_rollout_cache(self.rollout_cache)
        step.set_execution_time(time.perf_counter() - self._interaction_start)

        self._trajectory.append(self._current_workflow_step)


    def get_current_workflow_step(self) -> WorkflowStepOutput:
        return self._current_workflow_step


    async def update_context(self, messages: list[dict[str, str]], insert_skill: bool = True):
        '''Reset the rollout cache'''
        self._collect_workflow_step()
        self._track_workflow_step()
        self.get_current_workflow_step().set_prompt_messages(messages)
        self.logger.info(("Context Updated. New Prompt:\n") if self.trajectory else ("Inital Prompt:\n"))
        if insert_skill:
            messages = self.inject_skills_manifest(messages)
        for message in messages:
            self.logger.info(f"{message['role'].upper()} PROMPT:\n{message['content']}")
        
        self.rollout_cache = await self.model.prepare_rollout_cache(messages)
        self._rollout_caches.append(self.rollout_cache)

        self._has_prepared_cache = True
        self.messages = messages # for AgentInteraction.step


    async def _run(self) -> AgentWorkflowResult:
        self._trajectory: list[WorkflowStepOutput] = []
        self._rollout_caches: list = []
        self._global_step_idx = 0
        start_time = time.perf_counter()
        await self.run(self.raw_data)
        self._collect_workflow_step()
        execution_time = time.perf_counter() - start_time

        return AgentWorkflowResult(
            run_id = self.run_id,
            execution_time = execution_time,
            trajectory = self._trajectory,
            rollout_caches = self._rollout_caches,
            total_steps = self._global_step_idx,
            final_state = self.get_current_workflow_step()
        )

    @abstractmethod
    async def run(self, raw_data: dict):
        """
        Execute the workflow by calling `step` and `update_context` (no need to return anything)
        
        raw_data: {
            'raw_prompt': ... (Created by RLHFDataset)
            ...
            # The same as defined in your Dataset

        }
        """
