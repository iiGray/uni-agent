"""Task wrapper for MemAgent training and evaluation."""

from __future__ import annotations

from pydantic import Field

from uni_agent.context_management import ContextManagerResult

from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task
from .reward import compute_score


class MemAgentTaskConfig(TaskConfig):
    name: str = "mem_agent"
    ground_truth: list[str] = Field(
        default_factory=list,
        description="Accepted answers; falls back to metadata.reward_model.ground_truth.",
    )


@register_task("mem_agent")
class MemAgentTask(Task):
    """Run MemAgent and broadcast its final-answer reward to every context chain."""

    config_model = MemAgentTaskConfig

    async def run(self) -> TaskResult:
        cfg: MemAgentTaskConfig = self.config  # type: ignore[assignment]
        raw_data = dict(cfg.metadata)
        raw_data["prompt"] = cfg.prompt

        async with self.build_sandbox() as sandbox:
            agent = self.build_agent()
            agent_result = await agent.run(
                sandbox=sandbox,
                messages=cfg.prompt,
                raw_data=raw_data,
            )

        response = str(agent_result.output.get("response", ""))
        ground_truths = list(cfg.ground_truth)
        if not ground_truths:
            reward_model = cfg.metadata.get("reward_model", {})
            raw_ground_truth = reward_model.get("ground_truth", []) if isinstance(reward_model, dict) else []
            if isinstance(raw_ground_truth, str):
                ground_truths = [raw_ground_truth]
            elif isinstance(raw_ground_truth, list | tuple):
                ground_truths = [str(answer) for answer in raw_ground_truth]
            elif raw_ground_truth is not None:
                ground_truths = [str(raw_ground_truth)]

        reward = compute_score(response, ground_truths)
        context_manager_result = agent_result.output.get("context_manager_result")
        if isinstance(context_manager_result, ContextManagerResult):
            context_manager_result.set_reward(reward)

        return TaskResult(
            reward=reward,
            accuracy=reward,
            extra_info={
                "response": response,
                "num_contexts": agent_result.info.get("num_contexts", 0),
                "total_steps": agent_result.info.get("total_steps", 0),
            },
        )
