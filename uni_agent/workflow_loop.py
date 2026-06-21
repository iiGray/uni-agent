from __future__ import annotations

import asyncio
import json
import pickle
import uuid
from pathlib import Path
from typing import Any

import yaml, numpy as np

from uni_agent.async_logging import add_file_handler, cleanup_handlers, get_logger
from uni_agent.interaction import (
    AgentChatModel,
    AgentEnv,
    AgentEnvConfig,
    AgentInteraction,
    ToolsManager,
    ToolsManagerConfig,
)
from uni_agent.reward import load_reward_spec
from uni_agent.skills import SkillsManager, SkillsManagerConfig
from uni_agent.agent_loop import _deep_merge
from uni_agent.workflow.config import WorkflowConfig
from uni_agent.workflow.config import WorkflowConfig
from uni_agent.workflow.workflow import WorkflowStepOutput, AgentWorkflowBase, AgentWorkflowResult

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.experimental.agent_loop.utils import resolve_config_path



class UniAgentWorkflowLoop(AgentLoopBase):
    """Agent loop that executes a programmatic multi-step workflow.

    The user should provide a custom `WorkflowBase` subclass in `WorkflowConfig`
    that implements the exact turn-by-turn logic, memory/context switching, and state evaluation.
    """

    _semaphore: asyncio.Semaphore | None = None

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> list[AgentLoopOutput]:
        config_dict = self._init_config(sampling_params, **kwargs)
        self.mask_abnormal_exit_traj = config_dict.get("mask_abnormal_exit_traj", False)
        global_concurrent = config_dict.get("concurrency", 512)
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers
        worker_concurrent = max(global_concurrent // num_workers, 1)
        if UniAgentWorkflowLoop._semaphore is None:
            UniAgentWorkflowLoop._semaphore = asyncio.Semaphore(worker_concurrent)

        self.run_id = str(uuid.uuid4())
        self.logger = get_logger("workflow-loop", run_id=self.run_id)

        self.chat_model = self._init_chat_model(config_dict["model"])

        wf_config = WorkflowConfig(
            config_dict['workflow']['cls'], config_dict['tools'], config_dict['tool_parser']
        )

        self.tools_manager = self._init_tools_manager(
            tools_config_list = wf_config.tools,
            parser = wf_config.tool_parser,
        )
        
        self.skills_manager = self._init_skills_manager(config_dict.get("skills"))
        self.env = self._init_env(config_dict["env"])
        self.output_dir = Path(config_dict["log_dir"]) / self.run_id

        WorkflowClass: type[AgentWorkflowBase] = AgentWorkflowBase.cls_from_name(wf_config.workflow_class)

        self.workflow: AgentWorkflowBase = WorkflowClass(
            config_dict = config_dict['workflow']['config'],
            run_id = self.run_id,
            env = self.env,
            model = self.chat_model,
            tools_manager = self.tools_manager,
            raw_data = kwargs,
            skills_manager = self.skills_manager,
            ** wf_config['workflow']
        )

        if config_dict["reward"] is not None:
            reward_config = {
                **config_dict["reward"],
                "raw_data": kwargs,
                "run_id": self.run_id,
                "env": self.env,
            }
            self.reward_spec = load_reward_spec(reward_config)
        else:
            self.reward_spec = None


        async with self._semaphore:
            add_file_handler(self.output_dir / "run.log", self.run_id)

            self.logger.info(f"model name: {self.config.actor_rollout_ref.model.path}")
            self.logger.info(f"sampling_params: {sampling_params}")
            self.logger.info(f"environment config: {config_dict['env']}")
            self.logger.info(f"tools config: {config_dict['tools']}")
            self.logger.info(f"interaction config: {config_dict['interaction']}")
            self.logger.info(f"mask_abnormal_exit_traj: {self.mask_abnormal_exit_traj}")
            self.logger.info(f"output_dir: {self.output_dir}")
            try:
                await self.env.start()

                self.chat_model.set_tools_schemas(self.tools_manager.tools_schemas)
                await self.env.install_tools(self.tools_manager.tools)
                if self.skills_manager is not None:
                    await self.env.install_skills(self.skills_manager)

                workflow_result: AgentWorkflowResult = await self.workflow._run()

                if self.reward_spec is not None:
                    await self.reward_spec.set_workflow_reward(
                        workflow_result = workflow_result,
                    ) #TODO: check type is list
                else:
                    self.logger.warning("No reward spec is provided, reward score will be set to -100")
                    workflow_result.set_reward(-100)

                self._save_workflow_result(workflow_result)

                output = await self._convert_to_per_step_outputs(workflow_result)

            except Exception:
                self.logger.critical("Workflow failed before producing result", exc_info=True)
                output = await self._build_empty_agent_output(exit_reason="workflow_failed")
            finally:
                await self.env.close()
                cleanup_handlers(self.run_id)

            return output


    async def _build_empty_agent_output(self, exit_reason: str) -> list[AgentLoopOutput]:

        dummy_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if dummy_token_id is None:
            dummy_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(dummy_token_id, list):
            dummy_token_id = dummy_token_id[0] if dummy_token_id else 0
        if dummy_token_id is None:
            dummy_token_id = 0

        max_prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        max_response_length = self.config.actor_rollout_ref.rollout.response_length
        dummy_response_length = min(512, max_response_length)

        extra_fields = dict()
        # TODO: implement traj_mask in verl
        extra_fields["traj_masked"] = 1
        extra_fields["traj_exit_reason"] = exit_reason
        extra_fields["global_steps"] = 0
        extra_fields["min_global_steps"] = 0
        extra_fields["max_global_steps"] = 0

        return [AgentLoopOutput(
            prompt_ids = [0] * max_prompt_length,
            response_ids=[dummy_token_id] * dummy_response_length,
            response_mask=[0] * dummy_response_length,
            response_logprobs=[0.0] * dummy_response_length,
            routed_experts=self._synth_failed_routed_experts(dummy_response_length),
            multi_modal_data={},
            reward_score=0,
            num_turns=0,
            metrics={},
            extra_fields=extra_fields,
        )]

    def _synth_failed_routed_experts(self, length: int) -> np.ndarray | None:
        """Synthesize a zero ``routed_experts`` of shape ``(length, num_layers, top_k)``."""
        shape = self._get_routing_replay_shape()
        if shape is None:
            return None
        num_layers, top_k = shape
        return np.zeros((length, num_layers, top_k), dtype=np.int64)


    def _get_routing_replay_shape(self) -> tuple[int, int] | None:
        """Resolve and cache ``(num_hidden_layers, num_experts_per_tok)`` for the rollout
        model. Returns ``None`` if rollout routing replay is off or the model has no
        experts. The HF config is loaded at most once per worker process."""
        rollout_cfg = self.config.actor_rollout_ref.rollout
        if not bool(getattr(rollout_cfg, "enable_rollout_routing_replay", False)):
            return None
        cls = UniAgentWorkflowLoop #TODO: fix bug
        if not cls._routing_replay_resolved:
            from transformers import AutoConfig

            model_path = self.config.actor_rollout_ref.model.path
            model_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            # Newer Qwen3 nests MoE fields under ``text_config``; older configs keep them
            # at the top level. ``... or 0`` guards against fields explicitly set to None.
            text_cfg = getattr(model_cfg, "text_config", None) or model_cfg
            num_layers = int(getattr(text_cfg, "num_hidden_layers", 0) or 0) or int(
                getattr(model_cfg, "num_hidden_layers", 0) or 0
            )
            top_k = int(getattr(text_cfg, "num_experts_per_tok", 0) or 0) or int(
                getattr(model_cfg, "num_experts_per_tok", 0) or 0
            )
            cls._routing_replay_shape = (num_layers, top_k) if num_layers > 0 and top_k > 0 else None
            cls._routing_replay_resolved = True
            self.logger.info(f"routed_experts replay shape resolved: {cls._routing_replay_shape}")
        return cls._routing_replay_shape


    def _save_workflow_result(self, wf_result: AgentWorkflowResult, output_dir: Path):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with (output_dir / "rollout_cache.pkl").open("wb") as f:
            pickle.dump(wf_result.rollout_caches, f, protocol=pickle.HIGHEST_PROTOCOL)

        save_content: dict[str, Any] = {
            "run_id": wf_result.run_id,
            "execution_time": wf_result.execution_time,
            "final_state": wf_result.final_state.model_dump(),
            "trajectory": [s.model_dump() for s in wf_result.trajectory],
        }

        (output_dir / "workflow_result.json").write_text(
            json.dumps(save_content, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )



    def _init_config(self, sampling_params: dict[str, Any], **kwargs):
        agent_loop_config_path = self.config.actor_rollout_ref.rollout.agent.agent_loop_config_path
        assert agent_loop_config_path is not None, "agent_loop_config_path is None"
        resolved_path = resolve_config_path(agent_loop_config_path)
        base_config = yaml.safe_load(Path(resolved_path).read_text())[0]

        tools_kwargs = kwargs.get("tools_kwargs") or {}
        if "model" in tools_kwargs:
            raise ValueError(
                "tools_kwargs.model is reserved; the model config is always "
                "derived from the rollout config and cannot be overridden "
                "per-sample. Remove `model` from your dataset's tools_kwargs."
            )
        config_dict = _deep_merge(base_config, tools_kwargs)

        rollout_config = self.config.actor_rollout_ref.rollout
        max_model_len = (
            rollout_config.max_model_len
            if rollout_config.max_model_len is not None
            else rollout_config.prompt_length + rollout_config.response_length
        )
        config_dict["model"] = {
            "client": self.server_manager,
            "tokenizer": self.tokenizer,
            "max_model_len": max_model_len,
            "sampling_params": sampling_params,
        }

        if not config_dict.get("workflow"):
            raise ValueError(
                "UniAgentWorkflowLoop requires a 'workflow' key in agent config. "
                "Use UniAgentLoop for single-conversation tasks."
            )

        return config_dict

    def _init_chat_model(self, config_dict: dict) -> AgentChatModel:
        return AgentChatModel(**config_dict)

    def _init_tools_manager(self, tools_config_list: list[dict], parser: str = "qwen3_coder") -> ToolsManager:
        tools_manager_config = ToolsManagerConfig(tools=tools_config_list, parser=parser)
        return ToolsManager(tools_manager_config=tools_manager_config)


    def _init_skills_manager(self, skills_config: dict | None) -> SkillsManager | None:
        if not skills_config:
            return None
        cfg = SkillsManagerConfig(**skills_config)
        return SkillsManager.from_config(cfg)

    def _init_env(self, config_dict: dict) -> AgentEnv:
        env_config = AgentEnvConfig(**config_dict)
        return AgentEnv(run_id=self.run_id, env_config=env_config)



    async def convert_to_agent_output(self, interaction_result: dict) -> AgentLoopOutput:
        rollout_cache = interaction_result["rollout_cache"]
        reward_score = interaction_result.get("reward_score", None)

        if len(rollout_cache["response_mask"]) == 0:
            return await self._build_empty_agent_output(
                exit_reason="no_response",
            )

        num_turns = len(interaction_result["trajectory"])
        self.logger.info(f"num_turns: {num_turns}")

        prompt_ids = rollout_cache["prompt_ids"]
        traj_exit_reason = interaction_result["trajectory"][-1].exit_reason if num_turns > 0 else "unknown"
        should_mask_traj = self.mask_abnormal_exit_traj and traj_exit_reason != "finished"
        traj_masked = int(should_mask_traj)

        if should_mask_traj:
            response_mask = [0] * len(rollout_cache["response_mask"])
        else:
            response_mask = rollout_cache["response_mask"]
        response_logprobs = rollout_cache.get("response_logprobs") or []
        routed_experts = rollout_cache.get("routed_experts")
        metrics = interaction_result.get("metrics", rollout_cache.get("metrics", {}))
        extra_fields = dict(rollout_cache.get("extra_fields") or {})
        extra_fields["traj_masked"] = traj_masked
        extra_fields["traj_exit_reason"] = traj_exit_reason
        response_ids = prompt_ids[-len(response_mask) :]
        prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]

        max_prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        max_response_length = self.config.actor_rollout_ref.rollout.response_length

        if len(prompt_ids) > max_prompt_length:
            prompt_ids = prompt_ids[:max_prompt_length]
            self.logger.warning(
                f"prompt_ids length {len(prompt_ids)} exceeds max_prompt_length {max_prompt_length} "
                "truncate prompt_ids length"
            )
        if len(response_ids) > max_response_length:
            response_ids = response_ids[:max_response_length]
            response_mask = response_mask[:max_response_length]
            response_logprobs = response_logprobs[:max_response_length]
            self.logger.warning(
                f"response_ids length {len(response_ids)} exceeds max_response_length {max_response_length} "
                "truncate response_ids length"
            )

        self.logger.info(f"prompt_ids length: {len(prompt_ids)}")
        self.logger.info(f"response_ids length: {len(response_ids)}")
        self.logger.info(f"reward_score: {reward_score}")
        response_logprobs = response_logprobs if response_logprobs else None
        if routed_experts is not None:
            routed_experts = routed_experts[: len(prompt_ids) + len(response_ids)]

        multi_modal_data = {}
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            routed_experts=routed_experts,
            multi_modal_data=multi_modal_data,
            reward_score=reward_score,
            num_turns=num_turns,
            metrics=metrics,
            extra_fields=extra_fields,
        )


    async def _convert_to_per_step_outputs(
        self,
        wf_result: AgentWorkflowResult,
    ) -> list[AgentLoopOutput]:

        results = [w.to_json() for w in wf_result.trajectory]

        return [self.convert_to_agent_output(r) for r in results]