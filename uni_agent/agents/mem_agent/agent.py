"""MemAgent with its context-management policy kept inside the Agent."""

from __future__ import annotations

import copy
import time
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from pydantic import Field

from uni_agent.agents.react.model import OpenAICompatibleChatModel
from uni_agent.agents.registry import register_agent

from ..base import Agent, AgentConfig, AgentResult

if TYPE_CHECKING:
    from uni_agent.sandbox import Sandbox

TEMPLATE = (
    "You are presented with a problem, a section of an article that may contain the answer to the problem, and a "
    "previous memory. Please read the provided section carefully and update the memory with the new information that "
    "helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any "
    "new, useful information.\n\n"
    "<problem>\n{prompt}\n</problem>\n\n"
    "<memory>\n{memory}\n</memory>\n\n"
    "<section>\n{chunk}\n</section>\n\n"
    "Updated memory:\n"
)

TEMPLATE_FINAL_BOXED = (
    "You are presented with a problem and a previous memory. Please answer the problem based on the previous memory "
    "and put the answer in \\boxed{{}}.\n\n"
    "<problem>\n{prompt}\n</problem>\n\n"
    "<memory>\n{memory}\n</memory>\n\n"
    "Your answer:\n"
)


class MemAgentConfig(AgentConfig):
    """Configuration for chunked-context memory updates."""

    name: str = "mem_agent"
    tokenizer_path: str | None = Field(
        default=None,
        description="Tokenizer path used to split the long context into token chunks.",
    )
    chunk_size: int = Field(default=5000, gt=0)
    max_memorization_length: int = Field(default=1024, gt=0)
    max_chunks: int = Field(default=8, gt=0)
    max_final_response_length: int = Field(default=1024, gt=0)
    max_steps: int = Field(default=9, gt=0, description="Maximum model calls, including the final answer call.")


@lru_cache(maxsize=4)
def _load_tokenizer(tokenizer_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def process(item: dict[str, Any], tokenizer, chunk_size: int) -> tuple[str, list[str]]:
    """Split one long-context sample without changing the original prompt format."""

    question = item["prompt"][0]["content"]
    context = item["context"]
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    tokenized_chunks = [context_ids[i : i + chunk_size] for i in range(0, len(context_ids), chunk_size)]
    chunks = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in tokenized_chunks]
    return question, chunks


@register_agent("mem_agent")
class MemAgent(Agent):
    """Read long input in chunks and carry only a compact memory between contexts."""

    config_model = MemAgentConfig

    async def run(
        self,
        *,
        sandbox: Sandbox,
        messages: list[dict[str, Any]],
        raw_data: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Run the MemAgent-specific context-management policy."""

        del sandbox  # MemAgent does not require a sandbox-bound tool runtime.
        context_input = dict(raw_data or {})
        context_input.setdefault("prompt", messages)

        cfg: MemAgentConfig = self.config  # type: ignore[assignment]
        tokenizer_path = cfg.tokenizer_path or cfg.model.model_name
        if not tokenizer_path:
            raise ValueError("mem_agent requires agent.tokenizer_path or agent.model.model_name")
        if cfg.model.base_url is None:
            raise ValueError("mem_agent: config.model.base_url is not set (the endpoint the policy calls)")

        tokenizer = _load_tokenizer(tokenizer_path)
        prompt, chunks = process(context_input, tokenizer, cfg.chunk_size)
        model = OpenAICompatibleChatModel(
            base_url=cfg.model.base_url,
            api_key=cfg.model.api_key,
            model_name=cfg.model.model_name,
            sampling_params=cfg.model.sampling_params(),
        )

        started_at = time.perf_counter()
        transcript: list[dict[str, Any]] = []
        total_steps = 0
        total_completion_tokens = 0

        async def run_context(context_messages: list[dict[str, Any]], max_tokens: int) -> str:
            """Generate once from a newly constructed MemAgent context."""

            nonlocal total_steps, total_completion_tokens
            if total_steps >= cfg.max_steps:
                raise RuntimeError(f"MemAgent reached max_steps={cfg.max_steps}")

            sampling_params: dict[str, Any] = cfg.model.sampling_params()
            if cfg.model.max_tokens_per_turn is not None:
                max_tokens = min(max_tokens, cfg.model.max_tokens_per_turn)
            if cfg.model.max_total_tokens is not None:
                remaining = cfg.model.max_total_tokens - total_completion_tokens
                if remaining <= 0:
                    raise RuntimeError(f"MemAgent reached max_total_tokens={cfg.model.max_total_tokens}")
                max_tokens = min(max_tokens, remaining)
            sampling_params["max_tokens"] = max_tokens

            response, tool_calls, generation_info = await model.query(
                context_messages,
                sampling_params=sampling_params,
            )
            if tool_calls:
                raise RuntimeError("MemAgent does not support tool calls")

            total_steps += 1
            total_completion_tokens += int(generation_info["completion_tokens"])
            transcript.extend(copy.deepcopy(context_messages))
            transcript.append({"role": "assistant", "content": response})
            return response

        try:
            memory: str | None = None
            # Reserve one model call for the final answer.
            max_memory_steps = min(cfg.max_chunks, cfg.max_steps - 1)
            for chunk in chunks[:max_memory_steps]:
                conversation = [
                    {
                        "role": "user",
                        "content": TEMPLATE.format(
                            prompt=prompt,
                            memory=memory if memory else "No previous memory",
                            chunk=chunk,
                        ),
                    }
                ]
                memory = await run_context(conversation, cfg.max_memorization_length)

            final_conversation = [
                {
                    "role": "user",
                    "content": TEMPLATE_FINAL_BOXED.format(
                        prompt=prompt,
                        memory=memory if memory else "No previous memory",
                    ),
                }
            ]
            final_response = await run_context(final_conversation, cfg.max_final_response_length)
        finally:
            await model.aclose()

        return AgentResult(
            output={"response": final_response},
            transcript=transcript,
            info={
                "execution_time": time.perf_counter() - started_at,
                "total_steps": total_steps,
                "num_contexts": total_steps,
            },
        )
