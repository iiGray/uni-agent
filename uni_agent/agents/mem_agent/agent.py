"""MemAgent context-management example."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import Field

from uni_agent.agents.registry import register_agent
from uni_agent.agents.workflow import Workflow, WorkflowConfig


TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

<problem>
{prompt}
</problem>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""

TEMPLATE_FINAL_BOXED = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<problem>
{prompt}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""


class MemAgentConfig(WorkflowConfig):
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
class MemAgent(Workflow):
    """Read long input in chunks and carry only a compact memory between contexts."""

    config_model = MemAgentConfig

    async def workflow(self, raw_data: dict[str, Any]) -> None:
        cfg: MemAgentConfig = self.config  # type: ignore[assignment]
        tokenizer_path = cfg.tokenizer_path or cfg.model.model_name
        if not tokenizer_path:
            raise ValueError("mem_agent requires agent.tokenizer_path or agent.model.model_name")
        tokenizer = _load_tokenizer(tokenizer_path)
        prompt, chunks = process(raw_data, tokenizer, cfg.chunk_size)

        memory: str | None = None
        for chunk in chunks:
            if self.get_global_step_idx() >= cfg.max_chunks:
                break
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
            await self.update_context(conversation, insert_skill=False)
            step_output = await self.step(sampling_params={"max_tokens": cfg.max_memorization_length})
            memory = step_output.response

        conversation = [
            {
                "role": "user",
                "content": TEMPLATE_FINAL_BOXED.format(
                    prompt=prompt,
                    memory=memory if memory else "No previous memory",
                ),
            }
        ]
        await self.update_context(conversation, insert_skill=False)
        await self.step(sampling_params={"max_tokens": cfg.max_final_response_length})


# Compatibility name for existing configs that referred to the example class.
MemAgentWorkflow = MemAgent
