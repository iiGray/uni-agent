from __future__ import annotations

from typing import Any

import pytest

import uni_agent.agents.mem_agent.agent as mem_agent_module
import uni_agent.agents.workflow as workflow_module
from uni_agent.agents.mem_agent import MemAgent, MemAgentConfig
from uni_agent.tasks.mem_agent.reward import compute_score, last_boxed_only_string, remove_boxed


class _FakeTokenizer:
    def encode(self, value: str, *, add_special_tokens: bool) -> list[str]:
        assert not add_special_tokens
        return value.split()

    def decode(self, tokens: list[str], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return " ".join(tokens)


class _FakeModel:
    def __init__(self, **_: Any) -> None:
        self.calls: list[list[dict[str, Any]]] = []

    async def query(self, messages, *, sampling_params):
        del sampling_params
        self.calls.append(messages)
        return f"memory-{len(self.calls)}", [], {"prompt_tokens": 3, "completion_tokens": 2}

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_mem_agent_uses_each_chunk_as_a_new_context(monkeypatch):
    monkeypatch.setattr(workflow_module, "OpenAICompatibleChatModel", _FakeModel)
    monkeypatch.setattr(mem_agent_module, "_load_tokenizer", lambda _: _FakeTokenizer())
    agent = MemAgent(
        MemAgentConfig(
            tokenizer_path="fake-tokenizer",
            chunk_size=2,
            max_chunks=2,
            max_steps=3,
            model={"base_url": "http://gateway.invalid/v1", "model_name": "policy"},
        )
    )

    result = await agent.run(
        sandbox=object(),
        messages=[{"role": "user", "content": "question"}],
        raw_data={
            "prompt": [{"role": "user", "content": "question"}],
            "context": "one two three four",
        },
    )

    workflow_result = result.output["workflow_result"]
    assert len(workflow_result.trajectory) == 3
    assert "memory-1" in workflow_result.trajectory[1].prompt_messages[0]["content"]
    assert result.output["response"] == "memory-3"


def test_mem_agent_reward_scores_last_boxed_answer():
    response = "Earlier: \\boxed{wrong}. Final answer: \\boxed{alpha beta}"

    assert compute_score(response, ["alpha beta"]) == 1.0


def test_mem_agent_reward_returns_zero_without_boxed_answer():
    assert compute_score("alpha beta", ["alpha beta"]) == 0.0


def test_boxed_helpers_preserve_nested_braces():
    boxed = last_boxed_only_string("answer: \\boxed{\\text{alpha beta}}")

    assert boxed == "\\boxed{\\text{alpha beta}}"
    assert remove_boxed(boxed) == "alpha beta"
