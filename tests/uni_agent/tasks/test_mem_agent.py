from __future__ import annotations

from typing import Any

import pytest

import uni_agent.agents.mem_agent.agent as mem_agent_module
from examples.mem_agent.dataset import build_task_config, context_to_text
from uni_agent.agents import AgentResult
from uni_agent.agents.mem_agent import MemAgent, MemAgentConfig
from uni_agent.tasks.hotpotqa import HotpotQATask, HotpotQATaskConfig
from uni_agent.tasks.hotpotqa.reward import compute_score, last_boxed_only_string, remove_boxed


class _FakeTokenizer:
    def encode(self, value: str, *, add_special_tokens: bool) -> list[str]:
        assert not add_special_tokens
        return value.split()

    def decode(self, tokens: list[str], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return " ".join(tokens)


class _FakeModel:
    instances: list[_FakeModel] = []

    def __init__(self, **_: Any) -> None:
        self.calls: list[list[dict[str, Any]]] = []
        self.sampling_params: list[dict[str, Any]] = []
        self.closed = False
        type(self).instances.append(self)

    async def query(self, messages, *, sampling_params):
        self.calls.append(messages)
        self.sampling_params.append(sampling_params)
        return f"memory-{len(self.calls)}", [], {"prompt_tokens": 3, "completion_tokens": 2}

    async def aclose(self) -> None:
        self.closed = True


class _FakeSandbox:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None


class _FakeAgent:
    async def run(self, **_):
        return AgentResult(
            output={"response": "Final answer: \\boxed{alpha beta}"},
            info={"num_contexts": 3, "total_steps": 3},
        )


@pytest.mark.asyncio
async def test_mem_agent_uses_each_chunk_as_a_new_context(monkeypatch):
    _FakeModel.instances.clear()
    monkeypatch.setattr(mem_agent_module, "OpenAICompatibleChatModel", _FakeModel)
    monkeypatch.setattr(mem_agent_module, "_load_tokenizer", lambda _: _FakeTokenizer())
    agent = MemAgent(
        MemAgentConfig(
            tokenizer_path="fake-tokenizer",
            chunk_size=2,
            max_chunks=2,
            max_steps=3,
            max_memorization_length=11,
            max_final_response_length=7,
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

    assert result.output["response"] == "memory-3"
    assert result.info["num_contexts"] == 3
    assert "memory-1" in _FakeModel.instances[0].calls[1][0]["content"]
    assert [params["max_tokens"] for params in _FakeModel.instances[0].sampling_params] == [11, 11, 7]
    assert _FakeModel.instances[0].closed


def test_mem_agent_reward_scores_last_boxed_answer():
    response = "Earlier: \\boxed{wrong}. Final answer: \\boxed{alpha beta}"

    assert compute_score(response, ["alpha beta"]) == 1.0


def test_mem_agent_reward_returns_zero_without_boxed_answer():
    assert compute_score("alpha beta", ["alpha beta"]) == 0.0


def test_boxed_helpers_preserve_nested_braces():
    boxed = last_boxed_only_string("answer: \\boxed{\\text{alpha beta}}")

    assert boxed == "\\boxed{\\text{alpha beta}}"
    assert remove_boxed(boxed) == "alpha beta"


def test_mem_agent_dataset_builds_standard_task_payload():
    row = {
        "context": [["Title", ["alpha", "beta"]]],
        "reward_model": {"ground_truth": ["answer"]},
    }

    task = build_task_config(row)

    assert context_to_text(row["context"]) == "Title\nalpha\nbeta"
    assert task == {
        "name": "hotpotqa",
        "metadata": {
            "context": "Title\nalpha\nbeta",
            "reward_model": {"ground_truth": ["answer"]},
        },
    }


@pytest.mark.asyncio
async def test_hotpotqa_task_scores_final_response(monkeypatch):
    task = HotpotQATask(
        HotpotQATaskConfig(
            prompt=[{"role": "user", "content": "question"}],
            metadata={
                "context": "long context",
                "reward_model": {"ground_truth": ["alpha beta"]},
            },
            sandbox={"provider": "local"},
            agent=MemAgentConfig(),
        )
    )
    monkeypatch.setattr(task, "build_sandbox", lambda: _FakeSandbox())
    monkeypatch.setattr(task, "build_agent", _FakeAgent)

    result = await task.run()

    assert result.reward == 1.0
    assert result.accuracy == 1.0
    assert result.extra_info == {
        "response": "Final answer: \\boxed{alpha beta}",
        "num_contexts": 3,
        "total_steps": 3,
    }
