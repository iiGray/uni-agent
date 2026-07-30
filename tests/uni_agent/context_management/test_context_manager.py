from __future__ import annotations

import inspect
from typing import Any

import pytest

import uni_agent.context_management.context_manager as context_manager_module
from uni_agent.agents import Agent
from uni_agent.context_management import ContextManager, ContextManagerConfig


class _FakeModel:
    instances: list[_FakeModel] = []

    def __init__(self, **_: Any) -> None:
        self.calls: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
        self.closed = False
        type(self).instances.append(self)

    async def query(self, messages, *, sampling_params):
        self.calls.append((messages, sampling_params))
        response = f"response-{len(self.calls)}"
        return response, [], {"prompt_tokens": 3, "completion_tokens": 2}

    async def aclose(self) -> None:
        self.closed = True


class _TwoContextAgent(ContextManager, Agent):
    config_model = ContextManagerConfig

    async def run(self, *, sandbox, messages):
        async with self.context_session(sandbox=sandbox):
            await self.update_context(messages)
            await self.step({"max_tokens": 11})
            await self.update_context([{"role": "user", "content": "second context"}])
            await self.step({"max_tokens": 7})
        return self.build_agent_result()


class _MissingRunAgent(ContextManager, Agent):
    config_model = ContextManagerConfig


@pytest.mark.asyncio
async def test_update_context_materializes_independent_segments(monkeypatch):
    _FakeModel.instances.clear()
    monkeypatch.setattr(context_manager_module, "OpenAICompatibleChatModel", _FakeModel)
    agent = _TwoContextAgent(
        ContextManagerConfig(
            model={
                "base_url": "http://gateway.invalid/v1",
                "model_name": "policy",
            }
        )
    )

    result = await agent.run(
        sandbox=object(),
        messages=[{"role": "user", "content": "first context"}],
    )

    context_result = result.output["context_manager_result"]
    assert context_result.total_steps == 2
    assert len(context_result.trajectory) == 2
    assert context_result.trajectory[0].prompt_messages[0]["content"] == "first context"
    assert context_result.trajectory[1].prompt_messages[0]["content"] == "second context"
    assert result.output["response"] == "response-2"
    assert [call[1]["max_tokens"] for call in _FakeModel.instances[0].calls] == [11, 7]
    assert _FakeModel.instances[0].closed


def test_context_manager_does_not_implement_agent_run():
    assert inspect.isabstract(_MissingRunAgent)
    with pytest.raises(TypeError, match="abstract"):
        _MissingRunAgent(ContextManagerConfig())
