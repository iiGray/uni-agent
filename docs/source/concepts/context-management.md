# Context Management

`ContextManager` is a reusable mixin for Agents that need to control when a
trajectory context ends and a new one begins. It is independent of any
particular Agent implementation.

Use it when an Agent must process input in segments, compress prior information,
or deliberately discard earlier messages while keeping every segment available
for training.

## Compose It with an Agent

Put `ContextManager` before `Agent` in the base-class list, and derive the
Agent's configuration from `ContextManagerConfig`:

```python
from typing import Any

from uni_agent.agents import Agent
from uni_agent.context_management import ContextManager, ContextManagerConfig


class MyAgentConfig(ContextManagerConfig):
    name: str = "my_agent"


class MyAgent(ContextManager, Agent):
    config_model = MyAgentConfig

    async def context_management(self, raw_data: dict[str, Any]) -> None:
        first_context = [{"role": "user", "content": "Read the first section."}]
        await self.update_context(first_context)
        first_result = await self.step({"max_tokens": 512})

        next_context = [
            {
                "role": "user",
                "content": f"Continue from this summary:\n{first_result.response}",
            }
        ]
        await self.update_context(next_context)
        await self.step({"max_tokens": 512})
```

`ContextManager` supplies the normal `Agent.run()` implementation. A Task can
therefore build and run the composed Agent in the same way as any other Agent.

## Context Lifecycle

An Agent implements `context_management()` as its programmable control flow:

1. `update_context(messages)` finalizes the previous context segment and starts
   a new one from `messages`.
2. `step(sampling_params)` makes one model call in the active context and
   optionally executes returned Tools.
3. The Agent may repeat these methods in any order allowed by its policy.
4. `run()` collects the final segment and returns a `ContextManagerResult`
   through `AgentResult.output["context_manager_result"]`.

Each segment records its initial prompt messages, full messages, model turns,
execution time, and reward. `ContextManagerResult.set_reward()` assigns one
workflow-level reward to every segment.

## Training Behavior

All model calls use the same session-scoped Gateway endpoint. A call after
`update_context()` has a newly constructed message prefix, so the Gateway
tracks it as an independent trajectory chain. When the session is finalized,
all chains are emitted as training trajectories.

The MemAgent implementation under `uni_agent/agents/mem_agent/` is an example:
it reads a long input in chunks, updates a compact memory after each chunk,
starts a fresh context for the next chunk, and answers from the final memory.
