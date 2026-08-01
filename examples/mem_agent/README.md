# MemAgent

MemAgent is a self-contained Agent that reads a long document in token chunks,
starts a new model context for every chunk, carries forward a compact memory,
and answers the original question from the final memory.

The implementation is split by responsibility:

- `uni_agent/agents/mem_agent/` contains the MemAgent policy and its
  context-management logic.
- `uni_agent/tasks/hotpotqa/` contains the HotpotQA task and final-answer reward.
- `examples/mem_agent/dataset.py` adapts long-context dataset rows to the normal
  Uni-Agent task-runner payload.
- `examples/quickstart/training/task_config_hotpotqa.yaml` contains the
  HotpotQA task and MemAgent context-management defaults.
- `examples/quickstart/training/train_mem_agent.sh` is the canonical verl v1
  FSDP2 training recipe.

## Data

The Parquet rows must contain:

- `prompt`: an OpenAI-style message list containing the question.
- `context`: a string or nested list/dictionary containing the long document.
- `reward_model.ground_truth` (or `ground_truth`): one answer or a list of
  accepted answers.

`MemAgentDataset` moves the context and answer into
`tools_kwargs.task.metadata`, so the generic agent framework and task runner do
not need MemAgent-specific fields.

## Context Management

`MemAgent` implements its context-management policy directly in `run()`:

```python
class MemAgent(Agent):
    async def run(self, *, sandbox, messages):
        for chunk in chunks:
            context = build_memory_prompt(chunk, memory)
            memory = await run_context(context)

        final_context = build_answer_prompt(memory)
        return await run_context(final_context)
```

Every model call uses a newly constructed message list, so the Gateway records
it as a new trajectory chain. The Task scores the final boxed answer and posts
the session-level reward; the framework assigns that reward to every context
segment emitted by the session.

## Training

Set the required paths and launch from the repository root:

```bash
MODEL_PATH=/path/to/Qwen3-8B \
TRAIN_FILE=/path/to/hotpotqa_train.parquet \
VAL_FILE=/path/to/hotpotqa_dev.parquet \
bash examples/quickstart/training/train_mem_agent.sh
```

The recipe uses `verl.trainer.main_ppo` with the v1 `separate_async` topology.
Trainer and rollout GPU counts are configurable through environment variables
in the script.
