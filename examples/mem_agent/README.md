# MemAgent

MemAgent is a self-contained Agent that reads a long document in token chunks,
starts a new model context for every chunk, carries forward a compact memory,
and answers the original question from the final memory.

The implementation is split by responsibility:

- `uni_agent/agents/mem_agent/` contains the MemAgent policy and its
  context-management methods.
- `uni_agent/tasks/hotpotqa/` contains the HotpotQA task, final-answer reward,
  and dataset preprocessing.
- `examples/mem_agent/task_config.yaml` contains the HotpotQA task and MemAgent
  context-management defaults.
- `examples/mem_agent/train_mem_agent.sh` is the canonical verl v1 FSDP2
  training recipe.

## Data preprocessing

Like the SWE-bench Tasks, HotpotQA owns its preprocessing pipeline. Prepare the
standard HotpotQA distractor train and validation splits with the policy
tokenizer:

```bash
python -m uni_agent.tasks.hotpotqa.preprocess \
    --tokenizer-path /path/to/Qwen3-8B \
    --local-save-dir /path/to/processed_data \
    --context-chunk-size 5000
```

The preprocessor writes `hotpotqa_train.parquet` and `hotpotqa_dev.parquet`.
Each row contains the question and a serialized HotpotQA Task Config under
`extra_info.tools_kwargs.task`. The Task metadata owns the token-bounded context
chunks, while the Task Config owns the ground-truth answer. Training therefore
uses the standard verl Dataset path without a custom runtime Dataset class.

## Context Management

`MemAgent` implements the context-management methods directly:

```python
class MemAgent(Agent):
    async def run(self, *, sandbox, messages):
        async with self.context_session():
            await self.update_context(...)
            memory = (await self.step()).response
        return self.build_agent_result()
```

Every `update_context()` starts a new Gateway trajectory chain. The Task scores
the final boxed answer and posts the session-level reward; the framework assigns
that reward to every context segment emitted by the session.

## Training

Set the required paths and launch from the repository root:

```bash
MODEL_PATH=/path/to/Qwen3-8B \
TRAIN_FILE=/path/to/hotpotqa_train.parquet \
VAL_FILE=/path/to/hotpotqa_dev.parquet \
bash examples/mem_agent/train_mem_agent.sh
```

The recipe uses `verl.trainer.main_ppo` with the v1 `separate_async` topology.
Trainer and rollout GPU counts are configurable through environment variables
in the script. Change `--context-chunk-size` during preprocessing to adjust the
default 5000-token context chunks.
