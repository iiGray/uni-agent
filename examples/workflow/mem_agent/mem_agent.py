import time
from typing import Any

from uni_agent.interaction.interaction import AgentInteraction
from uni_agent.workflow.workflow import AgentWorkflowBase, AgentWorkflowResult


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



def process(item, tokenizer, chunk_size):
    question = item['prompt'][0]['content']
    context = item['context']

    context_ids = tokenizer.encode(context, add_special_tokens=False)
    tokenized_chunks = [context_ids[i: i + chunk_size] for i in range(0, len(context_ids), chunk_size)]
    chunks = [tokenizer.decode(c, skip_special_tokens = True) for c in tokenized_chunks]
    return question, chunks


class MemAgentWorkflow(AgentWorkflowBase):
    """A sample implementation of MemAgent using the programmatic workflow framework.
    
    In MemAgent, the agent is allowed to execute tools, but periodically 
    (or when triggered by a specific condition), the workflow intercepts the loop,
    summarizes the progress into a working memory, and resets the conversation context.
    This prevents the KV-cache from growing infinitely and keeps the agent focused.
    """
    async def run(self, raw_data: dict):
        prompt, chunks = process(raw_data, self.tokenizer, self.config_dict['chunk_size'])

        memory = None
        for chunk in chunks:
            if self.get_global_step_idx() >= self.config_dict['max_chunks']: break
            conversation = [
                {"role": "user", "content": TEMPLATE.format(
                    prompt = prompt,
                    memory = memory if memory else "No previous memory",
                    chunk = chunk
                )}
            ]
            await self.update_context(conversation, insert_skill = False)

            step_output = await self.step(sampling_params = dict(max_tokens = self.config_dict['max_memorization_length']))

            memory = step_output.response

        conversation = [
            {
                "role": "user",
                "content": TEMPLATE_FINAL_BOXED.format(
                    prompt = prompt,
                    memory = memory if memory else "No previous memory",
                ),
            }
        ]

        await self.update_context(conversation, insert_skill = False)

        await self.step(sampling_params = dict(max_tokens = self.config_dict['max_final_response_length']))

        
            
        