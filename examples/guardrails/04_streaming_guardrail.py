# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Streaming guardrail demo - Real-time content checking during streaming

This example shows how to:
1. Stream responses from a main agent
2. Run a guardrail agent in parallel to check content quality
3. Early-terminate streaming if the guardrail triggers

The expected output is that you'll see tokens stream in, then the guardrail
will trigger and stop the streaming when it detects content that is too complex
for a ten-year-old to understand.
"""
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pydantic import BaseModel, Field

from agentica import Agent, OpenAIChat, RunEvent

# ============================================================================
# Define Agents
# ============================================================================

# Main agent: generates verbose, detailed responses
agent = Agent(
    name="Assistant",
    model=OpenAIChat(id="gpt-4o-mini"),
    description=(
        "You are a helpful assistant. You ALWAYS write long responses, "
        "making sure to be verbose and detailed."
    ),
)


# Guardrail structured output
class GuardrailOutput(BaseModel):
    reasoning: str = Field(
        description="Reasoning about whether the response could be understood by a ten year old."
    )
    is_readable_by_ten_year_old: bool = Field(
        description="Whether the response is understandable by a ten year old."
    )


# Guardrail agent: judges content simplicity via structured output
guardrail_agent = Agent(
    name="Checker",
    model=OpenAIChat(id="gpt-4o-mini"),
    description=(
        "You will be given a question and a response. Your goal is to judge "
        "whether the response is simple enough to be understood by a ten year old."
    ),
    response_model=GuardrailOutput,
)


# ============================================================================
# Guardrail Check Function
# ============================================================================

async def check_guardrail(text: str) -> GuardrailOutput:
    """Run the guardrail agent on the given text and return the structured result."""
    result = await guardrail_agent.run(text)
    return result.content


# ============================================================================
# Main Streaming with Guardrail
# ============================================================================

async def main():
    question = "What is a black hole, and how does it behave?"
    print(f"Question: {question}\n")
    print("Streaming response (guardrail checks every 300 chars):\n")

    current_text = ""
    # Check the guardrail every N characters
    next_guardrail_check_len = 300
    guardrail_task = None

    async for chunk in agent.run_stream(question):
        if chunk.event == RunEvent.run_response and chunk.content:
            print(chunk.content, end="", flush=True)
            current_text += chunk.content

            # Start a guardrail check when we've accumulated enough text
            # Skip if there's already a check running
            if len(current_text) >= next_guardrail_check_len and not guardrail_task:
                print("\n[Guardrail check started...]")
                guardrail_task = asyncio.create_task(check_guardrail(current_text))
                next_guardrail_check_len += 300

        # Every iteration, check if the guardrail task has finished
        if guardrail_task and guardrail_task.done():
            guardrail_result = guardrail_task.result()
            if not guardrail_result.is_readable_by_ten_year_old:
                print("\n\n================\n")
                print(f"Guardrail triggered! Reasoning:\n{guardrail_result.reasoning}")
                break

    # Final check on the complete output
    print("\n\n[Running final guardrail check...]")
    guardrail_result = await check_guardrail(current_text)
    if not guardrail_result.is_readable_by_ten_year_old:
        print("\n================\n")
        print(f"Guardrail triggered on final output! Reasoning:\n{guardrail_result.reasoning}")
    else:
        print(f"\nGuardrail passed. Reasoning:\n{guardrail_result.reasoning}")


if __name__ == "__main__":
    asyncio.run(main())
