# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Routing demo - Triage agent routes to language-specific agents

This example shows the routing pattern:
1. A triage agent determines the language of user input
2. It routes (hands off) to the appropriate language-specific agent
3. The target agent stays active for subsequent turns until user types "reset"

Inspired by openai-agents-python routing.py, implemented with zero framework changes.
"""
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pydantic import BaseModel, Field

from agentica import Agent, OpenAIChat, RunEvent

# ============================================================================
# Language-specific Agents
# ============================================================================

chinese_agent = Agent(
    name="Chinese Agent",
    model=OpenAIChat(id="gpt-4o-mini"),
    description="You only speak Chinese. Respond to all messages in Chinese.",
)

spanish_agent = Agent(
    name="Spanish Agent",
    model=OpenAIChat(id="gpt-4o-mini"),
    description="You only speak Spanish. Respond to all messages in Spanish.",
)

english_agent = Agent(
    name="English Agent",
    model=OpenAIChat(id="gpt-4o-mini"),
    description="You only speak English. Respond to all messages in English.",
)

# ============================================================================
# Triage Agent with Structured Output
# ============================================================================

AGENT_MAP = {
    "chinese": chinese_agent,
    "spanish": spanish_agent,
    "english": english_agent,
}


class RoutingDecision(BaseModel):
    language: str = Field(
        description="The detected language to route to. One of: chinese, spanish, english."
    )


triage_agent = Agent(
    name="Triage Agent",
    model=OpenAIChat(id="gpt-4o-mini"),
    description=(
        "You are a routing agent. Detect the language of the user's message "
        "and output the target language. Supported: chinese, spanish, english. "
        "Do NOT answer the user's question. Just detect the language."
    ),
    response_model=RoutingDecision,
)


# ============================================================================
# Multi-turn Routing Loop
# ============================================================================

async def main():
    print("=" * 60)
    print("Routing Demo")
    print("=" * 60)
    print("We speak Chinese, Spanish and English.")
    print('Type "reset" to go back to triage, "quit" to exit.\n')

    current_agent = triage_agent

    while True:
        try:
            user_msg = input(f"[{current_agent.name}] You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_msg:
            continue
        if user_msg.lower() == "quit":
            print("Bye!")
            break
        if user_msg.lower() == "reset":
            current_agent = triage_agent
            print(">> Reset to Triage Agent\n")
            continue

        if current_agent is triage_agent:
            # Triage: structured output → routing decision
            result = await current_agent.run(user_msg)
            decision: RoutingDecision = result.content
            lang = decision.language.lower().strip()
            target = AGENT_MAP.get(lang, english_agent)
            current_agent = target
            print(f">> Routed to {current_agent.name}")

            # Re-run the original message through the target agent (streaming)
            async for chunk in current_agent.run_stream(user_msg):
                if chunk.event == RunEvent.run_response and chunk.content:
                    print(chunk.content, end="", flush=True)
            print("\n")
        else:
            # Target agent: stream response directly
            async for chunk in current_agent.run_stream(user_msg):
                if chunk.event == RunEvent.run_response and chunk.content:
                    print(chunk.content, end="", flush=True)
            print("\n")


if __name__ == "__main__":
    asyncio.run(main())
