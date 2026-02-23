# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Lifecycle hooks demo - Demonstrates AgentHooks and RunHooks

This example shows how to use the two-level lifecycle hooks system:
1. AgentHooks: per-agent hooks (on_start, on_end) - attached to a specific Agent
2. RunHooks: global run-level hooks (on_agent_start, on_agent_end, on_llm_start,
   on_llm_end, on_tool_start, on_tool_end, on_agent_transfer) - passed to run()

The demo creates:
- A math helper agent with a simple calculator tool
- A coordinator agent that delegates math tasks to the helper via team transfer
- Custom hooks that log every lifecycle event for observability
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import asyncio
from typing import Any, Dict, List, Optional

from agentica import Agent, AgentHooks, RunHooks, OpenAIChat


# ---------------------------------------------------------------------------
# 1. Define a simple tool for the math agent
# ---------------------------------------------------------------------------

def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b


# ---------------------------------------------------------------------------
# 2. AgentHooks — per-agent lifecycle hooks
# ---------------------------------------------------------------------------

class MyAgentHooks(AgentHooks):
    """Attached to a single Agent to observe its start/end."""

    async def on_start(self, agent: Any, **kwargs) -> None:
        print(f"  [AgentHooks] >>> Agent '{agent.name}' starting")

    async def on_end(self, agent: Any, output: Any, **kwargs) -> None:
        output_preview = str(output)[:80] if output else "(empty)"
        print(f"  [AgentHooks] <<< Agent '{agent.name}' finished, output: {output_preview}")


# ---------------------------------------------------------------------------
# 3. RunHooks — global run-level lifecycle hooks
# ---------------------------------------------------------------------------

class MyRunHooks(RunHooks):
    """Passed to agent.run(hooks=...) to observe the entire run."""

    def __init__(self):
        self.event_counter = 0

    def _log(self, tag: str, msg: str) -> None:
        self.event_counter += 1
        print(f"  [RunHooks #{self.event_counter:02d}] {tag}: {msg}")

    async def on_agent_start(self, agent: Any, **kwargs) -> None:
        self._log("AGENT_START", f"Agent '{agent.name}' begins execution")

    async def on_agent_end(self, agent: Any, output: Any, **kwargs) -> None:
        output_preview = str(output)[:60] if output else "(empty)"
        self._log("AGENT_END", f"Agent '{agent.name}' => {output_preview}")

    async def on_llm_start(self, agent: Any, messages: Optional[List[Dict[str, Any]]] = None, **kwargs) -> None:
        n_msgs = len(messages) if messages else 0
        self._log("LLM_START", f"Agent '{agent.name}' calling LLM with {n_msgs} messages")

    async def on_llm_end(self, agent: Any, response: Any = None, **kwargs) -> None:
        content_preview = ""
        if response and hasattr(response, "content") and response.content:
            content_preview = str(response.content)[:60]
        self._log("LLM_END", f"Agent '{agent.name}' LLM responded: {content_preview or '(tool calls)'}")

    async def on_tool_start(
        self, agent: Any, tool_name: str = "", tool_call_id: str = "",
        tool_args: Optional[Dict[str, Any]] = None, **kwargs
    ) -> None:
        self._log("TOOL_START", f"Agent '{agent.name}' calling tool '{tool_name}' with args={tool_args}")

    async def on_tool_end(
        self, agent: Any, tool_name: str = "", tool_call_id: str = "",
        tool_args: Optional[Dict[str, Any]] = None, result: Any = None,
        is_error: bool = False, elapsed: float = 0.0, **kwargs
    ) -> None:
        status = "ERROR" if is_error else "OK"
        self._log("TOOL_END", f"Tool '{tool_name}' [{status}] result={result} ({elapsed:.2f}s)")

    async def on_agent_transfer(self, from_agent: Any, to_agent: Any, **kwargs) -> None:
        self._log("TRANSFER", f"'{from_agent.name}' -> '{to_agent.name}'")


# ---------------------------------------------------------------------------
# 4. Build agents
# ---------------------------------------------------------------------------

# Math helper agent with tools and per-agent hooks
math_agent = Agent(
    name="Math Helper",
    model=OpenAIChat(id="gpt-4o-mini"),
    description="A math assistant that can add and multiply integers.",
    tools=[add, multiply],
    hooks=MyAgentHooks(),
)

# Coordinator agent that delegates math tasks via team transfer
coordinator = Agent(
    name="Coordinator",
    model=OpenAIChat(id="gpt-4o-mini"),
    description="You coordinate tasks. For any math calculation, transfer the task to Math Helper.",
    team=[math_agent],
    hooks=MyAgentHooks(),
)


# ---------------------------------------------------------------------------
# 5. Run the demo
# ---------------------------------------------------------------------------

async def main():
    run_hooks = MyRunHooks()

    # --- Demo 1: Single agent with tool calls ---
    print("=" * 60)
    print("Demo 1: Single agent with tool calls")
    print("=" * 60)
    response = await math_agent.run(
        "What is 3 + 5 and 4 * 7? Use the tools to compute.",
        hooks=run_hooks,
    )
    print(f"\nFinal response: {response.content}\n")

    # --- Demo 2: Team transfer (coordinator -> math agent) ---
    print("=" * 60)
    print("Demo 2: Team transfer (coordinator -> math helper)")
    print("=" * 60)
    run_hooks_2 = MyRunHooks()
    response = await coordinator.run(
        "Please calculate 12 + 8 for me.",
        hooks=run_hooks_2,
    )
    print(f"\nFinal response: {response.content}\n")

    print("=" * 60)
    print(f"Total lifecycle events: Demo1={run_hooks.event_counter}, Demo2={run_hooks_2.event_counter}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
