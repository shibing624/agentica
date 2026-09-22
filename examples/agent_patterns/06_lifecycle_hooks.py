# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Lifecycle hooks demo - Demonstrates AgentHooks, RunHooks, and ConversationArchiveHooks

This example shows how to use the lifecycle hooks system:
1. AgentHooks: per-agent hooks (on_start, on_end) - attached to a specific Agent
2. RunHooks: global run-level hooks (on_agent_start, on_agent_end, on_llm_start,
   on_llm_end, on_tool_start, on_tool_end, on_agent_transfer) - passed to run()
3. ConversationArchiveHooks: auto-archives conversations to workspace after each run
4. auto_archive via WorkspaceMemoryConfig: auto-injects ConversationArchiveHooks

The demo creates:
- A math helper agent with a simple calculator tool
- A coordinator agent that delegates math tasks to the helper via Agent.as_tool()
- Custom hooks that log every lifecycle event for observability
- A workspace agent with auto-archive enabled (via WorkspaceMemoryConfig)
- A workspace agent with ConversationArchiveHooks passed via RunConfig
"""
import sys
import os
import shutil
import tempfile
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import asyncio
from typing import Any, Dict, List, Optional

from agentica import Agent, AgentHooks, RunHooks, ConversationArchiveHooks, OpenAIChat
from agentica.agent.config import WorkspaceMemoryConfig
from agentica.run.config import RunConfig
from agentica.workspace import Workspace


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

# Coordinator agent that delegates math tasks via Agent.as_tool() composition
coordinator = Agent(
    name="Coordinator",
    model=OpenAIChat(id="gpt-4o-mini"),
    description="You coordinate tasks. For any math calculation, call the math_helper tool.",
    tools=[math_agent.as_tool(tool_name="math_helper", tool_description="Run the math helper agent.")],
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
    from agentica.run.config import RunConfig
    response = await math_agent.run(
        "What is 3 + 5 and 4 * 7? Use the tools to compute.",
        config=RunConfig(hooks=run_hooks),
    )
    print(f"\nFinal response: {response.content}\n")

    # --- Demo 2: as_tool composition (coordinator -> math agent) ---
    print("=" * 60)
    print("Demo 2: as_tool composition (coordinator -> math helper)")
    print("=" * 60)
    run_hooks_2 = MyRunHooks()
    response = await coordinator.run(
        "Please calculate 12 + 8 for me.",
        config=RunConfig(hooks=run_hooks_2),
    )
    print(f"\nFinal response: {response.content}\n")

    print("=" * 60)
    print(f"Total lifecycle events: Demo1={run_hooks.event_counter}, Demo2={run_hooks_2.event_counter}")
    print("=" * 60)

    # --- Demo 3: auto_archive via WorkspaceMemoryConfig ---
    # When auto_archive=True, Agent auto-injects ConversationArchiveHooks
    # so every run() archives the conversation to workspace — no manual setup needed.
    print("\n" + "=" * 60)
    print("Demo 3: auto_archive via WorkspaceMemoryConfig")
    print("=" * 60)

    ws_path = Path(tempfile.mkdtemp()) / "hooks_demo_workspace"
    workspace = Workspace(str(ws_path))
    workspace.initialize()

    archive_agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        workspace=workspace,
        long_term_memory_config=WorkspaceMemoryConfig(auto_archive=True),
    )
    print(f"auto_archive=True, _default_run_hooks injected: {archive_agent._default_run_hooks is not None}")

    response = await archive_agent.run("What is 2 + 2?")
    print(f"Response: {response.content}")

    # Check that conversation was archived
    conv_files = workspace.get_conversation_files(max_files=5)
    print(f"Archived conversation files: {len(conv_files)}")
    for f in conv_files:
        print(f"  - {Path(f).name}: {Path(f).read_text(encoding='utf-8')[:100]}...")

    # --- Demo 4: ConversationArchiveHooks via RunConfig ---
    # For explicit control: pass ConversationArchiveHooks directly to RunConfig.
    # This works even when auto_archive=False.
    print("\n" + "=" * 60)
    print("Demo 4: ConversationArchiveHooks via RunConfig (explicit)")
    print("=" * 60)

    ws_path_2 = Path(tempfile.mkdtemp()) / "hooks_explicit_workspace"
    workspace_2 = Workspace(str(ws_path_2))
    workspace_2.initialize()

    explicit_agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        workspace=workspace_2,
        # auto_archive is False by default
    )

    archive_hooks = ConversationArchiveHooks()
    response = await explicit_agent.run(
        "Tell me a fun fact about Python.",
        config=RunConfig(hooks=archive_hooks),
    )
    print(f"Response: {response.content}")

    conv_files_2 = workspace_2.get_conversation_files(max_files=5)
    print(f"Archived conversation files: {len(conv_files_2)}")
    for f in conv_files_2:
        print(f"  - {Path(f).name}: {Path(f).read_text(encoding='utf-8')[:100]}...")

    # Clean up
    shutil.rmtree(ws_path.parent, ignore_errors=True)
    shutil.rmtree(ws_path_2.parent, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
