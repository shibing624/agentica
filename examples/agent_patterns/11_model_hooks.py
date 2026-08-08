# -*- coding: utf-8 -*-
"""
Model-layer hooks demo: context overflow handling

Demonstrates an opt-in "deep agent" capability added directly to Agent
via ToolConfig — no subclass needed:

context_overflow_threshold: when estimated token usage reaches the
threshold fraction of context_window, the Runner first tries reversible
compression (summarize / truncate tool results) via the wired
CompressionManager, and only FIFO-evicts oldest non-system messages
when usage still exceeds the hard limit (threshold + 5pp) afterwards.
Prevents silent context overflow errors while preserving as much
information as possible.

Usage:
    python examples/agent_patterns/11_model_hooks.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import asyncio

from agentica import Agent, OpenAIChat
from agentica.agent.config import ToolConfig
from agentica.model.message import Message


# ---------------------------------------------------------------------------
# Demo 1: Context overflow handling (no real LLM needed)
# ---------------------------------------------------------------------------

def demo_context_overflow():
    print("=" * 60)
    print("Demo 1: Context Overflow Handling")
    print("=" * 60)
    print("""
    When estimated token usage reaches context_overflow_threshold × context_window:
      1. If a CompressionManager is wired, try reversible compression first.
      2. Only FIFO-evict oldest non-system messages if still above the hard
         limit (threshold + 5pp) after compression.
    System message is always preserved.
    """)

    agent = Agent(
        name="overflow-demo",
        model=OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY", "fake")),
        tool_config=ToolConfig(
            context_overflow_threshold=0.6,  # trigger at 60% of context_window
        ),
    )
    agent.update_model()
    # Use a tiny window so the demo is easy to trigger
    agent.model.context_window = 200

    hook = agent._build_pre_tool_hook()
    if hook is None:
        print("  [SKIP] Hook is None — context_overflow_threshold=0.0 (disabled)")
        return

    # Build a message history that exceeds 60% of 200 tokens (= 120 tokens = 480 chars)
    messages = [
        Message(role="system", content="You are a helpful AI assistant."),
        Message(role="user", content="A" * 160),
        Message(role="assistant", content="B" * 160),
        Message(role="user", content="C" * 160),
        Message(role="assistant", content="D" * 160),
    ]
    n_before = len(messages)
    total_chars_before = sum(len(str(m.content) or "") for m in messages)
    estimated_tokens_before = total_chars_before / 4

    skip = asyncio.run(hook(messages, []))

    total_chars_after = sum(len(str(m.content) or "") for m in messages)
    estimated_tokens_after = total_chars_after / 4

    print(f"  context_window: 200 tokens, threshold: 60%  (trigger at 120 tokens)")
    print(f"  Messages before: {n_before}")
    print(f"  Estimated tokens before: {estimated_tokens_before:.0f} "
          f"({estimated_tokens_before / 200:.0%} of window)")
    print(f"  Messages after: {len(messages)}")
    print(f"  Estimated tokens after:  {estimated_tokens_after:.0f} "
          f"({estimated_tokens_after / 200:.0%} of window)")
    print(f"  System message preserved: {messages[0].role == 'system'}")
    print(f"  Hook skip=False (handled, not skip): {not skip}")
    print()

    print("  Usage:")
    print("    agent = Agent(")
    print("        tool_config=ToolConfig(context_overflow_threshold=0.8),")
    print("        ...)")
    print("  Set context_overflow_threshold=0.0 to disable (default).")
    print("  Recommended: 0.8 for production (trigger at 80% of context window).")


# ---------------------------------------------------------------------------
# Demo 2: Enable on a full Agent
# ---------------------------------------------------------------------------

def demo_full_agent():
    print("=" * 60)
    print("Demo 2: Full Agent configuration")
    print("=" * 60)
    print("""
    A 'deep agent' equivalent: context overflow handling + tool call limit.
    No Agent subclass needed.
    """)

    agent = Agent(
        name="deep-agent",
        model=OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY", "fake")),
        tool_config=ToolConfig(
            context_overflow_threshold=0.8,   # compress-then-evict at 80% context
            tool_call_limit=50,                # hard stop at 50 total calls
        ),
    )

    print(f"  tool_config.context_overflow_threshold = {agent.tool_config.context_overflow_threshold}")
    print(f"  tool_config.tool_call_limit            = {agent.tool_config.tool_call_limit}")
    print()
    print("  Old pattern (DeepAgent subclass):")
    print("    from agentica import DeepAgent")
    print("    agent = DeepAgent(model=model, work_dir='/path')")
    print()
    print("  New pattern (ToolConfig fields on Agent):")
    print("    from agentica import Agent")
    print("    from agentica.tools.builtin import get_builtin_tools")
    print("    agent = Agent(")
    print("        model=model,")
    print("        tools=get_builtin_tools(work_dir='/path'),")
    print("        prompt_config=PromptConfig(enable_agentic_prompt=True),")
    print("        tool_config=ToolConfig(")
    print("            context_overflow_threshold=0.8,")
    print("        ),")
    print("    )")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Agentica: Model-layer hooks (context overflow handling)\n")
    demo_context_overflow()
    demo_full_agent()
    print("=" * 60)
    print("Summary: Enable deep-agent capabilities via ToolConfig fields.")
    print("No subclassing needed. The hook is disabled by default (zero overhead).")
    print("=" * 60)
