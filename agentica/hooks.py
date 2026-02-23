# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Lifecycle hooks for Agent runs.

Two levels of hooks:
- AgentHooks: per-agent hooks (on_start, on_end), set on Agent instance
- RunHooks: global run-level hooks (on_agent_start, on_agent_end, on_llm_start,
  on_llm_end, on_tool_start, on_tool_end, on_agent_transfer), passed to run()
"""
from typing import Any, Optional, List, Dict


class AgentHooks:
    """Per-agent lifecycle hooks.

    Subclass and override the methods you need. Attach to an Agent via
    ``Agent(hooks=MyHooks())``.

    Example::

        class LoggingHooks(AgentHooks):
            async def on_start(self, agent, **kwargs):
                print(f"{agent.name} starting")

            async def on_end(self, agent, output, **kwargs):
                print(f"{agent.name} produced: {output}")
    """

    async def on_start(self, agent: Any, **kwargs) -> None:
        """Called when this agent begins a run."""
        pass

    async def on_end(self, agent: Any, output: Any, **kwargs) -> None:
        """Called when this agent finishes a run."""
        pass


class RunHooks:
    """Global run-level lifecycle hooks.

    These hooks observe the entire run, including LLM calls, tool calls,
    and agent transfers. Pass to ``agent.run(hooks=MyRunHooks())``.

    Example::

        class MetricsHooks(RunHooks):
            def __init__(self):
                self.event_counter = 0

            async def on_agent_start(self, agent, **kwargs):
                self.event_counter += 1
                print(f"#{self.event_counter}: Agent {agent.name} started")

            async def on_llm_start(self, agent, messages, **kwargs):
                self.event_counter += 1
                print(f"#{self.event_counter}: LLM call started")

            async def on_llm_end(self, agent, response, **kwargs):
                self.event_counter += 1
                print(f"#{self.event_counter}: LLM call ended")

            async def on_tool_start(self, agent, tool_name, tool_call_id, tool_args, **kwargs):
                self.event_counter += 1
                print(f"#{self.event_counter}: Tool {tool_name} started")

            async def on_tool_end(self, agent, tool_name, tool_call_id, tool_args, result, **kwargs):
                self.event_counter += 1
                print(f"#{self.event_counter}: Tool {tool_name} ended")

            async def on_agent_transfer(self, from_agent, to_agent, **kwargs):
                self.event_counter += 1
                print(f"#{self.event_counter}: Transfer from {from_agent.name} to {to_agent.name}")

            async def on_agent_end(self, agent, output, **kwargs):
                self.event_counter += 1
                print(f"#{self.event_counter}: Agent {agent.name} ended")
    """

    async def on_agent_start(self, agent: Any, **kwargs) -> None:
        """Called when any agent begins execution within this run."""
        pass

    async def on_agent_end(self, agent: Any, output: Any, **kwargs) -> None:
        """Called when any agent finishes execution within this run."""
        pass

    async def on_llm_start(
        self,
        agent: Any,
        messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> None:
        """Called before each LLM API call."""
        pass

    async def on_llm_end(
        self,
        agent: Any,
        response: Any = None,
        **kwargs,
    ) -> None:
        """Called after each LLM API call returns."""
        pass

    async def on_tool_start(
        self,
        agent: Any,
        tool_name: str = "",
        tool_call_id: str = "",
        tool_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Called before a tool begins execution."""
        pass

    async def on_tool_end(
        self,
        agent: Any,
        tool_name: str = "",
        tool_call_id: str = "",
        tool_args: Optional[Dict[str, Any]] = None,
        result: Any = None,
        is_error: bool = False,
        elapsed: float = 0.0,
        **kwargs,
    ) -> None:
        """Called after a tool finishes execution."""
        pass

    async def on_agent_transfer(
        self,
        from_agent: Any,
        to_agent: Any,
        **kwargs,
    ) -> None:
        """Called when a task is transferred from one agent to another."""
        pass
