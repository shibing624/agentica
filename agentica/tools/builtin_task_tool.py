# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: BuiltinTaskTool — thin LLM-facing adapter over ``SubagentRegistry.spawn()``.

The real subagent runtime (model cloning, tool inheritance + filtering, depth
limit, registry tracking, event streaming, usage merge, timeout) lives in
``agentica.subagent.SubagentRegistry``. This tool only:

  1. Renders the user-facing system prompt (the available subagent table).
  2. Exposes a single ``task(description, subagent_type)`` LLM function that
     forwards to ``SubagentRegistry().spawn(parent_agent=self._parent_agent, ...)``.
  3. JSON-serializes the registry's structured result for the LLM.
"""
import json
from textwrap import dedent
from typing import Optional, Dict, Any, TYPE_CHECKING

from agentica.tools.base import Tool

if TYPE_CHECKING:
    from agentica.agent import Agent
    from agentica.model.base import Model


class BuiltinTaskTool(Tool):
    """LLM-facing wrapper around the subagent runtime.

    Subagent execution itself is implemented by ``SubagentRegistry.spawn()`` —
    this class is intentionally a thin adapter so there is exactly one place to
    fix subagent behavior.

    Supports the built-in subagent types (``explore`` / ``research`` / ``code``)
    and any custom types registered via ``register_custom_subagent``.
    """

    TASK_SYSTEM_PROMPT_TEMPLATE = dedent("""## task Tool (Subagent Spawner)

    Launch a subagent to handle complex, multi-step tasks autonomously.
    Each subagent runs in its own isolated context window and returns a single result.

    ### Available Subagent Types

    {subagent_table}

    ### Writing the Description (IMPORTANT)

    Brief the subagent like a smart colleague who just walked into the room — it hasn't seen
    this conversation, doesn't know what you've tried, doesn't understand why this task matters.

    - Explain what you're trying to accomplish and why
    - Describe what you've already learned or ruled out
    - Give enough context about the surrounding problem
    - If you need a short response, say so ("report in under 200 words")

    **Never delegate understanding.** Don't write "based on your findings, fix the bug" or
    "based on the research, implement it." Those phrases push synthesis onto the subagent
    instead of doing it yourself. You should synthesize the subagent's result yourself.

    ### Don't Peek, Don't Race

    After launching a subagent, you know nothing about what it found until it returns.
    - Do NOT fabricate or predict subagent results
    - Trust the returned output — the subagent's results should generally be trusted
    - Do NOT re-read files the subagent already examined unless you need to verify something specific

    ### Parallel Execution

    When you have **multiple independent tasks**, launch them in parallel:
    - Tasks execute simultaneously — total time = max(task_times), not sum(task_times)
    - Ideal for: exploring multiple directories, researching multiple topics, running multiple experiments

    ### When to Use

    - Research: open-ended exploration across many files or directories
    - Implementation: work that requires more than a couple of edits
    - Multi-part tasks: independent subtasks that can run in parallel

    ### When NOT to Use

    - Task is trivial (1-3 tool calls) — just do it directly
    - You need to see intermediate steps
    - Task depends heavily on the main conversation context
    - Reading a specific known file — use read_file instead
    - Searching for a specific definition — use grep/glob instead""")

    def __init__(self, model_override: Optional["Model"] = None):
        """
        Args:
            model_override: Optional model used by every subagent spawned through
                this tool instance. When ``None`` (default), the parent agent's
                model is cloned. Useful when the caller wants subagents to run on
                a different (cheaper/faster) model than the parent.
        """
        super().__init__(name="builtin_task_tool")
        self._model_override = model_override
        self._parent_agent: Optional["Agent"] = None
        self.register(self.task)
        self.functions["task"].manages_own_timeout = True
        self.functions["task"].interrupt_behavior = "block"

    def _build_subagent_table(self) -> str:
        """Build a markdown table of available subagent types."""
        from agentica.subagent import get_available_subagent_types

        lines = ["| Type | Name | Description |", "|------|------|-------------|"]
        for st in get_available_subagent_types():
            desc_first_line = st["description"].split("\n")[0]
            desc = desc_first_line[:60] + ("..." if len(desc_first_line) > 60 else "")
            lines.append(f"| `{st['type']}` | {st['name']} | {desc} |")
        return "\n".join(lines)

    def get_system_prompt(self) -> Optional[str]:
        """Render the available subagent types into the system prompt.

        Regenerated each call so newly registered custom subagents show up.
        """
        return self.TASK_SYSTEM_PROMPT_TEMPLATE.format(
            subagent_table=self._build_subagent_table(),
        )

    def set_parent_agent(self, agent: "Agent") -> None:
        """Bind to the parent agent so ``task()`` can spawn through the registry."""
        self._parent_agent = agent

    def clone(self) -> "BuiltinTaskTool":
        """Fresh instance so each agent owns its ``_parent_agent`` slot.

        Preserves the source's exposed ``functions`` keys so registry-side
        function filtering survives Agent re-cloning. ``task`` is on
        ``SubagentRegistry.BLOCKED_TOOLS`` so child agents normally do not
        inherit this tool at all, but the symmetry is worth keeping.
        """
        from collections import OrderedDict
        new = BuiltinTaskTool(model_override=self._model_override)
        if set(new.functions) != set(self.functions):
            new.functions = OrderedDict(
                (name, new.functions[name])
                for name in self.functions
                if name in new.functions
            )
        return new

    async def task(
        self,
        description: str,
        subagent_type: str = "code",
        timeout: Optional[int] = None,
        max_turns: Optional[int] = None,
        tool_call_limit: Optional[int] = None,
        system_prompt_override: Optional[str] = None,
        resume_from_run_id: Optional[str] = None,
    ) -> str:
        """Launch a subagent to handle a complex task.

        Args:
            description: Detailed description of the task. Brief the subagent
                like a colleague who has no prior context.
            subagent_type: Subagent type id. Built-ins:
                - ``explore``  — read-only codebase exploration
                - ``research`` — web search and document analysis
                - ``code``     — code generation and execution (default)
                Any custom type registered via ``register_custom_subagent`` is
                also accepted.
            timeout: Optional per-call override for the subagent's timeout in
                seconds. Use this to retry a task that hit ``status=timeout`` on
                the previous call — e.g. ``timeout=3600`` for a long research
                task. Default (``None``) keeps the subagent config's own value.
            max_turns: Optional per-call override for the ReAct turn budget.
                Use this to retry a task that hit ``status=max_turns``.
            tool_call_limit: Optional per-call override for the tool call
                budget. Use this to retry a task that hit ``status=tool_call_limit``.
            system_prompt_override: Optional replacement system prompt for
                this one call. Use this when the default subagent prompt is
                pulling the model off-task and you want tighter instructions.
            resume_from_run_id: Optional ``run_id`` from a previous call that
                returned ``partial=true``. When provided, the previous partial
                output is stitched into the continuation prompt so the
                subagent picks up where it left off instead of restarting.
                The failed call's ``next_action`` field tells you which
                ``run_id`` to pass here.

        Returns:
            JSON string with the subagent's final result and execution summary.
        """
        if self._parent_agent is None:
            return json.dumps({
                "success": False,
                "error": "task tool is not bound to a parent agent.",
            }, ensure_ascii=False)

        from agentica.subagent import SubagentRegistry

        result = await SubagentRegistry().spawn(
            parent_agent=self._parent_agent,
            task=description,
            agent_type=subagent_type,
            model_override=self._model_override,
            timeout_override=timeout,
            max_turns_override=max_turns,
            tool_call_limit_override=tool_call_limit,
            system_prompt_override=system_prompt_override,
            resume_from_run_id=resume_from_run_id,
        )

        status = result.get("status", "error")
        # ``completed`` = clean success. ``timeout`` / ``max_turns`` /
        # ``tool_call_limit`` / ``truncated`` = the subagent got interrupted
        # by a budget limit but still produced partial output that the caller
        # should see. We surface those as ``success=false`` (so the parent
        # model knows the task did not finish cleanly) but include ``result``,
        # ``tool_calls_summary`` and ``partial=true`` so the parent can still
        # use whatever work was done. Only genuine ``error`` / ``cancelled``
        # states drop into the bare error payload.
        if status == "completed":
            payload: Dict[str, Any] = {
                "success": True,
                "subagent_type": result["agent_type"],
                "subagent_name": result.get("subagent_name", result["agent_type"]),
                "result": result["content"],
                "tool_calls_summary": result.get("tool_calls_summary", []),
                "execution_time": result.get("execution_time", 0.0),
                "tool_count": result.get("tool_count", 0),
            }
            return json.dumps(payload, ensure_ascii=False, indent=2)

        if status in ("timeout", "max_turns", "tool_call_limit", "truncated"):
            return json.dumps({
                "success": False,
                "status": status,
                "partial": True,
                "error": result.get("error", f"Subagent stopped due to {status}."),
                "subagent_type": result.get("agent_type", subagent_type),
                "subagent_name": result.get("subagent_name", subagent_type),
                "result": result.get("content", ""),
                "tool_calls_summary": result.get("tool_calls_summary", []),
                "tool_count": result.get("tool_count", 0),
                "elapsed_seconds": result.get("elapsed_seconds", 0.0),
                # ``run_id`` + ``next_action`` are how the parent Agent's ReAct
                # loop learns it can resume this task instead of restarting.
                "run_id": result.get("run_id"),
                "next_action": result.get("next_action"),
                "description": description[:300],
            }, ensure_ascii=False, indent=2)

        # Genuine failure: still surface partial content if any was recovered
        # (e.g. exception mid-stream), otherwise fall back to bare error.
        payload = {
            "success": False,
            "status": status,
            "error": result.get("error", "Subagent failed without an error message."),
            "subagent_type": result.get("agent_type", subagent_type),
            "run_id": result.get("run_id"),
            "next_action": result.get("next_action"),
            "description": description[:300],
        }
        partial_content = result.get("content") or ""
        if partial_content:
            payload["result"] = partial_content
            payload["partial"] = True
            payload["tool_calls_summary"] = result.get("tool_calls_summary", [])
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @staticmethod
    def _format_tool_brief(tool_name: str, tool_args, content=None) -> str:
        """Format a one-line summary of a subagent tool call for CLI rendering.

        Used by ``SubagentRegistry._run_child_streaming`` to label each
        ``ToolCallStarted`` / ``ToolCallCompleted`` event the subagent emits.
        """
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except (json.JSONDecodeError, TypeError):
                tool_args = {}
        if not isinstance(tool_args, dict):
            tool_args = {}

        if tool_name == "read_file":
            fp = tool_args.get("file_path", "")
            if fp:
                fname = fp.rsplit("/", 1)[-1] if "/" in fp else fp
                lines = ""
                if tool_args.get("offset") or tool_args.get("limit"):
                    start = (tool_args.get("offset", 0) or 0) + 1
                    end = start + (tool_args.get("limit", 500) or 500) - 1
                    lines = f" (L{start}-{end})"
                if content:
                    line_count = str(content).count("\n") + 1
                    return f"Read {line_count} line(s) from {fname}"
                return f"{fname}{lines}"
        elif tool_name in ("grep", "search_content"):
            pattern = tool_args.get("pattern", "")
            if content and isinstance(content, str):
                match_count = content.count("\n") + 1 if content.strip() else 0
                return f'Found {match_count} match(es) for "{pattern[:40]}"'
            return f'"{pattern[:40]}"'
        elif tool_name in ("glob", "search_file"):
            pattern = tool_args.get("pattern", "")
            return f"pattern: {pattern}"
        elif tool_name == "ls":
            directory = tool_args.get("directory", ".")
            return directory.rsplit("/", 1)[-1] if "/" in directory else directory
        elif tool_name == "execute":
            cmd = tool_args.get("command", "")
            return cmd[:80] + ("..." if len(cmd) > 80 else "")
        elif tool_name == "write_file":
            fp = tool_args.get("file_path", "")
            return fp.rsplit("/", 1)[-1] if "/" in fp else fp
        elif tool_name == "edit_file":
            fp = tool_args.get("file_path", "")
            return fp.rsplit("/", 1)[-1] if "/" in fp else fp
        elif tool_name == "multi_edit_file":
            fp = tool_args.get("file_path", "")
            edits = tool_args.get("edits", [])
            fname = fp.rsplit("/", 1)[-1] if "/" in fp else fp
            return f"{fname} ({len(edits)} edits)"
        elif tool_name == "web_search":
            queries = tool_args.get("queries", "")
            if isinstance(queries, list):
                return ", ".join(str(q)[:30] for q in queries[:2])
            return str(queries)[:60]
        elif tool_name == "fetch_url":
            url = tool_args.get("url", "")
            return url[:60] + ("..." if len(url) > 60 else "")

        for k, v in tool_args.items():
            return f"{k}={str(v)[:50]}"
        return ""
