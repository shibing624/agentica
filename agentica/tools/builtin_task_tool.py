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
import re
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

    Available types come from package, user, and project Markdown definitions.
    """

    # Note the leading backslash: without it the first line carries no indent,
    # ``dedent`` finds a common prefix of "" and leaves every other line indented
    # by 4 spaces — which markdown renders as a code block and which splits the
    # substituted subagent table from its header row.
    TASK_SYSTEM_PROMPT_TEMPLATE = dedent("""\
    ## task Tool (Subagent Spawner)

    Launch a READ-ONLY subagent to investigate an open-ended question in its own
    context window; it returns one summary and you do all edits yourself.

    Use it only when you do not yet know where to look — the search would span
    directories you cannot enumerate. If you already know the target file,
    definition, or a handful of files, use `read_file` / `grep` / `glob`
    directly; task size is not the criterion. Do not use `task` for edits or
    state-changing commands (subagents are refused these), for work that needs
    this conversation's context, or merely to keep your own context small.

    The `Model` column below shows which model a type runs on. Treat every
    subagent result as evidence that you still reason over. Do not delegate code
    review, correctness verdicts, root-cause judgement, or release decisions;
    those belong to the main agent with the full conversation and diff context.

    {subagent_table}

    Briefing: the subagent cannot see this conversation. State what you want to
    know and why, the files or functions to start from, and ask for findings as
    path:line. Never ask it to edit or implement.

    - Launch independent tasks in one message to run them in parallel.
    - Once delegated, do not duplicate that work yourself.
    - On `partial=true`, default to synthesizing the partial output. If the
      result contains only a timeout/limit note and no substantive findings,
      do not resume merely to force a final answer; finish the work yourself.
      Resume at most once when real partial findings make continuation useful,
      using the SAME `description` plus `resume_from_run_id`.""")

    def __init__(self, auxiliary_model: Optional["Model"] = None):
        """
        Args:
            auxiliary_model: Model used by ``model_tier="auxiliary"`` subagents
                spawned through this tool. When ``None`` (default) the parent
                agent's ``resolve_auxiliary_model("task")`` decides.
        """
        super().__init__(name="builtin_task_tool")
        self._auxiliary_model = auxiliary_model
        self._parent_agent: Optional["Agent"] = None
        self.register(self.task)
        self.functions["task"].manages_own_timeout = True
        self.functions["task"].interrupt_behavior = "block"

    def _build_subagent_table(self) -> str:
        """Build a markdown table of available subagent types with their model tier."""
        from agentica.subagent import get_available_subagent_types

        lines = [
            "| Type | Name | Model | Description |",
            "|------|------|-------|-------------|",
        ]
        for st in get_available_subagent_types():
            desc_first_line = st["description"].split("\n")[0]
            desc = desc_first_line[:60] + ("..." if len(desc_first_line) > 60 else "")
            lines.append(
                f"| `{st['type']}` | {st['name']} | {st['model_tier']} | {desc} |"
            )
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
        new = BuiltinTaskTool(auxiliary_model=self._auxiliary_model)
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
        subagent_type: str = "explore",
        timeout: Optional[int] = None,
        max_turns: Optional[int] = None,
        system_prompt_override: Optional[str] = None,
        resume_from_run_id: Optional[str] = None,
    ) -> str:
        """Spawn a subagent. See the "task Tool" section for when this applies.

        Args:
            description: What to investigate. Brief the subagent like a
                colleague who has no prior context.
            subagent_type: One of the type ids listed in that section
                (default ``explore``).
            timeout: Per-call timeout override (seconds).
            max_turns: Per-call ReAct turn budget override.
            system_prompt_override: Replacement system prompt for this call.
            resume_from_run_id: ``run_id`` of a prior partial run to resume.
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
            auxiliary_model_override=self._auxiliary_model,
            timeout_override=timeout,
            max_turns_override=max_turns,
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
        elif tool_name == "apply_patch":
            patch = str(tool_args.get("patch", ""))
            count = len(re.findall(
                r"^\*\*\* (?:Add|Update|Delete) File: ", patch, re.MULTILINE
            ))
            return f"{count} {'file' if count == 1 else 'files'}"
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
