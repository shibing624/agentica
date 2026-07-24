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

    # Note the leading backslash: without it the first line carries no indent,
    # ``dedent`` finds a common prefix of "" and leaves every other line indented
    # by 4 spaces — which markdown renders as a code block and which splits the
    # substituted subagent table from its header row.
    TASK_SYSTEM_PROMPT_TEMPLATE = dedent("""\
    ## task Tool (Subagent Spawner)

    Launch a READ-ONLY subagent to investigate a complex task autonomously.
    Each subagent runs in its own isolated context window and returns a single result.

    ### Subagents are READ-ONLY (IMPORTANT)

    No subagent can edit files. It can read, search, analyze, and run commands
    that only *inspect* state (`git diff`, `git log`, `git show`, test and lint
    runners) — anything that changes state (commits, installs, writes) is
    refused. YOU (the main agent) do ALL edits and state-changing commands
    yourself, based on the subagent's findings. Never delegate implementation:
    delegate *investigation*, then implement the result yourself.

    ### Two Model Tiers (IMPORTANT)

    The `Model` column below tells you which model a type runs on.

    - `auxiliary` types run on a **cheaper, weaker** model. They are for
      **gathering facts**: where is X, which files touch Y, what does this module
      do, what does the web say. A weak model is reliable at retrieval.
    - `main` types run on **your own model** and cost real money. They are for
      **judgement**: is this code correct, what is the root cause, is this ready
      to ship. Use them sparingly and always narrowly scoped.

    **Never send a judgement question to an `auxiliary` type.** A weak model
    answers "looks fine" with total confidence and you will believe it — that is
    worse than not delegating at all. If the question is "is this right?", either
    answer it yourself or use a `main` type.

    Read the results accordingly: an `auxiliary` result is **evidence** (paths,
    snippets, quotes) that you still reason over yourself; a `main` result is an
    **opinion** you can weigh directly.

    ### Available Subagent Types

    {subagent_table}

    ### Scoping a `main`-tier Task

    A vague brief wastes the expensive model. Before launching one:
    - Name the exact files and functions to look at — it cannot see your
      conversation, though it can run `git diff` itself to see what changed
    - State the single question you want answered
    - Say what you already checked so it does not repeat your work
    - Ask for findings as path:line plus the concrete failure, not general advice

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
    - Take the reported facts (paths, snippets, quotes) at face value
    - Do NOT re-read files the subagent already examined unless you need to verify something specific
    - An `auxiliary` subagent's *conclusions* are not authoritative: if one draws
      a surprising conclusion, check it against the evidence it cited

    ### Parallel Execution

    When you have **multiple independent READ-ONLY tasks**, launch them in parallel:
    - Tasks execute simultaneously — total time = max(task_times), not sum(task_times)
    - Ideal for: exploring multiple directories, researching multiple topics
    - Do not fan out `main`-tier tasks in parallel; they are expensive

    ### When to Use

    - Exploration: open-ended search across many files or directories
    - Research: web search and document analysis
    - Code analysis: descriptive read-only questions (trace logic, summarize a module)
    - Review: a narrow correctness / root-cause question worth a fresh, unbiased
      read on your own model — give it the changed files, since it cannot see
      your diff
    - Multi-part READ-ONLY tasks that can run in parallel

    ### When NOT to Use

    - Editing / writing files — do it YOURSELF, do not delegate to a subagent
    - Running commands — do it YOURSELF
    - Any judgement question on an `auxiliary` type — answer it yourself or use `review`
    - Broad "review everything / check my work" sweeps — either scope it to a
      module and one question, or do it yourself
    - Avoiding context compression during a large refactor — make a compact
      implementation spec, investigate with a subagent if needed, then edit and
      verify one dependency-ordered phase at a time yourself
    - Task is trivial (1-3 tool calls) — just do it directly
    - You need to see intermediate steps
    - Task depends heavily on the main conversation context
    - Reading a specific known file — use read_file instead
    - Searching for a specific definition — use grep/glob instead

    ### Handling Partial Results (retry/resume)

    A subagent may stop early with ``partial=true`` (timeout or max_turns).
    When that happens:

    - **Default to synthesizing the partial output yourself.** It usually
      contains enough to proceed — do not reflexively relaunch the task.
    - Only resume when the task genuinely must run to completion. To resume,
      call ``task`` again with the SAME ``description`` and
      ``resume_from_run_id`` set to the failed call's ``run_id`` (from its
      ``next_action`` field), optionally raising ``timeout`` or ``max_turns``.
      The partial output is stitched in automatically.
    - **Resume at most once per task.** If a resumed call still stops early,
      synthesize what you have instead of looping.
    - Use ``system_prompt_override`` only when the default subagent prompt is
      pulling the model off-task.""")

    def __init__(self, auxiliary_model: Optional["Model"] = None):
        """
        Args:
            auxiliary_model: Model used by ``model_tier="auxiliary"`` subagents
                spawned through this tool. When ``None`` (default) the parent
                agent's ``resolve_auxiliary_model("task")`` decides. ``main``-tier
                types (e.g. ``review``) ignore this and always run on the parent's
                own model — judgement work must not be downgraded to a weak model.
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
        """Launch a read-only subagent to investigate a complex task.

        Args:
            description: Detailed description of the task. Brief the subagent
                like a colleague who has no prior context.
            subagent_type: Subagent type id (``explore`` / ``research`` / ``code``
                / ``review``, default ``explore``), or any custom type registered
                via ``register_custom_subagent``. All built-in types are READ-ONLY —
                they cannot edit files or run commands; the main agent does all
                edits based on the subagent's findings. ``explore`` / ``research``
                / ``code`` run on the cheap auxiliary model and are for gathering
                facts; ``review`` runs on the main model and is for judgement
                questions (correctness, root cause, readiness) that a weak model
                answers confidently and wrongly.
            timeout: Optional per-call timeout override (seconds).
            max_turns: Optional per-call ReAct turn budget override.
            system_prompt_override: Optional replacement system prompt for this call.
            resume_from_run_id: Optional ``run_id`` to resume a prior partial run.

        The retry/resume parameters above are only for continuing an
        interrupted run; see the task tool system prompt for guidance.

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
