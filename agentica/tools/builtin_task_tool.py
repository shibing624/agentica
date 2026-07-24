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

    Launch a READ-ONLY subagent to investigate an open-ended question in its own
    isolated context window. It returns one summary; you do all edits yourself.

    When broadly exploring the codebase to gather context for a large task —
    the search would span directories you cannot enumerate yet — prefer `task`
    with an exploration type over running the search tools yourself. Outside
    that case, search yourself.

    ### When NOT to use the task tool

    - Reading a specific known file — use `read_file`, it reaches the answer sooner
    - Finding a specific definition like `class Foo` — use `grep`, it reaches the answer sooner
    - Anything scoped to one file or a few known files — read them directly, it reaches the answer sooner
    - Editing files or running state-changing commands — subagents are refused
      these. Do it YOURSELF; delegate *investigation*, never implementation
    - A judgement question ("is this correct?", "is this ready?") on an
      `auxiliary` type — answer it yourself or use a `main` type
    - Work that depends on this conversation's context, or where you need to see
      the intermediate steps
    - No listed subagent type is a good fit — use the direct tools

    The test is **whether you already know where to look**, not how big the job
    is. A large task made of known targets is still direct-tool work; delegate
    only when the search space itself is the unknown. Never reach for `task`
    merely to keep your own context small.

    Examples:
    - "Where is CompressionManager defined?" → `grep` directly; the target is known
    - "Read runner.py and tell me what num_input_messages does" → `read_file` directly
    - "Fix the failing test in test_runner.py" → do it yourself; known file, and it needs edits
    - "How does compaction interact with session resume?" → `task` with `explore`;
      open-ended, spans files you cannot name yet
    - "Does this diff break the retry path?" → `task` with a `main` type, scoped to
      the changed files; it is a judgement question

    ### Two Model Tiers

    The `Model` column below says which model a type runs on. `auxiliary` is a
    cheaper, weaker model — reliable at **retrieval** (where is X, which files
    touch Y, what does the web say). `main` is your own model — for **judgement**,
    used sparingly and always narrowly scoped.

    **Never send a judgement question to an `auxiliary` type.** A weak model says
    "looks fine" with total confidence and you will believe it — worse than not
    delegating at all. Treat an `auxiliary` result as **evidence** you still
    reason over; a `main` result as an **opinion** you can weigh.

    ### Available Subagent Types

    {subagent_table}

    ### Briefing a Subagent

    It cannot see this conversation. State what you want to know and why, what
    you already ruled out, and the exact files or functions to start from. Ask
    for findings as path:line plus the concrete detail, not general advice.

    Never write "based on your findings, fix the bug" or "then implement it" —
    that pushes synthesis onto the subagent. **You** synthesize the result.

    After launching one you know nothing until it returns: do not predict its
    findings, and do not re-read what it examined unless you need to verify a
    specific claim.

    ### Usage Notes

    1. Launch independent READ-ONLY tasks in a single message to run them in
       parallel — total time becomes max, not sum. Do not fan out `main`-tier
       tasks; they are expensive.
    2. Once delegated, do not duplicate that work yourself.
    3. On `partial=true` (timeout or max_turns), default to synthesizing the
       partial output — it usually suffices. Resume at most once, by calling
       `task` again with the SAME `description` plus `resume_from_run_id` set to
       the failed `run_id`; the partial output is stitched in automatically.
    4. Use `system_prompt_override` only when the default subagent prompt is
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
        """Launch a read-only subagent to investigate an open-ended question.

        Use this only when you cannot name the files to look at yet. If you
        already know the target file, definition, or a handful of files, use
        `read_file` / `grep` / `glob` directly — they reach the answer sooner.
        Task size is not the criterion; a big job made of known targets is still
        direct-tool work.

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
