# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: BuiltinTaskTool — thin LLM-facing adapter over ``SubagentRegistry.spawn()``.

The real subagent runtime (model cloning, tool inheritance + filtering, depth
limit, registry tracking, event streaming, usage merge, timeout) lives in
``agentica.subagents.runtime.SubagentRegistry``. This tool only:

  1. Renders the user-facing system prompt (the available subagent table).
  2. Exposes a single ``task(description, prompt, subagent_type)`` LLM function that
     forwards to ``SubagentRegistry().spawn(parent_agent=self._parent_agent, ...)``.
  3. JSON-serializes the registry's structured result for the LLM.
"""
import asyncio
import json
import re
from textwrap import dedent
from typing import Optional, Dict, Any, TYPE_CHECKING

from agentica.compression.tool_result_storage import (
    DEFAULT_MAX_RESULT_SIZE_CHARS,
    can_recover_spill,
    maybe_persist_result,
)
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

    Use it only when you do not yet know where to look. If you already know the
    target file, definition, or a handful of files, look there yourself; task
    size is not the criterion. Not for edits or state-changing commands
    (subagents are refused these), work that needs this conversation's
    context, or merely to keep your own context small. Subagent results are
    evidence you still reason over; do not delegate code review, correctness
    verdicts, root-cause judgement, or release decisions.

    {subagent_table}

    Briefing: the subagent cannot see this conversation. State what you want to
    know and why, where to start, and ask for findings as path:line.

    - Independent tasks in one message run in parallel, three at a time.
    - Once delegated, do not duplicate that work yourself.
    - Each type has a default wall-clock `timeout` (column above). A broad
      investigation (many modules, a long review) can need more: pass a larger
      `timeout` (seconds) up front rather than resuming. Shortly before the
      limit the subagent is told to stop and report what it has.
    - On `partial=true`, synthesize the partial output; if it holds no
      findings, finish the work yourself. Resume at most once, with the SAME
      `prompt` plus `resume_from_run_id`.
    - Each completed task returns a `run_id`, child transcript path, and any
      child tool `output_path` locators. If a summary is insufficient, call
      `get_task_result(run_id)` and then read the transcript/output paths you need.""")

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
        self._slots: Optional[asyncio.Semaphore] = None
        self._slots_loop: Optional[asyncio.AbstractEventLoop] = None
        # A subagent is read-only (writes and state-changing commands are
        # refused, see SubagentRegistry) and owns entirely isolated state — its
        # own cloned model, HTTP client and Agent. It must be concurrency_safe
        # or the executor runs a batch of them one after another, which makes
        # "launch independent tasks in one message to run them in parallel"
        # (stated in this tool's own system prompt) a lie.
        self.register(self.task, concurrency_safe=True, is_read_only=True)
        self.register(self.get_task_result, concurrency_safe=True, is_read_only=True)
        task_fn = self.functions["task"]
        task_fn.manages_own_timeout = True
        task_fn.process_entrypoint()
        task_fn.parameters["properties"]["model_tier"]["enum"] = ["auxiliary", "main"]
        task_fn.skip_entrypoint_processing = True

    def _free_slot(self) -> asyncio.Semaphore:
        """Bound how many subagents this agent runs at once.

        ``task`` is ``concurrency_safe``, so a single assistant turn can put an
        unbounded number of subagents in flight — each one a full Agent with
        its own model and tool loop, and nobody is watching that bill. The
        limit matches ``SubagentRegistry.MAX_CONCURRENT``, which already bounds
        ``spawn_batch``; extra calls wait for a slot rather than being refused.

        Rebuilt per event loop: a Semaphore binds to the loop it is first
        awaited on, and one tool instance can outlive that loop (``run_sync``,
        tests calling ``asyncio.run`` more than once).
        """
        from agentica.subagents import SubagentRegistry

        loop = asyncio.get_running_loop()
        if self._slots is None or self._slots_loop is not loop:
            self._slots = asyncio.Semaphore(SubagentRegistry.MAX_CONCURRENT)
            self._slots_loop = loop
        return self._slots

    def _build_subagent_table(self) -> str:
        """Build a markdown table of available subagent types with their model tier."""
        from agentica.subagents import get_available_subagent_types

        lines = [
            "| Type | Name | Model | Timeout | Description |",
            "|------|------|-------|---------|-------------|",
        ]
        for st in get_available_subagent_types():
            desc_first_line = st["description"].split("\n")[0]
            desc = desc_first_line[:60] + ("..." if len(desc_first_line) > 60 else "")
            timeout = f"{st['timeout']}s" if st["timeout"] else "none"
            lines.append(
                f"| `{st['type']}` | {st['name']} | {st['model_tier']} | {timeout} | {desc} |"
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

    def _can_recover_spill(self) -> bool:
        """Whether this session has a tool that can read a spilled file back.

        ``maybe_persist_result`` writes ``Use read_file ... for the rest`` only
        when the caller says it is recoverable; an IM or otherwise
        tool-restricted session must get a truncation instead.
        """
        return can_recover_spill(self._parent_agent.model.functions or {})

    async def get_task_result(self, run_id: str) -> str:
        """Fetch a prior subagent run by run_id.

        Use this when an earlier ``task`` result was summarized or pushed out of
        context but you still have its ``run_id``. The response includes the
        final/partial answer, completed tool summaries, ``output_path`` locators,
        and the child transcript path when available.

        Args:
            run_id: The run_id returned by a previous task call.
        """
        from agentica.subagents import SubagentRegistry

        parent = self._parent_agent
        run = SubagentRegistry().get_for_owner(run_id, parent.user_id)
        if run is None:
            return json.dumps({"found": False, "run_id": run_id}, ensure_ascii=False, indent=2)

        result = run.result or ""
        if len(result) > DEFAULT_MAX_RESULT_SIZE_CHARS:
            # Same size bound as a freshly completed task: get_task_result must
            # not become a back door that pours a large answer back into the
            # context window after the original result was spilled away.
            result = maybe_persist_result(
                "get_task_result",
                f"{run.run_id}-result",
                result,
                session_id=parent.session_id or "default",
                cwd=parent.work_dir,
                user_id=parent.user_id,
                recoverable=self._can_recover_spill(),
            )
        return json.dumps({
            "found": True,
            "run_id": run.run_id,
            "status": run.status,
            "subagent_type": run.subagent_type,
            "task": run.task_description,
            "result": result,
            "tool_calls_summary": self._display_tool_calls(run.tool_calls_summary or []),
            "child_session_id": run.child_session_id or "",
            "transcript_path": run.transcript_path or "",
        }, ensure_ascii=False, indent=2)

    @staticmethod
    def _display_tool_calls(summary: Any) -> Any:
        """Keep parent-facing tool locators without replaying raw arguments."""
        if not isinstance(summary, list):
            return []
        return [
            {
                k: v
                for k, v in {
                    "name": item.get("name", ""),
                    "info": item.get("info", ""),
                    "output_path": item.get("output_path"),
                }.items()
                if v
            }
            for item in summary
            if isinstance(item, dict)
        ]

    def model_label_for(self, tool_args: Optional[dict] = None) -> Optional[str]:
        """Which model the subagent this call would spawn actually runs.

        Mirrors ``SubagentRegistry.spawn``'s tier selection exactly: a type
        declared ``model_tier: main`` runs the parent's own model, everything
        else runs the auxiliary model this tool was handed (or the parent's
        ``resolve_auxiliary_model("task")``). The user is told because "task"
        on its own does not say whether the work went to the cheap model or
        back to the main one.
        """
        from agentica.model.defaults import model_display_label
        from agentica.subagents import get_subagent_config

        args = tool_args or {}
        agent_type = str(args.get("subagent_type") or "explore").strip().lower()
        config = get_subagent_config(agent_type)
        # An unknown type is refused by spawn() anyway; label the tier it would
        # have resolved to so the call line does not go blank with the error.
        explicit_tier = str(args.get("model_tier") or "").strip().lower()
        tier = explicit_tier if explicit_tier in ("auxiliary", "main") else (config.model_tier if config is not None else "auxiliary")
        if tier == "main":
            return model_display_label(self._parent_agent.model if self._parent_agent is not None else None)
        auxiliary = self._auxiliary_model
        if auxiliary is None and self._parent_agent is not None:
            auxiliary = self._parent_agent.resolve_auxiliary_model("task")
        return model_display_label(auxiliary)

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
        prompt: str,
        subagent_type: str = "explore",
        timeout: Optional[int] = None,
        max_turns: Optional[int] = None,
        resume_from_run_id: Optional[str] = None,
        model_tier: Optional[str] = None,
    ) -> str:
        """Delegate an open-ended read-only investigation to a subagent in its own context window; returns one summary, not intermediate steps. See the "task Tool" section for available types and when this applies.

        Args:
            description: Short (3-5 word) label for this task, shown to the user.
            prompt: Complete, self-contained task for the subagent. It does not share this conversation's context, so include the goal, background, starting paths, and ask for findings as path:line.
            subagent_type: Subagent type id from the "task Tool" table (default "explore").
            timeout: Wall-clock limit in seconds for this call; defaults to the type's configured timeout. Pass only to raise it; a lower value is ignored.
            max_turns: ReAct turn budget; defaults to the type's own (100-200). Pass only to raise it, e.g. when resuming.
            resume_from_run_id: run_id of a prior partial subagent run to resume at most once.
            model_tier: Model tier override: "auxiliary" (fast/cheap default for fact-gathering) or "main" (parent's main model for complex analysis).
        """
        if self._parent_agent is None:
            return json.dumps({
                "success": False,
                "error": "task tool is not bound to a parent agent.",
            }, ensure_ascii=False)

        from agentica.subagents import SubagentRegistry, get_subagent_config

        tier_override = (
            model_tier.strip().lower()
            if isinstance(model_tier, str) and model_tier.strip().lower() in ("auxiliary", "main")
            else None
        )
        # The budget can only be raised from here. A model that guesses a
        # "small" number (12 turns, or 30 seconds) cuts a 200-turn / 1800s
        # investigation down to a stub and returns a timeout or max_turns stop
        # instead of a report. A raise is a real request and is honoured.
        config = get_subagent_config(subagent_type)
        if max_turns is not None and config is not None and max_turns < config.max_turns:
            max_turns = None
        if timeout is not None and config is not None and timeout < config.timeout:
            timeout = None
        async with self._free_slot():
            result = await SubagentRegistry().spawn(
                parent_agent=self._parent_agent,
                task=prompt,
                agent_type=subagent_type,
                auxiliary_model_override=self._auxiliary_model,
                timeout_override=timeout,
                max_turns_override=max_turns,
                resume_from_run_id=resume_from_run_id,
                model_tier_override=tier_override,  # type: ignore[arg-type]
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
            result_content = result["content"]
            if len(result_content) > DEFAULT_MAX_RESULT_SIZE_CHARS:
                parent = self._parent_agent
                result_content = maybe_persist_result(
                    "task",
                    f"{result.get('run_id') or 'subagent'}-result",
                    result_content,
                    session_id=parent.session_id or "default",
                    cwd=parent.work_dir,
                    user_id=parent.user_id,
                    recoverable=self._can_recover_spill(),
                )
            payload: Dict[str, Any] = {
                "success": True,
                "subagent_type": result["agent_type"],
                "subagent_name": result.get("subagent_name", result["agent_type"]),
                "model": result.get("model", ""),
                "model_tier": result.get("model_tier", ""),
                "result": result_content,
                "tool_calls_summary": self._display_tool_calls(result.get("tool_calls_summary", [])),
                "execution_time": result.get("execution_time", 0.0),
                "tool_count": result.get("tool_count", 0),
            }
            if result.get("run_id"):
                payload["run_id"] = result["run_id"]
            if result.get("child_session_id"):
                payload["child_session_id"] = result["child_session_id"]
            if result.get("transcript_path"):
                payload["transcript_path"] = result["transcript_path"]
            return json.dumps(payload, ensure_ascii=False, indent=2)

        if status in ("timeout", "max_turns", "tool_call_limit", "truncated"):
            return json.dumps({
                "success": False,
                "status": status,
                "partial": True,
                "error": result.get("error", f"Subagent stopped due to {status}."),
                "subagent_type": result.get("agent_type", subagent_type),
                "subagent_name": result.get("subagent_name", subagent_type),
                "model": result.get("model", ""),
                "model_tier": result.get("model_tier", ""),
                "result": result.get("content", ""),
                "tool_calls_summary": self._display_tool_calls(result.get("tool_calls_summary", [])),
                "tool_count": result.get("tool_count", 0),
                "execution_time": result.get("execution_time", result.get("elapsed_seconds", 0.0)),
                # ``run_id`` + ``next_action`` are how the parent Agent's ReAct
                # loop learns it can resume this task instead of restarting.
                "run_id": result.get("run_id"),
                "child_session_id": result.get("child_session_id"),
                "transcript_path": result.get("transcript_path"),
                "next_action": result.get("next_action"),
                "description": description[:300],
            }, ensure_ascii=False, indent=2)

        # Genuine failure: still surface partial content if any was recovered
        # (e.g. exception mid-stream), otherwise fall back to bare error.
        payload: Dict[str, Any] = {
            "success": False,
            "status": status,
            "error": result.get("error", "Subagent failed without an error message."),
            "subagent_type": result.get("agent_type", subagent_type),
            "model": result.get("model", ""),
            "model_tier": result.get("model_tier", ""),
            "run_id": result.get("run_id"),
            "child_session_id": result.get("child_session_id"),
            "transcript_path": result.get("transcript_path"),
            "next_action": result.get("next_action"),
            "description": description[:300],
        }
        partial_content = result.get("content") or ""
        if partial_content:
            payload["result"] = partial_content
            payload["partial"] = True
            payload["tool_calls_summary"] = self._display_tool_calls(result.get("tool_calls_summary", []))
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
        elif tool_name == "execute":
            cmd = tool_args.get("command", "")
            return cmd[:80] + ("..." if len(cmd) > 80 else "")
        elif tool_name == "write_file":
            fp = tool_args.get("file_path", "")
            return fp.rsplit("/", 1)[-1] if "/" in fp else fp
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
