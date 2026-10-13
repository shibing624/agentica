# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: ``acp_agent`` — hand a coding task to an external agent over ACP.

The interactive CLI owns one ``ACPSessionRegistry`` for its lifetime (like
``BackgroundProcessRegistry``), so a session opened in one turn can be
continued in the next and survives ``/model``; the CLI closes every session
on exit. The runner's permission requests become ordinary approval cards via
``Agent.approve`` under the tool name ``acp_permission`` (see
``agentica.agent.approvals.classify``): a session with nobody to ask denies.
"""
import asyncio
import os
import shutil
import threading
from typing import Any, Dict, List, Optional

from agentica.tools.acp_client import ACPClientError, ACPSession, PromptResult
from agentica.tools.base import Function, FunctionCall, Tool

ACP_AGENT_TOOL = "acp_agent"
ACP_PERMISSION_TOOL = "acp_permission"
MAX_SESSIONS = 3
DEFAULT_TIMEOUT_SECONDS = 1800
_MAX_TEXT_CHARS = 12000
_MAX_TOOL_CALLS_SHOWN = 40

# Verified against npm: @zed-industries/codex-acp, @zed-industries/claude-agent-acp.
DEFAULT_RUNNERS: Dict[str, Dict[str, Any]] = {
    "codex": {"command": "npx", "args": ["-y", "@zed-industries/codex-acp"]},
    "claude_code": {"command": "npx", "args": ["-y", "@zed-industries/claude-agent-acp"]},
    "opencode": {"command": "opencode", "args": ["acp"]},
}


def merge_runners(configured: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Defaults overlaid with config.yaml ``settings.acp_runners`` (same name replaces)."""
    runners = {name: dict(spec) for name, spec in DEFAULT_RUNNERS.items()}
    for name, spec in (configured or {}).items():
        runners[name] = {"command": spec["command"], "args": list(spec.get("args", [])),
                         "env": dict(spec.get("env", {}))}
    return runners


class ACPSessionRegistry:
    """Open ACP sessions of one CLI process, keyed by ``session_id``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: Dict[str, ACPSession] = {}
        self._runner_of: Dict[str, str] = {}

    def add(self, runner: str, session: ACPSession) -> None:
        with self._lock:
            self._sessions[session.session_id] = session
            self._runner_of[session.session_id] = runner

    def get(self, session_id: str) -> ACPSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"no open acp_agent session {session_id!r}; action='list' shows them")
        return session

    def runner_of(self, session_id: str) -> str:
        return self._runner_of[session_id]

    def items(self) -> List[ACPSession]:
        with self._lock:
            return list(self._sessions.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def close(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
            self._runner_of.pop(session_id, None)
        if session is not None:
            session.close()

    def close_all(self) -> None:
        with self._lock:
            sessions, self._sessions = list(self._sessions.values()), {}
            self._runner_of.clear()
        for session in sessions:
            session.close()


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit * 2 // 5
    tail = limit - head
    return f"{text[:head]}\n... [{len(text) - limit} chars omitted] ...\n{text[-tail:]}"


def format_result(runner: str, session_id: str, result: PromptResult) -> str:
    """The report the calling model reads."""
    lines = [f"[{runner} session {session_id} · stopReason: {result.stop_reason}]"]
    if result.timed_out:
        lines.append("Timed out: the turn was cancelled; the session is still open.")
    lines.append(_clip(result.text.strip(), _MAX_TEXT_CHARS) or "(no message text)")
    if result.tool_calls:
        lines.append(f"\nTool calls ({len(result.tool_calls)}):")
        for call in result.tool_calls[:_MAX_TOOL_CALLS_SHOWN]:
            lines.append(f"- [{call['status']}] {call['kind']}: {call['title']}")
        if len(result.tool_calls) > _MAX_TOOL_CALLS_SHOWN:
            lines.append(f"- ... {len(result.tool_calls) - _MAX_TOOL_CALLS_SHOWN} more")
    if result.plan:
        lines.append("\nPlan:")
        lines.extend(f"- [{entry['status']}] {entry['content']}" for entry in result.plan)
    if result.permissions_allowed or result.permissions_denied:
        lines.append(
            f"\nPermission requests: {result.permissions_allowed} allowed, "
            f"{result.permissions_denied} denied"
        )
    return "\n".join(lines)


class ACPAgentTool(Tool):
    """Drive Codex / Claude Code / OpenCode (or any ACP agent) as an executor."""

    def __init__(
        self,
        registry: ACPSessionRegistry,
        runners: Dict[str, Dict[str, Any]],
        work_dir: Optional[str] = None,
    ) -> None:
        super().__init__(name="acp_agent_tool")
        self._registry = registry
        self._runners = runners
        self._work_dir = work_dir
        self.register(self.acp_agent, is_destructive=True)
        # A turn of an external agent runs for minutes; the tool owns its timeout.
        self.functions[ACP_AGENT_TOOL].manages_own_timeout = True

    def _cwd(self, work_dir: str) -> str:
        raw = (work_dir or "").strip()
        base = self._work_dir or os.getcwd()
        if not raw:
            return os.path.abspath(base)
        path = os.path.expanduser(raw)
        return os.path.abspath(path if os.path.isabs(path) else os.path.join(base, path))

    async def _decide(self, runner: str, params: Dict[str, Any]) -> str:
        """Route one ``session/request_permission`` through this agent's approver."""
        agent = self.functions[ACP_AGENT_TOOL]._agent
        if agent is None or agent.approve is None:
            return "deny"
        tool_call = params["toolCall"]
        locations = tool_call.get("locations") or []
        fn = Function(name=ACP_PERMISSION_TOOL)
        fn.entrypoint = lambda **kwargs: None
        fc = FunctionCall(
            function=fn,
            arguments={
                "runner": runner,
                "title": tool_call.get("title") or "",
                "kind": tool_call.get("kind") or "other",
                "paths": [loc["path"] for loc in locations],
            },
            call_id=f"acp:{tool_call['toolCallId']}",
        )
        return await agent.approve(fc)

    def _list(self) -> str:
        lines = ["Runners:"]
        for name, spec in self._runners.items():
            found = "on PATH" if shutil.which(spec["command"]) else "NOT on PATH"
            lines.append(f"- {name}: {spec['command']} {' '.join(spec['args'])} ({found})")
        sessions = self._registry.items()
        lines.append(f"Open sessions ({len(sessions)}/{MAX_SESSIONS}):")
        for session in sessions:
            state = "busy" if session.busy else ("idle" if session.alive else "exited")
            lines.append(
                f"- {session.session_id}: {self._registry.runner_of(session.session_id)} "
                f"in {session.cwd}, pid {session.pid}, {state}"
            )
        return "\n".join(lines)

    async def _start(self, runner: str, message: str, work_dir: str, timeout: int) -> str:
        if runner not in self._runners:
            raise ValueError(f"unknown runner {runner!r}; configured: {', '.join(self._runners)}")
        if len(self._registry) >= MAX_SESSIONS:
            raise ValueError(
                f"{MAX_SESSIONS} acp_agent sessions are already open; close one first "
                "(action='list' shows them)"
            )
        spec = self._runners[runner]
        cwd = self._cwd(work_dir)
        if not os.path.isdir(cwd):
            raise ValueError(f"work_dir is not a directory: {cwd}")

        async def decide(params: Dict[str, Any]) -> str:
            return await self._decide(runner, params)

        session = ACPSession(
            [spec["command"], *spec["args"]], cwd=cwd, env=spec.get("env"),
            decide_permission=decide,
        )
        try:
            # npx may download the adapter on first use; bounded by the same budget.
            await asyncio.wait_for(session.open(), timeout=timeout)
        except BaseException:
            session.close()
            raise
        self._registry.add(runner, session)
        return await self._prompt(session, message, timeout)

    async def _prompt(self, session: ACPSession, message: str, timeout: int) -> str:
        runner = self._registry.runner_of(session.session_id)
        try:
            result = await session.prompt(message, timeout=timeout)
        except ACPClientError:
            if not session.alive:
                # A dead runner cannot take another message; do not leave it listed.
                self._registry.close(session.session_id)
            raise
        return format_result(runner, session.session_id, result)

    async def acp_agent(
        self,
        action: str,
        runner: str = "",
        message: str = "",
        session_id: str = "",
        work_dir: str = "",
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> str:
        """Hands a coding task to another coding agent (Codex, Claude Code, OpenCode)
        and returns its report.

        The other agent works in its own process with its own model, tools and
        context: it reads, edits and runs commands in ``work_dir`` on its own.
        Use it for a self-contained task you want that agent to do — an
        implementation in another checkout, a second-opinion review, a fix in
        an area it knows better — not for a step you can do with your own tools.
        Write ``message`` as a complete brief: the agent sees nothing of this
        conversation. When it asks for a permission outside its own sandbox,
        the user decides on an approval card.

        Actions:
        - ``list``: configured runners and open sessions.
        - ``start``: open a session with ``runner`` in ``work_dir`` (default:
          this session's directory) and send ``message``. Returns the session
          id with the report.
        - ``message``: send a follow-up ``message`` to ``session_id``; the agent
          keeps its context from earlier messages.
        - ``close``: end ``session_id`` and stop its process. Close sessions
          you are done with; at most three stay open.

        The report holds the agent's reply, the tool calls it made, its plan,
        and how many permission requests were allowed or denied. Verify its
        changes (diff, tests) before building on them.

        Args:
            action: list | start | message | close
            runner: for start: a runner name from list (codex, claude_code, opencode, ...)
            message: for start / message: the task or follow-up, self-contained
            session_id: for message / close
            work_dir: for start: directory the agent works in
            timeout: seconds to wait for this message's answer (default 1800);
                on expiry the turn is cancelled and the session stays open

        Returns:
            str: the agent's report, the session list, or a confirmation
        """
        if action == "list":
            return self._list()
        if action == "start":
            if not message.strip():
                raise ValueError("start needs a message: the task for the agent")
            return await self._start(runner, message, work_dir, timeout)
        if action == "message":
            if not message.strip():
                raise ValueError("message needs a message")
            return await self._prompt(self._registry.get(session_id), message, timeout)
        if action == "close":
            self._registry.get(session_id)
            self._registry.close(session_id)
            return f"Closed acp_agent session {session_id}."
        raise ValueError(f"unknown action {action!r}: use list, start, message or close")
