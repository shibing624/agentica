# -*- coding: utf-8 -*-
"""Tool-call approval: registry, classifier, session grants, ApproveFn factory.

The runner calls ``Agent.approve`` once per ``tool_call_id`` after
``tool_call_started`` and before ``fc.execute()``. A missing registry (no
LiveTurn / no TUI) is an immediate deny — never a hang. See
``agentica.agent.permissions`` for the three-tier product copy.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Tuple

from agentica.tools.base import FunctionCall
from agentica.tools.safety import is_read_only_command, split_compound_command

ApprovalDecision = Literal["allow", "allow_prefix", "deny"]
ApprovalRoute = Literal["allow", "ask"]

DENIED_TOOL_RESULT = "Tool call denied by user."

FILE_TOOLS = frozenset({"read_file", "write_file", "apply_patch", "glob", "grep"})
EXECUTE_TOOLS = frozenset({"execute", "bash", "shell", "run_command"})
NETWORK_TOOLS = frozenset({"web_search", "fetch_url"})

_BENIGN_REDIRECT = re.compile(r"\d*>&\d*|&>\s*/dev/null|\d*>>?\s*/dev/null")


@dataclass(frozen=True)
class PendingApproval:
    """One parked tool call, keyed by ``tool_call_id``."""

    tool_call_id: str
    name: str
    arguments: Dict[str, Any]
    question: str
    preview: str
    options: Tuple[ApprovalDecision, ...] = ("allow", "allow_prefix", "deny")


@dataclass
class _PendingEntry:
    pending: PendingApproval
    future: asyncio.Future


class ApprovalRegistry:
    """Pending approvals for one LiveTurn (key = ``tool_call_id``)."""

    def __init__(self) -> None:
        self._pending: Dict[str, _PendingEntry] = {}

    @property
    def size(self) -> int:
        return len(self._pending)

    def list(self) -> List[PendingApproval]:
        return [entry.pending for entry in self._pending.values()]

    def wait(self, pending: PendingApproval) -> Awaitable[ApprovalDecision]:
        """Register and wait. Re-registering the same id resolves the old wait as deny."""
        existing = self._pending.get(pending.tool_call_id)
        if existing is not None and not existing.future.done():
            existing.future.set_result("deny")
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[pending.tool_call_id] = _PendingEntry(pending=pending, future=future)
        return future

    def decide(self, tool_call_id: str, decision: ApprovalDecision) -> bool:
        """Submit a decision. Returns False if the id is unknown or already decided."""
        entry = self._pending.pop(tool_call_id, None)
        if entry is None or entry.future.done():
            return False
        entry.future.set_result(decision)
        return True

    def deny_all(self) -> None:
        """Resolve every pending wait as deny (interrupt / cancel)."""
        for tool_call_id in list(self._pending):
            self.decide(tool_call_id, "deny")


@dataclass
class SessionGrants:
    """In-session allow-once / allow-similar memory. Not persisted."""

    path_exact: set = field(default_factory=set)
    path_prefixes: set = field(default_factory=set)
    command_prefixes: List[Tuple[str, ...]] = field(default_factory=list)
    network_keys: set = field(default_factory=set)
    network_tools: set = field(default_factory=set)
    tool_names: set = field(default_factory=set)

    def covers(self, fc: FunctionCall, *, work_dir: Optional[str]) -> bool:
        name = fc.function.name
        if name in self.tool_names:
            return True
        if name in FILE_TOOLS:
            paths = call_paths(fc, work_dir=work_dir)
            if not paths:
                return False
            return all(self._path_covered(p) for p in paths)
        if name in EXECUTE_TOOLS:
            command = _call_command(fc)
            return bool(command) and self._command_covered(command)
        if name in NETWORK_TOOLS:
            if name in self.network_tools:
                return True
            key = _network_key(fc)
            return key is not None and key in self.network_keys
        return False

    def add_path(self, resolved: str, *, prefix: bool) -> None:
        if prefix:
            self.path_prefixes.add(resolved)
        else:
            self.path_exact.add(resolved)

    def add_command_prefix(self, command: str) -> None:
        tokens = _first_segment_tokens(command)
        if tokens and tokens not in self.command_prefixes:
            self.command_prefixes.append(tokens)

    def add_network(self, fc: FunctionCall, *, prefix: bool) -> None:
        name = fc.function.name
        if prefix:
            self.network_tools.add(name)
            return
        key = _network_key(fc)
        if key is not None:
            self.network_keys.add(key)

    def add_tool_name(self, name: str) -> None:
        self.tool_names.add(name)

    def _path_covered(self, resolved: str) -> bool:
        if resolved in self.path_exact:
            return True
        for prefix in self.path_prefixes:
            if _path_is_under(resolved, prefix):
                return True
        return False

    def _command_covered(self, command: str) -> bool:
        tokens = _first_segment_tokens(command)
        if tokens is None:
            return False
        for prefix in self.command_prefixes:
            if len(tokens) >= len(prefix) and tokens[: len(prefix)] == prefix:
                return True
        return False


def classify(
    mode: str,
    fc: FunctionCall,
    grants: SessionGrants,
    *,
    work_dir: Optional[str],
) -> ApprovalRoute:
    """Return ``allow`` (run now) or ``ask`` (park for a human)."""
    if mode == "allow-all":
        return "allow"
    if grants.covers(fc, work_dir=work_dir):
        return "allow"

    name = fc.function.name
    if name in FILE_TOOLS:
        paths = call_paths(fc, work_dir=work_dir)
        if not paths:
            return "ask"
        if any(_file_needs_approval(p, work_dir=work_dir) for p in paths):
            return "ask"
        return "allow"

    if name in EXECUTE_TOOLS:
        if mode == "ask":
            return "ask"
        command = _call_command(fc) or ""
        ok, _reason = is_read_only_command(command)
        return "allow" if ok else "ask"

    if name in NETWORK_TOOLS:
        return "ask" if mode == "ask" else "allow"

    flagged = fc.function.is_read_only or fc.function.is_destructive
    if not flagged:
        return "ask" if mode == "ask" else "allow"
    return "allow"


def command_allows_prefix(command: str) -> bool:
    """Redirects, command substitution, and heredocs are allow-once only."""
    if "$(" in command or "`" in command or "<<" in command:
        return False
    stripped = _BENIGN_REDIRECT.sub("", command)
    return ">" not in stripped


def call_paths(fc: FunctionCall, *, work_dir: Optional[str]) -> List[str]:
    """Resolved filesystem paths this file-tool call would touch. Empty = unknown."""
    args = fc.arguments or {}
    name = fc.function.name
    raw: List[str] = []
    if name in ("read_file", "write_file"):
        path = args.get("file_path")
        if isinstance(path, str) and path:
            raw.append(path)
    elif name in ("glob", "grep"):
        path = args.get("path", ".")
        if isinstance(path, str) and path:
            raw.append(path)
        else:
            raw.append(".")
    elif name == "apply_patch":
        patch = args.get("patch") or ""
        if not isinstance(patch, str):
            return []
        try:
            from agentica.tools.patch_tool import parse_patch_envelope

            raw.extend(op.path for op in parse_patch_envelope(patch))
        except (ValueError, TypeError):
            return []
    return [_resolve_tool_path(p, work_dir) for p in raw]


def describe_approval(fc: FunctionCall) -> Tuple[str, str]:
    """``(question, preview)`` for an approval card. No extra LLM call."""
    name = fc.function.name
    args = fc.arguments or {}
    if name in EXECUTE_TOOLS:
        preview = str(args.get("command") or "")
        return "Allow running the following command?", preview
    if name in FILE_TOOLS:
        preview = "; ".join(call_paths(fc, work_dir=None) or [str(args)])
        return f"Allow this {name} call?", preview
    if name == "web_search":
        preview = str(args.get("queries") or "")
        return "Allow this web search?", preview
    if name == "fetch_url":
        preview = str(args.get("url") or "")
        return "Allow fetching this URL?", preview
    preview = json.dumps(args, ensure_ascii=False, default=str)
    return f"Allow this {name} call?", preview


def apply_path_grant_on_agent(agent: Any, path: str, *, prefix: bool) -> None:
    """Write a session path grant onto every BuiltinFileTool the agent holds."""
    from agentica.tools.builtin.file_tool import BuiltinFileTool

    for tool in agent.tools or []:
        if isinstance(tool, BuiltinFileTool):
            tool.grant_path_access(path, prefix=prefix)


def make_approve(
    *,
    get_mode: Callable[[], str],
    get_grants: Callable[[], SessionGrants],
    get_registry: Callable[[], Optional[ApprovalRegistry]],
    get_work_dir: Callable[[], Optional[str]],
    publish: Callable[[PendingApproval], None],
    apply_path_grant: Callable[[str, bool], None],
) -> Callable[[FunctionCall], Awaitable[ApprovalDecision]]:
    """Build the ``Agent.approve`` callback.

    ``get_registry`` returning None is an immediate deny on the manual path
    (no LiveTurn, IM, cron, non-streaming POST, SDK without a TUI).
    """

    async def approve(fc: FunctionCall) -> ApprovalDecision:
        mode = get_mode()
        grants = get_grants()
        work_dir = get_work_dir()
        if classify(mode, fc, grants, work_dir=work_dir) == "allow":
            return "allow"
        registry = get_registry()
        if registry is None:
            return "deny"
        question, preview = describe_approval(fc)
        pending = PendingApproval(
            tool_call_id=fc.call_id or "",
            name=fc.function.name,
            arguments=dict(fc.arguments or {}),
            question=question,
            preview=preview,
        )
        waiter = registry.wait(pending)
        publish(pending)
        decision = await waiter
        if decision not in ("allow", "allow_prefix", "deny"):
            return "deny"
        if decision == "deny":
            return "deny"
        _record_grant(fc, grants, work_dir=work_dir, prefix=decision == "allow_prefix")
        if fc.function.name in FILE_TOOLS:
            for path in call_paths(fc, work_dir=work_dir):
                sensitive = _is_sensitive_path(path)
                apply_path_grant(path, (decision == "allow_prefix") and not sensitive)
        return decision

    return approve


def _record_grant(
    fc: FunctionCall,
    grants: SessionGrants,
    *,
    work_dir: Optional[str],
    prefix: bool,
) -> None:
    name = fc.function.name
    if name in FILE_TOOLS:
        for path in call_paths(fc, work_dir=work_dir):
            sensitive = _is_sensitive_path(path)
            granted = path
            if prefix and not sensitive:
                granted = path if Path(path).is_dir() else str(Path(path).parent)
            grants.add_path(granted, prefix=prefix and not sensitive)
        return
    if name in EXECUTE_TOOLS:
        command = _call_command(fc) or ""
        if prefix and command_allows_prefix(command):
            grants.add_command_prefix(command)
        return
    if name in NETWORK_TOOLS:
        grants.add_network(fc, prefix=prefix)
        return
    if prefix:
        grants.add_tool_name(name)


def _file_needs_approval(resolved: str, *, work_dir: Optional[str]) -> bool:
    if _is_sensitive_path(resolved):
        return True
    if not work_dir:
        return True
    return not _path_is_under(resolved, _resolve_tool_path(work_dir, None))


def _is_sensitive_path(filepath: str) -> bool:
    from agentica.tools.builtin.file_tool import is_sensitive_write_path

    return is_sensitive_write_path(filepath)


def _resolve_tool_path(path: str, work_dir: Optional[str]) -> str:
    expanded = Path(path).expanduser()
    if not expanded.is_absolute():
        root = Path(work_dir).expanduser() if work_dir else Path.cwd()
        expanded = root / expanded
    try:
        return str(expanded.resolve())
    except (OSError, ValueError):
        return str(expanded)


def _path_is_under(resolved: str, root: str) -> bool:
    if resolved == root:
        return True
    prefix = root.rstrip("/\\")
    return resolved.startswith(prefix + "/") or resolved.startswith(prefix + os.sep)


def _call_command(fc: FunctionCall) -> Optional[str]:
    command = (fc.arguments or {}).get("command")
    return command if isinstance(command, str) else None


def _network_key(fc: FunctionCall) -> Optional[str]:
    args = fc.arguments or {}
    if fc.function.name == "web_search":
        queries = args.get("queries")
        if queries is None:
            return None
        return json.dumps(queries, ensure_ascii=False, sort_keys=True, default=str)
    if fc.function.name == "fetch_url":
        url = args.get("url")
        return url if isinstance(url, str) else None
    return None


def _first_segment_tokens(command: str) -> Optional[Tuple[str, ...]]:
    segments = split_compound_command(command) or [command]
    if not segments:
        return None
    try:
        tokens = tuple(shlex.split(segments[0]))
    except ValueError:
        return None
    return tokens or None
