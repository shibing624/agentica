# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Ordered live tool blocks with Kimi-style prefix flush

Unfinished blocks stay in the prompt_toolkit live window. Completions
mark a block finished; only a leading run of finished blocks is flushed
to scrollback, so a later-finishing sibling cannot split an earlier
call from its result.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional


@dataclass
class LiveToolResult:
    """Payload captured at TOOL_COMPLETED, rendered only at prefix flush."""

    content: str
    is_error: bool = False
    elapsed: Optional[float] = None
    tool_args: Optional[dict] = None
    tool_display_meta: Optional[dict] = None


@dataclass
class LiveToolBlock:
    """One in-flight tool call, keyed by ``tool_call_id``."""

    tool_call_id: str
    tool_name: str
    tool_args: dict
    finished: bool = False
    result: Optional[LiveToolResult] = None
    sub_lines: List[str] = field(default_factory=list)


class LiveToolStore:
    """Insertion-ordered live tool blocks with prefix flush.

    ``drain_prefix()`` pops finished blocks from the front and stops at the
    first unfinished one — the same rule as kimi-cli's
    ``flush_finished_tool_calls``.
    """

    def __init__(self) -> None:
        self._blocks: "OrderedDict[str, LiveToolBlock]" = OrderedDict()
        self._run_to_parent: Dict[str, str] = {}

    def __contains__(self, tool_call_id: object) -> bool:
        return tool_call_id in self._blocks

    def __len__(self) -> int:
        return len(self._blocks)

    def get(self, tool_call_id: str) -> Optional[LiveToolBlock]:
        return self._blocks.get(tool_call_id)

    def blocks(self) -> Iterator[LiveToolBlock]:
        return iter(self._blocks.values())

    def start(self, tool_call_id: str, tool_name: str, tool_args: dict) -> LiveToolBlock:
        block = LiveToolBlock(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_args=dict(tool_args) if tool_args else {},
        )
        self._blocks[tool_call_id] = block
        return block

    def finish(self, tool_call_id: str, result: LiveToolResult) -> Optional[LiveToolBlock]:
        block = self._blocks.get(tool_call_id)
        if block is None or block.finished:
            return block
        block.finished = True
        block.result = result
        return block

    def drain_prefix(self) -> List[LiveToolBlock]:
        """Pop every finished block from the front; stop at the first unfinished."""
        flushed: List[LiveToolBlock] = []
        for key in list(self._blocks.keys()):
            block = self._blocks[key]
            if not block.finished:
                break
            flushed.append(self._blocks.pop(key))
        return flushed

    def drain_all(self) -> List[LiveToolBlock]:
        """Pop every remaining block, finished or not (turn end / interrupt)."""
        flushed = list(self._blocks.values())
        self._blocks.clear()
        return flushed

    def find_unfinished(self, tool_name: str) -> Optional[str]:
        """Match a result that omitted ``tool_call_id`` to one live block.

        Ambiguous (two unfinished calls of the same name) returns None so
        the caller falls back to immediate rendering instead of guessing.
        """
        found: Optional[str] = None
        for block in self._blocks.values():
            if block.tool_name == tool_name and not block.finished:
                if found is not None:
                    return None
                found = block.tool_call_id
        return found

    def bind_run(self, run_id: str) -> Optional[str]:
        """Attach a subagent ``run_id`` to the next unbound live ``task`` block."""
        if not run_id:
            return None
        existing = self._run_to_parent.get(run_id)
        if existing is not None and existing in self._blocks:
            return existing
        bound = set(self._run_to_parent.values())
        for block in self._blocks.values():
            if block.tool_name == "task" and block.tool_call_id not in bound:
                self._run_to_parent[run_id] = block.tool_call_id
                return block.tool_call_id
        return None

    def parent_for_run(self, run_id: Optional[str]) -> Optional[str]:
        if not run_id:
            return None
        parent = self._run_to_parent.get(run_id)
        if parent is not None and parent in self._blocks:
            return parent
        return None

    def attach_sub_line(self, parent_id: str, line: str) -> None:
        block = self._blocks.get(parent_id)
        if block is None:
            return
        block.sub_lines.append(line)
