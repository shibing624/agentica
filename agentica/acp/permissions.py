# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: IDE tool-permission policy for the ACP adapter.

When Agentica runs as an ACP agent inside an IDE, the editor (and user) should
control what the agent is allowed to do — especially file writes and shell
execution. This module provides a small, declarative permission policy that
classifies each tool call as allow / deny / ask.

    policy = ToolPermissionPolicy(mode=PermissionMode.CONFIRM_WRITES)
    policy.decide("read_file")    # ALLOW (read-only)
    policy.decide("write_file")   # ASK   (mutating)
    policy.decide("execute")      # ASK

Modes:
    AUTO           - allow everything (default; backward compatible)
    READ_ONLY      - allow read-only tools, deny mutating ones
    CONFIRM_WRITES - allow read-only tools, ASK for mutating ones
    DENY_ALL       - deny everything

``allow_tools`` / ``deny_tools`` are explicit overrides: deny wins over
everything, then allow, then the mode rule. Unknown tools are treated as
mutating (conservative).
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Set


class PermissionDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class PermissionMode(str, Enum):
    AUTO = "auto"
    READ_ONLY = "read_only"
    CONFIRM_WRITES = "confirm_writes"
    DENY_ALL = "deny_all"


# Default classification of the builtin ACP tools.
DEFAULT_READ_ONLY_TOOLS = frozenset({
    "read_file", "ls", "glob", "grep", "web_search", "fetch_url",
})
DEFAULT_WRITE_TOOLS = frozenset({
    "write_file", "edit_file", "multi_edit_file", "execute",
})


@dataclass
class ToolPermissionPolicy:
    """Decide whether a tool call is allowed in an ACP session."""
    mode: PermissionMode = PermissionMode.AUTO
    allow_tools: Set[str] = field(default_factory=set)
    deny_tools: Set[str] = field(default_factory=set)
    read_only_tools: Set[str] = field(default_factory=lambda: set(DEFAULT_READ_ONLY_TOOLS))
    write_tools: Set[str] = field(default_factory=lambda: set(DEFAULT_WRITE_TOOLS))

    def is_read_only(self, tool_name: str) -> bool:
        """Read-only iff explicitly listed read-only and not listed as write.

        Unknown tools default to mutating (False) so the safe modes err on the
        side of caution.
        """
        if tool_name in self.read_only_tools and tool_name not in self.write_tools:
            return True
        return False

    def decide(self, tool_name: str, arguments: Optional[dict] = None) -> PermissionDecision:
        """Return the permission decision for a tool call."""
        if tool_name in self.deny_tools:
            return PermissionDecision.DENY
        if tool_name in self.allow_tools:
            return PermissionDecision.ALLOW

        if self.mode == PermissionMode.AUTO:
            return PermissionDecision.ALLOW
        if self.mode == PermissionMode.DENY_ALL:
            return PermissionDecision.DENY

        read_only = self.is_read_only(tool_name)
        if self.mode == PermissionMode.READ_ONLY:
            return PermissionDecision.ALLOW if read_only else PermissionDecision.DENY
        if self.mode == PermissionMode.CONFIRM_WRITES:
            return PermissionDecision.ALLOW if read_only else PermissionDecision.ASK

        return PermissionDecision.ALLOW
