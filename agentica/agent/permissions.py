# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Unified 3-tier tool permission model.

Shared by the SDK (Agent/DeepAgent), the CLI, and the Gateway so all three
surfaces expose the exact same vocabulary and behavior:

  - "ask"       : only read-only tools are exposed to the LLM (write_file,
                  edit_file, execute are hidden from the tool schema).
  - "auto"      : every tool is exposed. Writes (write_file/edit_file) are
                  restricted to the agent's work_dir via SandboxConfig; reads
                  outside work_dir are also blocked for sensitive path
                  components (.ssh, .env, etc. — see SandboxConfig.blocked_paths).
                  Neither is a dead end: the model can call
                  ``request_path_access(path, reason)`` to ask the user for a
                  one-time yes/no approval, which then whitelists that path for
                  the rest of the session (see
                  ``BuiltinFileTool.request_path_access``). ``execute`` (shell)
                  has NO path restriction here — a shell command can `cd`/
                  redirect anywhere, so true path scoping would require
                  OS-level sandboxing (Docker/seccomp), which is out of scope.
                  Only the existing SandboxConfig.blocked_commands safety net
                  (best-effort dangerous-command blocklist) applies.
  - "allow-all" : every tool is exposed, no sandbox path restriction. The CLI's
                  actual default (see agentica.cli.runtime.parse_args). Note
                  that a small set of always-sensitive write targets (/etc,
                  ~/.ssh, ~/.aws/credentials, etc.) are still refused by
                  default even in this mode — the model should never silently
                  touch credentials or system files. This too is escalatable
                  via ``request_path_access``, so the user always has the
                  final say instead of the tool being a hard dead end.

Callers should not construct their own mode strings — always compare against
``PERMISSION_MODES`` / use ``validate_permission_mode`` so a typo fails loud
instead of silently falling through to the most permissive behavior.
"""
from typing import List, Optional, Set

PERMISSION_MODES = ("ask", "auto", "allow-all")

# Tools that only read — always safe to expose in "ask" mode.
READ_ONLY_TOOLS: Set[str] = frozenset({
    "ls", "read_file", "glob", "grep", "web_search", "fetch_url",
    "write_todos", "task",
})


def validate_permission_mode(mode: str) -> None:
    """Raise ValueError if `mode` is not one of PERMISSION_MODES."""
    if mode not in PERMISSION_MODES:
        raise ValueError(f"Invalid permission mode: {mode!r}. Must be one of {PERMISSION_MODES}.")


def read_only_whitelist(mode: str) -> Optional[List[str]]:
    """Query-level tool whitelist for `mode`. None means no restriction."""
    return list(READ_ONLY_TOOLS) if mode == "ask" else None


def sandbox_should_be_enabled(mode: str) -> bool:
    """Whether write operations should be path-restricted to work_dir for `mode`."""
    return mode != "allow-all"
