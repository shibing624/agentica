# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: OS-level sandbox for shell commands (macOS Seatbelt, Linux bubblewrap).

This is the real boundary behind ``SandboxConfig.os_sandbox``, the same shape
Codex CLI uses for ``--sandbox workspace-write``: the kernel, not a regex,
decides what a command may touch.

- Reads: everywhere, except paths that contain a ``SandboxConfig.blocked_paths``
  component (``.ssh``, ``.aws``, ``.env``, ...). On Linux bubblewrap cannot
  match components, so only ``$HOME/<component>`` is hidden there.
- Writes: only the writable roots — ``writable_dirs``, the session work_dir,
  every worktree of the git repositories those live in (plus the shared
  ``.git`` dir, or ``git commit`` in a worktree fails), the temp dirs and a
  few package caches. Inside them ``.git/hooks`` and ``.git/config`` stay
  read-only: both decide what *your own* later, unsandboxed ``git`` runs
  (hooks, ``core.hooksPath``, ``core.fsmonitor``), so writing them is a way
  out of the sandbox that needs no escalation.
- Network: off unless ``network_access``; loopback stays reachable on macOS
  so a test against a local dev server still works.

A command that needs more runs again with ``execute(escalate=True)``, which
the approval layer always routes to a human (see ``agentica.agent.approvals``).
"""
import asyncio
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

# Package caches a build writes into even offline. Only existing ones are added.
_CACHE_DIRS = ("~/.cache", "~/Library/Caches", "~/.npm")
_GIT_PROBE_TIMEOUT_SECONDS = 5
# Characters with a meaning in SBPL (POSIX extended) regex.
_REGEX_SPECIAL = set(".^$*+?()[]{}|\\")


@dataclass(frozen=True)
class SandboxPolicy:
    """What one sandboxed command may do. Paths are real (symlink-free) paths."""

    writable_roots: Tuple[str, ...]
    read_only_paths: Tuple[str, ...]
    blocked_components: Tuple[str, ...]
    network_access: bool


def unsupported_reason() -> Optional[str]:
    """Why the OS sandbox cannot run on this machine, or None when it can."""
    if sys.platform == "darwin":
        if shutil.which("sandbox-exec") is None:
            return "sandbox-exec is not on PATH"
        return None
    if sys.platform.startswith("linux"):
        if shutil.which("bwrap") is None:
            return "bubblewrap (bwrap) is not installed — e.g. `apt install bubblewrap`"
        return None
    return f"no OS sandbox is implemented for platform {sys.platform!r}"


def _real(path: str) -> str:
    return os.path.realpath(os.path.expanduser(path))


async def _git_lines(root: str, *args: str) -> List[str]:
    """stdout lines of ``git -C root <args>``; empty when ``root`` is no repository."""
    proc = await asyncio.create_subprocess_exec(
        "git", "-C", root, *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=_GIT_PROBE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return []
    if proc.returncode != 0:
        return []
    return out.decode("utf-8", errors="replace").splitlines()


async def _repository_paths(root: str) -> Tuple[List[str], List[str]]:
    """``(writable, read_only)`` for the git repository ``root`` is in.

    Writable: the shared git dir and every worktree. Read-only: the hooks dir
    and config of that git dir. Both empty when ``root`` is no repository.
    """
    common, worktrees = await asyncio.gather(
        _git_lines(root, "rev-parse", "--path-format=absolute", "--git-common-dir"),
        _git_lines(root, "worktree", "list", "--porcelain"),
    )
    git_dirs = [line for line in common if line]
    writable = list(git_dirs)
    writable.extend(line[len("worktree "):] for line in worktrees if line.startswith("worktree "))
    read_only = [os.path.join(d, name) for d in git_dirs for name in ("hooks", "config")]
    return writable, read_only


async def resolve_policy_paths(dirs: Iterable[str]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """``(writable_roots, read_only_paths)`` as real paths, deduplicated, stable order.

    ``dirs`` that do not exist are dropped: neither backend can grant a path
    that is not there, and a root created later is created by an escalated
    command anyway. Read-only paths are kept even when missing: Seatbelt then
    stops a command from creating them.
    """
    base = [_real(d) for d in dirs if d]
    repos = await asyncio.gather(*(_repository_paths(d) for d in base if os.path.isdir(d)))
    candidates = list(base)
    read_only: List[str] = []
    for writable, protected in repos:
        candidates.extend(_real(p) for p in writable)
        read_only.extend(_real(p) for p in protected if _real(p) not in read_only)
    candidates.extend(_real(p) for p in (tempfile.gettempdir(), "/tmp"))
    candidates.extend(_real(p) for p in _CACHE_DIRS)
    roots: List[str] = []
    for path in candidates:
        if os.path.exists(path) and path not in roots:
            roots.append(path)
    return tuple(roots), tuple(read_only)


def _sbpl_regex(component: str) -> str:
    escaped = "".join("\\" + ch if ch in _REGEX_SPECIAL else ch for ch in component)
    return f'#"(^|/){escaped}(/|$)"'


def seatbelt_profile(policy: SandboxPolicy) -> str:
    """The SBPL profile. Paths arrive as ``-D WRITABLE_ROOT_<i>`` / ``READ_ONLY_<i>``.

    In SBPL the last matching rule wins, so each ``allow`` narrows the
    ``deny`` above it, and the read-only paths and blocked components come
    last to beat both.
    """
    lines = ["(version 1)", "(allow default)", "(deny file-write*)", "(allow file-write*"]
    lines.extend(
        f'  (subpath (param "WRITABLE_ROOT_{i}"))' for i in range(len(policy.writable_roots))
    )
    lines.append(
        '  (literal "/dev/null") (literal "/dev/zero") (literal "/dev/stdout")'
        ' (literal "/dev/stderr") (literal "/dev/dtracehelper")'
        ' (regex #"^/dev/tty") (regex #"^/dev/fd/"))'
    )
    lines.extend(
        f'(deny file-write* (subpath (param "READ_ONLY_{i}")))'
        for i in range(len(policy.read_only_paths))
    )
    for component in policy.blocked_components:
        regex = _sbpl_regex(component)
        lines.append(f"(deny file-read-data (regex {regex}))")
        lines.append(f"(deny file-write* (regex {regex}))")
    if not policy.network_access:
        lines.append("(deny network-outbound (remote ip))")
        lines.append('(allow network-outbound (remote ip "localhost:*"))')
    return "\n".join(lines) + "\n"


def _seatbelt_argv(command: str, policy: SandboxPolicy) -> List[str]:
    argv = ["sandbox-exec", "-p", seatbelt_profile(policy)]
    for i, root in enumerate(policy.writable_roots):
        argv.extend(["-D", f"WRITABLE_ROOT_{i}={root}"])
    for i, path in enumerate(policy.read_only_paths):
        argv.extend(["-D", f"READ_ONLY_{i}={path}"])
    argv.extend(["/bin/sh", "-c", command])
    return argv


def _bwrap_argv(command: str, policy: SandboxPolicy) -> List[str]:
    argv = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc"]
    for root in policy.writable_roots:
        argv.extend(["--bind", root, root])
    # A bind needs an existing source, so a missing hooks dir stays creatable here.
    for path in policy.read_only_paths:
        if os.path.exists(path):
            argv.extend(["--ro-bind", path, path])
    home = os.path.expanduser("~")
    for component in policy.blocked_components:
        hidden = os.path.join(home, component)
        if os.path.isdir(hidden):
            argv.extend(["--tmpfs", hidden])
        elif os.path.exists(hidden):
            argv.extend(["--ro-bind", "/dev/null", hidden])
    if not policy.network_access:
        argv.append("--unshare-net")
    argv.extend(["--die-with-parent", "--", "/bin/sh", "-c", command])
    return argv


def wrap_command(command: str, policy: SandboxPolicy, platform: str = sys.platform) -> List[str]:
    """argv that runs ``command`` through ``/bin/sh -c`` inside the sandbox."""
    if platform == "darwin":
        return _seatbelt_argv(command, policy)
    if platform.startswith("linux"):
        return _bwrap_argv(command, policy)
    raise RuntimeError(f"no OS sandbox is implemented for platform {platform!r}")


def describe_policy(policy: SandboxPolicy, cwd: Optional[str]) -> str:
    """One line for the model after a sandboxed command failed."""
    if policy.network_access:
        network = "on"
    else:
        network = "off (localhost only)" if sys.platform == "darwin" else "off"
    in_root = cwd is not None and any(
        _real(cwd) == r or _real(cwd).startswith(r + os.sep) for r in policy.writable_roots
    )
    where = "" if in_root else f" This call's work_dir {cwd} is read-only."
    return (
        f"[sandbox: this command ran with writes limited to {', '.join(policy.writable_roots)}; "
        f"network {network}; reads of {', '.join(policy.blocked_components) or 'nothing'} "
        f"blocked.{where} If it failed because of that (typically 'Operation not "
        f"permitted', 'Read-only file system', or a connection error), call execute "
        f"again with the same command and escalate=true — that runs it outside the "
        f"sandbox and asks the user first. Do not work around the sandbox another way.]"
    )
