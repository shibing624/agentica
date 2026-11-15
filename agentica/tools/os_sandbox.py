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

Nothing here is installed or bundled: ``sandbox-exec`` ships with macOS and
``bwrap`` is a system package, so availability is a **functional** probe (apply
a real read-only profile once per process), not a ``which`` check. A host where
the runner exists but cannot apply a profile — most commonly because this very
process already runs inside a sandbox and macOS refuses to nest one — reports
unavailable and the sandboxed agent refuses to start rather than silently
running commands unconfined.
"""
import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

# Package caches a build writes into even offline. Only existing ones are added.
_CACHE_DIRS = ("~/.cache", "~/Library/Caches", "~/.npm")
_GIT_PROBE_TIMEOUT_SECONDS = 5
_PROBE_TIMEOUT_SECONDS = 5
# Characters with a meaning in SBPL (POSIX extended) regex.
_REGEX_SPECIAL = set(".^$*+?()[]{}|\\")


@dataclass(frozen=True)
class SandboxPolicy:
    """What one sandboxed command may do. Paths are real (symlink-free) paths."""

    writable_roots: Tuple[str, ...]
    read_only_paths: Tuple[str, ...]
    blocked_components: Tuple[str, ...]
    network_access: bool


@lru_cache(maxsize=1)
def _probe_sandbox() -> Optional[str]:
    """Run one real read-only profile; cached for the process lifetime."""
    if sys.platform == "darwin":
        binary = shutil.which("sandbox-exec")
        if binary is None:
            return "sandbox-exec is not on PATH"
    elif sys.platform.startswith("linux"):
        binary = shutil.which("bwrap")
        if binary is None:
            return "bubblewrap (bwrap) is not installed — e.g. `apt install bubblewrap`"
    else:
        return f"no OS sandbox is implemented for platform {sys.platform!r}"
    policy = SandboxPolicy(
        writable_roots=(), read_only_paths=(), blocked_components=(), network_access=False,
    )
    try:
        probe = subprocess.run(
            wrap_command("true", policy),
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return f"cannot start the sandbox runner {binary!r}: {e}"
    if probe.returncode == 0:
        return None
    detail = [line for line in (probe.stderr or probe.stdout or "").splitlines() if line.strip()]
    tail = f" ({detail[-1].strip()[:160]})" if detail else ""
    return (
        f"the OS sandbox cannot be applied on this host{tail} — a process that is "
        "itself already sandboxed cannot nest another sandbox"
    )


def unsupported_reason() -> Optional[str]:
    """Why the OS sandbox cannot run on this machine, or None when it can.

    A functional probe, not ``which``: the runner may be installed and still be
    unable to apply a profile (nested sandbox, locked-down kernel). Cached, so
    constructing one sandboxed agent per turn costs one probe per process.
    """
    return _probe_sandbox()


def reset_availability_probe() -> None:
    """Forget the cached probe verdict (tests that fake the host)."""
    _probe_sandbox.cache_clear()


def _real(path: str) -> str:
    return os.path.realpath(os.path.expanduser(path))


def writable_temp_roots() -> Tuple[str, ...]:
    """Real paths of the platform temp areas a workspace-write sandbox keeps writable.

    dsh keeps the temp areas writable in workspace-write mode: a scratch script,
    a test artifact or an editor's backup file does not need an escalation for
    it. The conventional ``/tmp`` is added alongside ``tempfile.gettempdir()``
    because the two differ on macOS (per-user ``/var/folders/.../T``) and a
    program may set ``TMPDIR`` while still writing to ``/tmp``. Only existing
    directories: neither backend can grant a path that is not there.

    Shared by both layers on purpose. ``execute`` already gets these roots via
    :func:`resolve_policy_paths`; the file tools must agree with it, or writing
    a scratch file is refused on a permission mode that lets the same command
    write it through the shell.
    """
    roots: List[str] = []
    home = _real("~")
    for candidate in (tempfile.gettempdir(), "/tmp"):
        path = _real(candidate)
        # ``TMPDIR`` is the user's environment, not a policy. Set to ``/`` or
        # to the home directory (or anything above it), it would make every
        # write there count as scratch — including ``~/.ssh``.
        if path == os.sep or home == path or home.startswith(path.rstrip(os.sep) + os.sep):
            continue
        if os.path.isdir(path) and path not in roots:
            roots.append(path)
    return tuple(roots)


def _git_output(root: str, *args: str) -> Optional[str]:
    """stdout of ``git -C root args``, or ``None`` when git fails or is missing."""
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "-C", root, *args],
            capture_output=True, text=True, timeout=_GIT_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _inside_a_repository(root: str) -> bool:
    """Is there a ``.git`` entry at ``root`` or above? A few stats, no process.

    A file counts too: a linked worktree and ``--separate-git-dir`` both
    leave a ``.git`` file pointing at the real git dir.
    """
    path = _real(root)
    while True:
        if os.path.lexists(os.path.join(path, ".git")):
            return True
        parent = os.path.dirname(path)
        if parent == path:
            return False
        path = parent


def _mtime_ns(path: str) -> Optional[int]:
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


_Token = Tuple[Optional[int], Tuple[Tuple[str, Optional[int]], ...]]
# root -> (git common dir, its stat token, (writable, read_only)).
_REPOSITORY_CACHE: Dict[str, Tuple[str, _Token, Tuple[Tuple[str, ...], Tuple[str, ...]]]] = {}


def _repository_token(common_dir: str) -> _Token:
    """Changes whenever the set of worktrees or where they live changes.

    ``git worktree add`` / ``remove`` / ``prune`` create or delete an entry in
    ``<common>/worktrees`` (the first add creates that directory, which
    changes ``<common>``); ``git worktree move`` rewrites the entry's
    ``gitdir`` file. A few stats instead of a git process per write.
    """
    entries_dir = os.path.join(common_dir, "worktrees")
    try:
        names = sorted(os.listdir(entries_dir))
    except OSError:
        names = []
    return _mtime_ns(common_dir), tuple(
        (name, _mtime_ns(os.path.join(entries_dir, name, "gitdir"))) for name in names
    )


def repository_paths(root: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """``(writable, read_only)`` real paths for the git repository ``root`` is in.

    Writable: the shared git dir and every worktree. Read-only: the hooks dir
    and config of that git dir — a write there becomes code the user later
    runs outside any sandbox (``git commit`` runs hooks as the user). Both
    empty when ``root`` is no repository.

    The one parser both layers use: ``execute``'s OS sandbox through
    :func:`resolve_policy_paths`, the file tools and the auto-tier approval
    directly. Cached per root and revalidated with two stats, so a write
    spawns git only after a worktree was added or removed.
    """
    key = _real(root)
    cached = _REPOSITORY_CACHE.get(key)
    if cached is not None and _repository_token(cached[0]) == cached[1]:
        return cached[2]
    if not _inside_a_repository(key):
        return (), ()
    common = (_git_output(key, "rev-parse", "--path-format=absolute", "--git-common-dir") or "").strip()
    listing = _git_output(key, "worktree", "list", "--porcelain", "-z")
    if not common or listing is None:
        return (), ()
    common = _real(common)
    # ``-z``: every attribute line ends in NUL, so a path with a newline or
    # quotes arrives verbatim instead of C-quoted.
    trees = [_real(field[len("worktree "):]) for field in listing.split("\0") if field.startswith("worktree ")]
    writable = tuple(dict.fromkeys([common, *trees]))
    read_only = tuple(os.path.join(common, name) for name in ("hooks", "config"))
    paths = (writable, read_only)
    _REPOSITORY_CACHE[key] = (common, _repository_token(common), paths)
    return paths


def cache_roots() -> Tuple[str, ...]:
    """Existing package caches a build writes into even offline."""
    return tuple(path for path in (_real(p) for p in _CACHE_DIRS) if os.path.isdir(path))


async def resolve_policy_paths(dirs: Iterable[str]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """``(writable_roots, read_only_paths)`` as real paths, deduplicated, stable order.

    ``dirs`` that do not exist are dropped: neither backend can grant a path
    that is not there, and a root created later is created by an escalated
    command anyway. Read-only paths are kept even when missing: Seatbelt then
    stops a command from creating them.
    """
    base = [_real(d) for d in dirs if d]
    repos = await asyncio.gather(*(asyncio.to_thread(repository_paths, d) for d in base if os.path.isdir(d)))
    candidates = list(base)
    read_only: List[str] = []
    for writable, protected in repos:
        candidates.extend(_real(p) for p in writable)
        read_only.extend(_real(p) for p in protected if _real(p) not in read_only)
    candidates.extend(writable_temp_roots())
    candidates.extend(cache_roots())
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
