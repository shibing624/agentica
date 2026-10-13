# -*- coding: utf-8 -*-
"""OS sandbox for ``execute`` (``SandboxConfig.os_sandbox``).

The profile / argv tests run everywhere. The tests that run commands through
the real kernel sandbox need macOS ``sandbox-exec`` and are skipped elsewhere
(bubblewrap is not on CI runners, and unprivileged user namespaces often are
not either).
"""
import asyncio
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from agentica.agent.approvals import (
    SessionGrants,
    _approval_options,
    approved_by_user,
    classify,
    describe_approval,
    is_sandbox_escalation,
)
from agentica.agent.config import SandboxConfig
from agentica.tools.background_processes import BackgroundProcessRegistry
from agentica.tools.base import Function, FunctionCall
from agentica.tools.builtin.execute_tool import BuiltinExecuteTool
from agentica.tools.os_sandbox import (
    SandboxPolicy,
    resolve_policy_paths,
    seatbelt_profile,
    wrap_command,
)


def _seatbelt_works() -> bool:
    if sys.platform != "darwin" or shutil.which("sandbox-exec") is None:
        return False
    # Already inside a sandbox (another agent running these tests) -> nesting fails.
    probe = subprocess.run(
        ["sandbox-exec", "-p", "(version 1)(allow default)", "/usr/bin/true"],
        capture_output=True,
    )
    return probe.returncode == 0


needs_seatbelt = pytest.mark.skipif(not _seatbelt_works(), reason="requires macOS sandbox-exec")
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="requires git")


def _policy(roots=("/w",), components=(".ssh",), network=False, read_only=()) -> SandboxPolicy:
    return SandboxPolicy(
        writable_roots=tuple(roots),
        read_only_paths=tuple(read_only),
        blocked_components=tuple(components),
        network_access=network,
    )


def _fc(arguments, name="execute") -> FunctionCall:
    fn = Function(name=name)
    fn.entrypoint = lambda **kwargs: "ran"
    return FunctionCall(function=fn, arguments=arguments, call_id="c1")


# --- profile / argv -------------------------------------------------------


def test_seatbelt_profile_orders_rules_so_the_narrow_ones_win():
    profile = seatbelt_profile(
        _policy(roots=("/a", "/b"), components=(".config/gcloud",), read_only=("/a/.git/hooks",))
    )
    lines = profile.splitlines()
    deny_all = lines.index("(deny file-write*)")
    allow_roots = lines.index("(allow file-write*")
    # SBPL: last matching rule wins, so allow must follow deny, components follow both.
    assert deny_all < allow_roots
    assert '(subpath (param "WRITABLE_ROOT_0"))' in profile
    assert '(subpath (param "WRITABLE_ROOT_1"))' in profile
    read_deny = next(i for i, line in enumerate(lines) if line.startswith("(deny file-read-data"))
    assert read_deny > allow_roots
    read_only = lines.index('(deny file-write* (subpath (param "READ_ONLY_0")))')
    assert read_only > allow_roots
    # '.' is escaped, '/' is not: a component with a slash still matches as a path.
    assert r'#"(^|/)\.config/gcloud(/|$)"' in profile


def test_seatbelt_network_rules_follow_network_access():
    off = seatbelt_profile(_policy(network=False))
    on = seatbelt_profile(_policy(network=True))
    assert "(deny network-outbound (remote ip))" in off
    assert '(allow network-outbound (remote ip "localhost:*"))' in off
    assert "network-outbound" not in on


def test_seatbelt_argv_passes_roots_as_parameters_and_command_verbatim():
    command = "echo 'a b' && printf \"%s\" $HOME"
    policy = _policy(roots=("/r0", "/r1"), read_only=("/r0/.git/config",))
    argv = wrap_command(command, policy, platform="darwin")
    assert argv[0] == "sandbox-exec"
    assert argv[argv.index("-p") + 1] == seatbelt_profile(policy)
    assert "WRITABLE_ROOT_0=/r0" in argv and "WRITABLE_ROOT_1=/r1" in argv
    assert "READ_ONLY_0=/r0/.git/config" in argv
    assert argv[-3:] == ["/bin/sh", "-c", command]


def test_bwrap_argv(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".ssh").mkdir()
    (tmp_path / ".netrc").write_text("x")
    hooks = tmp_path / "repo-hooks"
    hooks.mkdir()
    argv = wrap_command(
        "make",
        _policy(
            roots=("/w",),
            components=(".ssh", ".netrc", ".aws"),
            read_only=(str(hooks), str(tmp_path / "missing-config")),
        ),
        platform="linux",
    )
    assert argv[:4] == ["bwrap", "--ro-bind", "/", "/"]
    bind = argv.index("--bind")
    assert argv[bind : bind + 3] == ["--bind", "/w", "/w"]
    # read-only re-bind comes after the writable bind it narrows; a missing source is skipped
    assert argv.index(str(hooks)) > bind
    assert str(tmp_path / "missing-config") not in argv
    assert ["--tmpfs", str(tmp_path / ".ssh")] == argv[argv.index("--tmpfs") : argv.index("--tmpfs") + 2]
    assert str(tmp_path / ".netrc") in argv
    assert str(tmp_path / ".aws") not in argv  # does not exist -> nothing to hide
    assert "--unshare-net" in argv
    assert argv[-4:] == ["--", "/bin/sh", "-c", "make"]
    assert "--unshare-net" not in wrap_command("make", _policy(network=True), platform="linux")


def test_unsupported_platform_raises():
    with pytest.raises(RuntimeError, match="no OS sandbox"):
        wrap_command("ls", _policy(), platform="win32")


# --- writable roots -------------------------------------------------------


@needs_git
def test_writable_roots_cover_every_worktree_and_the_shared_git_dir(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(git + ["init", "-q", str(repo)], check=True)
    subprocess.run(git + ["-C", str(repo), "commit", "-q", "--allow-empty", "-m", "i"], check=True)
    sibling = tmp_path / "sibling-wt"
    subprocess.run(git + ["-C", str(repo), "worktree", "add", "-q", str(sibling)], check=True)

    roots, read_only = asyncio.run(
        resolve_policy_paths([str(sibling), str(tmp_path / "missing")])
    )

    real = os.path.realpath
    assert real(str(sibling)) in roots
    assert real(str(repo)) in roots  # the main checkout, reached through the worktree
    assert real(str(repo / ".git")) in roots  # `git commit` in the worktree writes here
    assert real(str(tmp_path / "missing")) not in roots
    assert len(roots) == len(set(roots))
    assert read_only == (real(str(repo / ".git" / "hooks")), real(str(repo / ".git" / "config")))


def test_writable_roots_for_a_plain_directory(tmp_path):
    roots, read_only = asyncio.run(resolve_policy_paths([str(tmp_path)]))
    assert roots[0] == os.path.realpath(str(tmp_path))
    assert read_only == ()
    assert os.path.realpath("/tmp") in roots


# --- approval routing -----------------------------------------------------


@pytest.mark.parametrize("mode", ["ask", "auto"])
def test_escalation_always_asks_in_ask_and_auto(mode, tmp_path):
    grants = SessionGrants()
    grants.add_command_prefix("git push origin")  # an earlier "allow similar"
    for command in ("ls", "git push origin main"):
        fc = _fc({"command": command, "escalate": True})
        assert classify(mode, fc, grants, work_dir=str(tmp_path)) == "ask"
    # Same calls without escalate keep their normal routes.
    assert classify(mode, _fc({"command": "ls"}), grants, work_dir=str(tmp_path)) == "allow"
    assert classify(
        mode, _fc({"command": "git push origin main", "escalate": False}), grants, work_dir=str(tmp_path)
    ) == "allow"


def test_escalation_in_allow_all_runs():
    fc = _fc({"command": "git push", "escalate": True})
    assert classify("allow-all", fc, SessionGrants(), work_dir=None) == "allow"


def test_escalation_predicate_errs_toward_asking():
    assert is_sandbox_escalation(_fc({"command": "x", "escalate": "yes"}))
    assert is_sandbox_escalation(_fc({"command": "x", "escalate": 1}))
    assert not is_sandbox_escalation(_fc({"command": "x"}))
    assert not is_sandbox_escalation(_fc({"command": "x", "escalate": False}))
    assert not is_sandbox_escalation(_fc({"escalate": True}, name="write_file"))


def test_escalation_card_is_allow_or_deny_and_says_outside():
    fc = _fc({"command": "git push", "escalate": True})
    question, preview = describe_approval(fc)
    assert "OUTSIDE the sandbox" in question
    assert preview == "git push"
    assert _approval_options(fc) == ("allow", "deny")


# --- execute tool ---------------------------------------------------------


def test_escalate_parameter_exists_only_with_the_os_sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr("agentica.tools.builtin.execute_tool.unsupported_reason", lambda: None)
    sandboxed = BuiltinExecuteTool(work_dir=str(tmp_path), sandbox_config=SandboxConfig(os_sandbox=True))
    plain = BuiltinExecuteTool(work_dir=str(tmp_path), sandbox_config=SandboxConfig())
    sandboxed_fn = sandboxed.functions["execute"]
    plain_fn = plain.functions["execute"]
    assert "escalate" in sandboxed_fn.parameters["properties"]
    assert "escalate=True" in sandboxed_fn.description
    assert "escalate" not in plain_fn.parameters["properties"]
    assert "escalate" not in plain_fn.description


def test_requesting_an_unavailable_sandbox_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "agentica.tools.builtin.execute_tool.unsupported_reason", lambda: "bwrap missing"
    )
    with pytest.raises(RuntimeError, match="bwrap missing"):
        BuiltinExecuteTool(work_dir=str(tmp_path), sandbox_config=SandboxConfig(os_sandbox=True))


def test_unapproved_escalation_does_not_run(tmp_path, monkeypatch):
    monkeypatch.setattr("agentica.tools.builtin.execute_tool.unsupported_reason", lambda: None)
    tool = BuiltinExecuteTool(work_dir=str(tmp_path), sandbox_config=SandboxConfig(os_sandbox=True))
    marker = tmp_path / "ran"
    with pytest.raises(PermissionError, match="needs the user's approval"):
        asyncio.run(tool.execute(f"touch {marker}", escalate=True))
    assert not marker.exists()


def _sandboxed_tool(work_dir: Path, **kwargs) -> BuiltinExecuteTool:
    return BuiltinExecuteTool(
        work_dir=str(work_dir),
        sandbox_config=SandboxConfig(os_sandbox=True, writable_dirs=[str(work_dir)]),
        **kwargs,
    )


@needs_seatbelt
def test_seatbelt_allows_work_dir_and_denies_outside(tmp_path):
    tool = _sandboxed_tool(tmp_path)
    assert asyncio.run(tool.execute("echo hi > a.txt && cat a.txt")) == "hi"

    # Every temp dir is a writable root, so "outside" has to be under $HOME.
    outside = Path.home() / f".agentica-sandbox-test-{uuid.uuid4().hex}"
    try:
        with pytest.raises(RuntimeError) as err:
            asyncio.run(tool.execute(f"echo x > {outside}"))
        assert not outside.exists()
        message = str(err.value)
        assert "Operation not permitted" in message
        # The model is told what the boundary was and how to cross it.
        assert "escalate=true" in message
        assert os.path.realpath(str(tmp_path)) in message
    finally:
        outside.unlink(missing_ok=True)


@needs_seatbelt
@needs_git
def test_seatbelt_commits_work_but_hooks_and_git_config_stay_read_only(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    tool = _sandboxed_tool(tmp_path)
    git = "git -c user.email=t@t -c user.name=t"
    asyncio.run(tool.execute(f"{git} commit -q --allow-empty -m one && git log --oneline | wc -l"))
    for command in ("echo 'curl evil' > .git/hooks/pre-commit", "git config core.hooksPath /tmp/x"):
        with pytest.raises(RuntimeError):
            asyncio.run(tool.execute(command))
    assert not (tmp_path / ".git" / "hooks" / "pre-commit").exists()
    assert "hooksPath" not in (tmp_path / ".git" / "config").read_text()


@needs_seatbelt
def test_seatbelt_blocks_credential_reads_and_escalation_lifts_it(tmp_path):
    (tmp_path / ".env").write_text("SECRET=1\n")
    tool = _sandboxed_tool(tmp_path)
    with pytest.raises(RuntimeError, match="Operation not permitted"):
        asyncio.run(tool.execute("cat .env"))

    token = approved_by_user.set(True)
    try:
        assert asyncio.run(tool.execute("cat .env", escalate=True)) == "SECRET=1"
    finally:
        approved_by_user.reset(token)


@needs_seatbelt
def test_seatbelt_network_off_but_loopback_reachable(tmp_path):
    tool = _sandboxed_tool(tmp_path)
    # TEST-NET-1: never routable, so outside a sandbox this would time out —
    # an immediate EPERM is the sandbox answering, no network needed.
    remote = (
        "python3 -c \"import socket; s=socket.socket(); s.settimeout(3); "
        "s.connect(('192.0.2.1', 80))\""
    )
    with pytest.raises(RuntimeError, match=r"Errno 1\]|Operation not permitted"):
        asyncio.run(tool.execute(remote))

    loopback = (
        "python3 -c \"import socket; srv=socket.socket(); srv.bind(('127.0.0.1', 0)); srv.listen(1); "
        "c=socket.create_connection(srv.getsockname(), timeout=3); print('connected')\""
    )
    assert asyncio.run(tool.execute(loopback)) == "connected"


@needs_seatbelt
def test_background_commands_are_sandboxed_too(tmp_path):
    (tmp_path / ".env").write_text("SECRET=1\n")
    registry = BackgroundProcessRegistry()
    try:
        tool = _sandboxed_tool(tmp_path, background_process_registry=registry)
        started = asyncio.run(tool.execute("cat .env", background=True))
        proc_id = started.split("id: ")[1].split(")")[0]
        report = asyncio.run(tool.wait(proc_id, timeout=20))
        assert "Operation not permitted" in report
        assert "SECRET=1" not in report
        item = registry.get(proc_id)
        # /ps and the log header show the command, not the sandbox wrapper.
        assert item.command == "cat .env"
        assert Path(item.log_path).read_text().splitlines()[0] == "$ cat .env"
    finally:
        registry.stop()


# --- SDK: DeepAgent(sandbox=...) -----------------------------------------


def _deep_agent(tmp_path, **kwargs):
    from unittest.mock import MagicMock, patch

    from agentica.agent.deep import DeepAgent

    with patch("agentica.agent.base.Agent._load_mcp_tools"), patch(
        "agentica.agent.base.Agent._merge_tool_system_prompts"
    ):
        return DeepAgent(
            model=MagicMock(), workspace=str(tmp_path / "ws"), work_dir=str(tmp_path),
            include_skills=False, **kwargs,
        )


def _execute_tool(agent) -> BuiltinExecuteTool:
    (tool,) = [t for t in agent.tools if isinstance(t, BuiltinExecuteTool)]
    return tool


def test_sdk_default_runs_on_the_host(tmp_path):
    agent = _deep_agent(tmp_path)
    assert agent.sandbox_config.os_sandbox is False
    assert "escalate" not in _execute_tool(agent).functions["execute"].parameters["properties"]


def test_sdk_sandbox_reaches_the_execute_tool(tmp_path, monkeypatch):
    monkeypatch.setattr("agentica.tools.builtin.execute_tool.unsupported_reason", lambda: None)
    agent = _deep_agent(tmp_path, sandbox="workspace-write", sandbox_network_access=True,
                        permission_mode="auto")
    tool = _execute_tool(agent)
    # One shared object: the tool confines with exactly what the agent reports.
    assert tool._sandbox_config is agent.sandbox_config
    assert agent.sandbox_config.os_sandbox is True
    assert agent.sandbox_config.network_access is True
    assert agent.sandbox_config.writable_dirs == [str(tmp_path)]
    assert "escalate" in tool.functions["execute"].parameters["properties"]


def test_sdk_rejects_an_unknown_mode_and_a_double_spec(tmp_path):
    with pytest.raises(ValueError, match="Invalid sandbox mode"):
        _deep_agent(tmp_path, sandbox="read-only")
    with pytest.raises(ValueError, match="not both"):
        _deep_agent(tmp_path, sandbox="workspace-write", sandbox_config=SandboxConfig())


@needs_seatbelt
def test_sdk_sandboxed_agent_confines_real_commands(tmp_path):
    agent = _deep_agent(tmp_path, sandbox="workspace-write")
    tool = _execute_tool(agent)
    assert asyncio.run(tool.execute("echo ok > inside.txt && cat inside.txt")) == "ok"
    outside = Path.home() / f".agentica-sdk-sandbox-test-{uuid.uuid4().hex}"
    try:
        with pytest.raises(RuntimeError, match="Operation not permitted"):
            asyncio.run(tool.execute(f"echo x > {outside}"))
        assert not outside.exists()
    finally:
        outside.unlink(missing_ok=True)
