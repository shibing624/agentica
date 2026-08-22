# -*- coding: utf-8 -*-
"""ApprovalRegistry, classify/grants, make_approve, and runner deny-as-result."""
import asyncio
import os
import tempfile
import time
import unittest
import weakref
from pathlib import Path

import pytest

from agentica.agent.approvals import (
    DENIED_TOOL_RESULT,
    ApprovalRegistry,
    PendingApproval,
    SessionGrants,
    classify,
    command_allows_prefix,
    make_approve,
)
from agentica.model.openai import OpenAIChat
from agentica.tools.base import Function, FunctionCall


def _fc(
    name: str,
    arguments=None,
    *,
    call_id="c1",
    is_read_only=False,
    is_destructive=False,
    concurrency_safe=False,
):
    fn = Function(name=name)
    fn.entrypoint = lambda **kwargs: "ran"
    fn.is_read_only = is_read_only
    fn.is_destructive = is_destructive
    fn.concurrency_safe = concurrency_safe
    return FunctionCall(function=fn, arguments=arguments or {}, call_id=call_id)


def _file_fc(name: str, file_path: str, **kwargs):
    destructive = name in ("write_file", "apply_patch")
    return _fc(
        name,
        {"file_path": file_path} if name != "apply_patch" else {"patch": file_path},
        is_read_only=name in ("read_file", "glob", "grep"),
        is_destructive=destructive,
        **kwargs,
    )


class TestApprovalRegistry(unittest.TestCase):
    def test_wait_decide_and_unknown_id(self):
        async def _run():
            registry = ApprovalRegistry()
            pending = PendingApproval(
                tool_call_id="t1", name="execute", arguments={"command": "rm x"},
                question="q", preview="rm x",
            )
            waiter = registry.wait(pending)
            self.assertEqual(registry.size, 1)
            self.assertEqual(registry.list()[0].tool_call_id, "t1")
            self.assertTrue(registry.decide("t1", "allow"))
            self.assertEqual(await waiter, "allow")
            self.assertEqual(registry.size, 0)
            self.assertFalse(registry.decide("t1", "deny"))

        asyncio.run(_run())

    def test_deny_all_resolves_pending(self):
        async def _run():
            registry = ApprovalRegistry()
            pending = PendingApproval(
                tool_call_id="t2", name="execute", arguments={},
                question="q", preview="",
            )
            waiter = registry.wait(pending)
            registry.deny_all()
            self.assertEqual(await waiter, "deny")
            self.assertEqual(registry.size, 0)

        asyncio.run(_run())

    def test_re_wait_same_id_denies_the_old_future(self):
        async def _run():
            registry = ApprovalRegistry()
            p = PendingApproval(tool_call_id="t3", name="x", arguments={}, question="q", preview="")
            first = registry.wait(p)
            second = registry.wait(p)
            self.assertEqual(await first, "deny")
            registry.decide("t3", "allow")
            self.assertEqual(await second, "allow")

        asyncio.run(_run())


class TestClassifyAndGrants(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.mkdtemp()
        self.grants = SessionGrants()

    def test_allow_all_never_asks(self):
        fc = _fc("execute", {"command": "rm -rf /tmp/x"}, is_destructive=True)
        self.assertEqual(classify("allow-all", fc, self.grants, work_dir=self.work), "allow")

    def test_ask_allows_workspace_write(self):
        path = os.path.join(self.work, "a.txt")
        fc = _file_fc("write_file", path)
        self.assertEqual(classify("ask", fc, self.grants, work_dir=self.work), "allow")

    def test_ask_parks_outside_workspace(self):
        fc = _file_fc("write_file", "/tmp/agentica-outside-approval.txt")
        self.assertEqual(classify("ask", fc, self.grants, work_dir=self.work), "ask")
        self.assertEqual(classify("auto", fc, self.grants, work_dir=self.work), "ask")

    def test_ask_parks_every_execute(self):
        ro = _fc("execute", {"command": "ls"}, is_destructive=True)
        rw = _fc("execute", {"command": "rm -f x"}, is_destructive=True)
        self.assertEqual(classify("ask", ro, self.grants, work_dir=self.work), "ask")
        self.assertEqual(classify("ask", rw, self.grants, work_dir=self.work), "ask")

    def test_auto_allows_read_only_execute(self):
        ro = _fc("execute", {"command": "ls"}, is_destructive=True)
        rw = _fc("execute", {"command": "rm -f x"}, is_destructive=True)
        self.assertEqual(classify("auto", ro, self.grants, work_dir=self.work), "allow")
        self.assertEqual(classify("auto", rw, self.grants, work_dir=self.work), "ask")

    def test_network_ask_parks_auto_allows(self):
        fc = _fc("web_search", {"queries": "news"}, is_read_only=True)
        self.assertEqual(classify("ask", fc, self.grants, work_dir=self.work), "ask")
        self.assertEqual(classify("auto", fc, self.grants, work_dir=self.work), "allow")

    def test_unlabeled_tool_ask_parks_auto_allows(self):
        fc = _fc("custom_tool", {"x": 1})
        self.assertEqual(classify("ask", fc, self.grants, work_dir=self.work), "ask")
        self.assertEqual(classify("auto", fc, self.grants, work_dir=self.work), "allow")

    def test_command_prefix_grant_covers_similar(self):
        first = _fc("execute", {"command": "rm -f /tmp/a.ini"}, is_destructive=True)
        self.grants.add_command_prefix("rm -f /tmp/a.ini")
        similar = _fc("execute", {"command": "rm -f /tmp/a.ini extra"}, is_destructive=True)
        other = _fc("execute", {"command": "rm -f /tmp/b.ini"}, is_destructive=True)
        self.assertEqual(classify("auto", similar, self.grants, work_dir=self.work), "allow")
        self.assertEqual(classify("auto", other, self.grants, work_dir=self.work), "ask")
        self.assertTrue(command_allows_prefix("rm -f /tmp/a.ini"))
        self.assertFalse(command_allows_prefix("rm -f /tmp/a.ini > /tmp/out"))
        self.assertFalse(command_allows_prefix("echo $(whoami)"))
        self.assertFalse(command_allows_prefix("cat <<EOF\nhi\nEOF"))

    def test_ampersand_compound_uses_first_segment_prefix(self):
        self.grants.add_command_prefix("rm -f /tmp/a.ini & echo done")
        similar = _fc("execute", {"command": "rm -f /tmp/a.ini"}, is_destructive=True)
        self.assertEqual(classify("auto", similar, self.grants, work_dir=self.work), "allow")


class TestMakeApprove(unittest.TestCase):
    def test_no_registry_denies_manual_path(self):
        async def _run():
            grants = SessionGrants()
            approve = make_approve(
                get_mode=lambda: "ask",
                get_grants=lambda: grants,
                get_registry=lambda: None,
                get_work_dir=lambda: "/tmp",
                publish=lambda p: None,
                apply_path_grant=lambda path, prefix: None,
            )
            fc = _fc("execute", {"command": "rm -f x"}, is_destructive=True)
            self.assertEqual(await approve(fc), "deny")

        asyncio.run(_run())

    def test_allow_prefix_records_command_grant(self):
        async def _run():
            grants = SessionGrants()
            registry = ApprovalRegistry()
            published = []
            approve = make_approve(
                get_mode=lambda: "auto",
                get_grants=lambda: grants,
                get_registry=lambda: registry,
                get_work_dir=lambda: "/tmp",
                publish=published.append,
                apply_path_grant=lambda path, prefix: None,
            )
            fc = _fc("execute", {"command": "rm -f /tmp/a.ini"}, is_destructive=True, call_id="x1")
            waiter = asyncio.create_task(approve(fc))
            await asyncio.sleep(0)
            self.assertEqual(len(published), 1)
            self.assertTrue(registry.decide("x1", "allow_prefix"))
            self.assertEqual(await waiter, "allow_prefix")
            again = _fc("execute", {"command": "rm -f /tmp/a.ini extra"}, is_destructive=True, call_id="x2")
            self.assertEqual(classify("auto", again, grants, work_dir="/tmp"), "allow")

        asyncio.run(_run())


class _HarnessAgent:
    tool_input_guardrails = []
    tool_output_guardrails = []
    context = None
    _run_hooks = None
    _cancelled = False
    approve = None
    _session_log = None
    agent_id = "a"
    name = "a"
    run_id = "r"


def _model():
    m = OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")
    m.metrics = {}
    m.function_call_stack = None
    m.tool_call_limit = None
    return m


def _exec_fc(call_id: str, command: str = "echo hi"):
    async def execute(command: str = "") -> str:
        execute.calls.append(command)
        return f"ran:{command}"

    execute.calls = []
    fn = Function.from_callable(execute)
    fn.name = "execute"
    fn.is_destructive = True
    fn.concurrency_safe = False
    return FunctionCall(function=fn, arguments={"command": command}, call_id=call_id), execute


class TestRunnerApprovalHook:
    @pytest.mark.asyncio
    async def test_deny_is_tool_result_not_sibling_abort(self):
        model = _model()
        agent = _HarnessAgent()
        events = []

        async def approve(fc):
            events.append(fc.call_id)
            return "deny" if fc.call_id == "c1" else "allow"

        agent.approve = approve
        model._agent_ref = weakref.ref(agent)
        fc1, ex1 = _exec_fc("c1", "rm -f a")
        fc2, ex2 = _exec_fc("c2", "echo b")
        results = []
        async for _ in model.run_function_calls([fc1, fc2], results):
            pass
        assert fc1.result == DENIED_TOOL_RESULT
        assert fc1.error == DENIED_TOOL_RESULT
        assert fc2.result == "ran:echo b"
        assert ex1.calls == []
        assert ex2.calls == ["echo b"]
        assert "Cancelled: sibling" not in (fc2.error or "")

    @pytest.mark.asyncio
    async def test_approve_exception_becomes_deny_result(self):
        model = _model()
        agent = _HarnessAgent()

        async def approve(fc):
            raise RuntimeError("boom")

        agent.approve = approve
        model._agent_ref = weakref.ref(agent)
        fc, ex = _exec_fc("c1")
        async for _ in model.run_function_calls([fc], []):
            pass
        assert fc.result == DENIED_TOOL_RESULT
        assert fc.error == DENIED_TOOL_RESULT
        assert ex.calls == []

    @pytest.mark.asyncio
    async def test_approve_is_outside_tool_timeout(self):
        model = _model()
        agent = _HarnessAgent()

        async def approve(fc):
            await asyncio.sleep(0.2)
            return "allow"

        agent.approve = approve
        model._agent_ref = weakref.ref(agent)

        async def quick(command: str = "") -> str:
            return "ok"

        fn = Function.from_callable(quick)
        fn.name = "execute"
        fn.timeout = 0.05
        fn.concurrency_safe = False
        fc = FunctionCall(function=fn, arguments={"command": "echo"}, call_id="slow")
        start = time.monotonic()
        async for _ in model.run_function_calls([fc], []):
            pass
        elapsed = time.monotonic() - start
        assert fc.result == "ok"
        assert fc.error is None
        assert elapsed >= 0.2

    @pytest.mark.asyncio
    async def test_writes_approval_decision_event(self):
        model = _model()
        agent = _HarnessAgent()
        logged = []

        class _Log:
            def append_event(self, name, **payload):
                logged.append((name, payload))
                return "u"

        agent._session_log = _Log()

        async def approve(fc):
            return "deny"

        agent.approve = approve
        model._agent_ref = weakref.ref(agent)
        fc, _ex = _exec_fc("tid")
        async for _ in model.run_function_calls([fc], []):
            pass
        assert logged == [("approval_decision", {"tool_call_id": "tid", "decision": "deny"})]


class TestGrantPathAccess(unittest.TestCase):
    def test_prefix_unlocks_sibling_files(self):
        from agentica.agent.config import SandboxConfig
        from agentica.tools.builtin import BuiltinFileTool

        with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as other:
            tool = BuiltinFileTool(
                work_dir=work,
                sandbox_config=SandboxConfig(enabled=True, writable_dirs=[work]),
            )
            a = os.path.join(other, "a.txt")
            b = os.path.join(other, "b.txt")
            tool.grant_path_access(a, prefix=True)
            asyncio.run(tool.write_file(a, "a"))
            asyncio.run(tool.write_file(b, "b"))
            self.assertEqual(Path(a).read_text(), "a")
            self.assertEqual(Path(b).read_text(), "b")

    def test_sensitive_grant_is_exact_file_never_parent(self):
        from agentica.tools.builtin import BuiltinFileTool

        with tempfile.TemporaryDirectory() as work:
            tool = BuiltinFileTool(work_dir=work)
            tool.grant_path_access("/etc/hosts", prefix=True)
            self.assertIsNone(tool._sensitive_write_guard("/etc/hosts"))
            self.assertIsNotNone(tool._sensitive_write_guard("/etc/passwd"))
