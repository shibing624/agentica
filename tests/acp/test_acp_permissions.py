# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for ACP IDE tool-permission policy + handler gate.
"""
import asyncio
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import websockets  # noqa: F401
    _HAS_WS = True
except ImportError:
    _HAS_WS = False

if _HAS_WS:
    from agentica.acp.permissions import (
        ToolPermissionPolicy, PermissionMode, PermissionDecision,
    )
    from agentica.acp.handlers import ACPHandlers
    from agentica.acp.types import ACPToolCall


@unittest.skipUnless(_HAS_WS, "acp extras (websockets) not installed")
class TestToolPermissionPolicy(unittest.TestCase):
    def test_auto_allows_all(self):
        p = ToolPermissionPolicy()  # AUTO default
        self.assertEqual(p.decide("write_file"), PermissionDecision.ALLOW)
        self.assertEqual(p.decide("execute"), PermissionDecision.ALLOW)

    def test_read_only_mode(self):
        p = ToolPermissionPolicy(mode=PermissionMode.READ_ONLY)
        self.assertEqual(p.decide("read_file"), PermissionDecision.ALLOW)
        self.assertEqual(p.decide("grep"), PermissionDecision.ALLOW)
        self.assertEqual(p.decide("write_file"), PermissionDecision.DENY)
        self.assertEqual(p.decide("execute"), PermissionDecision.DENY)

    def test_confirm_writes_mode(self):
        p = ToolPermissionPolicy(mode=PermissionMode.CONFIRM_WRITES)
        self.assertEqual(p.decide("read_file"), PermissionDecision.ALLOW)
        self.assertEqual(p.decide("write_file"), PermissionDecision.ASK)

    def test_deny_all_mode(self):
        p = ToolPermissionPolicy(mode=PermissionMode.DENY_ALL)
        self.assertEqual(p.decide("read_file"), PermissionDecision.DENY)

    def test_deny_list_overrides(self):
        p = ToolPermissionPolicy(mode=PermissionMode.AUTO, deny_tools={"execute"})
        self.assertEqual(p.decide("execute"), PermissionDecision.DENY)
        self.assertEqual(p.decide("read_file"), PermissionDecision.ALLOW)

    def test_allow_list_overrides_mode(self):
        p = ToolPermissionPolicy(mode=PermissionMode.READ_ONLY, allow_tools={"write_file"})
        self.assertEqual(p.decide("write_file"), PermissionDecision.ALLOW)

    def test_unknown_tool_treated_as_write(self):
        p = ToolPermissionPolicy(mode=PermissionMode.READ_ONLY)
        self.assertEqual(p.decide("some_unknown_tool"), PermissionDecision.DENY)


@unittest.skipUnless(_HAS_WS, "acp extras (websockets) not installed")
class TestHandlerPermissionGate(unittest.TestCase):
    def _handler(self, **kwargs):
        return ACPHandlers(**kwargs)

    def test_denied_tool_returns_error(self):
        handler = self._handler(
            permission_policy=ToolPermissionPolicy(mode=PermissionMode.READ_ONLY)
        )
        call = ACPToolCall(name="write_file", arguments={"file_path": "x.txt", "content": "y"})
        result = asyncio.run(handler._execute_tool(call))
        self.assertTrue(result.isError)
        self.assertIn("Permission denied", result.content)

    def test_ask_without_callback_denies(self):
        handler = self._handler(
            permission_policy=ToolPermissionPolicy(mode=PermissionMode.CONFIRM_WRITES)
        )
        call = ACPToolCall(name="write_file", arguments={"file_path": "x.txt", "content": "y"})
        result = asyncio.run(handler._execute_tool(call))
        self.assertTrue(result.isError)
        self.assertIn("not granted", result.content)

    def test_ask_with_granting_callback_proceeds(self):
        granted = {"called": False}

        async def cb(tool_call):
            granted["called"] = True
            return True

        handler = self._handler(
            permission_policy=ToolPermissionPolicy(mode=PermissionMode.CONFIRM_WRITES),
            permission_callback=cb,
        )
        # read_file on a missing file: permission passes (ASK->granted), the
        # tool then runs and returns its own (non-permission) result.
        call = ACPToolCall(name="write_file", arguments={"file_path": "/tmp/_acp_perm_test.txt", "content": "hi"})
        result = asyncio.run(handler._execute_tool(call))
        self.assertTrue(granted["called"])
        self.assertNotIn("not granted", str(result.content))
        # cleanup
        try:
            os.remove("/tmp/_acp_perm_test.txt")
        except OSError:
            pass

    def test_auto_mode_allows_read(self):
        handler = self._handler()  # AUTO
        call = ACPToolCall(name="ls", arguments={"directory": "."})
        result = asyncio.run(handler._execute_tool(call))
        self.assertFalse(result.isError)


if __name__ == "__main__":
    unittest.main()
