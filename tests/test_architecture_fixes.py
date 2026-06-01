# -*- coding: utf-8 -*-
"""
Tests for architecture fix verification.

Covers:
1. Runner-owned tool execution (run_tools=False)
2. Model.parse_tool_calls / format_tool_results
3. Loop helper methods (_loop_safety_checks, _loop_post_response)
4. SessionLog error handling
5. _file_locks race fix (setdefault)
6. CompressionManager circuit breaker reset

All tests mock LLM API keys -- no real API calls.
"""
import asyncio
import sys
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import agentica.utils.langfuse_integration as langfuse_integration
from agentica.model.base import Model
from agentica.model.message import Message
from agentica.model.loop_state import LoopState
from agentica.run_response import RunBreakReason
from agentica.runner import Runner
from agentica.tools.base import Function, FunctionCall


def _make_agent(name="test-agent"):
    """Create a minimal Agent with a fake OpenAI key."""
    from agentica.agent import Agent
    from agentica.model.openai import OpenAIChat
    return Agent(
        name=name,
        model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
    )


class TestRunnerRunToolsFalse(unittest.TestCase):
    """Verify Runner sets model.run_tools = False during execution."""

    def test_update_model_sets_agent_ref(self):
        """update_model() still sets _agent_ref for backward compat."""
        agent = _make_agent()
        agent.update_model()
        self.assertIsNotNone(agent.model._agent_ref)

    def test_run_tools_default_is_true(self):
        """Model.run_tools defaults to True (for direct use)."""
        from agentica.model.openai import OpenAIChat
        model = OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")
        self.assertTrue(model.run_tools)


class TestModelParseToolCalls(unittest.TestCase):
    """Test Model.parse_tool_calls default (OpenAI-compat) implementation."""

    def test_parse_empty_tool_calls(self):
        """No tool_calls -> empty list."""
        from agentica.model.openai import OpenAIChat
        model = OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")
        msg = Message(role="assistant", content="Hello")
        fcs, meta = model.parse_tool_calls(msg, [])
        self.assertEqual(fcs, [])
        self.assertEqual(meta, {})

    def test_parse_unknown_function(self):
        """Unknown function name -> error message appended to messages."""
        from agentica.model.openai import OpenAIChat
        model = OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")
        model.functions = {}  # empty, so no function will match

        msg = Message(
            role="assistant",
            content="",
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "unknown_fn", "arguments": "{}"}
            }],
        )
        messages = []
        fcs, meta = model.parse_tool_calls(msg, messages, tool_role="tool")
        self.assertEqual(fcs, [])
        # Error message should have been appended
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, "tool")
        self.assertIn("Could not find", messages[0].content)

    def test_format_tool_results_default(self):
        """Default format_tool_results extends messages."""
        from agentica.model.openai import OpenAIChat
        model = OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")
        results = [Message(role="tool", content="ok", tool_call_id="c1")]
        messages = []
        model.format_tool_results(results, messages, {})
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].content, "ok")


class TestLoopSafetyChecks(unittest.TestCase):
    """Test _loop_safety_checks helper."""

    def test_no_issues_returns_none(self):
        agent = _make_agent()
        agent.update_model()
        from agentica.cost_tracker import CostTracker
        agent.model._cost_tracker = CostTracker()
        agent._run_max_cost_usd = None
        runner = Runner(agent)
        loop_state = LoopState()
        result = runner._loop_safety_checks([], loop_state, agent)
        self.assertIsNone(result)

    def test_cost_exceeded_returns_message(self):
        agent = _make_agent()
        agent.update_model()
        from agentica.cost_tracker import CostTracker
        ct = CostTracker()
        ct.total_cost_usd = 10.0
        agent.model._cost_tracker = ct
        agent._run_max_cost_usd = 1.0
        runner = Runner(agent)
        loop_state = LoopState()
        result = runner._loop_safety_checks([], loop_state, agent)
        self.assertIsNotNone(result)
        self.assertEqual(result.reason, RunBreakReason.COST_BUDGET.value)
        self.assertIn("Cost budget exceeded", result.message)

    def test_death_spiral_returns_structured_reason(self):
        """Death spiral break carries a machine code and a clean message
        (no leading newlines / bracket tags that used to ride in content)."""
        agent = _make_agent()
        agent.update_model()
        from agentica.cost_tracker import CostTracker
        agent.model._cost_tracker = CostTracker()
        agent._run_max_cost_usd = None
        runner = Runner(agent)
        loop_state = LoopState(death_spiral_threshold=1)
        messages = [
            Message(role="assistant", content="", tool_calls=[{"id": "1"}]),
            Message(role="tool", content="boom", tool_call_error=True),
        ]
        result = runner._loop_safety_checks(messages, loop_state, agent)
        self.assertIsNotNone(result)
        self.assertEqual(result.reason, RunBreakReason.DEATH_SPIRAL.value)
        self.assertNotIn("[", result.message)
        self.assertFalse(result.message.startswith("\n"))

    def test_max_turns_returns_structured_reason(self):
        agent = _make_agent()
        agent.update_model()
        from agentica.cost_tracker import CostTracker
        agent.model._cost_tracker = CostTracker()
        agent._run_max_cost_usd = None
        runner = Runner(agent)
        loop_state = LoopState(max_turns=2)
        loop_state.turn_count = 2
        result = runner._loop_safety_checks([], loop_state, agent)
        self.assertIsNotNone(result)
        self.assertEqual(result.reason, RunBreakReason.MAX_TURNS.value)


class TestRunResponseBreakSignal(unittest.TestCase):
    """RunResponse exposes loop-break info as structured fields, keeping
    content clean for downstream consumers."""

    def test_defaults_complete(self):
        from agentica.run_response import RunResponse
        resp = RunResponse(content="hello")
        self.assertTrue(resp.is_complete)
        self.assertIsNone(resp.break_reason)
        self.assertIsNone(resp.break_message)

    def test_break_marks_incomplete(self):
        from agentica.run_response import RunResponse
        resp = RunResponse(
            content="partial answer",
            break_reason=RunBreakReason.DEATH_SPIRAL.value,
            break_message="All tool calls have failed repeatedly.",
        )
        self.assertFalse(resp.is_complete)
        self.assertEqual(resp.break_reason, "death_spiral")
        # Content stays clean — no internal error text leaked in.
        self.assertEqual(resp.content, "partial answer")


class TestLoopPostResponse(unittest.TestCase):
    """Test _loop_post_response helper."""

    def test_no_tool_calls_no_length_breaks(self):
        """No tool calls and no max_tokens -> should break (return False)."""
        from agentica.model.openai import OpenAIChat
        model = OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")
        model.last_finish_reason = "stop"
        loop_state = LoopState()
        result = Runner._loop_post_response([], model, loop_state, had_tool_calls=False)
        self.assertFalse(result)

    def test_no_tool_calls_length_continues(self):
        """Max-tokens truncation -> should continue (return True)."""
        from agentica.model.openai import OpenAIChat
        model = OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")
        model.last_finish_reason = "length"
        loop_state = LoopState()
        messages = []
        result = Runner._loop_post_response(messages, model, loop_state, had_tool_calls=False)
        self.assertTrue(result)
        # Should have appended "Continue" message
        self.assertEqual(len(messages), 1)
        self.assertIn("Continue", messages[0].content)

    def test_had_tool_calls_continues(self):
        """Tool calls processed -> should continue (return True)."""
        from agentica.model.openai import OpenAIChat
        model = OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")
        loop_state = LoopState()
        result = Runner._loop_post_response([], model, loop_state, had_tool_calls=True)
        self.assertTrue(result)

    def test_stop_after_tool_call_breaks(self):
        """stop_after_tool_call flag -> should break (return False)."""
        from agentica.model.openai import OpenAIChat
        model = OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")
        loop_state = LoopState()
        messages = [Message(role="tool", content="ok", stop_after_tool_call=True)]
        result = Runner._loop_post_response(messages, model, loop_state, had_tool_calls=True)
        self.assertFalse(result)


class TestSessionLogErrorHandling(unittest.TestCase):
    """SessionLog._append should not raise on disk errors."""

    def test_append_on_readonly_path_does_not_raise(self):
        """If write fails, should log warning but not raise."""
        from agentica.memory.session_log import SessionLog
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmp:
            log = SessionLog("test-session", base_dir=tmp)
            # Make path unwritable
            log.path = os.path.join(tmp, "nonexistent_dir", "deep", "test.jsonl")
            # Should not raise
            log.append("user", "hello")


class TestFileLockSetdefault(unittest.TestCase):
    """Verify _file_locks uses setdefault (no race condition)."""

    def test_get_file_lock_returns_same_lock(self):
        """Same path should return the same Lock object."""
        from agentica.tools.buildin_tools import BuiltinFileTool
        tool = BuiltinFileTool()
        lock1 = tool._get_file_lock("/tmp/test.txt")
        lock2 = tool._get_file_lock("/tmp/test.txt")
        self.assertIs(lock1, lock2)

    def test_get_file_lock_different_paths(self):
        """Different paths should return different Lock objects."""
        from agentica.tools.buildin_tools import BuiltinFileTool
        tool = BuiltinFileTool()
        lock1 = tool._get_file_lock("/tmp/a.txt")
        lock2 = tool._get_file_lock("/tmp/b.txt")
        self.assertIsNot(lock1, lock2)


class TestLangfuseContextManagers(unittest.TestCase):
    """Langfuse context managers must preserve the original body exception."""

    def test_trace_context_does_not_yield_dummy_after_body_exception(self):
        class FakeSpan:
            def update_trace(self, **_kwargs):
                pass

            def update(self, **_kwargs):
                pass

        class FakeObservation:
            def __enter__(self):
                return FakeSpan()

            def __exit__(self, _exc_type, _exc, _tb):
                return False

        class FakeClient:
            def start_as_current_observation(self, **_kwargs):
                return FakeObservation()

            def update_current_trace(self, **_kwargs):
                pass

        fake_langfuse = types.SimpleNamespace(get_client=lambda: FakeClient())
        with (
            patch.object(langfuse_integration, "is_langfuse_available", return_value=True),
            patch.dict(sys.modules, {"langfuse": fake_langfuse}),
        ):
            with self.assertRaisesRegex(ValueError, "root cause"):
                with langfuse_integration.langfuse_trace_context(name="test"):
                    raise ValueError("root cause")


class TestCompressionManagerResetRun(unittest.TestCase):
    """CompressionManager.reset_run_state() should reset circuit breaker."""

    def test_reset_clears_failure_counter(self):
        from agentica.compression import CompressionManager
        cm = CompressionManager()
        cm._consecutive_auto_compact_failures = 5
        cm.reset_run_state()
        self.assertEqual(cm._consecutive_auto_compact_failures, 0)

    def test_reset_clears_iterative_summary(self):
        from agentica.compression import CompressionManager
        cm = CompressionManager()
        cm._conversation_previous_summary = "summary from previous run"
        cm.reset_run_state()
        self.assertIsNone(cm._conversation_previous_summary)


class TestGetLastAssistantMessage(unittest.TestCase):
    """Test Runner._get_last_assistant_message helper."""

    def test_empty_messages(self):
        result = Runner._get_last_assistant_message([])
        self.assertIsNone(result)

    def test_finds_last_assistant(self):
        msgs = [
            Message(role="user", content="hi"),
            Message(role="assistant", content="first"),
            Message(role="user", content="again"),
            Message(role="assistant", content="second"),
        ]
        result = Runner._get_last_assistant_message(msgs)
        self.assertEqual(result.content, "second")

    def test_no_assistant(self):
        msgs = [Message(role="user", content="hi")]
        result = Runner._get_last_assistant_message(msgs)
        self.assertIsNone(result)


class TestBuiltinTaskToolExtraction(unittest.TestCase):
    """Verify BuiltinTaskTool is importable from all expected paths."""

    def test_import_from_builtin_task_tool(self):
        from agentica.tools.builtin_task_tool import BuiltinTaskTool
        self.assertIsNotNone(BuiltinTaskTool)

    def test_import_from_buildin_tools(self):
        from agentica.tools.buildin_tools import BuiltinTaskTool
        self.assertIsNotNone(BuiltinTaskTool)

    def test_import_from_tools_package(self):
        from agentica.tools import BuiltinTaskTool
        self.assertIsNotNone(BuiltinTaskTool)

    def test_import_from_top_level(self):
        from agentica import BuiltinTaskTool
        self.assertIsNotNone(BuiltinTaskTool)

    def test_same_class(self):
        """All import paths should resolve to the same class."""
        from agentica.tools.builtin_task_tool import BuiltinTaskTool as A
        from agentica.tools.buildin_tools import BuiltinTaskTool as B
        self.assertIs(A, B)


if __name__ == "__main__":
    unittest.main()
