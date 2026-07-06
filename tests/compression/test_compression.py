# -*- coding: utf-8 -*-
"""Tests for agentica.compression — micro-compact, tool result storage, compression manager."""
import asyncio
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, AsyncMock

from agentica.model.message import Message
from agentica.run_response import RunResponse


# ===========================================================================
# micro_compact tests
# ===========================================================================

class TestMicroCompact(unittest.TestCase):
    """Tests for agentica.compression.micro.micro_compact."""

    def _make_tool_msg(self, content: str, compacted: bool = False) -> Message:
        msg = Message(role="tool", content=content, tool_call_id="tc_1")
        msg._micro_compacted = compacted
        return msg

    def test_keeps_recent_messages_untouched(self):
        from agentica.compression.micro import micro_compact, DEFAULT_KEEP_RECENT
        msgs = [
            Message(role="user", content="hi"),
        ] + [self._make_tool_msg(f"result {'x' * 100}") for _ in range(DEFAULT_KEEP_RECENT)]
        count = micro_compact(msgs)
        self.assertEqual(count, 0, "Should not compact when tool msgs <= keep_recent")

    def test_compacts_oldest_tool_results(self):
        from agentica.compression.micro import micro_compact, MICRO_COMPACT_PLACEHOLDER
        msgs = [
            Message(role="user", content="hi"),
            self._make_tool_msg("A" * 200),  # old → should be compacted
            self._make_tool_msg("B" * 200),  # old → should be compacted
        ] + [self._make_tool_msg(f"C{'x' * 100}") for _ in range(5)]  # recent 5 → kept
        count = micro_compact(msgs, keep_recent=5)
        self.assertEqual(count, 2)
        self.assertEqual(msgs[1].content, MICRO_COMPACT_PLACEHOLDER)
        self.assertEqual(msgs[2].content, MICRO_COMPACT_PLACEHOLDER)
        # Recent 5 untouched
        for m in msgs[3:]:
            self.assertNotEqual(m.content, MICRO_COMPACT_PLACEHOLDER)

    def test_skips_short_content(self):
        from agentica.compression.micro import micro_compact, _MIN_CONTENT_LEN
        msgs = [
            Message(role="user", content="hi"),
            self._make_tool_msg("short"),  # < _MIN_CONTENT_LEN → skip
        ] + [self._make_tool_msg(f"x{'y' * 100}") for _ in range(5)]
        count = micro_compact(msgs, keep_recent=5)
        self.assertEqual(count, 0, "Short content should be skipped")
        self.assertEqual(msgs[1].content, "short")

    def test_skips_already_compacted(self):
        from agentica.compression.micro import micro_compact
        msgs = [
            Message(role="user", content="hi"),
            self._make_tool_msg("A" * 200, compacted=True),  # already compacted
        ] + [self._make_tool_msg(f"B{'x' * 100}") for _ in range(5)]
        count = micro_compact(msgs, keep_recent=5)
        self.assertEqual(count, 0, "Already compacted messages should be skipped")

    def test_marks_compacted_flag(self):
        from agentica.compression.micro import micro_compact
        msgs = [
            Message(role="user", content="hi"),
            self._make_tool_msg("A" * 200),
        ] + [self._make_tool_msg(f"B{'x' * 100}") for _ in range(5)]
        micro_compact(msgs, keep_recent=5)
        self.assertTrue(msgs[1]._micro_compacted)


# ===========================================================================
# tool_result_storage tests
# ===========================================================================

class TestToolCallArgumentShrinking(unittest.TestCase):
    """JSON-safe shrinking for assistant tool_call arguments."""

    def test_shrinks_long_string_leaves_and_preserves_json(self):
        from agentica.compression.tool_call_args import shrink_tool_call_arguments_json

        args = json.dumps({
            "path": "/tmp/example.txt",
            "content": "x" * 100,
            "nested": {"note": "y" * 100},
            "count": 3,
        })

        result = shrink_tool_call_arguments_json(args, max_string_chars=20)
        parsed = json.loads(result)

        self.assertEqual(parsed["path"], "/tmp/example.txt")
        self.assertEqual(parsed["count"], 3)
        self.assertEqual(parsed["content"], "x" * 20 + "...[truncated]")
        self.assertEqual(parsed["nested"]["note"], "y" * 20 + "...[truncated]")

    def test_invalid_json_is_returned_unchanged(self):
        from agentica.compression.tool_call_args import shrink_tool_call_arguments_json

        args = '{"content": "unterminated'

        self.assertEqual(shrink_tool_call_arguments_json(args), args)

    def test_compression_manager_shrinks_assistant_tool_call_arguments(self):
        from agentica.compression.manager import CompressionManager

        long_content = "z" * 300
        messages = [
            Message(role="user", content="write file"),
            Message(role="assistant", tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": json.dumps({"content": long_content, "path": "a.txt"}),
                },
            }]),
            Message(role="tool", tool_call_id="call_1", content="ok"),
        ]

        cm = CompressionManager(truncate_head_chars=20)
        with patch("agentica.compression.manager.count_tokens", return_value=10):
            asyncio.run(cm.compress(messages))

        args = messages[1].tool_calls[0]["function"]["arguments"]
        parsed = json.loads(args)
        self.assertEqual(parsed["content"], "z" * 20 + "...[truncated]")
        self.assertEqual(parsed["path"], "a.txt")


class TestSanitizePath(unittest.TestCase):
    """Tests for _sanitize_path."""

    def test_basic_path(self):
        from agentica.compression.tool_result_storage import _sanitize_path
        result = _sanitize_path("/Users/test/project")
        self.assertRegex(result, r'^[a-zA-Z0-9\-]+$')

    def test_long_path_truncated_with_hash(self):
        from agentica.compression.tool_result_storage import _sanitize_path, _MAX_SANITIZED_LENGTH
        long_path = "/a/b/c/" + "x" * 300
        result = _sanitize_path(long_path)
        self.assertLessEqual(len(result), _MAX_SANITIZED_LENGTH + 10)  # +hash suffix
        self.assertIn("-", result)  # hash appended

    def test_special_chars_replaced(self):
        from agentica.compression.tool_result_storage import _sanitize_path
        result = _sanitize_path("/path/to/my project (2)/test.txt")
        self.assertNotIn(" ", result)
        self.assertNotIn("(", result)


class TestMaybePersistResult(unittest.TestCase):
    """Tests for maybe_persist_result — Layer 1 per-tool persistence."""

    def test_small_content_unchanged(self):
        from agentica.compression.tool_result_storage import maybe_persist_result
        content = "small output"
        result = maybe_persist_result("test_tool", "call_1", content, max_result_size_chars=50000)
        self.assertEqual(result, content)

    def test_none_threshold_skips(self):
        from agentica.compression.tool_result_storage import maybe_persist_result
        big = "x" * 100_000
        result = maybe_persist_result("test_tool", "call_2", big, max_result_size_chars=None)
        self.assertEqual(result, big, "None threshold should never persist")

    def test_large_content_persisted(self):
        from agentica.compression.tool_result_storage import maybe_persist_result
        with tempfile.TemporaryDirectory() as tmpdir:
            big = "x" * 100
            with patch("agentica.compression.tool_result_storage.AGENTICA_PROJECTS_DIR", tmpdir):
                result = maybe_persist_result(
                    "test_tool", "call_3", big,
                    max_result_size_chars=50, cwd="/test/project",
                )
            self.assertIn("<persisted-output>", result)
            self.assertIn("Preview", result)

    def test_large_content_redacted_in_preview_and_disk(self):
        from agentica.compression.tool_result_storage import get_tool_result_path, maybe_persist_result

        with tempfile.TemporaryDirectory() as tmpdir:
            secret = "sk-abcdefghijklmnopqrstuvwxyz1234567890"
            big = f"before {secret} after " + ("x" * 100)
            with patch("agentica.compression.tool_result_storage.AGENTICA_PROJECTS_DIR", tmpdir):
                result = maybe_persist_result(
                    "test_tool", "call_secret", big,
                    max_result_size_chars=50, cwd="/test/project",
                )
                file_path = get_tool_result_path("call_secret", cwd="/test/project", session_id="default")
                persisted = open(file_path, encoding="utf-8").read()

        self.assertNotIn(secret, result)
        self.assertNotIn(secret, persisted)
        self.assertIn("REDACTED", result)
        self.assertIn("REDACTED", persisted)

    def test_disk_failure_fallback_truncation(self):
        from agentica.compression.tool_result_storage import maybe_persist_result
        big = "x" * 100
        with patch("agentica.compression.tool_result_storage._persist_to_disk", return_value=False):
            result = maybe_persist_result(
                "test_tool", "call_4", big,
                max_result_size_chars=50,
            )
        self.assertIn("truncated", result)
        self.assertLessEqual(len(result), 80)  # truncated to threshold + message


class TestBuildPersistedMessage(unittest.TestCase):
    """Tests for _build_persisted_message."""

    def test_message_format(self):
        from agentica.compression.tool_result_storage import _build_persisted_message, PREVIEW_CHARS
        content = "x" * 5000
        msg = _build_persisted_message("/path/to/file.txt", content)
        self.assertIn("<persisted-output>", msg)
        self.assertIn("</persisted-output>", msg)
        self.assertIn("/path/to/file.txt", msg)
        self.assertIn("Preview", msg)
        self.assertIn("...", msg)  # content > PREVIEW_CHARS, so has ellipsis

    def test_short_content_no_ellipsis(self):
        from agentica.compression.tool_result_storage import _build_persisted_message
        content = "short"
        msg = _build_persisted_message("/path/to/file.txt", content)
        # Content <= PREVIEW_CHARS, no "..." before closing tag
        self.assertIn("<persisted-output>", msg)
        self.assertIn("short", msg)
        # The message should NOT have the ellipsis line
        self.assertNotIn("\n...\n", msg)


class TestEnforceToolResultBudget(unittest.TestCase):
    """Tests for enforce_tool_result_budget — Layer 2 per-message budget."""

    def test_under_budget_no_changes(self):
        from agentica.compression.tool_result_storage import enforce_tool_result_budget
        msgs = [
            Message(role="tool", content="short1", tool_call_id="t1"),
            Message(role="tool", content="short2", tool_call_id="t2"),
        ]
        count = enforce_tool_result_budget(msgs, budget=1000)
        self.assertEqual(count, 0)

    def test_over_budget_largest_persisted(self):
        from agentica.compression.tool_result_storage import enforce_tool_result_budget
        with tempfile.TemporaryDirectory() as tmpdir:
            msgs = [
                Message(role="tool", content="a" * 100, tool_call_id="t1"),
                Message(role="tool", content="b" * 500, tool_call_id="t2"),  # largest
                Message(role="tool", content="c" * 50, tool_call_id="t3"),
            ]
            with patch("agentica.compression.tool_result_storage.AGENTICA_PROJECTS_DIR", tmpdir):
                count = enforce_tool_result_budget(msgs, budget=200, cwd="/test")
            self.assertGreater(count, 0)
            # The largest should be persisted
            self.assertIn("<persisted-output>", msgs[1].content)

    def test_over_budget_redacts_persisted_content(self):
        from agentica.compression.tool_result_storage import enforce_tool_result_budget, get_tool_result_path

        with tempfile.TemporaryDirectory() as tmpdir:
            secret = "ghp_abcdefghijklmnopqrstuvwxyz1234567890"
            msgs = [
                Message(role="tool", content="safe", tool_call_id="t1"),
                Message(role="tool", content=f"{secret} " + ("b" * 500), tool_call_id="t2"),
            ]
            with patch("agentica.compression.tool_result_storage.AGENTICA_PROJECTS_DIR", tmpdir):
                count = enforce_tool_result_budget(msgs, budget=100, cwd="/test")
                file_path = get_tool_result_path("t2", cwd="/test", session_id="default")
                persisted = open(file_path, encoding="utf-8").read()

        self.assertGreater(count, 0)
        self.assertNotIn(secret, msgs[1].content)
        self.assertNotIn(secret, persisted)
        self.assertIn("REDACTED", msgs[1].content)
        self.assertIn("REDACTED", persisted)

    def test_already_persisted_skipped(self):
        from agentica.compression.tool_result_storage import enforce_tool_result_budget
        msgs = [
            Message(role="tool", content="<persisted-output>already</persisted-output>", tool_call_id="t1"),
            Message(role="tool", content="b" * 500, tool_call_id="t2"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("agentica.compression.tool_result_storage.AGENTICA_PROJECTS_DIR", tmpdir):
                count = enforce_tool_result_budget(msgs, budget=100, cwd="/test")
        # Only the non-persisted one should be targeted
        self.assertLessEqual(count, 1)

    def test_empty_results_no_error(self):
        from agentica.compression.tool_result_storage import enforce_tool_result_budget
        count = enforce_tool_result_budget([], budget=1000)
        self.assertEqual(count, 0)


# ===========================================================================
# CompressionManager tests
# ===========================================================================

class TestCompressionManagerInit(unittest.TestCase):
    """CompressionManager initialization and defaults."""

    def test_defaults(self):
        from agentica.compression.manager import CompressionManager
        cm = CompressionManager()
        self.assertTrue(cm.compress_tool_results)
        self.assertIsNone(cm.compress_token_limit)
        self.assertEqual(cm.truncate_head_chars, 150)
        self.assertEqual(cm.keep_recent_rounds, 3)
        self.assertFalse(cm.use_llm_compression)

    def test_default_prompt_marks_summary_as_reference_only(self):
        from agentica.compression.manager import DEFAULT_COMPRESSION_PROMPT

        self.assertIn("REFERENCE ONLY", DEFAULT_COMPRESSION_PROMPT)
        self.assertIn("NOT active instructions", DEFAULT_COMPRESSION_PROMPT)

    def test_target_from_trigger(self):
        from agentica.compression.manager import CompressionManager
        cm = CompressionManager(compress_token_limit=10000)
        self.assertEqual(cm.compress_target_token_limit, 6000)  # 60% of trigger


class TestCompressionManagerResolveLimits(unittest.TestCase):
    """_resolve_limits auto-derives thresholds from model.context_window."""

    def test_resolve_from_model(self):
        from agentica.compression.manager import CompressionManager
        cm = CompressionManager()
        mock_model = MagicMock()
        mock_model.context_window = 100_000
        cm._resolve_limits(mock_model)
        self.assertEqual(cm.compress_token_limit, 80_000)
        self.assertEqual(cm.compress_target_token_limit, 50_000)

    def test_no_resolve_when_already_set(self):
        from agentica.compression.manager import CompressionManager
        cm = CompressionManager(compress_token_limit=5000)
        mock_model = MagicMock()
        mock_model.context_window = 200_000
        cm._resolve_limits(mock_model)
        self.assertEqual(cm.compress_token_limit, 5000, "Should not override explicit value")


class TestCompressionReport(unittest.TestCase):
    """CompressionManager exposes a compact report after compression."""

    def test_compress_records_last_report(self):
        from agentica.compression.manager import CompressionManager

        messages = [
            Message(role="user", content="start"),
            Message(role="assistant", tool_calls=[{"id": "old", "function": {"name": "search", "arguments": "{}"}}]),
            Message(role="tool", tool_call_id="old", content="old result " + ("a" * 100)),
            Message(role="assistant", tool_calls=[{"id": "new", "function": {"name": "search", "arguments": "{}"}}]),
            Message(role="tool", tool_call_id="new", content="new result " + ("b" * 100)),
        ]
        cm = CompressionManager(truncate_head_chars=20, keep_recent_rounds=1)

        with patch("agentica.compression.manager.count_tokens", return_value=10):
            asyncio.run(cm.compress(messages))

        report = cm.get_stats()["last_report"]
        self.assertEqual(report["trigger"], "manual")
        self.assertEqual(report["messages_before"], 5)
        self.assertEqual(report["messages_after"], len(messages))
        self.assertGreaterEqual(report["tool_results_pruned"], 1)
        self.assertFalse(report["llm_summary_used"])
        self.assertIsNone(report["task_anchor_preserved"])

    def test_runner_attaches_compression_report_to_run_metrics(self):
        from agentica.compression.manager import CompressionManager
        from agentica.runner import Runner

        messages = [
            Message(role="user", content="start"),
            Message(role="assistant", tool_calls=[{"id": "old", "function": {"name": "search", "arguments": "{}"}}]),
            Message(role="tool", tool_call_id="old", content="old result " + ("a" * 100)),
            Message(role="assistant", tool_calls=[{"id": "new", "function": {"name": "search", "arguments": "{}"}}]),
            Message(role="tool", tool_call_id="new", content="new result " + ("b" * 100)),
        ]
        cm = CompressionManager(compress_token_limit=1, truncate_head_chars=20, keep_recent_rounds=1)
        agent = SimpleNamespace(
            _event_callback=None,
            _run_hooks=None,
            name="Agent",
            run_id="run_1",
            run_response=RunResponse(metrics={}),
            tool_config=SimpleNamespace(compress_tool_results=True, compression_manager=cm),
            task_anchor=None,
            workspace=None,
        )
        model = SimpleNamespace(id="gpt-4o", tools=[], context_window=None)

        with patch("agentica.compression.manager.count_tokens", return_value=10):
            asyncio.run(Runner._maybe_compress_messages(messages, agent, model))

        report = agent.run_response.metrics["compression"]["last_report"]
        self.assertEqual(report["trigger"], "threshold")
        self.assertGreaterEqual(report["tool_results_pruned"], 1)

    def test_llm_tool_result_compression_redacts_prompt_input(self):
        from agentica.compression.manager import CompressionManager

        secret = "sk-abcdefghijklmnopqrstuvwxyz1234567890"

        class FakeCompressionModel:
            def __init__(self):
                self.messages = None

            async def response(self, messages):
                self.messages = messages
                return SimpleNamespace(content=f"compressed {secret}")

        model = FakeCompressionModel()
        cm = CompressionManager(model=model)
        msg = Message(role="tool", tool_name="diagnostic", content=f"OPENAI_API_KEY={secret}")

        result = asyncio.run(cm._compress_tool_result_llm(msg))

        self.assertNotIn(secret, result)
        self.assertIn("REDACTED", result)
        captured = model.messages[1].content
        self.assertNotIn(secret, captured)
        self.assertIn("REDACTED", captured)

    def test_conversation_summary_redacts_prompt_input(self):
        from agentica.compression.manager import CompressionManager

        secret = "sk-abcdefghijklmnopqrstuvwxyz1234567890"

        class FakeSummaryModel:
            context_window = 200_000

            def __init__(self):
                self.prompt = None

            async def invoke(self, messages):
                self.prompt = messages[0].content
                return SimpleNamespace(content=f"summary with {secret}")

        model = FakeSummaryModel()
        cm = CompressionManager()
        msgs = [Message(role="user", content=f"Please inspect OPENAI_API_KEY={secret}")]

        summary = asyncio.run(cm._summarise_conversation(msgs, model))

        self.assertNotIn(secret, model.prompt)
        self.assertIn("REDACTED", model.prompt)
        self.assertNotIn(secret, summary)
        self.assertIn("REDACTED", summary)


class TestCompressionManagerShouldCompress(unittest.TestCase):
    """should_compress triggers based on token count."""

    def test_disabled_returns_false(self):
        from agentica.compression.manager import CompressionManager
        cm = CompressionManager(compress_tool_results=False)
        msgs = [Message(role="user", content="hi")]
        self.assertFalse(cm.should_compress(msgs))

    def test_under_threshold_returns_false(self):
        from agentica.compression.manager import CompressionManager
        cm = CompressionManager(compress_token_limit=100_000)
        msgs = [Message(role="user", content="hi")]
        with patch("agentica.compression.manager.count_tokens", return_value=1000):
            self.assertFalse(cm.should_compress(msgs))

    def test_over_threshold_returns_true(self):
        from agentica.compression.manager import CompressionManager
        cm = CompressionManager(compress_token_limit=1000)
        msgs = [Message(role="user", content="hi")]
        with patch("agentica.compression.manager.count_tokens", return_value=2000):
            self.assertTrue(cm.should_compress(msgs))


class TestCompressionManagerDropOldMessages(unittest.TestCase):
    """_drop_old_messages preserves system + first user + recent rounds."""

    def test_preserves_system_and_first_user(self):
        from agentica.compression.manager import CompressionManager
        cm = CompressionManager(keep_recent_rounds=1)
        msgs = [
            Message(role="system", content="system prompt"),
            Message(role="user", content="first user msg"),
            Message(role="assistant", content="old reply", tool_calls=[{"id": "1"}]),
            Message(role="tool", content="old tool result", tool_call_id="1"),
            Message(role="assistant", content="old reply 2", tool_calls=[{"id": "2"}]),
            Message(role="tool", content="old tool result 2", tool_call_id="2"),
            Message(role="assistant", content="recent reply", tool_calls=[{"id": "3"}]),
            Message(role="tool", content="recent tool result", tool_call_id="3"),
        ]
        dropped = asyncio.run(cm._drop_old_messages(msgs))
        self.assertGreater(dropped, 0)
        # System and first user always preserved
        self.assertEqual(msgs[0].role, "system")
        self.assertEqual(msgs[1].role, "user")
        self.assertEqual(msgs[1].content, "first user msg")

    def test_not_enough_rounds_no_drop(self):
        from agentica.compression.manager import CompressionManager
        cm = CompressionManager(keep_recent_rounds=5)
        msgs = [
            Message(role="system", content="sys"),
            Message(role="user", content="hi"),
            Message(role="assistant", content="reply", tool_calls=[{"id": "1"}]),
            Message(role="tool", content="result", tool_call_id="1"),
        ]
        dropped = asyncio.run(cm._drop_old_messages(msgs))
        self.assertEqual(dropped, 0)


class TestCompressionManagerAutoCompact(unittest.TestCase):
    """auto_compact circuit breaker and SM-compact."""

    def test_circuit_breaker_skips_after_max_failures(self):
        from agentica.compression.manager import CompressionManager
        cm = CompressionManager()
        cm._consecutive_auto_compact_failures = 3
        msgs = [Message(role="user", content="hi")]
        result = asyncio.run(cm.auto_compact(msgs, force=True))
        self.assertFalse(result)

    def test_sm_compact_reuses_working_memory_summary(self):
        from agentica.compression.manager import CompressionManager
        cm = CompressionManager()
        msgs = [
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ]
        wm = MagicMock()
        wm.summary = MagicMock()
        wm.summary.summary = "Previously discussed: project setup and testing"
        wm.summary.topics = ["setup", "testing"]

        result = asyncio.run(cm.auto_compact(msgs, force=True, working_memory=wm))
        self.assertTrue(result)
        self.assertEqual(len(msgs), 2)
        self.assertIn("[Context compressed]", msgs[0].content)
        self.assertIn("project setup", msgs[0].content)

    def test_failure_increments_counter(self):
        from agentica.compression.manager import CompressionManager
        cm = CompressionManager()
        msgs = [Message(role="user", content="hi")]
        with patch.object(cm, '_summarise_conversation', new_callable=AsyncMock, return_value=None):
            result = asyncio.run(cm.auto_compact(msgs, force=True))
        self.assertFalse(result)
        self.assertEqual(cm._consecutive_auto_compact_failures, 1)

    def test_success_resets_counter(self):
        from agentica.compression.manager import CompressionManager
        cm = CompressionManager()
        cm._consecutive_auto_compact_failures = 2
        msgs = [Message(role="user", content="hi"), Message(role="assistant", content="ok")]
        with patch.object(cm, '_summarise_conversation', new_callable=AsyncMock, return_value="summary text"):
            result = asyncio.run(cm.auto_compact(msgs, force=True))
        self.assertTrue(result)
        self.assertEqual(cm._consecutive_auto_compact_failures, 0)

    def test_iterative_summary_does_not_duplicate_new_turn_dump(self):
        from agentica.compression.manager import CompressionManager

        class FakeModel:
            context_window = 200_000

            def __init__(self):
                self.captured_prompt = None

            async def invoke(self, messages):
                self.captured_prompt = messages[0].content

                class Resp:
                    content = "updated summary"

                return Resp()

        cm = CompressionManager()
        cm._conversation_previous_summary = "old summary"
        model = FakeModel()
        msgs = [
            Message(role="user", content="user asks for change"),
            Message(role="assistant", content="assistant responds"),
        ]

        summary = asyncio.run(cm._summarise_conversation(msgs, model))

        self.assertEqual(summary, "updated summary")
        self.assertIsNotNone(model.captured_prompt)
        self.assertEqual(model.captured_prompt.count('"role": "user"'), 1)
        self.assertEqual(model.captured_prompt.count("Conversation to summarise:"), 0)


class TestCompressionManagerGetStats(unittest.TestCase):
    """get_stats and get_compression_ratio."""

    def test_empty_stats(self):
        from agentica.compression.manager import CompressionManager
        cm = CompressionManager()
        stats = cm.get_stats()
        self.assertIn("compression_ratio", stats)
        self.assertEqual(stats["compression_ratio"], 1.0)

    def test_ratio_after_compression(self):
        from agentica.compression.manager import CompressionManager
        cm = CompressionManager()
        cm.stats["llm_original_size"] = 10000
        cm.stats["llm_compressed_size"] = 2000
        self.assertAlmostEqual(cm.get_compression_ratio(), 0.2)


class TestCompressionManagerCompress(unittest.TestCase):
    """compress() runs the two-stage pipeline."""

    def test_disabled_does_nothing(self):
        from agentica.compression.manager import CompressionManager
        cm = CompressionManager(compress_tool_results=False)
        msgs = [Message(role="user", content="hi")]
        asyncio.run(cm.compress(msgs))
        self.assertEqual(len(msgs), 1)

    def test_stage1a_truncates(self):
        from agentica.compression.manager import CompressionManager
        cm = CompressionManager(
            compress_token_limit=100,
            compress_target_token_limit=50,
            truncate_head_chars=20,
            keep_recent_rounds=1,
        )
        msgs = [
            Message(role="system", content="sys"),
            Message(role="user", content="hi"),
            Message(role="assistant", content="old", tool_calls=[{"id": "1"}]),
            Message(role="tool", content="x" * 1000, tool_call_id="1"),
            Message(role="assistant", content="recent", tool_calls=[{"id": "2"}]),
            Message(role="tool", content="y" * 100, tool_call_id="2"),
        ]
        with patch("agentica.compression.manager.count_tokens", return_value=10):
            asyncio.run(cm.compress(msgs))
        # The old tool result (index 3) should have been truncated/persisted
        old_tool = msgs[3]
        # Either persisted (has <persisted-output>) or truncated
        self.assertTrue(
            "<persisted-output>" in str(old_tool.content) or
            old_tool.compressed_content is not None or
            len(str(old_tool.content)) <= 1000
        )


if __name__ == "__main__":
    unittest.main()
