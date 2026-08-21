# -*- coding: utf-8 -*-
"""Tests for agentica.compression — eviction, tool result storage, compression manager."""
import asyncio
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, AsyncMock

from agentica.model.message import Message
from agentica.model.response import ModelResponse
from agentica.run_response import RunResponse


# ===========================================================================
# evict_tool_results tests
# ===========================================================================

class TestEvictToolResults(unittest.TestCase):
    """Tests for agentica.compression.evict.evict_tool_results.

    Messages are laid out the way the runner really builds them: each parallel
    tool round is preceded by the assistant message that requested it, and the
    list ends on the round the model has not seen yet.
    """

    WINDOW = 10_000
    OVER_THRESHOLD = 9_000  # 90% — well past the 80% trigger
    UNDER_THRESHOLD = 3_000  # 30% — nothing to buy

    def _body(self, tag, words=200):
        """~1k tokens of non-repeating text, so evictions save measurable tokens."""
        return " ".join(f"{tag}-token{i}" for i in range(words))

    def _batch(self, contents):
        msgs = [Message(role="assistant", content="calling tools")]
        for content in contents:
            msgs.append(Message(role="tool", content=content))
        return msgs

    def _conversation(self, *batch_sizes):
        msgs = [Message(role="user", content="hi")]
        for b, size in enumerate(batch_sizes):
            msgs += self._batch([self._body(f"b{b}r{i}") for i in range(size)])
        return msgs

    def _tools(self, msgs):
        return [m for m in msgs if m.role == "tool"]

    def test_gate_skips_when_context_is_roomy(self):
        from agentica.compression.evict import evict_tool_results
        msgs = self._conversation(*([1] * 8))
        count = evict_tool_results(
            msgs, context_tokens=self.UNDER_THRESHOLD, context_window=self.WINDOW,
        )
        self.assertEqual(count, 0, "Nothing to buy while the window has room")
        self.assertFalse(any(m._evicted for m in self._tools(msgs)))

    def test_no_context_window_disables_eviction(self):
        from agentica.compression.evict import evict_tool_results
        msgs = self._conversation(*([1] * 8))
        count = evict_tool_results(msgs, context_tokens=9_000, context_window=0)
        self.assertEqual(count, 0)

    def test_evicts_oldest_first_and_stops_at_target(self):
        from agentica.compression.evict import evict_tool_results
        msgs = self._conversation(*([1] * 8))

        count = evict_tool_results(
            msgs, context_tokens=self.OVER_THRESHOLD, context_window=self.WINDOW,
        )

        tools = self._tools(msgs)
        self.assertGreater(count, 0)
        self.assertLess(count, len(tools) - 1, "must stop at target, not evict everything")
        # Evicted results form a prefix: recency decides what survives.
        flags = [m._evicted for m in tools]
        self.assertEqual(flags, sorted(flags, reverse=True))
        self.assertEqual(count, sum(flags))

    def test_never_evicts_the_unseen_trailing_batch(self):
        """The regression: a 6-call round lost its first result before it was sent.

        A fixed "keep the last N results" budget always loses to a batch of
        N+1, so the model re-issued the same reads forever. Eviction now
        excludes the trailing batch outright, even under extreme pressure.
        """
        from agentica.compression.evict import evict_tool_results
        msgs = self._conversation(6)

        count = evict_tool_results(
            msgs, context_tokens=1_000_000, context_window=self.WINDOW,
        )

        self.assertEqual(count, 0)
        self.assertFalse(any(m._evicted for m in self._tools(msgs)))

    def test_extreme_pressure_still_spares_the_trailing_batch(self):
        from agentica.compression.evict import evict_tool_results
        msgs = self._conversation(2, 2, 2, 6)

        evict_tool_results(msgs, context_tokens=1_000_000, context_window=self.WINDOW)

        tools = self._tools(msgs)
        self.assertTrue(all(m._evicted for m in tools[:6]), "older rounds go first")
        self.assertFalse(any(m._evicted for m in tools[6:]), "current round survives")

    def test_skips_already_evicted(self):
        from agentica.compression.evict import evict_tool_results
        msgs = self._conversation(*([1] * 8))
        for msg in self._tools(msgs):
            msg._evicted = True

        count = evict_tool_results(
            msgs, context_tokens=self.OVER_THRESHOLD, context_window=self.WINDOW,
        )

        self.assertEqual(count, 0)

    def test_skips_content_shorter_than_the_placeholder(self):
        """Replacing a tiny result would make the request bigger, not smaller."""
        from agentica.compression.evict import evict_tool_results
        msgs = [Message(role="user", content="hi")]
        msgs += self._batch(["ok"])
        msgs += self._batch([self._body("b1")])

        evict_tool_results(
            msgs, context_tokens=self.OVER_THRESHOLD, context_window=self.WINDOW,
        )

        self.assertEqual(self._tools(msgs)[0].content, "ok")

    def test_skips_results_already_spilled_to_disk(self):
        """Their path is the only handle on output too large to hold in context."""
        from agentica.compression.evict import evict_tool_results
        msgs = [Message(role="user", content="hi")]
        msgs += self._batch(["<persisted-output>\n" + self._body("big") + "\n</persisted-output>"])
        msgs += self._batch([self._body("b1")])

        evict_tool_results(
            msgs, context_tokens=self.OVER_THRESHOLD, context_window=self.WINDOW,
        )

        self.assertIn("<persisted-output>", self._tools(msgs)[0].content)

    def test_placeholder_names_the_call_so_it_can_be_re_issued(self):
        from agentica.compression.evict import evict_tool_results
        msgs = [Message(role="user", content="hi")]
        msgs.append(Message(role="assistant", content="calling"))
        msgs.append(Message(
            role="tool",
            content=self._body("old"),
            tool_call_id="call_old",
            tool_name="read_file",
            tool_args={"file_path": "run.sh", "offset": 56},
        ))
        msgs += self._batch([self._body("b1")])

        evict_tool_results(
            msgs, context_tokens=self.OVER_THRESHOLD, context_window=self.WINDOW,
        )

        placeholder = self._tools(msgs)[0].content
        self.assertIn("read_file(", placeholder)
        self.assertIn("run.sh", placeholder)
        self.assertIn("offset=56", placeholder)


class TestEvictAnthropicToolResults(unittest.TestCase):
    """Layer 1 on the Anthropic transcript shape.

    Anthropic has no role="tool" message: a whole round is packed into the
    content list of one role="user" message as tool_result blocks. Scanning
    only role="tool" meant eviction never once ran on this path.
    """

    WINDOW = 10_000
    OVER_THRESHOLD = 9_000

    def _body(self, tag, words=200):
        return " ".join(f"{tag}-token{i}" for i in range(words))

    def _round(self, tag, size, name="read_file"):
        """One assistant tool_use message plus the user message answering it."""
        ids = [f"toolu_{tag}_{i}" for i in range(size)]
        assistant = Message(
            role="assistant",
            content="calling tools",
            tool_calls=[
                {
                    "type": "function",
                    "id": call_id,
                    "function": {
                        "name": name,
                        "arguments": json.dumps({"file_path": f"{tag}_{i}.py"}),
                    },
                }
                for i, call_id in enumerate(ids)
            ],
        )
        results = Message(role="user", content=[
            {"type": "tool_result", "tool_use_id": call_id, "content": self._body(f"{tag}{i}")}
            for i, call_id in enumerate(ids)
        ])
        return [assistant, results]

    def _conversation(self, *round_sizes):
        msgs = [Message(role="user", content="hi")]
        for i, size in enumerate(round_sizes):
            msgs += self._round(f"r{i}", size)
        return msgs

    def _blocks(self, msgs):
        from agentica.compression.evict import tool_result_blocks
        return [b for m in msgs for b in tool_result_blocks(m)]

    def test_anthropic_results_are_evicted_at_all(self):
        """The regression: this whole provider path was invisible to Layer 1."""
        from agentica.compression.evict import evict_tool_results
        msgs = self._conversation(*([1] * 8))

        count = evict_tool_results(
            msgs, context_tokens=self.OVER_THRESHOLD, context_window=self.WINDOW,
        )

        self.assertGreater(count, 0)
        blocks = self._blocks(msgs)
        evicted = [b for b in blocks if b["content"].startswith("[Tool result evicted")]
        self.assertEqual(len(evicted), count)

    def test_never_evicts_the_unseen_trailing_round(self):
        from agentica.compression.evict import evict_tool_results
        msgs = self._conversation(6)

        count = evict_tool_results(
            msgs, context_tokens=1_000_000, context_window=self.WINDOW,
        )

        self.assertEqual(count, 0)

    def test_evicts_block_by_block_oldest_first(self):
        from agentica.compression.evict import evict_tool_results
        msgs = self._conversation(3, 3, 3, 4)

        evict_tool_results(msgs, context_tokens=1_000_000, context_window=self.WINDOW)

        flags = [b["content"].startswith("[Tool result evicted") for b in self._blocks(msgs)]
        self.assertEqual(flags[:9], [True] * 9, "older rounds go first")
        self.assertEqual(flags[9:], [False] * 4, "current round survives")

    def test_placeholder_resolves_the_call_through_the_assistant_message(self):
        """A tool_result block carries only tool_use_id, never the tool name."""
        from agentica.compression.evict import evict_tool_results
        msgs = self._conversation(1, 1)

        evict_tool_results(
            msgs, context_tokens=self.OVER_THRESHOLD, context_window=self.WINDOW,
        )

        placeholder = self._blocks(msgs)[0]["content"]
        self.assertIn("read_file(", placeholder)
        self.assertIn("r0_0.py", placeholder)

    def test_partially_evicted_message_is_revisited(self):
        """A round is one message but many results; the flag must not close it early."""
        from agentica.compression.evict import evict_tool_results
        msgs = self._conversation(4, 1)
        results = msgs[2]

        evict_tool_results(
            msgs, context_tokens=self.OVER_THRESHOLD, context_window=self.WINDOW,
        )
        first_pass = sum(
            b["content"].startswith("[Tool result evicted") for b in results.content
        )
        self.assertGreater(first_pass, 0)
        if first_pass < 4:
            self.assertFalse(results._evicted, "more results left to reclaim here")
            evict_tool_results(
                msgs, context_tokens=1_000_000, context_window=self.WINDOW,
            )
            self.assertTrue(all(
                b["content"].startswith("[Tool result evicted") for b in results.content
            ))
        self.assertTrue(results._evicted)


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

    def _write_file_call(self, content):
        return [
            Message(role="user", content="write file"),
            Message(role="assistant", tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": json.dumps({"content": content, "path": "a.txt"}),
                },
            }]),
            Message(role="tool", tool_call_id="call_1", content="ok"),
        ]

    def test_shrinks_assistant_tool_call_arguments_under_pressure(self):
        """A write payload lives in the assistant message, out of eviction's reach."""
        from agentica.compression.evict import shrink_tool_call_arguments

        messages = self._write_file_call("z" * 300)

        shrunk = shrink_tool_call_arguments(
            messages, context_tokens=9_000, context_window=10_000, max_string_chars=20,
        )

        self.assertEqual(shrunk, 1)
        parsed = json.loads(messages[1].tool_calls[0]["function"]["arguments"])
        self.assertEqual(parsed["content"], "z" * 20 + "...[truncated]")
        self.assertEqual(parsed["path"], "a.txt")

    def test_leaves_tool_call_arguments_alone_when_roomy(self):
        from agentica.compression.evict import shrink_tool_call_arguments

        messages = self._write_file_call("z" * 300)

        shrunk = shrink_tool_call_arguments(
            messages, context_tokens=1_000, context_window=10_000, max_string_chars=20,
        )

        self.assertEqual(shrunk, 0)
        parsed = json.loads(messages[1].tool_calls[0]["function"]["arguments"])
        self.assertEqual(parsed["content"], "z" * 300)


class TestSanitizePath(unittest.TestCase):
    """Tests for sanitize_path."""

    def test_basic_path(self):
        from agentica.compression.tool_result_storage import sanitize_path
        result = sanitize_path("/Users/test/project")
        self.assertRegex(result, r'^[a-zA-Z0-9\-]+$')

    def test_long_path_truncated_with_hash(self):
        from agentica.compression.tool_result_storage import sanitize_path, _MAX_SANITIZED_LENGTH
        long_path = "/a/b/c/" + "x" * 300
        result = sanitize_path(long_path)
        self.assertLessEqual(len(result), _MAX_SANITIZED_LENGTH + 10)  # +hash suffix
        self.assertIn("-", result)  # hash appended

    def test_special_chars_replaced(self):
        from agentica.compression.tool_result_storage import sanitize_path
        result = sanitize_path("/path/to/my project (2)/test.txt")
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
            with patch.dict(os.environ, {"AGENTICA_PROJECTS_DIR": tmpdir}):
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
            with patch.dict(os.environ, {"AGENTICA_PROJECTS_DIR": tmpdir}):
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

    def test_disk_failure_falls_back_to_truncation(self):
        from agentica.compression.tool_result_storage import maybe_persist_result
        big = "x" * 100
        with patch("agentica.compression.tool_result_storage._persist_to_disk", return_value=False):
            result = maybe_persist_result(
                "test_tool", "call_4", big,
                max_result_size_chars=50,
            )
        self.assertIn("<truncated-output>", result)
        self.assertNotIn("<persisted-output>", result)

    def test_without_a_reader_nothing_is_written_and_no_path_is_offered(self):
        """A path is only useful to a session that can open it.

        An agent assembled from business tools has no read_file and no
        execute: handing it a filesystem path loses the data *and* invites it
        to call a tool it does not have.
        """
        from agentica.compression.tool_result_storage import (
            get_tool_results_dir, maybe_persist_result,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"AGENTICA_PROJECTS_DIR": tmpdir}):
                result = maybe_persist_result(
                    "claw_list_customers", "call_5", "x" * 100,
                    max_result_size_chars=50, cwd="/test/project",
                    recoverable=False,
                )
                spill_dir = get_tool_results_dir(cwd="/test/project", session_id="default")

        self.assertIn("<truncated-output>", result)
        self.assertNotIn("<persisted-output>", result)
        self.assertNotIn(tmpdir, result, "no path may be offered without a reader")
        self.assertFalse(os.path.isdir(spill_dir), "nothing should be written to disk")

    def test_with_a_reader_the_path_is_still_offered(self):
        from agentica.compression.tool_result_storage import maybe_persist_result
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"AGENTICA_PROJECTS_DIR": tmpdir}):
                result = maybe_persist_result(
                    "execute", "call_6", "x" * 100,
                    max_result_size_chars=50, cwd="/test/project",
                    recoverable=True,
                )
        self.assertIn("<persisted-output>", result)


class TestCanRecoverSpill(unittest.TestCase):
    """Which sessions may be handed a spill path."""

    def test_file_or_shell_tool_can_read_it_back(self):
        from agentica.compression.tool_result_storage import can_recover_spill
        self.assertTrue(can_recover_spill({"read_file", "web_search"}))
        self.assertTrue(can_recover_spill({"execute"}))

    def test_a_business_tool_set_cannot(self):
        from agentica.compression.tool_result_storage import can_recover_spill
        self.assertFalse(can_recover_spill({"claw_get_funnel", "claw_list_customers"}))
        self.assertFalse(can_recover_spill([]))


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


class TestEnforceToolBatchBudget(unittest.TestCase):
    """Layer 0 per-batch budget — bound one turn's fresh results by the window."""

    @staticmethod
    def _batch():
        return [
            Message(role="tool", content="alpha " * 40, tool_call_id="t1"),
            Message(role="tool", content="beta " * 400, tool_call_id="t2"),  # largest
            Message(role="tool", content="gamma " * 20, tool_call_id="t3"),
        ]

    def test_a_batch_the_window_has_room_for_is_untouched(self):
        from agentica.compression.tool_result_storage import enforce_tool_batch_budget
        msgs = self._batch()
        count = enforce_tool_batch_budget(msgs, context_window=200_000)
        self.assertEqual(count, 0)
        self.assertNotIn("output>", msgs[1].content)

    def test_no_window_is_a_no_op(self):
        """Without a window there is no way to tell a big batch from a batch
        this model has ample room for — the old fixed char budget guessed."""
        from agentica.compression.tool_result_storage import enforce_tool_batch_budget
        msgs = self._batch()
        self.assertEqual(enforce_tool_batch_budget(msgs, context_window=0), 0)
        self.assertNotIn("output>", msgs[1].content)

    def test_over_budget_shrinks_the_largest_first(self):
        from agentica.compression.tool_result_storage import enforce_tool_batch_budget
        with tempfile.TemporaryDirectory() as tmpdir:
            msgs = self._batch()
            with patch.dict(os.environ, {"AGENTICA_PROJECTS_DIR": tmpdir}):
                count = enforce_tool_batch_budget(
                    msgs, context_window=400, cwd="/test", recoverable=True,
                )
            self.assertGreater(count, 0)
            self.assertIn("<persisted-output>", msgs[1].content)

    def test_over_budget_redacts_what_it_writes(self):
        from agentica.compression.tool_result_storage import (
            enforce_tool_batch_budget, get_tool_result_path,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            secret = "ghp_abcdefghijklmnopqrstuvwxyz1234567890"
            msgs = [
                Message(role="tool", content="safe", tool_call_id="t1"),
                Message(role="tool", content=f"{secret} " + ("beta " * 400), tool_call_id="t2"),
            ]
            with patch.dict(os.environ, {"AGENTICA_PROJECTS_DIR": tmpdir}):
                count = enforce_tool_batch_budget(
                    msgs, context_window=200, cwd="/test", recoverable=True,
                )
                file_path = get_tool_result_path("t2", cwd="/test", session_id="default")
                persisted = open(file_path, encoding="utf-8").read()

        self.assertGreater(count, 0)
        self.assertNotIn(secret, msgs[1].content)
        self.assertNotIn(secret, persisted)

    def test_without_a_reader_the_batch_is_truncated_not_spilled(self):
        from agentica.compression.tool_result_storage import (
            enforce_tool_batch_budget, get_tool_results_dir,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            msgs = self._batch()
            with patch.dict(os.environ, {"AGENTICA_PROJECTS_DIR": tmpdir}):
                count = enforce_tool_batch_budget(
                    msgs, context_window=400, cwd="/test", recoverable=False,
                )
                spill_dir = get_tool_results_dir(cwd="/test", session_id="default")

        self.assertGreater(count, 0)
        self.assertIn("<truncated-output>", msgs[1].content)
        self.assertFalse(os.path.isdir(spill_dir))

    def test_an_already_shrunk_result_is_left_alone(self):
        from agentica.compression.tool_result_storage import enforce_tool_batch_budget
        msgs = [
            Message(role="tool", content="<persisted-output>already</persisted-output>", tool_call_id="t1"),
            Message(role="tool", content="beta " * 400, tool_call_id="t2"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"AGENTICA_PROJECTS_DIR": tmpdir}):
                count = enforce_tool_batch_budget(
                    msgs, context_window=200, cwd="/test", recoverable=True,
                )
        self.assertEqual(count, 1)
        self.assertEqual(msgs[0].content, "<persisted-output>already</persisted-output>")

    def test_empty_results_no_error(self):
        from agentica.compression.tool_result_storage import enforce_tool_batch_budget
        self.assertEqual(enforce_tool_batch_budget([], context_window=100_000), 0)


# ===========================================================================
# CompressionManager tests
# ===========================================================================

class TestCompressionManagerInit(unittest.TestCase):
    """CompressionManager initialization and defaults."""

    def test_defaults(self):
        from agentica.compression.manager import CompressionManager
        cm = CompressionManager()
        self.assertIsNone(cm.model)
        self.assertIsNone(cm.compress_token_limit)
        self.assertIsNone(cm.compress_target_token_limit)

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


class TestSummarisationRedaction(unittest.TestCase):
    """Secrets in the transcript must not reach the summarisation model."""

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


class TestAutoCompactPreservesRequiredMessages(unittest.TestCase):
    """auto_compact must not clear the system prompt or the pending turn.

    A blind `messages.clear()` left the rest of the run with no instructions and
    a conversation ending on an assistant turn, which providers reject with
    "does not support assistant message prefill".
    """

    def _compact(self, msgs):
        from agentica.compression.manager import CompressionManager
        cm = CompressionManager()
        with patch.object(cm, "_summarise_conversation",
                          new_callable=AsyncMock, return_value="the summary"):
            result = asyncio.run(cm.auto_compact(msgs, force=True))
        self.assertTrue(result)
        return msgs

    def test_system_prompt_survives(self):
        msgs = self._compact([
            Message(role="system", content="you are a helpful agent"),
            Message(role="user", content="old question"),
            Message(role="assistant", content="old answer"),
            Message(role="user", content="current question"),
        ])
        self.assertEqual(msgs[0].role, "system")
        self.assertEqual(msgs[0].content, "you are a helpful agent")

    def test_pending_user_question_survives_verbatim(self):
        msgs = self._compact([
            Message(role="system", content="sys"),
            Message(role="user", content="old question"),
            Message(role="assistant", content="old answer"),
            Message(role="user", content="current question"),
        ])
        self.assertEqual(msgs[-1].role, "user")
        self.assertEqual(msgs[-1].content, "current question")

    def test_never_ends_on_an_assistant_turn(self):
        msgs = self._compact([
            Message(role="user", content="q1"),
            Message(role="assistant", content="a1"),
            Message(role="user", content="q2"),
        ])
        self.assertNotEqual(msgs[-1].role, "assistant")

    def test_mid_turn_tool_pairing_is_kept(self):
        """Tool results must stay with the assistant tool_calls that produced them."""
        msgs = self._compact([
            Message(role="system", content="sys"),
            Message(role="user", content="old"),
            Message(role="assistant", content="old answer"),
            Message(role="user", content="current question"),
            Message(role="assistant", tool_calls=[{"id": "t1", "function": {"name": "glob", "arguments": "{}"}}]),
            Message(role="tool", tool_call_id="t1", content="file list"),
        ])
        self.assertEqual([m.role for m in msgs[-3:]], ["user", "assistant", "tool"])
        self.assertEqual(msgs[-1].tool_call_id, "t1")

    def test_old_turns_are_replaced_by_the_summary(self):
        msgs = self._compact([
            Message(role="user", content="old question"),
            Message(role="assistant", content="old answer"),
            Message(role="user", content="current question"),
        ])
        joined = " ".join(str(m.content) for m in msgs)
        self.assertIn("the summary", joined)
        self.assertNotIn("old answer", joined)

    def test_anthropic_tool_round_is_not_mistaken_for_the_pending_question(self):
        """Anthropic delivers a tool round as a user message of tool_result blocks.

        Cutting the tail there would keep results whose tool_use block lives in
        the assistant message the summary just replaced, which that API rejects.
        """
        msgs = self._compact([
            Message(role="system", content="sys"),
            Message(role="user", content="old question"),
            Message(role="assistant", content="old answer"),
            Message(role="user", content="current question"),
            Message(role="assistant", tool_calls=[
                {"id": "toolu_1", "function": {"name": "glob", "arguments": "{}"}},
            ]),
            Message(role="user", content=[
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "file list"},
            ]),
        ])
        tail = msgs[-3:]
        self.assertEqual([m.role for m in tail], ["user", "assistant", "user"])
        self.assertEqual(tail[0].content, "current question")
        self.assertEqual(tail[1].tool_calls[0]["id"], "toolu_1")


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
        self.assertIn("[Context compressed]", msgs[0].content)
        self.assertIn("project setup", msgs[0].content)
        # The trailing turn is kept verbatim after the summary pair.
        self.assertEqual([m.role for m in msgs], ["user", "assistant", "user", "assistant"])
        self.assertEqual(msgs[2].content, "hi")

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


    def test_anthropic_long_request_falls_back_to_streaming_summary(self):
        from agentica.compression.manager import CompressionManager

        class FakeAnthropicModel:
            context_window = 1_000_000

            def __init__(self):
                self.streamed = False
                self._agent_ref = None

            async def invoke(self, messages):
                raise ValueError(
                    "Streaming is required for operations that may take longer than 10 minutes. "
                    "See https://github.com/anthropics/anthropic-sdk-python#long-requests "
                    "for more details"
                )

            async def response_stream(self, messages):
                self.streamed = True
                yield ModelResponse(content="streamed ")
                yield ModelResponse(content="summary")

        cm = CompressionManager()
        model = FakeAnthropicModel()
        msgs = [Message(role="user", content="hi"), Message(role="assistant", content="hello")]

        result = asyncio.run(cm.auto_compact(msgs, model=model, force=True))

        self.assertTrue(result)
        self.assertTrue(model.streamed)
        self.assertEqual(cm._consecutive_auto_compact_failures, 0)
        self.assertIn("streamed summary", msgs[0].content)


class TestLayerThresholds(unittest.TestCase):
    """Layer 1 (evict) / Layer 2 (auto-compact) trigger points across windows.

    Both layers are pure ratios of the window (0.8 / 0.95), so the ordering
    layer1 < layer2 must hold for EVERY window size — the inverted-layer and
    negative-threshold regressions both come from mixing a ratio with an
    absolute buffer, which is what this class pins down.
    """

    WINDOWS = (8_192, 32_768, 128_000, 200_000, 1_000_000)

    def _layer2_trigger_point(self, window: int) -> int:
        """Bisect should_auto_compact for the smallest token count that fires."""
        from agentica.compression.manager import CompressionManager

        cm = CompressionManager()
        model = SimpleNamespace(context_window=window, id="test-model")
        lo, hi = 0, window  # below threshold / at-or-above (window always fires)
        while lo < hi:
            mid = (lo + hi) // 2
            if cm.should_auto_compact([], model, context_tokens=mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    def test_both_layers_ordered_and_within_window(self):
        from agentica.compression.evict import EVICT_THRESHOLD_RATIO
        from agentica.compression.manager import AUTO_COMPACT_THRESHOLD_RATIO

        for window in self.WINDOWS:
            layer1 = int(window * EVICT_THRESHOLD_RATIO)
            layer2 = int(window * AUTO_COMPACT_THRESHOLD_RATIO)
            with self.subTest(window=window):
                self.assertGreater(layer1, 0, "negative/zero evict threshold — small-window bug")
                self.assertLess(layer1, layer2, "layers inverted: summary would burn before free evict")
                self.assertLess(layer2, window, "threshold must leave headroom below the window")

    def test_layer2_trigger_bisects_to_ratio_and_is_monotonic(self):
        from agentica.compression.manager import AUTO_COMPACT_THRESHOLD_RATIO

        points = [self._layer2_trigger_point(w) for w in self.WINDOWS]
        for window, point in zip(self.WINDOWS, points):
            with self.subTest(window=window):
                self.assertEqual(point, int(window * AUTO_COMPACT_THRESHOLD_RATIO))
        self.assertEqual(points, sorted(points))
        self.assertEqual(len(set(points)), len(points), "trigger points must strictly increase with window")

    def test_small_window_really_full_still_triggers(self):
        """gpt-4 (8192): the old absolute buffer went negative and fired every
        turn; the ratio must still fire when the window is genuinely almost full."""
        from agentica.compression.manager import CompressionManager

        cm = CompressionManager()
        model = SimpleNamespace(context_window=8_192, id="gpt-4")
        self.assertTrue(cm.should_auto_compact([], model, context_tokens=8_000))
        self.assertFalse(cm.should_auto_compact([], model, context_tokens=7_000))


class TestNativeCompactionLimits(unittest.TestCase):
    """native_compaction_token_limit must never collapse to a degenerate value.

    Same bug family as the removed 13_000 auto-compact buffer: absolute
    headroom terms go negative on small windows, and a max(1, ...) floor then
    turns the limit into 1 — i.e. "compact every turn". Only models that set
    supports_native_compaction=True reach this code today.
    """

    WINDOWS = (8_192, 32_768, 128_000, 200_000, 1_000_000)

    def test_base_default_raises_so_declaring_models_must_implement(self):
        from agentica.model.base import Model

        with self.assertRaises(NotImplementedError):
            Model.native_compaction_token_limit(MagicMock())

    def test_responses_limit_bounded_and_monotonic(self):
        from agentica.model.openai.responses import OpenAIResponses

        prev = 0
        for window in self.WINDOWS:
            m = MagicMock()
            m.context_window = window
            m.max_output_tokens = None
            m.max_tokens = None
            limit = OpenAIResponses.native_compaction_token_limit(m)
            with self.subTest(window=window):
                self.assertGreaterEqual(limit, int(window * 0.8),
                                        "limit must never collapse below the 80% floor")
                self.assertLess(limit, window)
                self.assertGreater(limit, prev, "limit must grow with the window")
            prev = limit


class TestEvictThresholdOverride(unittest.TestCase):
    """P3-2: AGENTICA_EVICT_THRESHOLD_RATIO — the one user-facing knob."""

    def _with_env(self, value):
        if value is None:
            os.environ.pop("AGENTICA_EVICT_THRESHOLD_RATIO", None)
        else:
            os.environ["AGENTICA_EVICT_THRESHOLD_RATIO"] = value

    def tearDown(self):
        self._with_env(None)

    def test_default_is_point_eight(self):
        from agentica.compression.evict import evict_threshold_ratio
        self._with_env(None)
        self.assertEqual(evict_threshold_ratio(), 0.8)

    def test_env_override_applies(self):
        from agentica.compression.evict import evict_threshold_ratio, under_pressure
        self._with_env("0.7")
        self.assertEqual(evict_threshold_ratio(), 0.7)
        self.assertTrue(under_pressure(7_000, 10_000))
        self.assertFalse(under_pressure(6_999, 10_000))

    def test_garbage_env_falls_back_to_default(self):
        from agentica.compression.evict import evict_threshold_ratio
        self._with_env("banana")
        self.assertEqual(evict_threshold_ratio(), 0.8)

    def test_out_of_range_is_clamped_below_layer2(self):
        from agentica.compression.evict import evict_threshold_ratio
        self._with_env("0.99")
        ratio = evict_threshold_ratio()
        self.assertLess(ratio, 0.95, "must stay strictly below the Layer 2 trigger")
        self._with_env("0")
        self.assertGreater(evict_threshold_ratio(), 0.0)


class TestCompressionManagerGetStats(unittest.TestCase):
    """get_stats returns a snapshot the caller cannot mutate."""

    def test_empty_stats(self):
        from agentica.compression.manager import CompressionManager
        cm = CompressionManager()
        self.assertEqual(cm.get_stats(), {})

    def test_stats_are_copied(self):
        from agentica.compression.manager import CompressionManager
        cm = CompressionManager()
        cm.stats["auto_compact_count"] = 2
        stats = cm.get_stats()
        stats["auto_compact_count"] = 99
        self.assertEqual(cm.stats["auto_compact_count"], 2)


if __name__ == "__main__":
    unittest.main()
