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
    OVER_THRESHOLD = 9_000  # 90% — well past the 70% trigger
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

    def test_sanitize_tool_pairs_leaves_anthropic_transcripts_alone(self):
        """It only knows the role="tool" shape; rebuilding here would corrupt it."""
        from agentica.compression.tool_pairs import sanitize_tool_pairs
        msgs = self._conversation(2, 2)

        rebuilt = sanitize_tool_pairs(msgs)

        self.assertEqual([m.role for m in rebuilt], [m.role for m in msgs])
        self.assertFalse(any(m.role == "tool" for m in rebuilt))


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
            with patch.dict(os.environ, {"AGENTICA_PROJECTS_DIR": tmpdir}):
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
            with patch.dict(os.environ, {"AGENTICA_PROJECTS_DIR": tmpdir}):
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
            with patch.dict(os.environ, {"AGENTICA_PROJECTS_DIR": tmpdir}):
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
            Message(role="assistant", tool_calls=[{"id": "t1", "function": {"name": "ls", "arguments": "{}"}}]),
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
                {"id": "toolu_1", "function": {"name": "ls", "arguments": "{}"}},
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
