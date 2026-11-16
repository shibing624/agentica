# -*- coding: utf-8 -*-
"""Tests for agentica.compression — pruning, tool result storage, compression manager."""
import asyncio
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, AsyncMock

from agentica.model.message import Message
from agentica.model.response import ModelResponse
from agentica.run.response import RunResponse


# ===========================================================================
# prune_tool_results tests
# ===========================================================================

class TestPruneToolResults(unittest.TestCase):
    """Layer 1: keep the head and tail of older long tool results.

    Messages are laid out the way the runner really builds them: each parallel
    tool round is preceded by the assistant message that requested it, and the
    list ends on the round the model has not seen yet.
    """

    def _body(self, tag, chars=20_000):
        """Long, non-repeating text, so a prune frees measurable tokens."""
        words, i = [], 0
        while sum(len(w) + 1 for w in words) < chars:
            words.append(f"{tag}-token{i}")
            i += 1
        return " ".join(words)

    def _batch(self, contents, start=0):
        calls = [{
            "id": f"call_{start + i}", "type": "function",
            "function": {"name": "read_file", "arguments": json.dumps({"file_path": f"f{start + i}.py"})},
        } for i in range(len(contents))]
        msgs = [Message(role="assistant", content="calling tools", tool_calls=calls)]
        for i, content in enumerate(contents):
            msgs.append(Message(role="tool", tool_call_id=f"call_{start + i}", content=content))
        return msgs

    def _conversation(self, *batch_sizes):
        msgs = [Message(role="user", content="hi")]
        n = 0
        for b, size in enumerate(batch_sizes):
            msgs += self._batch([self._body(f"b{b}r{i}") for i in range(size)], start=n)
            n += size
        return msgs

    def _tools(self, msgs):
        return [m for m in msgs if m.role == "tool"]

    def _prune(self, msgs, target=None, recoverable=False):
        from agentica.compression.evict import prune_tool_results
        from agentica.utils.tokens import count_tokens
        return prune_tool_results(
            msgs, context_tokens=count_tokens(msgs, None, "gpt-4o"),
            target_tokens=target, model_id="gpt-4o", recoverable=recoverable,
        )

    def test_keeps_head_and_tail_of_every_older_long_result(self):
        from agentica.compression.evict import PRUNE_HEAD_CHARS, PRUNE_TAIL_CHARS
        msgs = self._conversation(2, 2, 1)
        originals = [m.content for m in self._tools(msgs)]

        result = self._prune(msgs)

        self.assertEqual(result.pruned, 4)
        self.assertGreater(result.freed_tokens, 0)
        for msg, original in zip(self._tools(msgs)[:4], originals[:4]):
            self.assertTrue(msg.content.startswith(original[:PRUNE_HEAD_CHARS]))
            self.assertTrue(msg.content.endswith(original[-PRUNE_TAIL_CHARS:]))
            cut = len(original) - PRUNE_HEAD_CHARS - PRUNE_TAIL_CHARS
            self.assertIn(f"{cut} chars pruned from the middle", msg.content)
        self.assertEqual(self._tools(msgs)[4].content, originals[4], "the unseen batch survives")

    def test_a_pruned_result_is_never_pruned_again(self):
        msgs = self._conversation(2, 1)
        self._prune(msgs)
        after_first = [m.content for m in msgs]

        again = self._prune(msgs)

        self.assertEqual(again.pruned, 0)
        self.assertEqual([m.content for m in msgs], after_first)

    def test_all_or_nothing_against_the_target(self):
        """A prune that cannot reach the target is not applied at all: it would
        break the cache and still leave the request near the line."""
        msgs = self._conversation(2, 1)
        before = [m.content for m in msgs]

        missed = self._prune(msgs, target=1)

        self.assertEqual(missed.pruned, 0)
        self.assertEqual([m.content for m in msgs], before, "history must be byte-identical")

        reached = self._prune(msgs, target=1_000_000)
        self.assertEqual(reached.pruned, 2)

    def test_short_results_are_left_alone(self):
        from agentica.compression.evict import PRUNE_MIN_CHARS
        msgs = [Message(role="user", content="hi")]
        msgs += self._batch(["x" * PRUNE_MIN_CHARS], start=0)
        msgs += self._batch([self._body("live")], start=1)

        self.assertEqual(self._prune(msgs).pruned, 0)

    def test_results_already_spilled_to_disk_are_left_alone(self):
        spilled = "<persisted-output>\n" + self._body("spill")
        msgs = [Message(role="user", content="hi")]
        msgs += self._batch([spilled], start=0)
        msgs += self._batch([self._body("live")], start=1)

        self.assertEqual(self._prune(msgs).pruned, 0)
        self.assertEqual(msgs[2].content, spilled)

    def test_tool_call_arguments_are_never_rewritten(self):
        msgs = self._conversation(1, 1)
        msgs[1].tool_calls[0]["function"]["arguments"] = json.dumps({"content": "z " * 4000})
        calls_before = [json.dumps(m.tool_calls) for m in msgs if m.tool_calls]

        self._prune(msgs)

        self.assertEqual([json.dumps(m.tool_calls) for m in msgs if m.tool_calls], calls_before)

    def test_marker_quotes_a_handle_only_when_the_log_is_readable(self):
        with_log, without_log = self._conversation(1, 1), self._conversation(1, 1)
        self._prune(with_log, recoverable=True)
        self._prune(without_log, recoverable=False)

        self.assertIn('search_session(uuid="call_0")', with_log[2].content)
        self.assertNotIn("search_session", without_log[2].content)


class TestPruneAnthropicToolResults(unittest.TestCase):
    """Layer 1 on the Anthropic transcript shape: a round is one role="user"
    message of tool_result blocks, so the unit is a block, not a message."""

    def _body(self, tag):
        return " ".join(f"{tag}-token{i}" for i in range(3000))

    def _round(self, tag, size):
        ids = [f"toolu_{tag}_{i}" for i in range(size)]
        assistant = Message(role="assistant", content="calling tools", tool_calls=[{
            "type": "function", "id": call_id,
            "function": {"name": "read_file", "arguments": json.dumps({"file_path": f"{tag}_{i}.py"})},
        } for i, call_id in enumerate(ids)])
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

    def test_older_blocks_are_pruned_and_the_unseen_round_survives(self):
        from agentica.compression.evict import prune_tool_results
        msgs = self._conversation(3, 3, 4)
        live_before = [b["content"] for b in self._blocks(msgs)[6:]]

        result = prune_tool_results(
            msgs, context_tokens=1_000_000, target_tokens=None,
            model_id="gpt-4o", recoverable=True,
        )

        self.assertEqual(result.pruned, 6)
        blocks = self._blocks(msgs)
        self.assertTrue(all("pruned from the middle" in b["content"] for b in blocks[:6]))
        self.assertIn('search_session(uuid="toolu_r0_0")', blocks[0]["content"])
        self.assertEqual([b["content"] for b in blocks[6:]], live_before)


# ===========================================================================
# tool_result_storage tests
# ===========================================================================

class TestToolCallArgumentShrinking(unittest.TestCase):
    """JSON-safe shrinking for assistant tool_call arguments."""

    def _write_file_call(self, content, *, call_id="call_1", path="a.txt"):
        return [
            Message(role="user", content=f"write {path}"),
            Message(role="assistant", tool_calls=[{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": json.dumps({"content": content, "path": path}),
                },
            }]),
            Message(role="tool", tool_call_id=call_id, content="ok"),
        ]

    def _content_of(self, messages, assistant_index):
        return json.loads(
            messages[assistant_index].tool_calls[0]["function"]["arguments"]
        )["content"]

    def test_anthropic_tool_use_input_and_thinking_survive_a_prune(self):
        """Claude keeps the payload in tool_use.input next to a signed thinking
        block; neither may change when the old round's result is pruned."""
        from agentica.compression.evict import prune_tool_results

        old_payload = "z " * 4000
        old = Message(
            role="assistant",
            content=[
                {"type": "thinking", "thinking": "plan", "signature": "sig"},
                {
                    "type": "tool_use",
                    "id": "call_old",
                    "name": "write_file",
                    "input": {"path": "old.txt", "content": old_payload},
                },
            ],
            tool_calls=[{
                "id": "call_old",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": json.dumps({"path": "old.txt", "content": old_payload}),
                },
            }],
        )
        messages = [
            Message(role="user", content="first"),
            old,
            Message(role="user", content=[{
                "type": "tool_result", "tool_use_id": "call_old",
                "content": " ".join(f"old-result-{i}" for i in range(3000)),
            }]),
            Message(role="user", content="second"),
            Message(role="assistant", content="done"),
        ]
        content_before = json.dumps(old.content)

        result = prune_tool_results(
            messages, context_tokens=1_000_000, target_tokens=None, model_id="gpt-4o",
        )

        self.assertEqual(result.pruned, 1)
        self.assertIn("pruned from the middle", messages[2].content[0]["content"])
        self.assertEqual(json.dumps(old.content), content_before)
        self.assertEqual(self._content_of(messages, 1), old_payload)

    def test_omitted_arg_is_an_object_not_a_string(self):
        """A string marker is a tool argument. The stand-in must not be one."""
        from agentica.compression.tool_call_args import OMITTED_ARG_KEY, omitted_tool_arg

        marker = omitted_tool_arg(1183)
        self.assertIsInstance(marker, dict)
        self.assertEqual(marker, {OMITTED_ARG_KEY: 1183})

    def test_upgrade_rewrites_old_string_marker_even_when_roomy(self):
        """Live JSONL still has the string form. Leave it and the model copies it."""
        from agentica.compression.evict import prune_tool_results, upgrade_legacy_omitted_args
        from agentica.compression.tool_call_args import omitted_tool_arg

        old = "<evicted-tool-arg chars=1183>"
        messages = [
            Message(role="user", content="ping"),
            Message(role="assistant", tool_calls=[{
                "id": "s1",
                "type": "function",
                "function": {
                    "name": "write_note",
                    "arguments": json.dumps({"target": "peer", "message": old}),
                },
            }]),
            Message(role="tool", tool_call_id="s1", content="queued"),
            Message(role="user", content="next"),
            Message(role="assistant", content="ok"),
        ]

        n = upgrade_legacy_omitted_args(messages)
        self.assertGreaterEqual(n, 1)
        parsed = json.loads(messages[1].tool_calls[0]["function"]["arguments"])
        self.assertEqual(parsed["message"], omitted_tool_arg(1183))
        self.assertIsInstance(parsed["message"], dict)

        prune_tool_results(
            messages, context_tokens=100, target_tokens=None, model_id="gpt-4o",
        )
        parsed = json.loads(messages[1].tool_calls[0]["function"]["arguments"])
        self.assertEqual(parsed["message"], omitted_tool_arg(1183))

    def test_upgrade_rewrites_anthropic_input_and_drops_thinking(self):
        """Mutating tool_use.input without dropping thinking 400s Claude."""
        from agentica.compression.evict import upgrade_legacy_omitted_args
        from agentica.compression.tool_call_args import omitted_tool_arg

        old = "<evicted-tool-arg chars=1183>"
        msg = Message(
            role="assistant",
            content=[
                {"type": "thinking", "thinking": "plan", "signature": "sig"},
                {
                    "type": "tool_use",
                    "id": "s1",
                    "name": "write_note",
                    "input": {"target": "peer", "message": old},
                },
            ],
            tool_calls=[{
                "id": "s1",
                "type": "function",
                "function": {
                    "name": "write_note",
                    "arguments": json.dumps({"target": "peer", "message": old}),
                },
            }],
        )

        self.assertGreaterEqual(upgrade_legacy_omitted_args([msg]), 1)
        self.assertFalse(any(b.get("type") == "thinking" for b in msg.content))
        tool_use = next(b for b in msg.content if b.get("type") == "tool_use")
        self.assertEqual(tool_use["input"]["message"], omitted_tool_arg(1183))
        self.assertEqual(
            json.loads(msg.tool_calls[0]["function"]["arguments"])["message"],
            omitted_tool_arg(1183),
        )


class TestSessionLogToolCallHandle(unittest.TestCase):
    """The handle in a pruned result's marker must resolve to the full result."""

    def test_find_entries_resolves_a_tool_call_id(self):
        from agentica.memory.session_log import SessionLog

        with tempfile.TemporaryDirectory() as tmp:
            log = SessionLog("s-handle", base_dir=tmp)
            log.append("user", "run it")
            log.append("tool", "FULL-OUTPUT-XYZ", tool_call_id="call_h1", tool_name="execute")

            hits = log.find_entries("call_h1")

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["type"], "tool")
        self.assertIn("FULL-OUTPUT-XYZ", hits[0]["content"])

    def test_a_relogged_tool_row_is_not_ambiguous(self):
        from agentica.memory.session_log import SessionLog

        with tempfile.TemporaryDirectory() as tmp:
            log = SessionLog("s-handle2", base_dir=tmp)
            log.append("tool", "same", tool_call_id="call_h2", tool_name="execute")
            log.append("tool", "same", tool_call_id="call_h2", tool_name="execute")

            self.assertEqual(len(log.find_entries("call_h2")), 1)

    def test_handle_survives_a_compact_that_shadows_the_original_row(self):
        from agentica.memory.session_log import SessionLog

        with tempfile.TemporaryDirectory() as tmp:
            log = SessionLog("s-handle3", base_dir=tmp)
            log.append("user", "fetch the report")
            log.append("tool", "FULL-OUTPUT-XYZ", tool_call_id="toolu_h3", tool_name="execute")
            log.append_compact_boundary("", window_id=1)
            # The preserved tail is re-logged: same question, and on the
            # Anthropic wire the result comes back as a user row with no id.
            log.append("user", "fetch the report")
            log.append("user", '[{"type": "tool_result", "tool_use_id": "toolu_h3", "content": "[evicted]"}]')

            hits = log.find_entries("toolu_h3")

        self.assertEqual(len(hits), 1)
        self.assertIn("FULL-OUTPUT-XYZ", hits[0]["content"])


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

    def test_already_shrunk_is_left_alone(self):
        from agentica.compression.tool_result_storage import maybe_persist_result
        body = "<persisted-output>\npreview\n</persisted-output>"
        result = maybe_persist_result("execute", "call_x", body, max_result_size_chars=10)
        self.assertEqual(result, body)

    def test_persist_failure_still_returns_a_truncated_preview(self):
        from agentica.compression.tool_result_storage import maybe_persist_result
        def boom(*_a, **_k):
            raise RuntimeError("disk full")
        with patch(
            "agentica.compression.tool_result_storage._maybe_persist_result_inner",
            boom,
        ):
            out = maybe_persist_result(
                "execute", "call_fail", "Z" * 8000, max_result_size_chars=50,
            )
        self.assertIn("<truncated-output>", out)
        self.assertNotIn("Z" * 8000, out)

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
            self.assertIn("lines", result)

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

    def test_incomplete_header_does_not_invite_read_file(self):
        from agentica.compression.tool_result_storage import _build_persisted_message
        msg = _build_persisted_message(
            "/path/to/file.txt", "preview", size_bytes=64 * 1024 * 1024,
            n_lines=10, incomplete=True,
        )
        self.assertIn("INCOMPLETE", msg)
        self.assertIn("Do not read_file", msg)
        self.assertNotIn("Full output saved", msg)
        self.assertNotIn("Use read_file", msg)

    def test_complete_spill_still_points_at_read_file(self):
        from agentica.compression.tool_result_storage import _build_persisted_message
        msg = _build_persisted_message("/path/to/file.txt", "preview")
        self.assertIn("Full output saved", msg)
        self.assertIn("Use read_file", msg)
        self.assertNotIn("INCOMPLETE", msg)


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


class TestAutoCompactPreservesRequiredMessages(unittest.TestCase):
    """auto_compact must not clear the system prompt or the pending turn."""

    def _compact(self, msgs):
        from agentica.compression.manager import CompressionManager
        cm = CompressionManager()
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
        self.assertIn("current question", msgs[-1].content)

    def test_never_ends_on_an_assistant_turn(self):
        msgs = self._compact([
            Message(role="user", content="q1"),
            Message(role="assistant", content="a1"),
            Message(role="user", content="q2"),
        ])
        self.assertNotEqual(msgs[-1].role, "assistant")

    def test_mid_turn_tool_pairing_is_kept(self):
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

    def test_old_turns_are_dropped_without_a_summary(self):
        msgs = self._compact([
            Message(role="user", content="old question"),
            Message(role="assistant", content="old answer"),
            Message(role="user", content="current question"),
        ])
        joined = " ".join(str(m.content) for m in msgs)
        self.assertIn("<context_window>", joined)
        self.assertIn("current question", joined)
        self.assertFalse(any(m.role == "assistant" for m in msgs))
        self.assertNotIn("[Context compressed]", joined)

    def test_anthropic_tool_round_is_not_mistaken_for_the_pending_question(self):
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
        self.assertIn("current question", tail[0].content)
        self.assertEqual(tail[1].tool_calls[0]["id"], "toolu_1")


class TestCompressionManagerAutoCompact(unittest.TestCase):
    """auto_compact installs a new window; no LLM, no SM-compact."""

    def test_new_window_does_not_call_an_llm(self):
        from agentica.compression.manager import CompressionManager
        cm = CompressionManager()
        msgs = [
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
            Message(role="user", content="now"),
        ]
        result = asyncio.run(cm.auto_compact(msgs, force=True))
        self.assertTrue(result)
        self.assertEqual(cm.window_id, 1)
        self.assertTrue(any("<context_window>" in str(m.content) for m in msgs))
        self.assertNotIn("[Context compressed]", " ".join(str(m.content) for m in msgs))

    def test_dropping_trailing_turn_leaves_a_blank_page(self):
        from agentica.compression.manager import CompressionManager
        cm = CompressionManager()
        msgs = [
            Message(role="system", content="sys"),
            Message(role="user", content="old question"),
            Message(role="assistant", content="old answer"),
            Message(role="user", content="current question"),
        ]
        result = asyncio.run(cm.auto_compact(msgs, force=True, keep_trailing_turn=False))
        self.assertTrue(result)
        self.assertEqual([m.role for m in msgs], ["system", "user"])
        self.assertIn("New context window started", msgs[-1].content)
        self.assertFalse(
            any(m.role == "user" and m.content == "current question" for m in msgs)
        )

    def test_should_native_compact_is_always_false(self):
        from agentica.compression.manager import CompressionManager
        cm = CompressionManager()
        model = SimpleNamespace(context_window=200_000, id="gpt-4o", supports_native_compaction=True)
        self.assertFalse(cm.should_native_compact([], model))

    def test_threshold_still_uses_window_ratio(self):
        from agentica.compression.manager import CompressionManager

        cm = CompressionManager()
        model = SimpleNamespace(context_window=8_192, id="gpt-4")
        self.assertTrue(cm.should_auto_compact([], model, context_tokens=8_000))
        self.assertFalse(cm.should_auto_compact([], model, context_tokens=7_000))


class TestCompactTokenLimit(unittest.TestCase):
    """compact_token_limit is a cap on the existing layers, not a third window."""

    def test_unset_matches_ratio(self):
        from agentica.compression.manager import (
            AUTO_COMPACT_THRESHOLD_RATIO,
            auto_compact_threshold,
            working_context_window,
        )

        window = 1_000_000
        self.assertEqual(auto_compact_threshold(window), int(window * AUTO_COMPACT_THRESHOLD_RATIO))
        self.assertEqual(working_context_window(window), window)

    def test_cap_below_window_is_absolute_layer2_trigger(self):
        from agentica.compression.manager import (
            CompressionManager,
            auto_compact_threshold,
            working_context_window,
        )

        window, cap = 1_000_000, 300_000
        self.assertEqual(auto_compact_threshold(window, cap), cap)
        self.assertEqual(working_context_window(window, cap), cap)
        cm = CompressionManager(compact_token_limit=cap)
        model = SimpleNamespace(context_window=window, id="gpt-5.6-sol")
        self.assertFalse(cm.should_auto_compact([], model, context_tokens=cap - 1))
        self.assertTrue(cm.should_auto_compact([], model, context_tokens=cap))

    def test_cap_above_window_is_ignored(self):
        from agentica.compression.manager import (
            AUTO_COMPACT_THRESHOLD_RATIO,
            auto_compact_threshold,
            working_context_window,
        )

        window, cap = 32_768, 128_000
        self.assertEqual(
            auto_compact_threshold(window, cap),
            int(window * AUTO_COMPACT_THRESHOLD_RATIO),
        )
        self.assertEqual(working_context_window(window, cap), window)

    def test_small_cap_on_32k_fires_at_cap(self):
        from agentica.compression.manager import CompressionManager, auto_compact_threshold

        window, cap = 32_768, 8_000
        self.assertEqual(auto_compact_threshold(window, cap), cap)
        cm = CompressionManager(compact_token_limit=cap)
        model = SimpleNamespace(context_window=window, id="test")
        self.assertFalse(cm.should_auto_compact([], model, context_tokens=7_999))
        self.assertTrue(cm.should_auto_compact([], model, context_tokens=8_000))

    def test_parse_rejects_junk(self):
        from agentica.compression.manager import parse_compact_token_limit

        self.assertIsNone(parse_compact_token_limit(None))
        self.assertIsNone(parse_compact_token_limit(0))
        self.assertIsNone(parse_compact_token_limit(-1))
        self.assertIsNone(parse_compact_token_limit("nope"))
        self.assertEqual(parse_compact_token_limit("8000"), 8000)
        self.assertEqual(parse_compact_token_limit(8000), 8000)


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
