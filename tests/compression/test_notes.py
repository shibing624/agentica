# -*- coding: utf-8 -*-
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

from agentica.compression.manager import CompressionManager
from agentica.compression.notes import (
    compose_transcript_digest,
    notes_are_ready,
    rollover_handover,
)
from agentica.memory.session_log import SessionLog
from agentica.model.message import Message


class TestComposeTranscriptDigest(unittest.TestCase):
    def test_keeps_user_prose_and_skips_pad(self):
        text = compose_transcript_digest([
            Message(role="system", content="sys"),
            Message(role="user", content="工单 ZX-41827，泄漏在 evict.py:412。不要改 compress.py。"),
            Message(role="assistant", content="记下了，改用 KeyDB"),
            Message(role="user", content="[pad 0001] Unrelated monsoon notes."),
            Message(role="user", content="Read this background dump 9.\n[pad 0009] Unrelated monsoon notes."),
        ])
        self.assertIn("ZX-41827", text)
        self.assertIn("不要改 compress.py", text)
        self.assertIn("改用 KeyDB", text)
        self.assertLess(text.index("ZX-41827"), text.index("改用 KeyDB"))
        self.assertIn("## Turns", text)
        self.assertIn("# Dropped span", text)
        self.assertNotIn("monsoon", text)
        self.assertNotIn("background dump", text)
        self.assertNotIn("## Facts", text)
        self.assertIn("search_session", text)

    def test_keeps_order_and_timestamps(self):
        text = compose_transcript_digest([
            Message(role="user", content="先问工单", created_at=1_725_000_000),
            Message(role="assistant", content="答工单", created_at=1_725_000_060),
            Message(role="user", content="再问路经", created_at=1_725_000_120),
        ])
        self.assertLess(text.index("先问工单"), text.index("答工单"))
        self.assertLess(text.index("答工单"), text.index("再问路经"))
        self.assertIn("2024-08-30", text)

    def test_records_tool_args_and_results(self):
        text = compose_transcript_digest([
            Message(role="user", content="查一下", created_at=100),
            Message(
                role="assistant",
                content="在搜",
                created_at=101,
                tool_calls=[{
                    "id": "c1",
                    "function": {"name": "grep", "arguments": '{"pattern":"ZX-41827"}'},
                }],
            ),
            Message(
                role="tool",
                content="hit in evict.py",
                tool_name="grep",
                created_at=102,
            ),
        ])
        self.assertIn("## Tools", text)
        self.assertIn("grep args", text)
        self.assertIn("ZX-41827", text)
        self.assertIn("grep result", text)
        self.assertIn("evict.py", text)

    def test_keeps_needle_in_the_tail_of_a_long_tool_result(self):
        pad = "noise " * 2000
        text = compose_transcript_digest([
            Message(role="tool", content=pad + "\nSECRET_LEASE=hv-9917-omega\n"),
        ])
        self.assertIn("SECRET_LEASE=hv-9917-omega", text)
        self.assertIn("## Tools", text)


class TestRolloverHandover(unittest.TestCase):
    def test_model_authored_notes_are_kept_and_digest_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "s.notes.md")
            Path(path).write_text("Constraint: do not rewrite auth\n", encoding="utf-8")
            notes, span = rollover_handover(
                [Message(role="user", content="工单 ZX-41827")],
                path,
            )
            self.assertEqual(notes, "Constraint: do not rewrite auth\n")
            self.assertIsNone(span)
            self.assertEqual(
                Path(path).read_text(encoding="utf-8"),
                "Constraint: do not rewrite auth\n",
            )
            self.assertTrue(notes_are_ready(path))

    def test_empty_notes_digest_is_not_written_to_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "s.notes.md")
            notes, span = rollover_handover(
                [Message(role="user", content="工单 ZX-41827")],
                path,
            )
            self.assertIsNone(notes)
            self.assertIn("ZX-41827", span)
            self.assertFalse(Path(path).exists())
            self.assertFalse(notes_are_ready(path))


class TestAutoCompactWritesNotes(unittest.TestCase):
    def test_empty_file_stays_empty_and_injects_dropped_span(self):
        import asyncio

        with tempfile.TemporaryDirectory() as tmp:
            slog = SessionLog("notes-roll", base_dir=tmp)
            slog.append("user", "工单 ZX-41827")
            cm = CompressionManager()
            agent = type("A", (), {"_session_log": slog})()
            model = type("M", (), {"id": "gpt-4o", "context_window": 128_000})()
            model._agent_ref = lambda: agent
            msgs = [
                Message(role="system", content="sys"),
                Message(role="user", content="工单 ZX-41827，不要改 compress.py。"),
                Message(role="assistant", content="记下了"),
                Message(role="user", content="继续"),
            ]
            asyncio.run(cm.auto_compact(msgs, model=model, force=True))
            notes = Path(slog.path).with_name("notes-roll.notes.md")
            self.assertFalse(notes.exists())
            joined = " ".join(str(m.content) for m in msgs)
            self.assertIn("<dropped_span", joined)
            self.assertNotIn("<session_notes", joined)
            self.assertIn("ZX-41827", joined)
            self.assertIn("继续", joined)
