# -*- coding: utf-8 -*-
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

from agentica.compression.manager import CompressionManager
from agentica.compression.notes import (
    compose_transcript_digest,
    ensure_rollover_notes,
    notes_are_ready,
    persist_notes,
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
        self.assertNotIn("monsoon", text)
        self.assertNotIn("background dump", text)
        self.assertNotIn("## Facts", text)

    def test_keeps_needle_in_the_tail_of_a_long_tool_result(self):
        pad = "noise " * 2000
        text = compose_transcript_digest([
            Message(role="tool", content=pad + "\nSECRET_LEASE=hv-9917-omega\n"),
        ])
        self.assertIn("SECRET_LEASE=hv-9917-omega", text)
        self.assertIn("Long excerpts", text)

    def test_persist_does_not_clobber_model_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "s.notes.md")
            Path(path).write_text("Goal: keep auth\n", encoding="utf-8")
            persist_notes(path, "# Session notes (transcript digest)\n- ZX-1\n")
            body = Path(path).read_text(encoding="utf-8")
            self.assertEqual(body, "Goal: keep auth\n")
            self.assertTrue(notes_are_ready(path))


class TestAutoCompactWritesNotes(unittest.TestCase):
    def test_empty_file_gets_a_digest(self):
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
            self.assertTrue(notes.is_file())
            body = notes.read_text(encoding="utf-8")
            self.assertIn("ZX-41827", body)
            self.assertIn("transcript digest", body)
            joined = " ".join(str(m.content) for m in msgs)
            self.assertIn("<session_notes", joined)
            self.assertIn("ZX-41827", joined)

    def test_model_authored_notes_are_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "s.notes.md")
            Path(path).write_text("Constraint: do not rewrite auth\n", encoding="utf-8")
            text = ensure_rollover_notes(
                [Message(role="user", content="工单 ZX-41827")],
                path,
                window_id=1,
            )
            self.assertEqual(text, "Constraint: do not rewrite auth\n")
            self.assertEqual(
                Path(path).read_text(encoding="utf-8"),
                "Constraint: do not rewrite auth\n",
            )
