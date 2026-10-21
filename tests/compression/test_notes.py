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

    def test_header_only_names_paths_that_exist(self):
        """The header may not advertise tools the agent does not have.

        With a session log the model can search it and the notes are injected
        beside the digest. With no session log, ``search_session`` answers "No
        session log on this agent" — naming it sends the model after nothing,
        on the one window whose skim is the only copy of those turns.
        """
        msgs = [
            Message(role="user", content="工单 ZX-41827"),
            Message(role="assistant", content="记下了，改用 KeyDB"),
        ]
        with_path = compose_transcript_digest(msgs, notes_path="/tmp/s.notes.md")
        without = compose_transcript_digest(msgs)
        self.assertIn("search_session", with_path)
        self.assertIn("session notes", with_path)
        self.assertNotIn("search_session", without)
        self.assertNotIn("session notes", without)
        # Both keep the content; only the instructions differ.
        for text in (with_path, without):
            self.assertIn("ZX-41827", text)
            self.assertIn("KeyDB", text)

    def test_keeps_order_without_timestamps(self):
        text = compose_transcript_digest([
            Message(role="user", content="先问工单", created_at=1_725_000_000),
            Message(role="assistant", content="答工单", created_at=1_725_000_060),
            Message(role="user", content="再问路经", created_at=1_725_000_120),
        ])
        self.assertLess(text.index("先问工单"), text.index("答工单"))
        self.assertLess(text.index("答工单"), text.index("再问路经"))
        self.assertNotIn("2024-08-30", text)
        self.assertNotIn("1970", text)

    def test_strips_preamble_and_keeps_the_folded_question(self):
        """The question survives, and so does what a previous cut skimmed.

        It used to assert the opposite for ``ZX-41827`` (count == 0): the
        strip treated the prior digest body as chrome. That discarded the only
        copy of those turns on the next cut — fatal for an SDK agent with no
        session log, where the prompt is the history. The chrome
        (``<context_window>`` markers, headers) still must not survive.
        """
        text = compose_transcript_digest([
            Message(
                role="user",
                content=(
                    "<context_window>\nCurrent context window 1.\n"
                    "</context_window>\n\n"
                    "<dropped_span>\n# Dropped span\n"
                    "## Turns\n"
                    "- user: 工单 ZX-41827\n</dropped_span>\n\n"
                    "现在怎么办？"
                ),
            ),
            Message(role="assistant", content="先查 JSONL"),
        ])
        self.assertIn("现在怎么办？", text)
        self.assertIn("先查 JSONL", text)
        self.assertLess(text.index("ZX-41827"), text.index("现在怎么办？"))
        self.assertLess(text.index("现在怎么办？"), text.index("先查 JSONL"))
        # Carried forward, not dropped: the prompt is the only copy.
        self.assertEqual(text.count("ZX-41827"), 1)
        # Chrome still goes: markers and section titles are not content.
        self.assertNotIn("<context_window>", text)
        self.assertNotIn("<dropped_span>", text)
        self.assertEqual(text.count("# Dropped span"), 1)
        self.assertEqual(text.count("## Turns"), 1)

    def test_long_assistant_turn_keeps_its_closing_ask(self):
        """The tail of an assistant turn is the ask the next request answers.

        Head-only clipping dropped it: the model reasoned out loud, ended with
        「要我把 notes.md 里对应的行号一并订正吗？」, the user replied "ok", and
        the new window showed neither the question nor what "ok" agreed to.
        """
        ask = "要我把 notes.md 里对应的行号一并订正吗？"
        text = compose_transcript_digest([
            Message(role="assistant", content="全部核对完毕。结论：" + ("细节 " * 300) + ask),
        ])
        self.assertIn(ask, text)
        self.assertIn("全部核对完毕", text)

    def test_long_user_turn_keeps_its_closing_ask(self):
        ask = "先答这一条，别的不急。"
        text = compose_transcript_digest([
            Message(role="user", content=("把日志贴一下 " * 200) + ask),
        ])
        self.assertIn(ask, text)
        self.assertIn("把日志贴一下", text)

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
    @staticmethod
    def _folded(question: str, prior_digest: str = "", window_id: int = 1) -> str:
        """The user row a window cut actually produces.

        ``_preamble`` builds this from ``full_window_text`` +
        ``dropped_span_excerpt`` + the preserved turn, so the test uses those
        same pieces instead of hand-writing tags that could drift from them.
        """
        from agentica.compression.new_window import dropped_span_excerpt
        from agentica.compression.token_budget import full_window_text

        parts = [full_window_text(window_id, 512000, None)]
        span = dropped_span_excerpt(prior_digest) if prior_digest else None
        if span:
            parts.append(span)
        return "\n\n".join(parts + [question])

    def test_carried_rows_survive_a_second_cut(self):
        """What cut 1 skimmed must still be there after cut 2.

        Cut 2's dropped span is cut 1's preamble row, so the prior digest body
        arrives as an injected tag. ``strip_window_preamble`` peels it (the
        search index wants the chrome gone) — the digest must carry the rows
        forward instead, or every fact leaves one cut after it arrived.
        """
        prior = compose_transcript_digest([
            Message(role="user", content="工单 ZX-41827，泄漏在 evict.py:412"),
            Message(role="assistant", content="记下了，改用 KeyDB"),
        ])
        text = compose_transcript_digest([
            Message(role="user", content=self._folded("现在怎么办？", prior)),
            Message(role="assistant", content="先查 JSONL"),
        ])
        self.assertIn("ZX-41827", text)
        self.assertIn("evict.py:412", text)
        self.assertIn("KeyDB", text)
        self.assertIn("现在怎么办？", text)
        self.assertIn("先查 JSONL", text)
        # Carried rows stay chronological and do not nest the other headers.
        self.assertLess(text.index("ZX-41827"), text.index("现在怎么办？"))
        self.assertEqual(text.count("# Dropped span"), 1)
        self.assertEqual(text.count("## Turns"), 1)

    def test_carrying_is_flat_not_nested_over_many_cuts(self):
        """Ten cuts must not stack ten headers — that is header soup.

        Re-emitting the previous header/titles each time grew the digest by a
        full copy per cut. Only the ``- `` rows are carried, so the structure
        stays flat and the length stays linear and bounded.
        """
        text = compose_transcript_digest([
            Message(role="user", content="工单 ZX-41827"),
            Message(role="assistant", content="记下了，改用 KeyDB"),
        ])
        for _ in range(10):
            text = compose_transcript_digest([
                Message(role="user", content=self._folded("继续？", text)),
            ])
        self.assertEqual(text.count("# Dropped span"), 1)
        self.assertEqual(text.count("## Turns"), 1)
        self.assertIn("ZX-41827", text)
        self.assertIn("KeyDB", text)
        self.assertLess(len(text), 8000, "digest must stay bounded")

    def test_session_notes_are_not_copied_into_dropped_span(self):
        """<session_notes> is re-injected on every cut, so carried_rows must not copy it into <dropped_span>."""
        from agentica.compression.new_window import dropped_span_excerpt, notes_excerpt
        from agentica.compression.token_budget import full_window_text

        notes = notes_excerpt(None, notes_text=(
            "- Constraint: do not rewrite auth\n- Ticket ZX-41827\n"
        ))
        prior_span = dropped_span_excerpt(
            "# Dropped span\n\n## Turns\n- user: 窗口1问题 WIN1-8080\n"
        )
        folded = "\n\n".join([
            full_window_text(1, 512000, None),
            notes,
            prior_span,
        ]) + "\n\n继续？"
        text = compose_transcript_digest([
            Message(role="user", content=folded),
        ])
        self.assertNotIn("do not rewrite auth", text)
        self.assertNotIn("ZX-41827", text)
        self.assertIn("WIN1-8080", text)
        self.assertIn("继续？", text)
        self.assertNotIn("<session_notes", text)

    def test_model_authored_notes_keep_both_notes_and_dropped_span(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "s.notes.md")
            Path(path).write_text("Constraint: do not rewrite auth\n", encoding="utf-8")
            notes, span = rollover_handover(
                [Message(role="user", content="工单 ZX-41827")],
                path,
            )
            self.assertEqual(notes, "Constraint: do not rewrite auth\n")
            self.assertIsNotNone(span)
            self.assertIn("ZX-41827", span)
            self.assertEqual(
                Path(path).read_text(encoding="utf-8"),
                "Constraint: do not rewrite auth\n",
            )
            self.assertTrue(notes_are_ready(path))

    def test_dropped_span_drops_tool_call_narration_before_user_and_final_answers(self):
        msgs = [
            Message(role="user", content="请排查 Ticket-ALPHA-001 并修复"),
        ]
        # 50 intermediate assistant tool-call narrations (exceeding _MAX_TURNS=40)
        for i in range(50):
            msgs.append(
                Message(
                    role="assistant",
                    content=f"我先调用 grep 检查第 {i} 个文件",
                    tool_calls=[{"id": f"tc_{i}", "function": {"name": "grep", "arguments": "{}"}}],
                )
            )
        msgs.append(Message(role="assistant", content="最终结论：根因在 parser.py:88"))
        digest = compose_transcript_digest(msgs)
        self.assertIn("Ticket-ALPHA-001", digest)
        self.assertIn("最终结论：根因在 parser.py:88", digest)
        self.assertNotIn("检查第 0 个文件", digest)

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

    def test_existing_notes_inject_both_session_notes_and_dropped_span(self):
        import asyncio

        with tempfile.TemporaryDirectory() as tmp:
            slog = SessionLog("notes-both", base_dir=tmp)
            notes = Path(slog.path).with_name("notes-both.notes.md")
            notes.write_text("Goal: keep compatibility\n", encoding="utf-8")
            cm = CompressionManager()
            agent = type("A", (), {"_session_log": slog})()
            model = type("M", (), {"id": "gpt-4o", "context_window": 128_000})()
            model._agent_ref = lambda: agent
            msgs = [
                Message(role="system", content="sys"),
                Message(role="user", content="第二窗口新增约束：端口号固定 9091"),
                Message(role="assistant", content="好的，已改为 9091"),
                Message(role="user", content="继续"),
            ]
            asyncio.run(cm.auto_compact(msgs, model=model, force=True))
            joined = " ".join(str(m.content) for m in msgs)
            self.assertIn("<session_notes", joined)
            self.assertIn("Goal: keep compatibility", joined)
            self.assertIn("<dropped_span", joined)
            self.assertIn("9091", joined)

    def test_long_notes_do_not_bleed_into_dropped_span_on_second_compact(self):
        import asyncio
        from agentica.compression.notes import _DROPPED_SPAN_CLOSE, _DROPPED_SPAN_OPEN, _tag_body

        with tempfile.TemporaryDirectory() as tmp:
            slog = SessionLog("notes-no-bleed", base_dir=tmp)
            notes = Path(slog.path).with_name("notes-no-bleed.notes.md")
            long_notes = "\n".join(
                f"- STANDING-NOTE-{i:02d}: keep architectural invariant #{i:02d} intact across all modules"
                for i in range(40)
            )
            notes.write_text(long_notes + "\n", encoding="utf-8")
            cm = CompressionManager()
            agent = type("A", (), {"_session_log": slog})()
            model = type("M", (), {"id": "gpt-4o", "context_window": 128_000})()
            model._agent_ref = lambda: agent

            # Compact #1
            msgs = [
                Message(role="system", content="sys"),
                Message(role="user", content="窗口1任务：修复 WIN1-BUG-7788"),
                Message(role="assistant", content="窗口1已完成 WIN1-BUG-7788 修复"),
                Message(role="user", content="进入窗口2：端口号改为 PORT-9091"),
            ]
            asyncio.run(cm.auto_compact(msgs, model=model, force=True))

            # Compact #2
            msgs.append(Message(role="assistant", content="窗口2已将端口改为 PORT-9091"))
            msgs.append(Message(role="user", content="进入窗口3：继续验证"))
            asyncio.run(cm.auto_compact(msgs, model=model, force=True))

            final_user = msgs[-1].content
            span_body = _tag_body(final_user, _DROPPED_SPAN_OPEN, _DROPPED_SPAN_CLOSE)
            self.assertTrue(span_body)
            self.assertNotIn("STANDING-NOTE-", span_body)
            self.assertIn("WIN1-BUG-7788", span_body)
            self.assertIn("PORT-9091", span_body)
            self.assertIn("STANDING-NOTE-00", final_user)
