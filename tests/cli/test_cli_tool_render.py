# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for CLI module.
"""

import logging
import os
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.cost_tracker import CostTracker
from agentica.cli import (
    TOOL_ICONS,
    TOOL_REGISTRY,
)
from agentica.cli import commands as cli_commands
from agentica.cli import setup as cli_setup
from agentica.goals import CONTINUATION_PROMPT_PREFIX
from agentica.memory.session_log import SessionLog



class TestCLIToolRender(unittest.TestCase):
    """CLI helpers tests (TestCLIToolRender)."""

    @staticmethod
    def _render_compact_event(event):
        from agentica.cli.display import StreamDisplayManager

        fake = MagicMock()
        fake.width = 80
        StreamDisplayManager(fake).handle_event(event)
        return "\n".join(str(call) for call in fake.print.call_args_list)


    def test_tool_icon_lookup(self):
        """Test looking up tool icons."""
        # Test existing icon
        icon = TOOL_ICONS.get("read_file", TOOL_ICONS["default"])
        self.assertIsNotNone(icon)

        # Test default fallback
        icon = TOOL_ICONS.get("nonexistent_tool", TOOL_ICONS["default"])
        self.assertEqual(icon, TOOL_ICONS["default"])


    def test_edit_diff_uses_configured_work_dir(self):
        """Relative write paths must resolve against the file tool's work_dir."""
        from agentica.cli.display import StreamDisplayManager

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "pkg" / "sample.py"
            target.parent.mkdir()
            target.write_text("value = 1\n", encoding="utf-8")

            fake = MagicMock()
            fake.width = 80
            manager = StreamDisplayManager(fake, work_dir=root)
            manager._capture_file_before_call(
                "edit_file",
                {"file_path": "pkg/sample.py", "old_string": "value = 1\n", "new_string": "value = 2\n"},
                "call-1",
            )
            target.write_text("value = 2\n", encoding="utf-8")

            manager._display_edit_merged(
                "edit_file",
                {"file_path": "pkg/sample.py", "old_string": "value = 1\n", "new_string": "value = 2\n"},
                "Successfully applied 1 edit",
                False,
                " (100ms)",
                "call-1",
            )
            rendered = "\n".join(
                str(call.args[0]) for call in fake.print.call_args_list if call.args
            )
            syntax = fake.print.call_args_list[-1].args[0]
            diff_text = str(syntax.code)

        self.assertIn("Edited 1 file (+1 -1)", rendered)
        self.assertIn("pkg/sample.py", rendered)
        self.assertIn("diff -- pkg/sample.py", diff_text)
        self.assertEqual(diff_text.count("pkg/sample.py"), 1)
        self.assertNotIn("#1", diff_text)

    def test_same_file_edit_batch_renders_each_tool_result_snapshot(self):
        """Each completion renders the exact change reported by its tool call."""
        from agentica.cli.display import StreamDisplayManager

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "sample.py"
            target.write_text("first = 1\nsecond = 1\n", encoding="utf-8")
            first = {
                "file_path": "sample.py",
                "old_string": "first = 1",
                "new_string": "first = 2",
            }
            second = {
                "file_path": "sample.py",
                "old_string": "second = 1",
                "new_string": "second = 2",
            }
            fake = MagicMock()
            fake.width = 80
            manager = StreamDisplayManager(fake, work_dir=root)

            manager.display_tool("edit_file", first, tool_call_id="edit-1")
            manager.display_tool("edit_file", second, tool_call_id="edit-2")
            target.write_text("first = 2\nsecond = 2\n", encoding="utf-8")

            manager.display_tool_result(
                "edit_file", "Successfully replaced 1 occurrence", elapsed=0.1,
                tool_args=first, tool_call_id="edit-1",
                tool_display_meta={"files": [{
                    "path": str(target), "action": "update",
                    "before": "first = 1\nsecond = 1\n",
                    "after": "first = 2\nsecond = 1\n",
                }]},
            )
            manager.display_tool_result(
                "edit_file", "Successfully replaced 1 occurrence", elapsed=0.1,
                tool_args=second, tool_call_id="edit-2",
                tool_display_meta={"files": [{
                    "path": str(target), "action": "update",
                    "before": "first = 2\nsecond = 1\n",
                    "after": "first = 2\nsecond = 2\n",
                }]},
            )

            diffs = [
                str(call.args[0].code)
                for call in fake.print.call_args_list
                if call.args and hasattr(call.args[0], "code")
            ]

        self.assertEqual(len(diffs), 2)
        self.assertIn("-first = 1", diffs[0])
        self.assertIn("+first = 2", diffs[0])
        self.assertNotIn("-second = 1", diffs[0])
        self.assertNotIn("+second = 2", diffs[0])
        self.assertIn("-second = 1", diffs[1])
        self.assertIn("+second = 2", diffs[1])
        self.assertNotIn("-first = 1", diffs[1])
        self.assertNotIn("+first = 2", diffs[1])

    def test_write_file_uses_tool_result_snapshot_not_arguments(self):
        from agentica.cli.display import StreamDisplayManager

        fake = MagicMock()
        fake.width = 80
        manager = StreamDisplayManager(fake)
        manager.display_tool_result(
            "write_file",
            "Updated file",
            tool_args={"file_path": "sample.py", "content": "stale argument\n"},
            tool_display_meta={"files": [{
                "path": "sample.py", "action": "update",
                "before": "@old\n", "after": "@new\n",
            }]},
        )

        diff_text = str(fake.print.call_args_list[-1].args[0].code)
        self.assertIn("-@old", diff_text)
        self.assertIn("+@new", diff_text)
        self.assertNotIn("stale argument", diff_text)

    def test_apply_patch_uses_all_tool_result_snapshots(self):
        from agentica.cli.display import StreamDisplayManager

        fake = MagicMock()
        fake.width = 80
        manager = StreamDisplayManager(fake)
        manager.display_tool_result(
            "apply_patch",
            "Successfully applied patch to 2 files (+1 -1):\nM a.py (+1 -1)\nA b.py (+1 -0)",
            tool_args={"patch": "not needed by the renderer"},
            tool_display_meta={"files": [
                {"path": "a.py", "action": "update", "before": "a = 1\n", "after": "a = 2\n"},
                {"path": "b.py", "action": "add", "before": None, "after": "b = 1\n"},
            ]},
        )

        diff_text = str(fake.print.call_args_list[-1].args[0].code)
        self.assertIn("diff -- a.py", diff_text)
        self.assertIn("diff -- b.py", diff_text)


    def test_write_file_summary_uses_configured_work_dir(self):
        from agentica.cli.display import StreamDisplayManager

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "pkg" / "created.py"
            target.parent.mkdir()
            args = {"file_path": str(target), "content": "value = 1\n"}
            fake = MagicMock()
            fake.width = 80
            manager = StreamDisplayManager(fake, work_dir=root)
            manager.display_tool("write_file", args, tool_call_id="write-1")
            target.write_text("value = 1\n", encoding="utf-8")
            manager.display_tool_result(
                "write_file",
                f"Created file, absolute path: {target}",
                elapsed=0.1,
                tool_args=args,
                tool_call_id="write-1",
            )

            rendered = "\n".join(
                str(call.args[0]) for call in fake.print.call_args_list if call.args
            )
            syntax = fake.print.call_args_list[-1].args[0]

        self.assertIn("pkg/created.py", rendered)
        self.assertNotIn(str(root), rendered)
        self.assertIn("diff -- pkg/created.py", str(syntax.code))
        self.assertEqual(str(syntax.code).count("pkg/created.py"), 1)


    def test_tool_result_sequencer_flushes_parallel_results_in_call_order(self):
        """Out-of-order completions must remain complete, call-ordered blocks."""
        from agentica.cli.interactive.session_state import _ToolResultSequencer

        sequencer = _ToolResultSequencer()
        sequencer.on_start("call-a", "grep")
        sequencer.on_start("call-b", "execute")
        sequencer.on_complete("call-b", {"content": "result-b"})
        self.assertEqual(list(sequencer.drain()), [])
        sequencer.on_complete("call-a", {"content": "result-a"})

        self.assertEqual(
            [item["content"] for item in sequencer.drain()],
            ["result-a", "result-b"],
        )


    def test_display_tool_result_suppresses_write_todos_footer(self):
        """write_todos drops the result footer on success (call line lists tasks)."""
        from agentica.cli.display import StreamDisplayManager

        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        dm.display_tool_result("write_todos", '{"message":"ok"}', is_error=False, elapsed=0.002)
        fake.print.assert_not_called()


    def test_list_agents_result_is_shown_in_full(self):
        """Peer discovery is multi-line and must not be folded behind Ctrl+O."""
        from agentica.cli.display import StreamDisplayManager, clear_truncated_blocks, get_truncated_blocks

        clear_truncated_blocks()
        body = "\n".join(
            [
                "1 other live session(s). You are 'nlp-5f' [peer=abc].",
                "Your session_id: aaaa",
                "",
                "- payments [peer=def]",
                "  session_id: bbbb",
                "  cwd: /repos/payments",
                "  project: /tmp/projects/payments-hash",
                "  session_log: /tmp/projects/payments-hash/bbbb.jsonl",
                "  log_file (INFO): /tmp/home/.agentica/logs/20260809-80403.log",
                "  memory: /tmp/ws/users/default/MEMORY.md",
                "  working on: adding keys",
            ]
        )
        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        dm.display_tool_result("list_agents", body, is_error=False, elapsed=0.004)

        rendered = "\n".join(
            str(call.args[0]) for call in fake.print.call_args_list if call.args
        )
        self.assertIn("session_log:", rendered)
        self.assertIn("log_file (INFO):", rendered)
        self.assertIn("MEMORY.md", rendered)
        self.assertIn("working on: adding keys", rendered)
        self.assertNotIn("more lines", rendered)
        self.assertEqual(get_truncated_blocks(), [])

    ASK_PROMPT = (
        "实测发现 storage 不可复用（v2 用完即删，316 题记忆只剩 3 个残留）。"
        "方案 A 必须改为重跑 ingest。修正成本：方案 A' = 重跑 qwen ingest"
        "（~2880 次，本地免费）+ venus QA（~84 次，~0.5M tok，~1-2h 并行）；"
        "方案 B = 全链路 ds-v4（~5880 次付费，~25M tok）。选哪个？"
    )
    ASK_OPTIONS = [
        "方案 A'（qwen 重跑 ingest + venus QA，~2880 免费 + ~84 付费，推荐）",
        "方案 B（全链路 ds-v4，~5880 次付费）",
    ]

    def _render(self, dm, fake) -> str:
        return "\n".join(
            str(call.args[0]) for call in fake.print.call_args_list if call.args
        )

    def test_ask_user_question_call_line_does_not_repeat_the_question(self):
        """The TUI prompt widget already shows the question; the call line
        used to print a clipped second copy plus 'options=[3 items]'."""
        from agentica.cli.display import StreamDisplayManager

        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        dm.display_tool(
            "ask_user_question",
            {"prompt": self.ASK_PROMPT, "options": self.ASK_OPTIONS},
            tool_call_id="ask-1",
        )

        rendered = self._render(dm, fake)
        self.assertIn("ask_user_question", rendered)
        self.assertNotIn("items]", rendered)
        self.assertNotIn("实测发现", rendered)
        self.assertNotIn("...", rendered)

    def test_ask_user_question_result_replays_question_and_answer_in_full(self):
        """The question widget is transient, so the result block is the only
        record left to scroll back to."""
        import json

        from agentica.cli.display import StreamDisplayManager, clear_truncated_blocks, get_truncated_blocks

        clear_truncated_blocks()
        answer = self.ASK_OPTIONS[0]
        result = json.dumps(
            {"prompt": self.ASK_PROMPT, "response": answer},
            ensure_ascii=False,
        )
        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        dm.display_tool_result(
            "ask_user_question", result, is_error=False, elapsed=12.5,
            tool_args={"prompt": self.ASK_PROMPT}, tool_call_id="ask-1",
        )

        rendered = self._render(dm, fake)
        self.assertIn(self.ASK_PROMPT, rendered.replace("\n", "").replace("   ", ""))
        self.assertIn(answer, rendered)
        self.assertNotIn("more lines", rendered)
        self.assertEqual(get_truncated_blocks(), [])

    def test_multiline_answer_keeps_every_line(self):
        import json

        from agentica.cli.display import StreamDisplayManager

        answer = "先跑方案 A'\n如果 QA 超时再切 B\n预算上限 5 美元"
        result = json.dumps(
            {"prompt": "接下来怎么做？", "response": answer},
            ensure_ascii=False,
        )
        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        dm.display_tool_result("ask_user_question", result, is_error=False, elapsed=3.0)

        rendered = self._render(dm, fake)
        for line in answer.splitlines():
            self.assertIn(line, rendered)

    def test_raw_input_rationale_is_shown_in_transcript(self):
        """When the user typed "3, because workers=10 is ok" and it resolved to
        option 3, scrolling back must still show the rationale the user gave."""
        import json
        from io import StringIO

        from rich.console import Console
        from agentica.cli.display import StreamDisplayManager

        output = StringIO()
        console = Console(file=output, force_terminal=False, color_system=None)
        dm = StreamDisplayManager(console)
        dm.display_tool_result(
            "ask_user_question",
            json.dumps({
                "prompt": "选哪个？",
                "response": "100 题（~3.5 小时）",
                "raw_input": "3 , 100题, workers=10 is ok",
            }, ensure_ascii=False),
            is_error=False,
            elapsed=5.0,
        )

        rendered = output.getvalue()
        self.assertIn("100 题（~3.5 小时）", rendered)
        self.assertIn("workers=10 is ok", rendered)

    def test_unparseable_ask_result_falls_back_to_generic_rendering(self):
        """A payload with no 'response' isn't an exchange — it must still print."""
        from agentica.cli.display import StreamDisplayManager

        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        dm.display_tool_result(
            "ask_user_question",
            '{"error": "user aborted", "prompt": "pick one"}',
            is_error=False,
            elapsed=0.01,
        )

        rendered = self._render(dm, fake)
        self.assertIn("user aborted", rendered)

    def test_ask_exchange_renders_markup_as_literal_text(self):
        """A question or answer mentioning '[bold]' is content, not styling."""
        import json
        from io import StringIO

        from rich.console import Console
        from agentica.cli.display import StreamDisplayManager

        output = StringIO()
        console = Console(file=output, force_terminal=False, color_system=None)
        dm = StreamDisplayManager(console)
        dm.display_tool_result(
            "ask_user_question",
            json.dumps({
                "prompt": "Should the banner use [bold] or [/red]?",
                "response": "use [bold]",
            }),
            is_error=False,
            elapsed=0.5,
        )

        rendered = output.getvalue()
        self.assertIn("[bold]", rendered)
        self.assertIn("[/red]", rendered)

    def test_tool_returns_the_whole_prompt(self):
        """The result JSON is what the CLI replays, so it can't clip at 200."""
        import asyncio
        import json

        from agentica.tools.ask_user_question_tool import AskUserQuestionTool

        long_prompt = self.ASK_PROMPT * 3
        tool = AskUserQuestionTool(input_callback=lambda prompt, options: "ok")
        payload = json.loads(asyncio.run(tool.ask_user_question(prompt=long_prompt)))

        self.assertEqual(payload["prompt"], long_prompt)
        self.assertEqual(payload["response"], "ok")

    def test_wait_result_shows_full_command(self):
        """wait's Command: line is the whole reason you looked — never ellipsis it."""
        from agentica.cli.display import StreamDisplayManager, clear_truncated_blocks, get_truncated_blocks

        clear_truncated_blocks()
        command = (
            "cd /apdcephfs_qy3/share_7435715/flemingxu/nlp/exp/dual_mem_exp/benchmarks "
            "&& RUN_FROM=P1 bash run_overnight.sh --phase all --gpus 8"
        )
        log_path = (
            "/root/.agentica/projects/default/"
            "-apdcephfs-qy3-share-7435715-flemingxu-nlp-6115aec9/background/"
            "20260810-013023-term_4.log"
        )
        body = "\n".join(
            [
                "Background command #4 (term_4) is still running after 1:38:18; "
                "this wait timed out but it was not stopped. If it has already "
                "outlasted a wait or two, stop waiting: end your turn and let "
                "the completion notice the user gets drive the next step.",
                f"Command: {command}",
                f"Log: {log_path}",
            ]
        )
        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        dm.display_tool_result("wait", body, is_error=False, elapsed=300.1)

        rendered = "\n".join(
            str(call.args[0]) for call in fake.print.call_args_list if call.args
        )
        self.assertIn(command, rendered)
        self.assertIn(log_path, rendered)
        self.assertNotIn("run_overnight....", rendered)
        self.assertNotIn("more lines", rendered)
        self.assertEqual(get_truncated_blocks(), [])

    def test_delegate_result_is_shown_in_full(self):
        """delegate's Log: path is actionable — same FULL treatment as wait."""
        from agentica.cli.display import StreamDisplayManager, clear_truncated_blocks, get_truncated_blocks

        clear_truncated_blocks()
        log_path = (
            "/root/.agentica/projects/default/"
            "-apdcephfs-qy3-share-7435715-flemingxu-nlp-6115aec9/background/"
            "20260810-013023-term_7.log"
        )
        body = (
            f'Delegated "overnight bench" to a separate agentica session: term_7 '
            f"(PID 4242).\n"
            f"Log: {log_path}\n"
            f"It is running now and you are not blocked. Its report is delivered "
            f'to this conversation when it finishes; call wait(id="term_7") only '
            f"if your next step needs the answer before you can continue."
        )
        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        dm.display_tool_result("delegate", body, is_error=False, elapsed=0.05)

        rendered = "\n".join(
            str(call.args[0]) for call in fake.print.call_args_list if call.args
        )
        self.assertIn(log_path, rendered)
        self.assertIn('wait(id="term_7")', rendered)
        self.assertNotIn("more lines", rendered)
        self.assertEqual(get_truncated_blocks(), [])

    def test_background_execute_start_result_is_shown_in_full(self):
        """background=True start text carries Log: — never head/tail or 120-char cut."""
        from agentica.cli.display import StreamDisplayManager, clear_truncated_blocks, get_truncated_blocks

        clear_truncated_blocks()
        log_path = (
            "/root/.agentica/projects/default/"
            "-apdcephfs-qy3-share-7435715-flemingxu-nlp-6115aec9/background/"
            "20260810-013023-term_4.log"
        )
        body = (
            f"Started background command #4 (PID 99, id: term_4).\n"
            f"Log: {log_path}\n"
            f"It is detached: its exit is reported to the user, not to you. If a "
            f'later step needs its result, call wait(id="term_4") — it returns '
            f"the moment the command exits."
        )
        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        dm.display_tool_result(
            "execute",
            body,
            is_error=False,
            elapsed=0.02,
            tool_args={"command": "bash run.sh", "background": True},
        )

        rendered = "\n".join(
            str(call.args[0]) for call in fake.print.call_args_list if call.args
        )
        self.assertIn(log_path, rendered)
        self.assertIn('wait(id="term_4")', rendered)
        self.assertNotIn("hidden lines", rendered)
        self.assertNotIn("20260810-013023-...", rendered)
        self.assertEqual(get_truncated_blocks(), [])

    def test_background_execute_call_shows_full_command(self):
        """background execute call lines show the whole command, not a 3-line fold."""
        from io import StringIO
        from rich.console import Console

        from agentica.cli import display as disp
        from agentica.cli.display import StreamDisplayManager

        command = "python3 -m personamem.run " + " ".join(
            f"--question-id value-{index}" for index in range(12)
        )
        output = StringIO()
        console = Console(file=output, width=60, force_terminal=False, no_color=True)
        manager = StreamDisplayManager(console)
        disp.clear_truncated_blocks()

        manager.display_tool(
            "execute",
            {"command": command, "background": True},
            tool_call_id="exec-bg",
        )

        rendered = output.getvalue()
        self.assertIn("value-11", rendered)
        self.assertNotIn("Ctrl+O", rendered)
        self.assertEqual(disp.get_truncated_blocks(), [])

    def test_send_message_call_shows_full_body(self):
        """send_message must not hide the handoff behind the default 40-char truncate."""
        from agentica.cli.display.tool_format import format_tool_display, _display_tool_impl

        long_msg = (
            "用户决定：合并 schema 改动，并说明这是用户的决定。"
            "请对方在采纳前仍向其用户确认 agent 消息边界。"
        )
        display = format_tool_display(
            "send_message",
            {"target": "temp-af", "message": long_msg},
        )
        self.assertIn("→ temp-af", display)
        self.assertIn(long_msg, display)
        self.assertNotIn("...", display)

        fake = MagicMock()
        fake.width = 80
        _display_tool_impl(fake, "send_message", {"target": "temp-af", "message": long_msg})
        rendered = "\n".join(str(call.args[0]) for call in fake.print.call_args_list if call.args)
        self.assertIn(long_msg, rendered)

    def test_task_and_delegate_calls_show_full_brief(self):
        """task / delegate must not truncate the handoff instruction in the CLI."""
        from agentica.cli.display.tool_format import format_tool_display, _display_tool_impl

        long_task = (
            "写一个快速排序算法（Python），包含以下内容：\n"
            "1. 实现 quicksort 函数\n"
            "2. 边界用例与简单测试\n"
            "3. 时间复杂度说明"
        )
        for name, args, body_key in (
            ("delegate", {"task": long_task, "label": "quicksort-delegate"}, "task"),
            ("task", {"description": long_task, "subagent_type": "code"}, "description"),
        ):
            display = format_tool_display(name, args)
            self.assertIn(args[body_key].splitlines()[0], display)
            self.assertIn("2. 边界用例与简单测试", display)
            self.assertNotIn("...", display)

            fake = MagicMock()
            fake.width = 80
            _display_tool_impl(fake, name, args)
            rendered = "\n".join(
                str(call.args[0]) for call in fake.print.call_args_list if call.args
            )
            self.assertIn("2. 边界用例与简单测试", rendered)
            if name == "delegate":
                self.assertIn("quicksort-delegate", rendered)
            else:
                self.assertIn("subagent_type='code'", rendered)

    def test_display_tool_defers_read_only_call_line(self):
        """Read-only tools skip the start-time call line (deferred to completion)."""
        from agentica.cli.display import StreamDisplayManager

        for name in ("read_file", "ls", "glob", "grep", "web_search", "fetch_url"):
            fake = MagicMock()
            fake.width = 80
            dm = StreamDisplayManager(fake)
            dm.display_tool(name, {"file_path": "x.py"})
            fake.print.assert_not_called(), f"{name} call line must be deferred"


    def test_display_tool_result_merged_single_line_for_read_ops(self):
        """Read-only tools collapse call + result into one line with elapsed."""
        from agentica.cli.display import StreamDisplayManager

        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        dm.display_tool_result(
            "grep",
            "path/a.py:1:match\npath/b.py:2:match\npath/c.py:3:match",
            is_error=False,
            elapsed=2.23,
            tool_args={"pattern": "foo", "path": "dual_mem"},
        )
        text = "\n".join(str(c) for c in fake.print.call_args_list)
        # one merged line: icon name params - count (elapsed)
        self.assertIn("grep", text)
        self.assertIn("'foo'", text)
        self.assertIn("3 lines", text)
        self.assertIn("(2.23s)", text)
        # matched content must not leak into the CLI
        self.assertNotIn("match", text)
        # no separate ⎿ footer — everything is on the merged line
        self.assertNotIn("⎿", text)


    def test_display_tool_result_merged_line_hides_fast_elapsed(self):
        """Fast tools render no timing; only slow calls (>= 1s) surface it."""
        from agentica.cli.display import StreamDisplayManager

        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        dm.display_tool_result(
            "read_file",
            "\n".join(f"line {i}" for i in range(311)),
            is_error=False,
            elapsed=0.027,
            tool_args={"file_path": "config.py", "offset": 130, "limit": 40},
        )
        text = "\n".join(str(c) for c in fake.print.call_args_list)
        self.assertIn("config.py", text)
        self.assertIn("311 lines", text)
        self.assertNotIn("ms)", text)
        self.assertNotIn("0.0", text)


    def test_display_tool_result_surfaces_errors_even_for_deferred_tools(self):
        from agentica.cli.display import StreamDisplayManager

        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        dm.display_tool_result(
            "read_file",
            "FileNotFoundError: nope",
            is_error=True,
            elapsed=0.01,
            tool_args={"file_path": "x.py"},
        )
        text = "\n".join(str(c) for c in fake.print.call_args_list)
        self.assertIn("FileNotFoundError", text)


    def test_read_file_error_keeps_external_absolute_path_visible(self):
        from agentica.cli.display import StreamDisplayManager

        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        path = "/apdcephfs_qy3/share_7435715/flemingxu/nlp/dual-mem/dual_mem/memory_extractor.py"

        dm.display_tool_result(
            "read_file",
            "File not found: " + path,
            is_error=True,
            elapsed=0.01,
            tool_args={"file_path": path, "offset": 0, "limit": 300},
        )

        text = "\n".join(str(c) for c in fake.print.call_args_list)
        self.assertIn(path, text)
        self.assertIn("memory_extractor.py", text)

    def test_read_file_error_prints_full_plain_error(self):
        from agentica.cli.display import StreamDisplayManager, clear_truncated_blocks, get_truncated_blocks

        clear_truncated_blocks()
        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        error = (
            "Path does not exist: agentica/cli/status_bar.py. "
            "Resolve relative paths from the nearest existing parent; "
            "do not retry speculative absolute paths."
        )

        dm.display_tool_result(
            "read_file",
            error,
            is_error=True,
            elapsed=0.01,
            tool_args={"file_path": "agentica/cli/status_bar.py", "offset": 65, "limit": 30},
        )

        text = "\n".join(str(c) for c in fake.print.call_args_list)
        self.assertIn("- error: " + error, text)
        self.assertNotIn("[yellow]", text)
        self.assertNotIn("error: ...", text)
        self.assertEqual(get_truncated_blocks(), [])


    def test_display_tool_defers_edit_tools_call_line(self):
        """Write-diff tools defer the call line until completion."""
        from agentica.cli.display import StreamDisplayManager

        for name in ("edit_file", "apply_patch"):
            fake = MagicMock()
            fake.width = 80
            dm = StreamDisplayManager(fake)
            args = (
                {"patch": "*** Begin Patch\n*** Delete File: old.py\n*** End Patch"}
                if name == "apply_patch"
                else {"file_path": "/abs/path/to/config.py"}
            )
            dm.display_tool(name, args)
            fake.print.assert_not_called(), f"{name} call line must be deferred"


    def test_display_edit_file_merged_shows_real_file_diff_and_summary(self):
        """edit_file diffs the real pre/post file and reports changed lines."""
        import tempfile
        from agentica.cli.display import StreamDisplayManager

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "config.py")
            with open(path, "w") as f:
                f.write("DEBUG = False\nKEEP = 1\n")
            args = {
                "file_path": path,
                "old_string": "DEBUG = False\n",
                "new_string": "DEBUG = True\n",
            }
            fake = MagicMock()
            fake.width = 80
            dm = StreamDisplayManager(fake)
            dm.display_tool("edit_file", args, tool_call_id="edit-1")
            with open(path, "w") as f:
                f.write("DEBUG = True\nKEEP = 1\n")
            dm.display_tool_result(
                "edit_file",
                "Successfully applied 1 edit to config.py",
                is_error=False,
                elapsed=0.12,
                tool_args=args,
                tool_call_id="edit-1",
            )

        text = "\n".join(str(c) for c in fake.print.call_args_list)
        self.assertIn("edit_file", text)
        self.assertIn("config.py", text)
        self.assertNotIn(td, text)
        self.assertIn("Edited 1 file (+1 -1)", text)
        self.assertNotIn("ms)", text)
        syntax_args = [c.args[0] for c in fake.print.call_args_list if c.args and "Syntax" in type(c.args[0]).__name__]
        self.assertEqual(len(syntax_args), 1)
        code = getattr(syntax_args[0], "code", "")
        self.assertIn("diff -- config.py", code)
        self.assertEqual(code.count("config.py"), 1)
        self.assertNotIn("--- a/", code)
        self.assertNotIn("+++ b/", code)
        self.assertIn("-DEBUG = False", code)
        self.assertIn("+DEBUG = True", code)


    def test_display_apply_patch_renders_real_multi_file_diff_with_relative_paths(self):
        from agentica.cli.display import StreamDisplayManager

        patch_text = """*** Begin Patch
*** Update File: pkg/app.py
@@
-VALUE = 1
+VALUE = 2
*** Add File: tests/test_app.py
+def test_value():
+    assert True
*** End Patch"""
        args = {"patch": patch_text}
        fake = MagicMock()
        fake.width = 80
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = root / "pkg" / "app.py"
            app.parent.mkdir()
            app.write_text("VALUE = 1\n")
            dm = StreamDisplayManager(fake, work_dir=root)

            dm.display_tool("apply_patch", args, tool_call_id="patch-1")
            app.write_text("VALUE = 2\n")
            test_app = root / "tests" / "test_app.py"
            test_app.parent.mkdir()
            test_app.write_text("def test_value():\n    assert True")
            dm.display_tool_result(
                "apply_patch",
                "Successfully applied patch to 2 files (+3 -1):\n"
                "M pkg/app.py (+1 -1)\nA tests/test_app.py (+2 -0)",
                elapsed=0.25,
                tool_args=args,
                tool_call_id="patch-1",
            )

        text = "\n".join(str(c) for c in fake.print.call_args_list)
        self.assertIn("apply_patch", text)
        self.assertIn("Edited 2 files (+3 -1)", text)
        self.assertNotIn("ms)", text)
        syntax_args = [
            call.args[0] for call in fake.print.call_args_list
            if call.args and "Syntax" in type(call.args[0]).__name__
        ]
        self.assertEqual(len(syntax_args), 1)
        code = getattr(syntax_args[0], "code", "")
        self.assertIn("diff -- pkg/app.py", code)
        self.assertEqual(code.count("pkg/app.py"), 1)
        self.assertIn("-VALUE = 1", code)
        self.assertIn("+VALUE = 2", code)
        self.assertIn("diff -- tests/test_app.py", code)
        self.assertEqual(code.count("tests/test_app.py"), 1)
        self.assertIn("+def test_value():", code)
        self.assertNotIn(tmp, code)


    def test_display_apply_patch_error_keeps_relative_hunk_context(self):
        from agentica.cli.display import StreamDisplayManager

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "pkg" / "app.py"
            error = "\n".join([
                "Patch preflight failed for 1 file; no files were changed.",
                f"- {target}:",
                "  Hunk 1: context not found from line 1.",
                "  Expected context:",
                "    STALE_FIRST = 1",
                "  Hunk 2: context not found from line 1.",
                "  Expected context:",
                "    STALE_SECOND = 2",
                "Read or re-read each failed region with read_file.",
            ])
            fake = MagicMock()
            fake.width = 100
            manager = StreamDisplayManager(fake, work_dir=root)

            manager.display_tool_result(
                "apply_patch",
                error,
                is_error=True,
                elapsed=0.003,
                tool_call_id="patch-error",
            )

            rendered = "\n".join(
                str(call.args[0]) for call in fake.print.call_args_list if call.args
            )

        self.assertIn("pkg/app.py", rendered)
        self.assertNotIn(str(root), rendered)
        self.assertIn("Hunk 1: context not found", rendered)
        self.assertIn("STALE_FIRST = 1", rendered)
        self.assertIn("Hunk 2: context not found", rendered)
        self.assertIn("Ctrl+O to expand", rendered)


    def test_display_apply_patch_error_renders_rich_markup_as_literal_text(self):
        from rich.console import Console
        from agentica.cli.display import StreamDisplayManager

        output = StringIO()
        console = Console(file=output, force_terminal=False, color_system=None)
        manager = StreamDisplayManager(console)

        manager.display_tool_result(
            "apply_patch",
            "Patch preflight failed.\nExpected context: x = [/red] and [bold]",
            is_error=True,
            elapsed=0.003,
            tool_call_id="patch-markup-error",
        )

        rendered = output.getvalue()
        self.assertIn("x = [/red] and [bold]", rendered)


    def test_display_path_keeps_lexical_symlink_path(self):
        from agentica.cli.display import StreamDisplayManager

        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            root = Path(tmp)
            link = root / "linked"
            link.symlink_to(outside, target_is_directory=True)
            console = MagicMock()
            console.width = 80
            manager = StreamDisplayManager(console, work_dir=root)

            self.assertEqual(manager._display_path(str(link / "app.py")), "linked/app.py")


    def test_shorten_workdir_text_respects_path_boundaries(self):
        from agentica.cli.display import StreamDisplayManager

        console = MagicMock()
        console.width = 80
        manager = StreamDisplayManager(console, work_dir=Path("/tmp/foo"))

        self.assertEqual(
            manager._shorten_workdir_text("/tmp/foobar/app.py /tmp/foo/pkg/app.py"),
            "/tmp/foobar/app.py pkg/app.py",
        )


    def test_display_write_file_merged_shows_summary_and_diff(self):
        """write_file: one summary line (created/updated + line count) + a diff."""
        import tempfile
        from agentica.cli.display import StreamDisplayManager

        new_content = "\n".join(f"line {i}" for i in range(20)) + "\n"
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "new_file.py")
            fake = MagicMock()
            fake.width = 80
            dm = StreamDisplayManager(fake)
            # Call-start stashes pre-write content (file is new → empty old).
            dm.display_tool("write_file", {"file_path": path, "content": new_content})
            dm.display_tool_result(
                "write_file",
                f"Created file, absolute path: {path}",
                is_error=False,
                elapsed=0.12,
                tool_args={"file_path": path, "content": new_content},
            )
        text = "\n".join(str(c) for c in fake.print.call_args_list)
        self.assertIn("write_file", text)
        self.assertIn("new_file.py", text)
        self.assertIn("created", text)
        self.assertNotIn("ms)", text)
        # A diff Syntax block is rendered; for a new file it's all additions.
        syntax_args = [c.args[0] for c in fake.print.call_args_list if c.args and "Syntax" in type(c.args[0]).__name__]
        self.assertTrue(syntax_args, "expected a diff Syntax block")
        self.assertIn("line 0", getattr(syntax_args[0], "code", ""))


    def test_display_write_file_diff_against_old_content(self):
        """write_file on an existing file diffs old→new (not all-additions)."""
        import tempfile
        from agentica.cli.display import StreamDisplayManager

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "cfg.py")
            with open(path, "w") as f:
                f.write("DEBUG = False\nKEEP = 1\n")
            new_content = "DEBUG = True\nKEEP = 1\n"
            fake = MagicMock()
            fake.width = 80
            dm = StreamDisplayManager(fake)
            dm.display_tool("write_file", {"file_path": path, "content": new_content})
            dm.display_tool_result(
                "write_file",
                f"Updated file, absolute path: {path}",
                is_error=False,
                elapsed=0.10,
                tool_args={"file_path": path, "content": new_content},
            )
        syntax_args = [c.args[0] for c in fake.print.call_args_list if c.args and "Syntax" in type(c.args[0]).__name__]
        self.assertTrue(syntax_args)
        code = getattr(syntax_args[0], "code", "")
        # Real diff: -False / +True, and unchanged KEEP has no +/-.
        self.assertIn("-DEBUG = False", code)
        self.assertIn("+DEBUG = True", code)
        text = "\n".join(str(c) for c in fake.print.call_args_list)
        self.assertIn("updated", text)


    def test_display_execute_tail_window(self):
        """execute shows short output inline; long output keeps only the tail."""
        from agentica.cli.display import StreamDisplayManager

        # 10 lines: <= inline threshold, fully shown, no fold hint.
        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        dm.display_tool_result(
            "execute",
            "\n".join(f"line {i}" for i in range(10)),
            is_error=False,
            elapsed=0.1,
        )
        text = "\n".join(str(c) for c in fake.print.call_args_list)
        self.assertNotIn("… +", text)
        self.assertNotIn("...", text)

        # 30 lines: tail-only window (last 6) — head folded into a hint line.
        fake2 = MagicMock()
        fake2.width = 80
        dm2 = StreamDisplayManager(fake2)
        dm2.display_tool_result(
            "execute",
            "\n".join(f"line {i}" for i in range(30)),
            is_error=False,
            elapsed=0.1,
        )
        text2 = "\n".join(str(c) for c in fake2.print.call_args_list)
        # Tail present; head folded into a leading "+N lines" hint.
        self.assertIn("+24 lines", text2)
        self.assertIn("line 29", text2)
        self.assertNotIn("line 0", text2)
        self.assertNotIn("line 15", text2)


    def test_display_execute_diagnostics_uses_short_warning_window(self):
        from agentica.cli.display import StreamDisplayManager

        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        output = "\n".join(f"diag {i}" for i in range(18))
        output += "\n\n[Exit code: 1]\n(Note: Diagnostics found)"

        dm.display_tool_result("execute", output, is_error=False, elapsed=0.1)

        text = "\n".join(str(c) for c in fake.print.call_args_list)
        self.assertIn("⚠", text)
        self.assertIn("+13 lines", text)
        self.assertIn("diag 17", text)
        self.assertIn("[Exit code: 1]", text)
        self.assertNotIn("diag 0", text)
        self.assertNotIn("diag 9", text)


    def test_display_execute_filters_internal_repeat_failure_notice(self):
        from agentica.cli.display import StreamDisplayManager

        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        output = (
            "real error\n\n"
            "[Notice: This exact call has failed 2 times this run with the same error. "
            "Consider a different approach.]"
        )

        dm.display_tool_result("execute", output, is_error=True, elapsed=0.1)

        text = "\n".join(str(c) for c in fake.print.call_args_list)
        self.assertIn("real error", text)
        self.assertNotIn("This exact call has failed", text)


    def test_display_execute_wraps_long_command_and_retains_full_command(self):
        """Long commands use a width-aware preview and remain expandable."""
        from io import StringIO
        from rich.console import Console

        from agentica.cli import display as disp
        from agentica.cli.display import StreamDisplayManager

        command = "python3 -m personamem.run " + " ".join(
            f"--question-id value-{index}" for index in range(12)
        )
        output = StringIO()
        console = Console(file=output, width=60, force_terminal=False, no_color=True)
        manager = StreamDisplayManager(console)
        disp.clear_truncated_blocks()

        manager.display_tool("execute", {"command": command}, tool_call_id="exec-long")

        rendered = output.getvalue()
        self.assertIn("execute python3 -m personamem.run", rendered)
        self.assertEqual(rendered.count("│"), 3)
        self.assertRegex(rendered, r"… \+\d+ lines \(Ctrl\+O to expand\)")
        self.assertNotIn("value-11", rendered)
        self.assertEqual(disp.get_truncated_blocks(), [{
            "title": "Command · execute",
            "content": command,
        }])


    def test_display_execute_heredoc_uses_same_multiline_preview(self):
        """Heredoc scripts show their leading lines without Python-specific parsing."""
        from io import StringIO
        from rich.console import Console

        from agentica.cli import display as disp
        from agentica.cli.display import StreamDisplayManager

        command = "\n".join([
            "python3 - <<'PY'",
            "import csv, json, pathlib",
            "base = pathlib.Path('exp/dual_mem_exp/benchmarks')",
            "rows = list(csv.DictReader(open(base / 'questions.csv')))",
            "print(json.dumps(rows[0]))",
            "PY",
        ])
        output = StringIO()
        console = Console(file=output, width=100, force_terminal=False, no_color=True)
        manager = StreamDisplayManager(console)
        disp.clear_truncated_blocks()

        manager.display_tool("execute", {"command": command}, tool_call_id="exec-heredoc")

        rendered = output.getvalue()
        self.assertIn("execute python3 - <<'PY'", rendered)
        self.assertIn("│ import csv, json, pathlib", rendered)
        self.assertIn("│ base = pathlib.Path", rendered)
        self.assertIn("… +3 lines (Ctrl+O to expand)", rendered)
        self.assertNotIn("print(json.dumps", rendered)
        self.assertEqual(disp.get_last_truncated()["content"], command)


    def test_execute_command_and_output_are_separate_expandable_blocks(self):
        """Ctrl+O retains both a folded command and its folded output."""
        from io import StringIO
        from rich.console import Console

        from agentica.cli import display as disp
        from agentica.cli.display import StreamDisplayManager

        command = "python3 -m benchmark " + " ".join(
            f"--case case-{index}" for index in range(16)
        )
        long_output = "\n".join(f"output {index}" for index in range(30))
        output = StringIO()
        console = Console(file=output, width=60, force_terminal=False, no_color=True)
        manager = StreamDisplayManager(console)
        disp.clear_truncated_blocks()

        manager.display_tool("execute", {"command": command}, tool_call_id="exec-both")
        manager.display_tool_result(
            "execute",
            long_output,
            elapsed=0.5,
            tool_args={"command": command},
            tool_call_id="exec-both",
        )

        self.assertEqual(disp.get_truncated_blocks(), [
            {"title": "Command · execute", "content": command},
            {"title": "Tool output · execute", "content": long_output},
        ])




if __name__ == "__main__":
    unittest.main()
