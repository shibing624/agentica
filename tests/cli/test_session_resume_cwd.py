# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Resuming a session started in another working directory.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.cli import session_resume as sr
from agentica.cli.commands import session as cli_session
from agentica.cli.commands.context import CommandContext
from agentica.memory.session_log import SessionLog
from agentica.run_response import AgentCancelledError


class _ProjectStore:
    """Isolated ``~/.agentica/projects`` root for one test."""

    def __init__(self, stack: unittest.TestCase):
        self._tmp = tempfile.TemporaryDirectory()
        stack.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.projects = self.root / "projects"
        self.projects.mkdir(parents=True, exist_ok=True)
        prev = os.environ.get("AGENTICA_PROJECTS_DIR")
        os.environ["AGENTICA_PROJECTS_DIR"] = str(self.projects)

        def _restore_projects_dir() -> None:
            if prev is None:
                os.environ.pop("AGENTICA_PROJECTS_DIR", None)
            else:
                os.environ["AGENTICA_PROJECTS_DIR"] = prev

        stack.addCleanup(_restore_projects_dir)

    def work_dir(self, name: str) -> str:
        path = self.root / name
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def session(self, work_dir: str, session_id: str, text: str = "hello") -> SessionLog:
        log = SessionLog(session_id, work_dir=work_dir, user_id="default")
        log.append("user", text)
        return log


class TestProjectMarker(unittest.TestCase):
    def setUp(self):
        self.store = _ProjectStore(self)

    def test_marker_records_work_dir(self):
        work_dir = self.store.work_dir("alpha")
        log = self.store.session(work_dir, "11111111-aaaa")

        self.assertEqual(SessionLog.project_work_dir(log.base_dir), work_dir)

    def test_marker_is_written_once(self):
        work_dir = self.store.work_dir("alpha")
        log = self.store.session(work_dir, "11111111-aaaa")
        marker = Path(log.base_dir) / "project.json"
        marker.write_text(json.dumps({"work_dir": "/kept"}), encoding="utf-8")

        self.store.session(work_dir, "22222222-bbbb")

        self.assertEqual(SessionLog.project_work_dir(log.base_dir), "/kept")

    def test_legacy_dir_falls_back_to_transcript_cwd(self):
        work_dir = self.store.work_dir("alpha")
        log = self.store.session(work_dir, "11111111-aaaa")
        (Path(log.base_dir) / "project.json").unlink()

        # No marker: the cwd stamped on the first entry is the next best source.
        self.assertEqual(SessionLog.project_work_dir(log.base_dir), os.getcwd())

    def test_explicit_base_dir_writes_no_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            SessionLog("33333333-cccc", base_dir=directory).append("user", "hi")
            self.assertFalse((Path(directory) / "project.json").exists())


class TestFindSessions(unittest.TestCase):
    def setUp(self):
        self.store = _ProjectStore(self)
        self.alpha = self.store.work_dir("alpha")
        self.beta = self.store.work_dir("beta")
        self.store.session(self.alpha, "aaaaaaaa-1111")
        self.store.session(self.beta, "bbbbbbbb-2222")

    def test_finds_session_from_another_project(self):
        found = SessionLog.find_sessions("bbbbbbbb", user_id="default")

        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["session_id"], "bbbbbbbb-2222")
        self.assertEqual(found[0]["work_dir"], self.beta)

    def test_unknown_prefix_returns_nothing(self):
        self.assertEqual(SessionLog.find_sessions("zzzz", user_id="default"), [])

    def test_blank_needle_never_matches_everything(self):
        self.assertEqual(SessionLog.find_sessions("  ", user_id="default"), [])

    def test_truncated_picker_form_is_accepted(self):
        found = SessionLog.find_sessions("aaaaaaaa...1111", user_id="default")

        self.assertEqual([s["session_id"] for s in found], ["aaaaaaaa-1111"])

    def test_glob_metacharacters_are_literal(self):
        self.assertEqual(SessionLog.find_sessions("*", user_id="default"), [])

    def test_list_projects_reports_both(self):
        projects = SessionLog.list_projects(user_id="default")

        self.assertEqual(
            sorted(p["work_dir"] for p in projects), sorted([self.alpha, self.beta])
        )

    def test_local_match_wins_over_global(self):
        local = SessionLog.list_sessions(work_dir=self.alpha, user_id="default")

        found = sr.find_sessions_by_id("a", local, user_id="default")

        self.assertEqual([s["session_id"] for s in found], ["aaaaaaaa-1111"])


class TestChooseResumeWorkDir(unittest.TestCase):
    def setUp(self):
        self.store = _ProjectStore(self)
        self.session_dir = self.store.work_dir("alpha")
        self.current_dir = self.store.work_dir("beta")
        self.saved = {}
        patcher = patch.object(sr, "set_setting", lambda k, v: self.saved.update({k: v}))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _choose(self, answer, preference="ask"):
        asks = []

        def asker(prompt, options):
            asks.append((prompt, options))
            if isinstance(answer, Exception):
                raise answer
            return answer

        with patch.object(sr, "get_setting", return_value=preference):
            choice = sr.choose_resume_work_dir(
                self.session_dir, self.current_dir, asker=asker
            )
        return choice, asks

    def test_same_directory_asks_nothing(self):
        with patch.object(sr, "get_setting", return_value="ask"):
            choice = sr.choose_resume_work_dir(
                self.session_dir,
                self.session_dir,
                asker=lambda *_: self.fail("must not ask"),
            )
        self.assertIsNone(choice.work_dir)

    def test_symlinked_directory_counts_as_same(self):
        link = os.path.join(self.store.root, "link-to-alpha")
        os.symlink(self.session_dir, link)
        with patch.object(sr, "get_setting", return_value="ask"):
            choice = sr.choose_resume_work_dir(
                self.session_dir, link, asker=lambda *_: self.fail("must not ask")
            )
        self.assertIsNone(choice.work_dir)

    def test_option_one_uses_session_directory(self):
        choice, asks = self._choose("1")

        self.assertEqual(choice.work_dir, self.session_dir)
        self.assertEqual(len(asks[0][1]), 4)
        self.assertEqual(self.saved, {})

    def test_option_two_keeps_current_directory(self):
        choice, _ = self._choose("2")

        self.assertIsNone(choice.work_dir)
        self.assertEqual(self.saved, {})

    def test_answering_with_option_text_works(self):
        choice, asks = self._choose(f"Use session directory ({self.session_dir})")

        self.assertEqual(choice.work_dir, self.session_dir)

    def test_always_session_is_persisted(self):
        choice, _ = self._choose("3")

        self.assertEqual(choice.work_dir, self.session_dir)
        self.assertEqual(self.saved, {"resume_cwd": "session"})

    def test_always_current_is_persisted(self):
        choice, _ = self._choose("4")

        self.assertIsNone(choice.work_dir)
        self.assertEqual(self.saved, {"resume_cwd": "current"})

    def test_saved_preference_skips_the_question(self):
        choice, asks = self._choose("1", preference="session")
        self.assertEqual(choice.work_dir, self.session_dir)
        self.assertEqual(asks, [])

        choice, asks = self._choose("1", preference="current")
        self.assertIsNone(choice.work_dir)
        self.assertEqual(asks, [])

    def test_unknown_preference_falls_back_to_asking(self):
        choice, asks = self._choose("2", preference="nonsense")

        self.assertIsNone(choice.work_dir)
        self.assertEqual(len(asks), 1)

    def test_blank_answer_defaults_to_session_directory(self):
        choice, _ = self._choose("")

        self.assertEqual(choice.work_dir, self.session_dir)

    def test_cancelling_aborts_the_resume(self):
        choice, _ = self._choose(AgentCancelledError("ctrl-c"))

        self.assertTrue(choice.cancelled)
        self.assertIsNone(choice.work_dir)

    def test_deleted_session_directory_is_not_offered(self):
        gone = os.path.join(self.store.root, "deleted")
        messages = []
        with patch.object(sr, "get_setting", return_value="ask"):
            choice = sr.choose_resume_work_dir(
                gone,
                self.current_dir,
                asker=lambda *_: self.fail("must not ask"),
                printer=messages.append,
            )

        self.assertIsNone(choice.work_dir)
        self.assertIn("no longer exists", " ".join(messages))


class TestPrepareStartupResume(unittest.TestCase):
    def setUp(self):
        self.store = _ProjectStore(self)
        self.alpha = self.store.work_dir("alpha")
        self.current = self.store.work_dir("beta")
        self.log = self.store.session(self.alpha, "aaaaaaaa-1111")
        self.addCleanup(os.chdir, os.getcwd())

    def _config(self, session_id):
        return {"session_id": session_id, "work_dir": self.current}

    def test_prefix_resolves_and_switches_directory(self):
        config = self._config("aaaaaaaa")

        with patch.object(sr, "get_setting", return_value="session"):
            ok = sr.prepare_startup_resume(config, user_id="default")

        self.assertTrue(ok)
        self.assertEqual(config["session_id"], "aaaaaaaa-1111")
        self.assertEqual(config["session_base_dir"], str(self.log.base_dir))
        self.assertEqual(config["work_dir"], self.alpha)
        self.assertEqual(os.path.realpath(os.getcwd()), os.path.realpath(self.alpha))

    def test_current_directory_keeps_transcript_where_it_is(self):
        config = self._config("aaaaaaaa-1111")

        with patch.object(sr, "get_setting", return_value="current"):
            ok = sr.prepare_startup_resume(config, user_id="default")

        self.assertTrue(ok)
        self.assertEqual(config["work_dir"], self.current)
        self.assertEqual(config["session_base_dir"], str(self.log.base_dir))

    def test_unknown_session_reports_and_stops(self):
        messages = []
        config = self._config("nope")

        ok = sr.prepare_startup_resume(config, user_id="default", printer=messages.append)

        self.assertFalse(ok)
        self.assertIn("No session found", " ".join(messages))

    def test_ambiguous_prefix_reports_and_stops(self):
        self.store.session(self.store.work_dir("gamma"), "aaaaaaaa-9999")
        messages = []

        ok = sr.prepare_startup_resume(
            self._config("aaaaaaaa"), user_id="default", printer=messages.append
        )

        self.assertFalse(ok)
        self.assertIn("matches 2 sessions", " ".join(messages))


class TestResumeCommandAcrossProjects(unittest.TestCase):
    """`/resume <id>` typed in a directory the session does not belong to."""

    def setUp(self):
        self.store = _ProjectStore(self)
        self.alpha = self.store.work_dir("alpha")
        self.current = self.store.work_dir("beta")
        self.log = self.store.session(self.alpha, "aaaaaaaa-1111")
        self.store.session(self.current, "bbbbbbbb-2222")
        self.addCleanup(os.chdir, os.getcwd())

    def _run(self, target, answer="1"):
        agent = MagicMock()
        agent.user_id = "default"
        agent._session_log = SessionLog(
            "bbbbbbbb-2222", work_dir=self.current, user_id="default"
        )
        ctx = CommandContext(
            agent_config={"work_dir": self.current},
            current_agent=agent,
            tui_state={},
            ask_user_question_callback=lambda prompt, options: answer,
        )
        console = MagicMock()
        with (
            patch("agentica.cli.commands.session.get_console", return_value=console),
            patch("agentica.cli.commands.session.create_agent") as create_agent,
            patch("agentica.cli.commands.session.GoalManager") as goal_manager,
            patch.object(sr, "set_setting"),
        ):
            resumed = MagicMock()
            resumed._session_log = None
            resumed.working_memory.runs = []
            create_agent.return_value = resumed
            result = cli_session._cmd_resume(ctx, target)
        return ctx, create_agent, result, console

    def test_session_directory_choice_rebinds_agent_and_storage(self):
        with patch.object(sr, "get_setting", return_value="ask"):
            ctx, create_agent, result, _ = self._run("aaaaaaaa")

        passed = create_agent.call_args[0][0]
        self.assertEqual(passed["session_id"], "aaaaaaaa-1111")
        self.assertEqual(passed["work_dir"], self.alpha)
        self.assertEqual(passed["session_base_dir"], str(self.log.base_dir))
        self.assertEqual(result["work_dir"], self.alpha)
        self.assertEqual(os.path.realpath(os.getcwd()), os.path.realpath(self.alpha))

    def test_current_directory_choice_leaves_cwd_alone(self):
        before = os.getcwd()
        with patch.object(sr, "get_setting", return_value="ask"):
            ctx, create_agent, result, _ = self._run("aaaaaaaa", answer="2")

        passed = create_agent.call_args[0][0]
        self.assertEqual(passed["work_dir"], self.current)
        self.assertEqual(passed["session_base_dir"], str(self.log.base_dir))
        self.assertNotIn("work_dir", result)
        self.assertEqual(os.getcwd(), before)

    def test_cancelling_the_prompt_builds_no_agent(self):
        def cancel(prompt, options):
            raise AgentCancelledError("ctrl-c")

        agent = MagicMock()
        agent.user_id = "default"
        agent._session_log = SessionLog(
            "bbbbbbbb-2222", work_dir=self.current, user_id="default"
        )
        ctx = CommandContext(
            agent_config={"work_dir": self.current},
            current_agent=agent,
            tui_state={},
            ask_user_question_callback=cancel,
        )
        console = MagicMock()
        with (
            patch("agentica.cli.commands.session.get_console", return_value=console),
            patch("agentica.cli.commands.session.create_agent") as create_agent,
            patch.object(sr, "get_setting", return_value="ask"),
        ):
            result = cli_session._cmd_resume(ctx, "aaaaaaaa")

        self.assertIsNone(result)
        create_agent.assert_not_called()

    def test_local_session_never_prompts(self):
        with patch.object(sr, "get_setting", return_value="ask"):
            ctx, create_agent, result, _ = self._run(
                "bbbbbbbb", answer="unused — must not be asked"
            )

        passed = create_agent.call_args[0][0]
        self.assertEqual(passed["session_id"], "bbbbbbbb-2222")
        self.assertEqual(passed["work_dir"], self.current)
        self.assertNotIn("work_dir", result)

    def test_resume_all_lists_every_project_and_numbers_them(self):
        agent = MagicMock()
        agent.user_id = "default"
        agent.session_id = "bbbbbbbb-2222"
        agent._session_log = SessionLog(
            "bbbbbbbb-2222", work_dir=self.current, user_id="default"
        )
        ctx = CommandContext(
            agent_config={"work_dir": self.current}, current_agent=agent, tui_state={}
        )
        console = MagicMock()
        with patch("agentica.cli.commands.session.get_console", return_value=console):
            cli_session._cmd_resume(ctx, "all")

        printed = "\n".join(str(c.args[0]) for c in console.print.call_args_list if c.args)
        self.assertIn("aaaaaaaa", printed)
        self.assertIn("bbbbbbbb", printed)
        self.assertIn(self.alpha, printed)
        self.assertEqual(len(ctx.tui_state["resume_picker"]), 2)

    def test_number_after_resume_all_uses_that_listing(self):
        agent = MagicMock()
        agent.user_id = "default"
        agent.session_id = "bbbbbbbb-2222"
        agent._session_log = SessionLog(
            "bbbbbbbb-2222", work_dir=self.current, user_id="default"
        )
        tui_state = {}
        ctx = CommandContext(
            agent_config={"work_dir": self.current},
            current_agent=agent,
            tui_state=tui_state,
            ask_user_question_callback=lambda prompt, options: "1",
        )
        console = MagicMock()
        with (
            patch("agentica.cli.commands.session.get_console", return_value=console),
            patch("agentica.cli.commands.session.create_agent") as create_agent,
            patch("agentica.cli.commands.session.GoalManager") as goal_manager,
            patch.object(sr, "get_setting", return_value="ask"),
            patch.object(sr, "set_setting"),
        ):
            resumed = MagicMock()
            resumed._session_log = None
            resumed.working_memory.runs = []
            create_agent.return_value = resumed
            cli_session._cmd_resume(ctx, "all")
            listed = [s["session_id"] for s in tui_state["resume_picker"]]
            cli_session._cmd_resume(ctx, "1")

        self.assertEqual(create_agent.call_args[0][0]["session_id"], listed[0])


if __name__ == "__main__":
    unittest.main()
