# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: CLI preferences (`/reasoning`, `/statusbar`, `/debug`,
`/permissions`) that must survive the process that set them.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.cli import prefs as cli_prefs
from agentica.cli.commands import model_config as cli_model_config
from agentica.cli.commands import tools_skills as cli_tools_skills
from agentica.cli.commands.context import CommandContext
from agentica.cli.interactive.app import _resume_session_cli_prefs
from agentica.memory.session_log import SessionLog
from agentica.project_store import project_base_dir, read_project_file


class _WorkDir:
    """A throwaway work_dir (its project dir is isolated per test by conftest)."""

    def __init__(self, stack: unittest.TestCase):
        self._tmp = tempfile.TemporaryDirectory()
        stack.addCleanup(self._tmp.cleanup)
        self.path = self._tmp.name

    def project_file(self):
        return read_project_file(project_base_dir(self.path))


def _ctx(work_dir, agent=None, tui_state=None, **agent_config):
    config = {"work_dir": work_dir}
    config.update(agent_config)
    return CommandContext(
        agent_config=config,
        current_agent=agent,
        tui_state={} if tui_state is None else tui_state,
    )


def _agent_with_log(work_dir):
    """An Agent stand-in carrying a real SessionLog for the given work_dir."""
    from agentica.agent import Agent

    log = SessionLog("s-prefs-1", work_dir=work_dir, user_id="default")
    agent = Agent()
    agent._session_log = log
    agent.work_dir = work_dir
    return agent, log


class TestToggleCommandsPersist(unittest.TestCase):
    """The four toggles are written to the session sidecar AND project.json."""

    def setUp(self):
        self.work = _WorkDir(self)

    def test_reasoning_off_persists_in_both_scopes(self):
        agent, log = _agent_with_log(self.work.path)
        tui = {"show_reasoning": True}
        ctx = _ctx(self.work.path, agent=agent, tui_state=tui)

        with patch.object(cli_model_config, "get_console", return_value=MagicMock()):
            cli_model_config._cmd_reasoning(ctx, "off")

        self.assertFalse(tui["show_reasoning"])
        self.assertEqual(log.get_cli_prefs(), {"show_reasoning": False})
        self.assertEqual(self.work.project_file().get("cli"), {"show_reasoning": False})

    def test_reasoning_on_is_recorded_too(self):
        agent, log = _agent_with_log(self.work.path)
        tui = {"show_reasoning": False}
        ctx = _ctx(self.work.path, agent=agent, tui_state=tui)

        with patch.object(cli_model_config, "get_console", return_value=MagicMock()):
            cli_model_config._cmd_reasoning(ctx, "on")

        self.assertTrue(tui["show_reasoning"])
        self.assertEqual(log.get_cli_prefs(), {"show_reasoning": True})

    def test_statusbar_toggle_persists(self):
        agent, log = _agent_with_log(self.work.path)
        tui = {"statusbar_visible": True}
        ctx = _ctx(self.work.path, agent=agent, tui_state=tui)

        with patch.object(cli_model_config, "get_console", return_value=MagicMock()):
            cli_model_config._cmd_statusbar(ctx, "")

        self.assertFalse(tui["statusbar_visible"])
        self.assertEqual(log.get_cli_prefs(), {"statusbar_visible": False})

    def test_debug_off_persists(self):
        agent, log = _agent_with_log(self.work.path)
        ctx = _ctx(self.work.path, agent=agent, tui_state={"debug": True}, debug=True)

        with (
            patch.object(cli_model_config, "get_console", return_value=MagicMock()),
            patch.object(cli_model_config, "set_log_level_to_info"),
            patch.object(cli_model_config, "suppress_console_logging"),
        ):
            cli_model_config._cmd_debug(ctx, "off")

        self.assertFalse(ctx.agent_config["debug"])
        self.assertEqual(log.get_cli_prefs(), {"debug": False})

    def test_permissions_also_updates_agent_config(self):
        """A rebuild reads agent_config, not the live agent — without this the
        tier snapped back to allow-all on `/resume` or `/model`."""
        from agentica.agent import Agent

        agent = Agent()
        agent._session_log = SessionLog("s-prefs-2", work_dir=self.work.path, user_id="default")
        ctx = _ctx(self.work.path, agent=agent)

        with patch.object(cli_tools_skills, "get_console", return_value=MagicMock()):
            cli_tools_skills._cmd_permissions(ctx, "ask")

        self.assertEqual(agent.tool_config.permission_mode, "ask")
        self.assertEqual(ctx.agent_config["permissions"], "ask")
        self.assertEqual(self.work.project_file()["cli"], {"permissions": "ask"})

    def test_invalid_permission_mode_is_not_persisted(self):
        from agentica.agent import Agent

        agent = Agent()
        agent._session_log = SessionLog("s-prefs-3", work_dir=self.work.path, user_id="default")
        ctx = _ctx(self.work.path, agent=agent)

        with patch.object(cli_tools_skills, "get_console", return_value=MagicMock()):
            cli_tools_skills._cmd_permissions(ctx, "strict")

        self.assertEqual(agent.tool_config.permission_mode, "allow-all")
        self.assertNotIn("cli", self.work.project_file())
        self.assertEqual(agent._session_log.get_cli_prefs(), {})

    def test_a_failing_sidecar_write_does_not_break_the_command(self):
        """The in-process view is already updated; a read-only sidecar is not
        a reason for `/reasoning off` to raise."""
        agent, _log = _agent_with_log(self.work.path)
        tui = {"show_reasoning": True}
        ctx = _ctx(self.work.path, agent=agent, tui_state=tui)

        with (
            patch.object(cli_model_config, "get_console", return_value=MagicMock()),
            patch.object(
                SessionLog, "set_cli_prefs", side_effect=OSError("read-only")
            ),
        ):
            cli_model_config._cmd_reasoning(ctx, "off")

        self.assertFalse(tui["show_reasoning"])
        self.assertEqual(ctx.agent_config["_cli_prefs"], {"show_reasoning": False})


class TestStartupMerge(unittest.TestCase):
    """What a new CLI in the same work_dir starts with."""

    def setUp(self):
        self.work = _WorkDir(self)

    def test_saved_project_prefs_apply_to_a_fresh_agent_config(self):
        cli_prefs.write_project_prefs(self.work.path, {"show_reasoning": False, "permissions": "auto"})

        agent_config = {}
        cli_prefs.apply_cli_prefs(agent_config, cli_prefs.read_project_prefs(self.work.path))

        self.assertEqual(agent_config["permissions"], "auto")
        tui = {}
        cli_prefs.sync_view_prefs_to_tui(tui, agent_config)
        self.assertFalse(tui["show_reasoning"])
        # Not saved → built-in default, never a missing key.
        self.assertTrue(tui["statusbar_visible"])
        self.assertFalse(tui["debug"])

    def test_explicit_flag_beats_the_saved_value(self):
        cli_prefs.write_project_prefs(self.work.path, {"permissions": "auto", "debug": True})

        agent_config = {
            "_debug_explicit": False,
            "_permissions_explicit": True,
            "permissions": "allow-all",
            "debug": False,
        }
        cli_prefs.apply_cli_prefs(agent_config, cli_prefs.read_project_prefs(self.work.path))

        self.assertEqual(agent_config["permissions"], "allow-all")
        self.assertTrue(agent_config["debug"])

    def test_reads_nothing_from_a_directory_without_project_json(self):
        self.assertEqual(cli_prefs.read_project_prefs(self.work.path), {})

    def test_no_work_dir_is_not_an_error(self):
        self.assertEqual(cli_prefs.read_project_prefs(None), {})
        cli_prefs.write_project_prefs(None, {"debug": True})  # no raise

    def test_junk_values_are_dropped_not_carried_into_agent_config(self):
        self.assertEqual(
            cli_prefs.normalize_cli_prefs(
                {"permissions": "strict", "debug": "yes", "show_reasoning": False, "unknown": 1}
            ),
            {"show_reasoning": False},
        )

    def test_clearing_the_tool_list_is_stored_as_a_removal(self):
        self.assertEqual(cli_prefs.normalize_cli_prefs({"extra_tools": []}), {"extra_tools": None})

    def test_a_tool_set_of_unknown_names_is_ignored_entirely(self):
        """A hand-edit or a tool this version dropped must not be read as
        "clear the set" — that would forget the user's tools on a downgrade."""
        self.assertEqual(cli_prefs.normalize_cli_prefs({"extra_tools": ["gone_tool"]}), {})

    def test_junk_in_project_json_does_not_set_a_bogus_tier(self):
        base = Path(project_base_dir(self.work.path))
        base.mkdir(parents=True, exist_ok=True)
        (base / "project.json").write_text(
            json.dumps({"work_dir": self.work.path, "cli": {"permissions": "nope"}}),
            encoding="utf-8",
        )

        agent_config = {}
        cli_prefs.apply_cli_prefs(agent_config, cli_prefs.read_project_prefs(self.work.path))

        self.assertNotIn("permissions", agent_config)


class TestMainWiring(unittest.TestCase):
    """`main()` is what reads the saved preferences before the first agent."""

    def setUp(self):
        self.work = _WorkDir(self)

    def _main(self, argv):
        import importlib

        cli_main = importlib.import_module("agentica.cli.main")
        console = MagicMock()
        with (
            patch.object(sys, "argv", argv),
            patch.object(cli_main, "_enable_cli_file_logging"),
            patch.object(cli_main, "refresh_model_catalog_in_background"),
            patch.object(
                cli_main,
                "resolve_model_config",
                return_value={"model_provider": "openai", "model_name": "gpt-4o", "base_url": None},
            ),
            patch.object(cli_main, "get_console", return_value=console),
            patch("agentica.cli.interactive.run_interactive") as run_interactive,
        ):
            cli_main.main()
        return run_interactive.call_args

    def _argv(self, *extra):
        return ["agentica", "--work_dir", self.work.path, "--no-workspace", "--no-experience", *extra]

    def test_saved_preferences_reach_the_agent_config(self):
        cli_prefs.write_project_prefs(
            self.work.path, {"permissions": "ask", "debug": True, "show_reasoning": False}
        )

        call = self._main(self._argv())
        agent_config = call.args[0]

        self.assertEqual(agent_config["permissions"], "ask")
        self.assertTrue(agent_config["debug"])
        self.assertFalse(agent_config["_cli_prefs"]["show_reasoning"])

    def test_startup_flags_still_win(self):
        cli_prefs.write_project_prefs(self.work.path, {"permissions": "ask"})

        call = self._main(self._argv("--allow-all"))

        self.assertEqual(call.args[0]["permissions"], "allow-all")

    def test_nothing_saved_leaves_the_defaults_alone(self):
        call = self._main(self._argv())
        agent_config = call.args[0]

        self.assertEqual(agent_config["permissions"], "allow-all")
        self.assertFalse(agent_config["debug"])
        self.assertNotIn("_cli_prefs", agent_config)


class TestSessionScope(unittest.TestCase):
    """The sidecar carries the toggles, and resume prefers them."""

    def setUp(self):
        self.work = _WorkDir(self)

    def _log(self, session_id="s-list"):
        log = SessionLog(session_id, work_dir=self.work.path, user_id="default")
        log.append("user", "hello")
        return log

    def test_sidecar_merges_keys_instead_of_replacing_them(self):
        log = self._log()
        log.set_cli_prefs({"show_reasoning": False})
        log.set_cli_prefs({"permissions": "ask"})

        self.assertEqual(
            log.get_cli_prefs(), {"show_reasoning": False, "permissions": "ask"}
        )
        # The sidecar is shared with name/archived/profile — they survive.
        log.set_name("my session")
        self.assertEqual(log.get_name(), "my session")
        self.assertEqual(log.get_cli_prefs(), {"show_reasoning": False, "permissions": "ask"})

    def test_none_removes_a_key_so_a_pref_can_return_to_default(self):
        log = self._log()
        log.set_cli_prefs({"show_reasoning": False, "permissions": "ask"})
        log.set_cli_prefs({"show_reasoning": None})

        self.assertEqual(log.get_cli_prefs(), {"permissions": "ask"})

    def test_session_listing_exposes_the_cli_block(self):
        log = self._log("s-listed")
        log.set_cli_prefs({"show_reasoning": False})

        entries = {s["session_id"]: s for s in SessionLog.list_sessions(base_dir=str(log.base_dir))}

        self.assertEqual(entries["s-listed"]["cli"], {"show_reasoning": False})
        # A session that never saved one reports {} rather than None, so
        # `apply_cli_prefs(agent_config, entry["cli"])` is always safe.
        other = self._log("s-plain")
        entries = {s["session_id"]: s for s in SessionLog.list_sessions(base_dir=str(other.base_dir))}
        self.assertEqual(entries["s-plain"]["cli"], {})

    def test_fork_carries_the_view_preferences(self):
        log = self._log("s-source")
        log.set_cli_prefs({"show_reasoning": False, "permissions": "auto"})

        forked = log.fork("s-branch")

        self.assertEqual(
            forked.get_cli_prefs(), {"show_reasoning": False, "permissions": "auto"}
        )
        self.assertEqual(forked.get_forked_from(), "s-source")

    def test_resume_prefers_the_session_over_the_project(self):
        """Two sessions in one work_dir can disagree; resuming is a request for
        the session as it was, so its sidecar wins over project.json."""
        cli_prefs.write_project_prefs(self.work.path, {"show_reasoning": True})
        log = self._log("s-quiet")
        log.set_cli_prefs({"show_reasoning": False})
        entry = next(
            s for s in SessionLog.list_sessions(base_dir=str(log.base_dir)) if s["session_id"] == "s-quiet"
        )

        agent_config = {}
        cli_prefs.apply_cli_prefs(agent_config, cli_prefs.read_project_prefs(self.work.path))
        cli_prefs.apply_session_cli_prefs(agent_config, entry["cli"])
        tui = {}
        cli_prefs.sync_view_prefs_to_tui(tui, agent_config)

        self.assertFalse(tui["show_reasoning"])

    def test_the_project_value_survives_the_keys_the_session_does_not_name(self):
        agent_config = {}
        cli_prefs.apply_cli_prefs(
            agent_config, {"statusbar_visible": False, "permissions": "auto"}
        )

        cli_prefs.apply_session_cli_prefs(agent_config, {"show_reasoning": False})

        tui = {}
        cli_prefs.sync_view_prefs_to_tui(tui, agent_config)
        self.assertFalse(tui["show_reasoning"])   # from the session
        self.assertFalse(tui["statusbar_visible"])  # from the project
        self.assertEqual(agent_config["permissions"], "auto")

    def test_a_second_resume_does_not_inherit_the_first_sessions_view(self):
        """`/resume` twice in one process used to leave the first session's
        reasoning/status-bar choice in place for the second."""
        agent_config = {}
        cli_prefs.apply_cli_prefs(agent_config, {})
        cli_prefs.apply_session_cli_prefs(agent_config, {"show_reasoning": False})

        cli_prefs.apply_session_cli_prefs(agent_config, {"permissions": "ask"})

        tui = {}
        cli_prefs.sync_view_prefs_to_tui(tui, agent_config)
        self.assertTrue(tui["show_reasoning"])  # back to the default, not inherited
        self.assertEqual(agent_config["permissions"], "ask")

    def test_resume_requested_reads_the_sidecar_of_that_session(self):
        log = SessionLog("s-resumed", work_dir=self.work.path, user_id="default")
        log.set_cli_prefs({"permissions": "ask", "show_reasoning": False})

        agent_config = {
            "session_id": "s-resumed",
            "session_base_dir": str(log.base_dir),
            "work_dir": self.work.path,
            "_resume_requested": True,
        }
        self.assertEqual(
            _resume_session_cli_prefs(agent_config),
            {"permissions": "ask", "show_reasoning": False},
        )

    def test_no_resume_means_no_session_lookup(self):
        self.assertEqual(_resume_session_cli_prefs({"session_id": "s-x"}), {})
        self.assertEqual(_resume_session_cli_prefs({}), {})

    def test_missing_sidecar_returns_empty(self):
        agent_config = {
            "session_id": "never-existed",
            "session_base_dir": str(Path(self.work.path) / "nope"),
            "work_dir": self.work.path,
            "_resume_requested": True,
        }
        self.assertEqual(_resume_session_cli_prefs(agent_config), {})

    def test_sidecar_cli_block_is_ignored_when_it_is_not_a_dict(self):
        log = self._log("s-broken")
        log.set_name("canary")  # writes the sidecar with a shape we then corrupt
        payload = json.loads(log.meta_path.read_text(encoding="utf-8"))
        payload["cli"] = "off"
        log.meta_path.write_text(json.dumps(payload), encoding="utf-8")

        self.assertEqual(log.get_cli_prefs(), {})


if __name__ == "__main__":
    unittest.main()
