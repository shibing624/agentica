# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for project-scoped profile overrides.

These verify the two-layer profile resolution introduced in
``global_config.resolve_active_profile_name``:

  1. Project override (``~/.agentica/projects/<key>/project.json`` ``active_profile``)
  2. Global default (``config.yaml -> active_profile``)

Key contract: the project override is keyed by ``work_dir`` (same hash as
SessionLog), NOT by git toplevel. This aligns with Workspace / AGENTICA_HOME
and needs no fallback logic for non-git directories.
"""

import logging
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import global_config as gc


class TestProjectProfile(unittest.TestCase):
    def setUp(self):
        # Isolate BOTH config.yaml AND ~/.agentica/projects/ into tmp.
        self._tmp = tempfile.TemporaryDirectory()
        home = os.path.join(self._tmp.name, "agentica_home")
        os.makedirs(home, exist_ok=True)
        self._home = home

        self._patch_cfg = patch.object(
            gc,
            "global_config_path",
            return_value=os.path.join(home, "config.yaml"),
        )
        self._patch_cfg.start()

        self._env_snapshot = dict(os.environ)
        os.environ["AGENTICA_HOME"] = home

        # Two work dirs that will get their own project overrides.
        self._proj_a = os.path.realpath(os.path.join(self._tmp.name, "proj_a"))
        self._proj_b = os.path.realpath(os.path.join(self._tmp.name, "proj_b"))
        os.makedirs(self._proj_a, exist_ok=True)
        os.makedirs(self._proj_b, exist_ok=True)

        # Seed two profiles + a global default.
        gc.upsert_profile("work", {"model_provider": "openai", "model_name": "gpt-4o"})
        gc.upsert_profile(
            "personal",
            {"model_provider": "deepseek", "model_name": "deepseek-v4-flash"},
            make_active=False,
        )
        # Global default is now "work" (last upsert_profile with make_active=True).
        self.assertEqual(gc.get_active_profile_name(), "work")

    def tearDown(self):
        self._patch_cfg.stop()
        self._tmp.cleanup()
        os.environ.clear()
        os.environ.update(self._env_snapshot)

    # ------------------------------------------------------------------ core

    def test_no_override_returns_global_default(self):
        name, source = gc.resolve_active_profile_name(work_dir=self._proj_a)
        self.assertEqual(name, "work")
        self.assertEqual(source, "global")

    def test_project_override_wins_over_global(self):
        gc.set_project_profile(self._proj_a, "personal")
        name, source = gc.resolve_active_profile_name(work_dir=self._proj_a)
        self.assertEqual(name, "personal")
        self.assertEqual(source, "project")

    def test_two_workdirs_are_independent(self):
        """Core use case: same config.yaml, different work_dirs, independent overrides."""
        gc.set_project_profile(self._proj_a, "personal")
        gc.set_project_profile(self._proj_b, "work")

        name_a, source_a = gc.resolve_active_profile_name(work_dir=self._proj_a)
        name_b, source_b = gc.resolve_active_profile_name(work_dir=self._proj_b)

        self.assertEqual((name_a, source_a), ("personal", "project"))
        self.assertEqual((name_b, source_b), ("work", "project"))

    def test_clear_project_profile_falls_back_to_global(self):
        gc.set_project_profile(self._proj_a, "personal")
        self.assertTrue(gc.clear_project_profile(self._proj_a))
        name, source = gc.resolve_active_profile_name(work_dir=self._proj_a)
        self.assertEqual(name, "work")
        self.assertEqual(source, "global")

    def test_clear_project_profile_when_absent_is_noop(self):
        self.assertFalse(gc.clear_project_profile(self._proj_a))

    # ------------------------------------------------------------------ fallback / robustness

    def test_override_pointing_at_missing_profile_falls_back_with_warning(self):
        gc.set_project_profile(self._proj_a, "ghost")
        with self.assertLogs(gc.logger, level="WARNING") as cm:
            name, source = gc.resolve_active_profile_name(work_dir=self._proj_a)
        self.assertEqual(name, "work")
        self.assertEqual(source, "global")
        self.assertTrue(any("ghost" in msg for msg in cm.output))

    def test_no_workdir_returns_global(self):
        name, source = gc.resolve_active_profile_name(work_dir=None)
        self.assertEqual(name, "work")
        self.assertEqual(source, "global")

    def test_empty_workdir_returns_global(self):
        name, source = gc.resolve_active_profile_name(work_dir="")
        self.assertEqual(name, "work")
        self.assertEqual(source, "global")

    # ------------------------------------------------------------------ key semantics

    def test_symlink_path_is_independent_key(self):
        """Symlink path and real path hash differently (same as SessionLog).

        Session storage keys on the work_dir string as given, not
        ``os.path.realpath``. Profile overrides must use that same key so
        ``project.json`` lands next to the session files.
        """
        link = os.path.join(self._tmp.name, "link_to_a")
        os.symlink(self._proj_a, link)
        gc.set_project_profile(self._proj_a, "personal")
        name, source = gc.resolve_active_profile_name(work_dir=link)
        self.assertEqual((name, source), ("work", "global"))
        self.assertEqual(gc.get_project_profile(self._proj_a), "personal")
        self.assertIsNone(gc.get_project_profile(link))

    def test_project_key_uses_workdir_not_git_toplevel(self):
        """Two sibling subdirs get INDEPENDENT keys even inside one git repo.

        Keying is the work_dir string (SessionLog / sanitize_path), not
        ``git rev-parse --show-toplevel``, so siblings never collapse.
        """
        sub_a = os.path.join(self._proj_a, "frontend")
        sub_b = os.path.join(self._proj_a, "backend")
        os.makedirs(sub_a)
        os.makedirs(sub_b)
        gc.set_project_profile(sub_a, "personal")
        gc.set_project_profile(sub_b, "work")
        self.assertEqual(gc.get_project_profile(sub_a), "personal")
        self.assertEqual(gc.get_project_profile(sub_b), "work")

    def test_override_file_perms_are_restrictive(self):
        gc.set_project_profile(self._proj_a, "personal")
        path = gc._project_meta_path(self._proj_a)
        self.assertTrue(os.path.exists(path))
        import stat
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)

    def test_get_project_profile_returns_none_when_unset(self):
        self.assertIsNone(gc.get_project_profile(self._proj_a))

    def test_get_project_profile_strips_whitespace(self):
        # Pad active_profile in project.json; get_project_profile must strip.
        gc.set_project_profile(self._proj_a, "personal")
        from agentica.project_store import project_base_dir, read_project_file, write_project_file
        base = project_base_dir(self._proj_a)
        data = read_project_file(base)
        data["active_profile"] = "  personal  \n"
        write_project_file(base, data)
        self.assertEqual(gc.get_project_profile(self._proj_a), "personal")


if __name__ == "__main__":
    unittest.main()