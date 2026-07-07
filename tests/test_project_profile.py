# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for project-scoped profile overrides.

These verify the two-layer profile resolution introduced in
``global_config.resolve_active_profile_name``:

  1. Project override (``~/.agentica/projects/<key>/profile``)
  2. Global default (``config.yaml -> active_profile``)

Key contract: the project override is keyed by ``realpath(work_dir)``, NOT
by git toplevel. This aligns with Workspace / AGENTICA_HOME and needs no
fallback logic for non-git directories.
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

    def test_realpath_symlink_consistency(self):
        """Symlink pointing at the same real dir must resolve to the same key."""
        link = os.path.join(self._tmp.name, "link_to_a")
        os.symlink(self._proj_a, link)
        gc.set_project_profile(self._proj_a, "personal")
        # Resolving via the symlink should still see the override.
        name, source = gc.resolve_active_profile_name(work_dir=link)
        self.assertEqual((name, source), ("personal", "project"))

    def test_project_key_uses_workdir_not_git_toplevel(self):
        """Two sibling subdirs get INDEPENDENT keys even inside one git repo.

        We don't actually initialize git here — the point is that the key
        function only looks at realpath, so it can't accidentally collapse
        sibling directories together the way ``git rev-parse --show-toplevel``
        would.
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
        path = gc._project_profile_path(self._proj_a)
        self.assertTrue(os.path.exists(path))
        import stat
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)

    def test_get_project_profile_returns_none_when_unset(self):
        self.assertIsNone(gc.get_project_profile(self._proj_a))

    def test_get_project_profile_strips_whitespace(self):
        # Write a file with trailing whitespace/newlines manually to make sure
        # get_project_profile normalises reads.
        gc.set_project_profile(self._proj_a, "personal")
        path = gc._project_profile_path(self._proj_a)
        with open(path, "w", encoding="utf-8") as f:
            f.write("  personal  \n\n")
        self.assertEqual(gc.get_project_profile(self._proj_a), "personal")


if __name__ == "__main__":
    unittest.main()