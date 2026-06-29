# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for CLI self-management primitives and the self_manage tool.
"""
import json
import os
import tempfile
import unittest


class TestSelfManagePrimitives(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._orig_dotenv = os.environ.get("AGENTICA_DOTENV_PATH")
        os.environ["AGENTICA_DOTENV_PATH"] = os.path.join(self._tmp, ".env")
        import importlib
        from agentica import config as cfg
        importlib.reload(cfg)
        from agentica.cli import self_manage as sm
        importlib.reload(sm)
        self.sm = sm

    def tearDown(self):
        if self._orig_dotenv is None:
            os.environ.pop("AGENTICA_DOTENV_PATH", None)
        else:
            os.environ["AGENTICA_DOTENV_PATH"] = self._orig_dotenv

    def test_mask_secret(self):
        self.assertEqual(self.sm.mask_secret("api_key", "sk-1234567890abcdef"), "sk-1...cdef")
        self.assertEqual(self.sm.mask_secret("model_name", "gpt-4o"), "gpt-4o")
        self.assertEqual(self.sm.mask_secret("token", "abc"), "****")

    def test_version_comparison(self):
        self.assertTrue(self.sm.is_upgrade_available("1.0.0", "1.0.1"))
        self.assertTrue(self.sm.is_upgrade_available("1.4.6", "2.0.0"))
        self.assertFalse(self.sm.is_upgrade_available("1.4.6", "1.4.6"))
        self.assertFalse(self.sm.is_upgrade_available("1.4.6", None))

    def test_dotenv_round_trip(self):
        self.sm.set_dotenv_var("FOO", "bar")
        self.assertEqual(self.sm.read_dotenv()["FOO"], "bar")
        self.assertEqual(os.environ["FOO"], "bar")
        # update
        self.sm.set_dotenv_var("FOO", "baz")
        self.assertEqual(self.sm.read_dotenv()["FOO"], "baz")
        # delete
        self.sm.set_dotenv_var("FOO", None)
        self.assertNotIn("FOO", self.sm.read_dotenv())
        self.assertNotIn("FOO", os.environ)

    def test_dotenv_secret_masking(self):
        self.sm.set_dotenv_var("OPENAI_API_KEY", "sk-secret-12345678")
        self.assertIn("...", self.sm.read_dotenv()["OPENAI_API_KEY"])
        self.assertEqual(
            self.sm.read_dotenv(reveal_secrets=True)["OPENAI_API_KEY"],
            "sk-secret-12345678",
        )

    def test_dotenv_invalid_key(self):
        with self.assertRaises(ValueError):
            self.sm.set_dotenv_var("bad key!", "x")

    def test_set_profile_field_rejects_unknown(self):
        with self.assertRaises(ValueError):
            self.sm.set_profile_field("evil_field", "x")


class TestSelfManageTool(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._orig_dotenv = os.environ.get("AGENTICA_DOTENV_PATH")
        os.environ["AGENTICA_DOTENV_PATH"] = os.path.join(self._tmp, ".env")
        import importlib
        from agentica import config as cfg
        importlib.reload(cfg)
        from agentica.cli import self_manage as sm
        importlib.reload(sm)
        from agentica.tools.self_manage_tool import self_manage, SelfManageTool
        self.self_manage = self_manage
        self.SelfManageTool = SelfManageTool

    def tearDown(self):
        if self._orig_dotenv is None:
            os.environ.pop("AGENTICA_DOTENV_PATH", None)
        else:
            os.environ["AGENTICA_DOTENV_PATH"] = self._orig_dotenv

    def test_tool_object_exposes_function(self):
        tool = self.SelfManageTool()
        self.assertEqual([f.__name__ for f in tool.functions], ["self_manage"])

    def test_show(self):
        out = json.loads(self.self_manage(action="show"))
        self.assertIn("current_version", out)
        self.assertIn("config_path", out)

    def test_set_env_and_mask(self):
        out = json.loads(self.self_manage(action="set_env", key="X_KEY", value="v"))
        self.assertTrue(out["success"])
        self.assertEqual(os.environ["X_KEY"], "v")

    def test_set_env_requires_key(self):
        out = json.loads(self.self_manage(action="set_env"))
        self.assertIn("error", out)

    def test_upgrade_requires_confirm(self):
        import agentica.tools.self_manage_tool as smt
        orig = smt.sm.get_latest_version
        smt.sm.get_latest_version = lambda *a, **k: "99.0.0"
        try:
            out = json.loads(self.self_manage(action="upgrade"))
            self.assertIn("error", out)
            self.assertIn("confirm", out["error"].lower())
        finally:
            smt.sm.get_latest_version = orig

    def test_unknown_action(self):
        out = json.loads(self.self_manage(action="nope"))
        self.assertIn("error", out)

    def test_slash_commands_registered(self):
        from agentica.cli import commands
        self.assertIn("/upgrade", commands.COMMAND_REGISTRY)
        self.assertIn("/config", commands.COMMAND_REGISTRY)


if __name__ == "__main__":
    unittest.main()