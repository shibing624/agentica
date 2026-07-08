# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for CLI self-management primitives and the self_manage tool.
"""
import json
import os
import tempfile
import unittest
from unittest.mock import patch


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

    def test_set_profile_field_rejects_invalid_value(self):
        """config.yaml is core: `/model set temperature 99` is refused, not written."""
        from agentica import global_config as gc
        tmp = tempfile.mkdtemp()
        import shutil
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        path = os.path.join(tmp, "config.yaml")
        with patch("agentica.global_config.global_config_path", return_value=path):
            gc.upsert_profile("default", {
                "model_provider": "openai", "model_name": "gpt-4o",
                "base_url": "https://api.openai.com/v1", "api_key": "sk-x",
            }, make_active=True)
            with self.assertRaises(ValueError):
                self.sm.set_profile_field("temperature", "99")
            with self.assertRaises(ValueError):
                self.sm.set_profile_field("base_url", "not-a-url")
            with self.assertRaises(ValueError):
                self.sm.set_profile_field("top_p", "2.0")
            # Nothing invalid was written.
            self.assertNotIn("temperature", gc.get_profile())
            self.assertEqual(gc.get_profile()["base_url"], "https://api.openai.com/v1")
            # A valid value is accepted and persisted.
            self.sm.set_profile_field("temperature", "0.5")
            self.assertEqual(gc.get_profile()["temperature"], 0.5)

    def test_set_profile_field_extra_body_accepts_json_object(self):
        from agentica import global_config as gc
        tmp = tempfile.mkdtemp()
        import shutil
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        path = os.path.join(tmp, "config.yaml")
        with patch("agentica.global_config.global_config_path", return_value=path):
            gc.upsert_profile("default", {
                "model_provider": "openai", "model_name": "hy3",
                "base_url": "http://api.taiji.woa.com/openapi/v2", "api_key": "sk-x",
            }, make_active=True)
            self.sm.set_profile_field(
                "extra_body", '{"chat_template_kwargs": {"reasoning_effort": "high"}}'
            )
            self.assertEqual(
                gc.get_profile()["extra_body"],
                {"chat_template_kwargs": {"reasoning_effort": "high"}},
            )

    def test_set_profile_field_extra_body_rejects_non_json(self):
        from agentica import global_config as gc
        tmp = tempfile.mkdtemp()
        import shutil
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        path = os.path.join(tmp, "config.yaml")
        with patch("agentica.global_config.global_config_path", return_value=path):
            gc.upsert_profile("default", {
                "model_provider": "openai", "model_name": "hy3",
                "base_url": "http://api.taiji.woa.com/openapi/v2", "api_key": "sk-x",
            }, make_active=True)
            with self.assertRaises(ValueError):
                self.sm.set_profile_field("extra_body", "not json")
            with self.assertRaises(ValueError):
                self.sm.set_profile_field("extra_body", "[1, 2, 3]")  # valid JSON, not an object
            self.assertNotIn("extra_body", gc.get_profile())

    def test_set_profile_field_extra_body_none_clears(self):
        from agentica import global_config as gc
        tmp = tempfile.mkdtemp()
        import shutil
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        path = os.path.join(tmp, "config.yaml")
        with patch("agentica.global_config.global_config_path", return_value=path):
            gc.upsert_profile("default", {
                "model_provider": "openai", "model_name": "hy3",
                "base_url": "http://api.taiji.woa.com/openapi/v2", "api_key": "sk-x",
                "extra_body": {"a": 1},
            }, make_active=True)
            self.sm.set_profile_field("extra_body", "none")
            self.assertNotIn("extra_body", gc.get_profile())


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
        from agentica.tools.base import Tool
        tool = self.SelfManageTool()
        # Must be a real Tool so model.add_tool registers `self_manage`.
        self.assertIsInstance(tool, Tool)
        self.assertEqual(list(tool.functions.keys()), ["self_manage"])

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

    def test_show_includes_settings_block(self):
        from agentica import global_config as gc
        config_path = os.path.join(self._tmp, "config.yaml")
        with patch("agentica.global_config.global_config_path", return_value=config_path):
            gc.upsert_profile("default", {
                "model_provider": "openai", "model_name": "gpt-4o",
                "base_url": "https://api.openai.com/v1", "api_key": "sk-x",
            }, make_active=True)
            gc.set_setting("cron.enabled", True)
            gc.set_setting("cron.interval", 30)
            out = json.loads(self.self_manage(action="show"))
        self.assertEqual(out["config"]["settings"]["cron.enabled"], True)
        self.assertEqual(out["config"]["settings"]["cron.interval"], 30)

    def test_default_restart_hint_defers_to_deployment_owner(self):
        """The gateway default must not tell a chat user to restart a CLI they don't have."""
        from agentica.tools.self_manage_tool import DEFAULT_RESTART_HINT
        self.assertIn("deployment", DEFAULT_RESTART_HINT)
        self.assertNotIn("CLI", DEFAULT_RESTART_HINT)

    def test_self_manage_tool_uses_custom_restart_hint(self):
        from agentica.tools.self_manage_tool import CLI_RESTART_HINT
        tool = self.SelfManageTool(restart_hint=CLI_RESTART_HINT)
        self.assertIn(CLI_RESTART_HINT, tool.description)


if __name__ == "__main__":
    unittest.main()