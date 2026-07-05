# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for the unified config.yaml (SDK + CLI shared, YAML).
"""

import os
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import global_config as gc


class TestGlobalConfig(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._patch = patch.object(
            gc,
            "global_config_path",
            return_value=os.path.join(self._tmp.name, "config.yaml"),
        )
        self._patch.start()
        # Snapshot env so injected keys don't leak between tests.
        self._env_snapshot = dict(os.environ)

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()
        os.environ.clear()
        os.environ.update(self._env_snapshot)

    def test_load_returns_empty_when_missing(self):
        self.assertEqual(gc.load_global_config(), {})

    def test_upsert_profile_and_make_active(self):
        gc.upsert_profile(
            "default",
            {
                "model_provider": "deepseek",
                "model_name": "deepseek-v4-flash",
                "base_url": "https://api.deepseek.com",
                "api_key": "sk-1",
            },
        )
        self.assertEqual(gc.get_active_profile_name(), "default")
        self.assertIn("default", gc.get_profiles())
        profile = gc.get_profile()
        self.assertEqual(profile["model_provider"], "deepseek")

    def test_upsert_drops_none_values(self):
        gc.upsert_profile(
            "p",
            {
                "model_provider": "openai",
                "model_name": "gpt-4o",
                "base_url": None,
                "api_key": None,
            },
        )
        profile = gc.get_profile("p")
        self.assertNotIn("base_url", profile)
        self.assertNotIn("api_key", profile)

    def test_set_active_profile_unknown_returns_false(self):
        gc.upsert_profile("default", {"model_provider": "deepseek"})
        self.assertFalse(gc.set_active_profile("nope"))
        self.assertTrue(gc.set_active_profile("default"))

    def test_file_written_with_restrictive_perms(self):
        gc.upsert_profile("default", {"model_provider": "deepseek", "api_key": "sk"})
        path = gc.global_config_path()
        self.assertTrue(os.path.exists(path))
        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)

    def test_apply_injects_active_profile_api_key(self):
        gc.upsert_profile(
            "default",
            {
                "model_provider": "deepseek",
                "api_key": "sk-deepseek",
            },
        )
        os.environ.pop("DEEPSEEK_API_KEY", None)
        gc.apply_global_config()
        self.assertEqual(os.environ.get("DEEPSEEK_API_KEY"), "sk-deepseek")

    def test_shell_env_wins_over_config(self):
        gc.upsert_profile(
            "default",
            {
                "model_provider": "openai",
                "api_key": "config-key",
            },
        )
        os.environ["OPENAI_API_KEY"] = "shell-key"
        gc.apply_global_config()
        # setdefault semantics: an already-present env var is never overwritten.
        self.assertEqual(os.environ.get("OPENAI_API_KEY"), "shell-key")

    def test_apply_injects_free_form_env_block(self):
        gc.save_global_config(
            {
                "active_profile": "default",
                "profiles": {},
                "env": {"SERPER_API_KEY": "serper-123"},
            }
        )
        os.environ.pop("SERPER_API_KEY", None)
        gc.apply_global_config()
        self.assertEqual(os.environ.get("SERPER_API_KEY"), "serper-123")

    def test_apply_empty_config_is_noop(self):
        # No file -> returns empty dict, injects nothing.
        self.assertEqual(gc.apply_global_config(), {})

    def test_custom_openai_endpoint_seeds_base_url(self):
        gc.upsert_profile(
            "default",
            {
                "model_provider": "openai",
                "base_url": "https://my-llm.local/v1",
                "api_key": "k",
            },
        )
        os.environ.pop("OPENAI_BASE_URL", None)
        os.environ.pop("OPENAI_API_KEY", None)
        gc.apply_global_config()
        self.assertEqual(os.environ.get("OPENAI_BASE_URL"), "https://my-llm.local/v1")

    def test_malformed_yaml_returns_empty(self):
        path = gc.global_config_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("active_profile: default\n  profiles: [this is : not : valid yaml\n")
        self.assertEqual(gc.load_global_config(), {})

    def test_find_profile_for_provider(self):
        gc.upsert_profile("default", {
            "model_provider": "deepseek", "model_name": "x",
            "base_url": "https://api.deepseek.com", "api_key": "sk-d",
        })
        gc.upsert_profile("zhipuai", {
            "model_provider": "zhipuai", "model_name": "glm",
            "base_url": "https://open.bigmodel.cn/api/paas/v4", "api_key": "sk-z",
        }, make_active=False)
        self.assertEqual(gc.find_profile_for_provider("zhipuai").get("api_key"), "sk-z")
        # base_url scoping: zhipuai with a different base_url is not matched.
        self.assertEqual(gc.find_profile_for_provider("zhipuai", "https://other/v4"), {})
        self.assertEqual(gc.find_profile_for_provider("anthropic"), {})

    def test_auxiliary_model_block_round_trips(self):
        gc.upsert_profile("default", {
            "model_provider": "anthropic", "model_name": "claude",
            "base_url": "https://api.anthropic.com", "api_key": "sk-main",
            "auxiliary_model": {
                "model_provider": "zhipuai", "model_name": "glm-flash",
                "base_url": "https://open.bigmodel.cn/api/paas/v4", "api_key": "sk-z",
            },
        })
        am = gc.get_profile().get("auxiliary_model")
        self.assertEqual(am["model_provider"], "zhipuai")
        self.assertEqual(am["api_key"], "sk-z")

    def test_upsert_preserves_user_comments(self):
        gc.write_commented_template()
        gc.upsert_profile("default", {
            "model_provider": "deepseek", "model_name": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com", "api_key": "sk-x",
        })
        txt = open(gc.global_config_path()).read()
        self.assertIn("Hand-edit freely", txt)

    def test_set_and_get_setting_round_trip(self):
        gc.save_global_config({"profiles": {}})

        gc.set_setting("cli_markdown", "on")

        data = gc.load_global_config()
        self.assertEqual(data["settings"]["cli_markdown"], "on")
        self.assertEqual(gc.get_setting("cli_markdown"), "on")


if __name__ == "__main__":
    unittest.main()
