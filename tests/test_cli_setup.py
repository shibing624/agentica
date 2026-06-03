# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for the CLI first-run model provider onboarding.
"""
import argparse
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.cli import setup as cli_setup


def _make_args(**overrides):
    """Build an argparse.Namespace like parse_args() would produce."""
    base = dict(model_provider=None, model_name=None, base_url=None, api_key=None)
    base.update(overrides)
    return argparse.Namespace(**base)


class TestCliConfigPersistence(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._patch = patch.object(
            cli_setup, "CLI_CONFIG_PATH",
            os.path.join(self._tmp.name, "cli_config.json"),
        )
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def test_load_missing_returns_empty(self):
        self.assertEqual(cli_setup.load_cli_config(), {})

    def test_save_and_load_roundtrip(self):
        cli_setup.save_cli_config({"onboarded": True, "model_provider": "openai"})
        loaded = cli_setup.load_cli_config()
        self.assertTrue(loaded["onboarded"])
        self.assertEqual(loaded["model_provider"], "openai")

    def test_config_complete_requires_provider_model_and_base_url(self):
        self.assertFalse(cli_setup.is_cli_config_complete({"onboarded": True}))
        self.assertTrue(cli_setup.is_cli_config_complete({
            "onboarded": True,
            "model_provider": "deepseek",
            "model_name": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com",
        }))

    def test_is_onboarded(self):
        self.assertFalse(cli_setup.is_onboarded())
        cli_setup.save_cli_config({"onboarded": True})
        self.assertTrue(cli_setup.is_onboarded())

    def test_invalid_json_returns_empty(self):
        with open(cli_setup.CLI_CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write("{ not valid json")
        self.assertEqual(cli_setup.load_cli_config(), {})


class TestProviderHelpers(unittest.TestCase):
    def test_provider_env_var_known(self):
        self.assertEqual(cli_setup.provider_env_var("deepseek"), "DEEPSEEK_API_KEY")
        self.assertEqual(cli_setup.provider_env_var("zhipuai"), "ZAI_API_KEY")

    def test_provider_env_var_unknown_falls_back_to_openai(self):
        self.assertEqual(cli_setup.provider_env_var("whatever"), "OPENAI_API_KEY")

    def test_provider_defaults(self):
        self.assertEqual(cli_setup.default_base_url("deepseek"), "https://api.deepseek.com")
        self.assertEqual(cli_setup.default_model_name("openai"), "gpt-4o")
        self.assertIsNone(cli_setup.default_base_url("azure"))

    def test_has_api_key_reads_env(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "x"}, clear=False):
            self.assertTrue(cli_setup.has_api_key("deepseek"))

    def test_has_api_key_zhipuai_legacy_fallback(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("ZAI_API_KEY", "ZHIPUAI_API_KEY")}
        env["ZHIPUAI_API_KEY"] = "legacy"
        with patch.dict(os.environ, env, clear=True):
            self.assertTrue(cli_setup.has_api_key("zhipuai"))


class TestSaveApiKeyToEnv(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._env_path = os.path.join(self._tmp.name, ".env")
        self._patch = patch.object(cli_setup, "AGENTICA_DOTENV_PATH", self._env_path)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def test_writes_new_key(self):
        cli_setup.save_api_key_to_env("DEEPSEEK_API_KEY", "sk-abc")
        with open(self._env_path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("DEEPSEEK_API_KEY=sk-abc", content)
        self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "sk-abc")

    def test_updates_existing_key(self):
        with open(self._env_path, "w", encoding="utf-8") as f:
            f.write("OTHER=1\nDEEPSEEK_API_KEY=old\n")
        cli_setup.save_api_key_to_env("DEEPSEEK_API_KEY", "new")
        with open(self._env_path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("DEEPSEEK_API_KEY=new", content)
        self.assertNotIn("DEEPSEEK_API_KEY=old", content)
        self.assertIn("OTHER=1", content)


class TestResolveModelConfig(unittest.TestCase):
    """resolve_model_config: args > saved config > defaults; onboarding gating."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._patch = patch.object(
            cli_setup, "CLI_CONFIG_PATH",
            os.path.join(self._tmp.name, "cli_config.json"),
        )
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def test_defaults_when_nothing_set(self):
        # No console -> never onboards; falls back to hardcoded defaults.
        resolved = cli_setup.resolve_model_config(_make_args(), console=None)
        self.assertEqual(resolved["model_provider"], cli_setup.DEFAULT_PROVIDER)
        self.assertEqual(resolved["model_name"], cli_setup.DEFAULT_MODEL)
        self.assertEqual(resolved["base_url"], "https://api.deepseek.com")

    def test_saved_config_used(self):
        cli_setup.save_cli_config({
            "onboarded": True,
            "model_provider": "openai",
            "model_name": "gpt-4o",
            "base_url": "https://example/v1",
        })
        resolved = cli_setup.resolve_model_config(_make_args(), console=None)
        self.assertEqual(resolved["model_provider"], "openai")
        self.assertEqual(resolved["model_name"], "gpt-4o")
        self.assertEqual(resolved["base_url"], "https://example/v1")

    def test_cli_args_override_saved(self):
        cli_setup.save_cli_config({
            "onboarded": True,
            "model_provider": "openai",
            "model_name": "gpt-4o",
            "base_url": "https://api.openai.com/v1",
        })
        args = _make_args(model_provider="deepseek", model_name="deepseek-chat")
        resolved = cli_setup.resolve_model_config(args, console=None)
        self.assertEqual(resolved["model_provider"], "deepseek")
        self.assertEqual(resolved["model_name"], "deepseek-chat")
        self.assertEqual(resolved["base_url"], "https://api.deepseek.com")

    def test_cli_model_override_keeps_saved_base_url_for_same_provider(self):
        cli_setup.save_cli_config({
            "onboarded": True,
            "model_provider": "openai",
            "model_name": "gpt-4o",
            "base_url": "https://proxy.example/v1",
        })
        args = _make_args(model_name="gpt-5")
        resolved = cli_setup.resolve_model_config(args, console=None)
        self.assertEqual(resolved["model_provider"], "openai")
        self.assertEqual(resolved["model_name"], "gpt-5")
        self.assertEqual(resolved["base_url"], "https://proxy.example/v1")

    def test_explicit_provider_flag_skips_onboarding(self):
        # Even on a TTY, passing --model_provider must not trigger the wizard.
        mock_console = MagicMock()
        with patch.object(cli_setup, "run_onboarding") as mock_wizard:
            args = _make_args(model_provider="deepseek")
            cli_setup.resolve_model_config(args, console=mock_console)
            mock_wizard.assert_not_called()

    def test_onboarding_triggered_and_applied(self):
        mock_console = MagicMock()
        with patch.object(cli_setup, "should_onboard", return_value=True), \
             patch.object(cli_setup, "run_onboarding", return_value={
                 "model_provider": "openai",
                 "model_name": "gpt-4o",
                 "base_url": "https://api.openai.com/v1",
             }) as mock_wizard:
            resolved = cli_setup.resolve_model_config(_make_args(), console=mock_console)
            mock_wizard.assert_called_once()
            self.assertEqual(resolved["model_provider"], "openai")
            self.assertEqual(resolved["model_name"], "gpt-4o")


class TestShouldOnboard(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._patch = patch.object(
            cli_setup, "CLI_CONFIG_PATH",
            os.path.join(self._tmp.name, "cli_config.json"),
        )
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def _tty(self):
        return patch.multiple(
            cli_setup.sys,
            stdin=MagicMock(isatty=MagicMock(return_value=True)),
            stdout=MagicMock(isatty=MagicMock(return_value=True)),
        )

    def test_skips_when_config_complete_and_key_present(self):
        cli_setup.save_cli_config({
            "onboarded": True,
            "model_provider": "deepseek",
            "model_name": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com",
        })
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "x"}, clear=True), self._tty():
            self.assertFalse(cli_setup.should_onboard("deepseek"))

    def test_onboards_when_key_present_but_config_incomplete(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "x"}, clear=True), self._tty():
            self.assertTrue(cli_setup.should_onboard("deepseek"))

    def test_onboards_when_config_complete_but_key_missing(self):
        cli_setup.save_cli_config({
            "onboarded": True,
            "model_provider": "deepseek",
            "model_name": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com",
        })
        with patch.dict(os.environ, {}, clear=True), self._tty():
            self.assertTrue(cli_setup.should_onboard("deepseek"))

    def test_requires_tty(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(cli_setup.sys.stdin, "isatty", return_value=False):
            self.assertFalse(cli_setup.should_onboard("deepseek"))


class TestRunOnboarding(unittest.TestCase):
    """Drive the wizard end-to-end with mocked prompt input."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._cfg_patch = patch.object(
            cli_setup, "CLI_CONFIG_PATH",
            os.path.join(self._tmp.name, "cli_config.json"),
        )
        self._env_patch = patch.object(
            cli_setup, "AGENTICA_DOTENV_PATH",
            os.path.join(self._tmp.name, ".env"),
        )
        self._cfg_patch.start()
        self._env_patch.start()

    def tearDown(self):
        self._cfg_patch.stop()
        self._env_patch.stop()
        self._tmp.cleanup()

    def test_preset_provider_flow(self):
        console = MagicMock()
        console.width = 80
        # prompts in order: provider pick("2"=openai), base_url(""=default),
        # api_key("sk-test"), model_name(""=default gpt-4o)
        inputs = iter(["2", "", "sk-test", ""])
        with patch.object(cli_setup, "pt_prompt", side_effect=lambda *a, **k: next(inputs)):
            result = cli_setup.run_onboarding(console)

        self.assertEqual(result["model_provider"], "openai")
        self.assertEqual(result["model_name"], "gpt-4o")
        self.assertEqual(result["base_url"], "https://api.openai.com/v1")

        saved = cli_setup.load_cli_config()
        self.assertTrue(saved["onboarded"])
        self.assertEqual(saved["model_provider"], "openai")
        self.assertEqual(saved["model_name"], "gpt-4o")
        self.assertEqual(saved["base_url"], "https://api.openai.com/v1")
        self.assertEqual(os.environ.get("OPENAI_API_KEY"), "sk-test")

    def test_custom_endpoint_flow(self):
        console = MagicMock()
        console.width = 80
        # provider pick (custom = last index after all presets), base_url, api_key, model_name
        custom_index = str(len(cli_setup._PROVIDER_ORDER) + 1)
        inputs = iter([custom_index, "https://my-llm.local/v1", "sk-custom", "my-model"])
        with patch.object(cli_setup, "pt_prompt", side_effect=lambda *a, **k: next(inputs)):
            result = cli_setup.run_onboarding(console)

        self.assertEqual(result["model_provider"], "openai")
        self.assertEqual(result["base_url"], "https://my-llm.local/v1")
        self.assertEqual(result["model_name"], "my-model")
        saved = cli_setup.load_cli_config()
        self.assertEqual(saved["base_url"], "https://my-llm.local/v1")
        self.assertEqual(saved["model_name"], "my-model")
        self.assertEqual(os.environ.get("OPENAI_API_KEY"), "sk-custom")


if __name__ == "__main__":
    unittest.main()
