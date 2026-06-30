# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Unit tests for CLI first-run onboarding + model config resolution.

Configuration lives in ~/.agentica/config.yaml (YAML, named profiles). There is
no cli_config.json anymore. Each profile has a main model and an optional
``auxiliary_model`` sub-block (the cheap model for background LLM work + the `task`
subagent tool).
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
    base = dict(
        model_provider=None,
        model_name=None,
        base_url=None,
        api_key=None,
        auxiliary_model_provider=None,
        auxiliary_model_name=None,
        auxiliary_base_url=None,
        auxiliary_api_key=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


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
        env = {k: v for k, v in os.environ.items() if k not in ("ZAI_API_KEY", "ZHIPUAI_API_KEY")}
        env["ZHIPUAI_API_KEY"] = "legacy"
        with patch.dict(os.environ, env, clear=True):
            self.assertTrue(cli_setup.has_api_key("zhipuai"))

    def test_custom_endpoint_does_not_fall_back_to_openai_env(self):
        """Custom (openai + non-default base_url) must NOT inherit OPENAI_API_KEY."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with (
            patch("agentica.global_config.global_config_path", return_value=os.path.join(tmp.name, "config.yaml")),
            patch.dict(os.environ, {"OPENAI_API_KEY": "real-openai"}, clear=True),
        ):
            self.assertFalse(
                cli_setup.has_api_key("openai", base_url="https://my-llm.local/v1")
            )
            # Canonical OpenAI endpoint still picks the env var up.
            self.assertTrue(
                cli_setup.has_api_key("openai", base_url="https://api.openai.com/v1")
            )

    def test_is_custom_openai(self):
        self.assertTrue(cli_setup._is_custom_openai("openai", "https://my-llm.local/v1"))
        self.assertFalse(cli_setup._is_custom_openai("openai", "https://api.openai.com/v1"))
        self.assertFalse(cli_setup._is_custom_openai("deepseek", "https://anywhere"))

    def test_get_profile_api_key_reads_profile(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with patch("agentica.global_config.global_config_path", return_value=os.path.join(tmp.name, "config.yaml")):
            from agentica import global_config as gc
            gc.upsert_profile("zhipuai", {
                "model_provider": "zhipuai", "model_name": "glm-flash",
                "base_url": cli_setup.default_base_url("zhipuai"), "api_key": "sk-zhipu",
            }, make_active=False)
            self.assertEqual(cli_setup.get_profile_api_key("zhipuai"), "sk-zhipu")
            self.assertIsNone(cli_setup.get_profile_api_key("deepseek"))


class TestResolveModelConfig(unittest.TestCase):
    """resolve_model_config: args > config.yaml profile > defaults; onboarding gating."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._gc_patch = patch(
            "agentica.global_config.global_config_path",
            return_value=os.path.join(self._tmp.name, "config.yaml"),
        )
        self._gc_patch.start()

    def tearDown(self):
        self._gc_patch.stop()
        self._tmp.cleanup()

    def _write_profile(self, profile: dict, name: str = "default", make_active: bool = True):
        from agentica import global_config as gc
        gc.upsert_profile(name, profile, make_active=make_active)

    def test_defaults_when_nothing_set(self):
        # No console -> never onboards; falls back to hardcoded defaults.
        resolved = cli_setup.resolve_model_config(_make_args(), console=None)
        self.assertEqual(resolved["model_provider"], cli_setup.DEFAULT_PROVIDER)
        self.assertEqual(resolved["model_name"], cli_setup.DEFAULT_MODEL)
        self.assertEqual(resolved["base_url"], "https://api.deepseek.com")

    def test_profile_used(self):
        self._write_profile({
            "model_provider": "openai", "model_name": "gpt-4o",
            "base_url": "https://example/v1", "api_key": "sk-x",
        })
        resolved = cli_setup.resolve_model_config(_make_args(), console=None)
        self.assertEqual(resolved["model_provider"], "openai")
        self.assertEqual(resolved["model_name"], "gpt-4o")
        self.assertEqual(resolved["base_url"], "https://example/v1")
        self.assertEqual(resolved["api_key"], "sk-x")

    def test_cli_args_override_profile(self):
        self._write_profile({
            "model_provider": "openai", "model_name": "gpt-4o",
            "base_url": "https://api.openai.com/v1", "api_key": "sk-x",
        })
        args = _make_args(model_provider="deepseek", model_name="deepseek-chat")
        resolved = cli_setup.resolve_model_config(args, console=None)
        self.assertEqual(resolved["model_provider"], "deepseek")
        self.assertEqual(resolved["model_name"], "deepseek-chat")
        self.assertEqual(resolved["base_url"], "https://api.deepseek.com")

    def test_cli_model_override_keeps_profile_base_url_for_same_provider(self):
        self._write_profile({
            "model_provider": "openai", "model_name": "gpt-4o",
            "base_url": "https://proxy.example/v1", "api_key": "sk-x",
        })
        args = _make_args(model_name="gpt-5")
        resolved = cli_setup.resolve_model_config(args, console=None)
        self.assertEqual(resolved["model_provider"], "openai")
        self.assertEqual(resolved["model_name"], "gpt-5")
        self.assertEqual(resolved["base_url"], "https://proxy.example/v1")

    def test_explicit_provider_flag_skips_onboarding(self):
        mock_console = MagicMock()
        with patch.object(cli_setup, "run_onboarding") as mock_wizard:
            args = _make_args(model_provider="deepseek")
            cli_setup.resolve_model_config(args, console=mock_console)
            mock_wizard.assert_not_called()

    def test_onboarding_triggered_and_applied(self):
        mock_console = MagicMock()
        with (
            patch.object(cli_setup, "should_onboard", return_value=True),
            patch.object(
                cli_setup,
                "run_onboarding",
                return_value={
                    "model_provider": "openai", "model_name": "gpt-4o",
                    "base_url": "https://api.openai.com/v1", "api_key": "sk-from-wizard",
                    "auxiliary_model_provider": None, "auxiliary_model_name": None,
                    "auxiliary_base_url": None, "auxiliary_api_key": None,
                },
            ) as mock_wizard,
        ):
            resolved = cli_setup.resolve_model_config(_make_args(), console=mock_console)
            mock_wizard.assert_called_once()
            self.assertEqual(resolved["model_provider"], "openai")
            self.assertEqual(resolved["api_key"], "sk-from-wizard")

    def test_returns_profile_api_key(self):
        self._write_profile({
            "model_provider": "deepseek", "model_name": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com", "api_key": "sk-from-profile",
        })
        resolved = cli_setup.resolve_model_config(_make_args(), console=None)
        self.assertEqual(resolved["api_key"], "sk-from-profile")


class TestResolveAuxModel(unittest.TestCase):
    """Optional auxiliary model resolution: CLI flag > profile auxiliary_model block >
    main model (same provider) / preset + matching profile (cross provider)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._gc_patch = patch(
            "agentica.global_config.global_config_path",
            return_value=os.path.join(self._tmp.name, "config.yaml"),
        )
        self._gc_patch.start()

    def tearDown(self):
        self._gc_patch.stop()
        self._tmp.cleanup()

    def _write_profile(self, profile: dict, name: str = "default", make_active: bool = True):
        from agentica import global_config as gc
        gc.upsert_profile(name, profile, make_active=make_active)

    def test_no_block_no_flags_returns_none(self):
        self._write_profile({
            "model_provider": "deepseek", "model_name": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com", "api_key": "sk-main",
        })
        resolved = cli_setup.resolve_model_config(_make_args(), console=None)
        for k in ("auxiliary_model_name", "auxiliary_model_provider", "auxiliary_base_url", "auxiliary_api_key"):
            self.assertIsNone(resolved[k], k)

    def test_same_provider_inherits_main_base_and_key(self):
        self._write_profile({
            "model_provider": "deepseek", "model_name": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com", "api_key": "sk-main",
            "auxiliary_model": {"model_name": "deepseek-chat"},  # only name; same provider
        })
        resolved = cli_setup.resolve_model_config(_make_args(), console=None)
        self.assertEqual(resolved["auxiliary_model_provider"], "deepseek")
        self.assertEqual(resolved["auxiliary_model_name"], "deepseek-chat")
        self.assertEqual(resolved["auxiliary_base_url"], "https://api.deepseek.com")
        self.assertEqual(resolved["auxiliary_api_key"], "sk-main")

    def test_cross_provider_uses_preset_base_and_block_key(self):
        self._write_profile({
            "model_provider": "deepseek", "model_name": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com", "api_key": "sk-main",
            "auxiliary_model": {
                "model_provider": "zhipuai", "model_name": "glm-4.7-flash",
                "api_key": "sk-zhipu",
            },
        })
        resolved = cli_setup.resolve_model_config(_make_args(), console=None)
        self.assertEqual(resolved["auxiliary_model_provider"], "zhipuai")
        self.assertEqual(resolved["auxiliary_model_name"], "glm-4.7-flash")
        # base_url absent in block -> zhipuai preset default, NOT deepseek's.
        self.assertEqual(resolved["auxiliary_base_url"], cli_setup.default_base_url("zhipuai"))
        # block key used, NOT main key.
        self.assertEqual(resolved["auxiliary_api_key"], "sk-zhipu")

    def test_cross_provider_no_block_key_uses_matching_profile(self):
        self._write_profile({
            "model_provider": "deepseek", "model_name": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com", "api_key": "sk-main",
            "auxiliary_model": {"model_provider": "zhipuai", "model_name": "glm-4.7-flash"},
        })
        # A separate profile stores the zhipuai key (not active).
        self._write_profile({
            "model_provider": "zhipuai", "model_name": "glm-flash",
            "base_url": cli_setup.default_base_url("zhipuai"), "api_key": "sk-saved-zhipu",
        }, name="zhipuai", make_active=False)
        resolved = cli_setup.resolve_model_config(_make_args(), console=None)
        self.assertEqual(resolved["auxiliary_api_key"], "sk-saved-zhipu")

    def test_cross_provider_no_key_anywhere_returns_none_not_main(self):
        self._write_profile({
            "model_provider": "deepseek", "model_name": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com", "api_key": "sk-main",
            "auxiliary_model": {"model_provider": "zhipuai", "model_name": "glm-4.7-flash"},
        })
        with patch.dict(os.environ, {}, clear=True):
            resolved = cli_setup.resolve_model_config(_make_args(), console=None)
        self.assertIsNone(resolved["auxiliary_api_key"])  # must NOT fall back to sk-main
        self.assertEqual(resolved["auxiliary_base_url"], cli_setup.default_base_url("zhipuai"))

    def test_cli_flag_overrides_profile_block(self):
        self._write_profile({
            "model_provider": "deepseek", "model_name": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com", "api_key": "sk-main",
            "auxiliary_model": {"model_provider": "zhipuai", "model_name": "glm-4.7-flash"},
        })
        args = _make_args(auxiliary_model_provider="deepseek", auxiliary_model_name="deepseek-chat")
        resolved = cli_setup.resolve_model_config(args, console=None)
        self.assertEqual(resolved["auxiliary_model_name"], "deepseek-chat")
        self.assertEqual(resolved["auxiliary_model_provider"], "deepseek")
        self.assertEqual(resolved["auxiliary_base_url"], "https://api.deepseek.com")
        self.assertEqual(resolved["auxiliary_api_key"], "sk-main")


class TestShouldOnboard(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._gc_patch = patch(
            "agentica.global_config.global_config_path",
            return_value=os.path.join(self._tmp.name, "config.yaml"),
        )
        self._gc_patch.start()

    def tearDown(self):
        self._gc_patch.stop()
        self._tmp.cleanup()

    def _tty(self):
        return patch.multiple(
            cli_setup.sys,
            stdin=MagicMock(isatty=MagicMock(return_value=True)),
            stdout=MagicMock(isatty=MagicMock(return_value=True)),
        )

    def test_skips_when_profile_complete(self):
        from agentica import global_config as gc
        gc.upsert_profile("default", {
            "model_provider": "deepseek", "model_name": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com", "api_key": "sk-x",
        })
        with self._tty():
            self.assertFalse(cli_setup.should_onboard("deepseek"))

    def test_onboards_when_profile_missing_key(self):
        from agentica import global_config as gc
        gc.upsert_profile("default", {
            "model_provider": "deepseek", "model_name": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com",  # no api_key
        })
        with patch.dict(os.environ, {}, clear=True), self._tty():
            self.assertTrue(cli_setup.should_onboard("deepseek"))

    def test_skips_when_a_profile_has_key(self):
        from agentica import global_config as gc
        # No active profile, but a profile for the provider has a key.
        gc.upsert_profile("deepseek", {
            "model_provider": "deepseek", "model_name": "x",
            "base_url": "https://api.deepseek.com", "api_key": "sk-x",
        })
        with self._tty():
            self.assertFalse(cli_setup.should_onboard("deepseek"))

    def test_requires_tty(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(cli_setup.sys.stdin, "isatty", return_value=False):
            self.assertFalse(cli_setup.should_onboard("deepseek"))


class TestRunOnboarding(unittest.TestCase):
    """Drive the wizard end-to-end with mocked prompt input."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._gc_patch = patch(
            "agentica.global_config.global_config_path",
            return_value=os.path.join(self._tmp.name, "config.yaml"),
        )
        self._gc_patch.start()

    def tearDown(self):
        self._gc_patch.stop()
        self._tmp.cleanup()

    def test_preset_provider_flow(self):
        from agentica import global_config as gc
        console = MagicMock()
        console.width = 80
        # prompts: provider("2"=openai), base_url(""=default), api_key("sk-test"),
        # model_name(""=default gpt-4o), advanced("n"), auxiliary("n")
        inputs = iter(["2", "", "sk-test", "", "n", "n", "n"])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(cli_setup, "pt_prompt", side_effect=lambda *a, **k: next(inputs)),
        ):
            result = cli_setup.run_onboarding(console)

        self.assertEqual(result["model_provider"], "openai")
        self.assertEqual(result["model_name"], "gpt-4o")
        self.assertEqual(result["base_url"], "https://api.openai.com/v1")
        self.assertEqual(result["api_key"], "sk-test")
        profile = gc.get_profile()
        self.assertEqual(profile["model_provider"], "openai")
        self.assertEqual(profile["api_key"], "sk-test")
        self.assertNotIn("auxiliary_model", profile)
        # Wizard must NOT touch ~/.agentica/.env anymore.
        self.assertFalse(os.path.exists(os.path.join(self._tmp.name, ".env")))
        # And must NOT have exported the entered key under generic OpenAI env names.
        self.assertNotEqual(os.environ.get("OPENAI_API_KEY"), "sk-test")

    def test_custom_endpoint_flow(self):
        from agentica import global_config as gc
        console = MagicMock()
        console.width = 80
        custom_index = str(len(cli_setup._PROVIDER_ORDER) + 1)
        # provider(custom), base_url, api_key, model_name, advanced("n"), auxiliary("n")
        inputs = iter([custom_index, "https://my-llm.local/v1", "sk-custom", "my-model", "n", "n", "n"])
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "real-openai"}, clear=True),
            patch.object(cli_setup, "pt_prompt", side_effect=lambda *a, **k: next(inputs)),
        ):
            result = cli_setup.run_onboarding(console)

        self.assertEqual(result["model_provider"], "openai")
        self.assertEqual(result["base_url"], "https://my-llm.local/v1")
        self.assertEqual(result["model_name"], "my-model")
        self.assertEqual(result["api_key"], "sk-custom")
        profile = gc.get_profile()
        # Custom endpoint -> host-suffixed profile name, isolated from real openai.
        self.assertEqual(profile["base_url"], "https://my-llm.local/v1")
        self.assertEqual(profile["api_key"], "sk-custom")
        self.assertNotIn("auxiliary_model", profile)
        self.assertFalse(os.path.exists(os.path.join(self._tmp.name, ".env")))

    def test_advanced_params_saved_to_profile(self):
        from agentica import global_config as gc
        console = MagicMock()
        console.width = 80
        # provider("2"), base_url(""), api_key("sk"), model_name(""), advanced("y"),
        # reasoning("high"), max_tokens("4096"), context("500000"), temp("0.3"),
        # top_p("0.9"), auxiliary("n")
        inputs = iter(["2", "", "sk", "", "y", "high", "4096", "500000", "0.3", "0.9", "n", "n"])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(cli_setup, "pt_prompt", side_effect=lambda *a, **k: next(inputs)),
        ):
            result = cli_setup.run_onboarding(console)

        self.assertEqual(result["reasoning_effort"], "high")
        self.assertEqual(result["max_tokens"], 4096)
        self.assertEqual(result["context_window"], 500000)
        profile = gc.get_profile()
        self.assertEqual(profile["reasoning_effort"], "high")
        self.assertEqual(profile["max_tokens"], 4096)
        self.assertEqual(profile["context_window"], 500000)

    def test_auxiliary_model_skipped_when_answered_no(self):
        from agentica import global_config as gc
        console = MagicMock()
        console.width = 80
        inputs = iter(["2", "", "sk-main", "", "n", "n", "n"])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(cli_setup, "pt_prompt", side_effect=lambda *a, **k: next(inputs)),
        ):
            cli_setup.run_onboarding(console)
        self.assertNotIn("auxiliary_model", gc.get_profile())

    def test_cache_control_opt_in_persisted_to_profile(self):
        from agentica import global_config as gc
        console = MagicMock()
        console.width = 80
        # provider("2"), base_url(""), api_key("sk"), model_name(""), advanced("n"),
        # cache: yes("y"), messages("2"), header("Venus-Session-Id"), auxiliary("n")
        inputs = iter(["2", "", "sk", "", "n", "y", "2", "Venus-Session-Id", "n"])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(cli_setup, "pt_prompt", side_effect=lambda *a, **k: next(inputs)),
        ):
            cli_setup.run_onboarding(console)
        profile = gc.get_profile()
        self.assertTrue(profile.get("enable_cache_control"))
        self.assertEqual(profile.get("cache_control_messages"), 2)
        self.assertEqual(profile.get("cache_control_session_header"), "Venus-Session-Id")

    def test_cache_control_skipped_when_answered_no(self):
        from agentica import global_config as gc
        console = MagicMock()
        console.width = 80
        inputs = iter(["2", "", "sk", "", "n", "n", "n"])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(cli_setup, "pt_prompt", side_effect=lambda *a, **k: next(inputs)),
        ):
            cli_setup.run_onboarding(console)
        profile = gc.get_profile()
        self.assertNotIn("enable_cache_control", profile)

    def test_auxiliary_model_configured_and_persisted(self):
        from agentica import global_config as gc
        console = MagicMock()
        console.width = 80
        # main: openai("2"), default base, key("sk-main"), default model, adv("n");
        # auxiliary: yes("y"), zhipuai("4"), default base(""), key("sk-zhipu"), default model
        zhipu_idx = str(cli_setup._PROVIDER_ORDER.index("zhipuai") + 1)
        inputs = iter(["2", "", "sk-main", "", "n", "n", "y", zhipu_idx, "", "sk-zhipu", ""])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(cli_setup, "pt_prompt", side_effect=lambda *a, **k: next(inputs)),
        ):
            cli_setup.run_onboarding(console)
        profile = gc.get_profile()
        self.assertIn("auxiliary_model", profile)
        am = profile["auxiliary_model"]
        self.assertEqual(am["model_provider"], "zhipuai")
        self.assertEqual(am["model_name"], "glm-4.7-flash")
        self.assertEqual(am["base_url"], cli_setup.default_base_url("zhipuai"))
        self.assertEqual(am["api_key"], "sk-zhipu")

    def test_rerun_keeps_existing_on_all_enter(self):
        """A re-run walked through with Enter/decline preserves the whole profile."""
        from agentica import global_config as gc

        # Pre-seed a fully-configured active profile (main + tuning + cache + auxiliary).
        gc.upsert_profile(
            "openai",
            {
                "model_provider": "openai",
                "model_name": "gpt-4o",
                "base_url": "https://api.openai.com/v1",
                "api_key": "sk-keep-main",
                "reasoning_effort": "high",
                "max_tokens": 4096,
                "enable_cache_control": True,
                "cache_control_messages": 3,
                "auxiliary_model": {
                    "model_provider": "zhipuai",
                    "model_name": "glm-4.7-flash",
                    "base_url": cli_setup.default_base_url("zhipuai"),
                    "api_key": "sk-keep-auxiliary",
                },
            },
            make_active=True,
        )
        console = MagicMock()
        console.width = 80
        # 7 prompts, all "keep": provider(Enter), base_url(Enter), api_key(Enter),
        # model_name(Enter), advanced(edit? n), cache(edit? n), auxiliary(reconfigure? n).
        inputs = iter(["", "", "", "", "n", "n", "n"])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(cli_setup, "pt_prompt", side_effect=lambda *a, **k: next(inputs)),
        ):
            result = cli_setup.run_onboarding(console)

        # Main model kept.
        self.assertEqual(result["model_provider"], "openai")
        self.assertEqual(result["model_name"], "gpt-4o")
        self.assertEqual(result["base_url"], "https://api.openai.com/v1")
        self.assertEqual(result["api_key"], "sk-keep-main")
        profile = gc.get_profile()
        # Tuning kept without re-entry.
        self.assertEqual(profile.get("reasoning_effort"), "high")
        self.assertEqual(profile.get("max_tokens"), 4096)
        # Cache kept.
        self.assertTrue(profile.get("enable_cache_control"))
        self.assertEqual(profile.get("cache_control_messages"), 3)
        # Auxiliary block kept verbatim (not dropped by the decline).
        self.assertIn("auxiliary_model", profile)
        am = profile["auxiliary_model"]
        self.assertEqual(am["model_provider"], "zhipuai")
        self.assertEqual(am["model_name"], "glm-4.7-flash")
        self.assertEqual(am["api_key"], "sk-keep-auxiliary")

    def test_rerun_shows_api_key_in_full(self):
        """The api key prompt must NOT be masked (is_password=False) in setup."""
        from agentica import global_config as gc

        gc.upsert_profile(
            "openai",
            {
                "model_provider": "openai",
                "model_name": "gpt-4o",
                "base_url": "https://api.openai.com/v1",
                "api_key": "sk-visible",
            },
            make_active=True,
        )
        console = MagicMock()
        console.width = 80
        inputs = iter(["", "", "", "", "n", "n", "n"])
        calls: list = []

        def fake_prompt(*args, **kwargs):
            calls.append((args, kwargs))
            return next(inputs)

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(cli_setup, "pt_prompt", side_effect=fake_prompt),
        ):
            cli_setup.run_onboarding(console)

        api_key_calls = [c for c in calls if "API key" in (c[0][0] if c[0] else "")]
        self.assertTrue(api_key_calls, "expected an API key prompt")
        self.assertFalse(
            api_key_calls[0][1].get("is_password", False),
            "API key must be shown in full during setup (not masked)",
        )


class TestMultiProfileWizard(unittest.TestCase):
    """The wizard supports multiple named profiles per provider.

    Pre-refactor: profile name was derived from provider, so a user
    couldn't keep both ``opus`` and ``gpt5-fast`` on different OpenAI-style
    endpoints — re-running setup just overwrote the previous profile. Now
    the wizard prompts for a profile name and offers a "configure another?"
    loop so the same session can land multiple profiles.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._gc_patch = patch(
            "agentica.global_config.global_config_path",
            return_value=os.path.join(self._tmp.name, "config.yaml"),
        )
        self._gc_patch.start()

    def tearDown(self):
        self._gc_patch.stop()
        self._tmp.cleanup()

    def test_profile_name_prompt_creates_named_profile(self):
        """User-supplied name becomes the YAML key, not the provider slug."""
        from agentica import global_config as gc
        console = MagicMock()
        console.width = 80
        # provider("2"=openai), base_url(""), api_key("sk"), model_name(""),
        # advanced("n"), cache("n"), auxiliary("n"), profile_name("opus"),
        # configure_another("n").
        inputs = iter(["2", "", "sk-a", "", "n", "n", "n", "opus", "n"])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(cli_setup, "pt_prompt", side_effect=lambda *a, **k: next(inputs)),
            patch.object(cli_setup, "_prompt_cron", return_value=None),
        ):
            cli_setup.run_onboarding(console)

        profiles = gc.get_profiles()
        self.assertIn("opus", profiles)
        self.assertEqual(gc.get_active_profile_name(), "opus")

    def test_two_profiles_same_provider_coexist(self):
        """Two OpenAI-style profiles with different names both survive."""
        from agentica import global_config as gc
        console = MagicMock()
        console.width = 80
        # First profile: opus
        inputs1 = iter(["2", "", "sk-opus", "gpt-5", "n", "n", "n", "opus", "y"])
        # Second profile (loop iteration 2): gpt5-fast — provider+model differ
        inputs2 = iter(["2", "", "sk-fast", "gpt-5-mini", "n", "n", "n", "gpt5-fast", "n"])
        all_inputs = iter(list(inputs1) + list(inputs2))
        # Active picker: pick "gpt5-fast" (index 2).
        # Wrapped above into one stream + a final picker answer.
        all_with_picker = iter(list(all_inputs) + ["2"])

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(cli_setup, "pt_prompt", side_effect=lambda *a, **k: next(all_with_picker)),
            patch.object(cli_setup, "_prompt_cron", return_value=None),
        ):
            cli_setup.run_onboarding(console)

        profiles = gc.get_profiles()
        self.assertIn("opus", profiles)
        self.assertIn("gpt5-fast", profiles)
        self.assertEqual(profiles["opus"]["model_name"], "gpt-5")
        self.assertEqual(profiles["gpt5-fast"]["model_name"], "gpt-5-mini")
        self.assertEqual(profiles["opus"]["api_key"], "sk-opus")
        self.assertEqual(profiles["gpt5-fast"]["api_key"], "sk-fast")
        self.assertEqual(gc.get_active_profile_name(), "gpt5-fast")

    def test_suggest_profile_name_auto_suffixes_on_collision(self):
        """``_suggest_profile_name`` never returns a name that already exists."""
        existing = {"openai": {}, "openai-2": {}}
        suggested = cli_setup._suggest_profile_name("openai", None, existing)
        self.assertEqual(suggested, "openai-3")
        # No collision => seed is used unchanged.
        self.assertEqual(
            cli_setup._suggest_profile_name("deepseek", None, existing),
            "deepseek",
        )


class TestConfigYamlRoundTrip(unittest.TestCase):
    """config.yaml is YAML with ruamel round-trip: user comments survive writes."""

    def test_upsert_preserves_comments(self):
        from agentica import global_config as gc
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "config.yaml")
        with patch("agentica.global_config.global_config_path", return_value=path):
            gc.write_commented_template()
            gc.upsert_profile("default", {
                "model_provider": "anthropic", "model_name": "claude",
                "base_url": "https://api.anthropic.com", "api_key": "sk-x",
            })
            txt = open(path).read()
            # Template header comments must survive the upsert write.
            self.assertIn("Hand-edit freely", txt)
            # And the data round-trips.
            self.assertEqual(gc.get_profile()["model_provider"], "anthropic")


class TestProfileValidation(unittest.TestCase):
    """config.yaml is core: invalid input is re-prompted, invalid profiles are
    never written. Tests the validators, the per-field re-prompt loops, and the
    final pre-write gate in run_onboarding()."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._gc_patch = patch(
            "agentica.global_config.global_config_path",
            return_value=os.path.join(self._tmp.name, "config.yaml"),
        )
        self._gc_patch.start()

    def tearDown(self):
        self._gc_patch.stop()
        self._tmp.cleanup()

    # ── _validate_base_url ──────────────────────────────────────────────────
    def test_validate_base_url_accepts_http_https(self):
        self.assertIsNone(cli_setup._validate_base_url("https://api.example.com/v1"))
        self.assertIsNone(cli_setup._validate_base_url("http://localhost:8080"))

    def test_validate_base_url_rejects_bad_input(self):
        self.assertIsNotNone(cli_setup._validate_base_url(""))
        self.assertIsNotNone(cli_setup._validate_base_url("foo"))
        self.assertIsNotNone(cli_setup._validate_base_url("ftp://files.example.com"))
        self.assertIsNotNone(cli_setup._validate_base_url("https:///no-host"))

    # ── _validate_profile ───────────────────────────────────────────────────
    def test_validate_profile_valid(self):
        errors = cli_setup._validate_profile({
            "model_provider": "openai",
            "model_name": "gpt-4o",
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-x",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 4096,
        })
        self.assertEqual(errors, [])

    def test_validate_profile_flags_each_invalid_field(self):
        errors = cli_setup._validate_profile({
            "model_provider": "not-a-provider",
            "model_name": "",
            "base_url": "no-scheme",
            "reasoning_effort": "ultra",
            "max_tokens": 0,
            "context_window": -1,
            "temperature": 5.0,
            "top_p": 2.0,
            "enable_cache_control": "yes",
            "cache_control_messages": 0,
            "cache_control_session_header": "   ",
            "auxiliary_model": {"model_provider": "x", "model_name": "", "base_url": "bad"},
        })
        joined = " | ".join(errors)
        for needle in (
            "model_provider", "model_name", "base_url", "reasoning_effort",
            "max_tokens", "context_window", "temperature", "top_p",
            "enable_cache_control", "cache_control_messages",
            "cache_control_session_header", "auxiliary_model",
        ):
            self.assertIn(needle, joined)

    # ── per-field re-prompt loops ────────────────────────────────────────────
    def test_prompt_base_url_reprompts_until_valid(self):
        console = MagicMock()
        inputs = iter(["foo", "ftp://x", "https://api.example.com/v1"])
        with patch.object(cli_setup, "pt_prompt", side_effect=lambda *a, **k: next(inputs)):
            url = cli_setup._prompt_base_url(console, label="  Base URL: ")
        self.assertEqual(url, "https://api.example.com/v1")

    def test_prompt_float_range_reprompts_until_in_range(self):
        console = MagicMock()
        inputs = iter(["99", "-1", "0.5"])
        with patch.object(cli_setup, "pt_prompt", side_effect=lambda *a, **k: next(inputs)):
            val = cli_setup._prompt_float_range(console, "  Temperature: ", 0.0, 2.0)
        self.assertEqual(val, 0.5)

    def test_prompt_base_url_accepts_blank_default_when_valid(self):
        console = MagicMock()
        inputs = iter([""])
        with patch.object(cli_setup, "pt_prompt", side_effect=lambda *a, **k: next(inputs)):
            url = cli_setup._prompt_base_url(
                console, label="  Base URL [https://api.openai.com/v1]: ",
                default="https://api.openai.com/v1",
            )
        self.assertEqual(url, "https://api.openai.com/v1")

    # ── final pre-write gate ────────────────────────────────────────────────
    def test_onboarding_reprompts_invalid_base_url(self):
        from agentica import global_config as gc
        console = MagicMock()
        console.width = 80
        custom_index = str(len(cli_setup._PROVIDER_ORDER) + 1)
        # provider(custom), base_url("foo"->re-prompt->valid), key, model, adv, cache, auxiliary
        inputs = iter([custom_index, "foo", "https://my-llm.local/v1", "sk", "m", "n", "n", "n"])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(cli_setup, "pt_prompt", side_effect=lambda *a, **k: next(inputs)),
        ):
            result = cli_setup.run_onboarding(console)
        self.assertEqual(result["base_url"], "https://my-llm.local/v1")
        self.assertEqual(gc.get_profile()["base_url"], "https://my-llm.local/v1")

    def test_onboarding_gate_blocks_invalid_prefill(self):
        """A corrupted existing profile (out-of-range temperature) carried
        through by declining to edit must NOT be re-written — the gate aborts."""
        from agentica import global_config as gc
        gc.upsert_profile("default", {
            "model_provider": "openai",
            "model_name": "gpt-4o",
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-x",
            "temperature": 99,  # invalid, seeded directly (upsert has no gate)
        }, make_active=True)
        console = MagicMock()
        console.width = 80
        # All-Enter keeps existing values; "n" declines advanced/cache/auxiliary edit.
        inputs = iter(["", "", "", "", "n", "n", "n"])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(cli_setup, "pt_prompt", side_effect=lambda *a, **k: next(inputs)),
        ):
            with self.assertRaises(SystemExit):
                cli_setup.run_onboarding(console)
        # The invalid temperature must still be on disk (upsert never ran).
        self.assertEqual(gc.get_profile().get("temperature"), 99)


if __name__ == "__main__":
    unittest.main()
