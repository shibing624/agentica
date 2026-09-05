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



class TestCLIModelParams(unittest.TestCase):
    """get_model() and resolution should honour the extended tuning params."""

    def test_get_model_passes_top_p_and_context_window(self):
        from agentica.cli.runtime import get_model

        model = get_model(
            "deepseek",
            "deepseek-v4-flash",
            api_key="fake_key",
            max_tokens=4096,
            temperature=0.3,
            reasoning_effort="high",
            top_p=0.9,
            context_window=500000,
        )
        self.assertEqual(model.top_p, 0.9)
        self.assertEqual(model.context_window, 500000)
        self.assertEqual(model.max_tokens, 4096)
        self.assertEqual(model.reasoning_effort, "high")

    def test_get_model_context_window_overrides_catalog(self):
        from agentica.cli.runtime import get_model

        # Without an explicit value the catalog fills it (deepseek -> 1_000_000);
        # an explicit value must win.
        default_model = get_model("deepseek", "deepseek-v4-flash", api_key="k")
        override_model = get_model(
            "deepseek",
            "deepseek-v4-flash",
            api_key="k",
            context_window=42000,
        )
        self.assertNotEqual(default_model.context_window, 42000)
        self.assertEqual(override_model.context_window, 42000)

    def test_anthropic_accepts_top_p_and_reasoning_effort(self):
        # Anthropic now takes reasoning_effort too: the Claude model maps it to
        # adaptive thinking (thinking.type=adaptive + output_config.effort).
        from agentica.cli.runtime import get_model

        model = get_model(
            "anthropic",
            "claude-opus-4-8",
            api_key="k",
            top_p=0.8,
            context_window=300000,
            reasoning_effort="high",
            base_url="https://ignored",
        )
        self.assertEqual(model.top_p, 0.8)
        self.assertEqual(model.context_window, 300000)
        self.assertEqual(model.reasoning_effort, "high")
        # And it must actually enable adaptive thinking in the request kwargs.
        kwargs = model.request_kwargs
        self.assertEqual(kwargs.get("thinking"), {"type": "adaptive"})
        self.assertEqual(kwargs.get("extra_body", {}).get("output_config"), {"effort": "high"})
        # Adaptive thinking requires temperature=1; it must be forced.
        self.assertEqual(kwargs.get("temperature"), 1)

    def test_claude_opus_5_defaults_to_adaptive_high_thinking(self):
        from agentica.cli.runtime import get_model

        model = get_model(
            "anthropic",
            "anthropic/claude-opus-5",
            api_key="k",
        )

        self.assertEqual(model.reasoning_effort, "high")
        self.assertEqual(model.request_kwargs.get("thinking"), {"type": "adaptive"})
        self.assertEqual(
            model.request_kwargs.get("extra_body", {}).get("output_config"),
            {"effort": "high"},
        )

    def test_claude_opus_5_can_disable_thinking_explicitly(self):
        from agentica.cli.runtime import get_model

        model = get_model(
            "anthropic",
            "anthropic/claude-opus-5",
            api_key="k",
            reasoning_effort="off",
        )

        self.assertIsNone(model.reasoning_effort)
        self.assertIsNone(model.thinking)
        # "off" means we send no thinking field; the API then uses its own
        # default. The status bar still shows `off` when config.yaml says so.
        self.assertEqual(model.describe_thinking_mode(), "default")

    def test_claude_opus_5_supports_all_effort_levels(self):
        from agentica.cli.runtime import get_model

        for effort in ("low", "medium", "high", "extra-high", "max"):
            with self.subTest(effort=effort):
                model = get_model(
                    "anthropic",
                    "anthropic/claude-opus-5",
                    api_key="k",
                    reasoning_effort=effort,
                )

            self.assertEqual(model.reasoning_effort, effort)
            self.assertEqual(model.describe_thinking_mode(), f"adaptive(effort={effort})")
            self.assertEqual(
                model.request_kwargs.get("extra_body", {}).get("output_config"),
                {"effort": effort},
            )

    def test_openai_compatible_opus_5_unset_effort_is_default_not_off(self):
        """OpenOneAPI / openai/claude-opus-5 with no config.yaml effort.

        We send no reasoning_effort field; the gateway uses its own default.
        The status-bar label must not claim thinking is off.
        """
        from agentica.cli.runtime import get_model

        model = get_model(
            "openai",
            "openai/claude-opus-5",
            api_key="k",
        )
        self.assertIsNone(model.reasoning_effort)
        self.assertNotIn("reasoning_effort", model.request_kwargs)
        self.assertEqual(model.describe_thinking_mode(), "default")

    def test_openai_compatible_explicit_off_still_describes_as_off(self):
        from agentica.cli.runtime import get_model

        model = get_model(
            "openai",
            "openai/claude-opus-5",
            api_key="k",
            reasoning_effort="off",
        )
        self.assertEqual(model.reasoning_effort, "off")
        self.assertEqual(model.describe_thinking_mode(), "off")

    def test_get_model_passes_extra_body_and_extra_headers(self):
        from agentica.cli.runtime import get_model

        model = get_model(
            "openai",
            "hy3",
            api_key="k",
            base_url="http://api.taiji.woa.com/openapi/v2",
            extra_body={"chat_template_kwargs": {"reasoning_effort": "high"}},
            extra_headers={"X-Custom": "value"},
        )
        self.assertEqual(model.extra_body, {"chat_template_kwargs": {"reasoning_effort": "high"}})
        self.assertEqual(model.extra_headers, {"X-Custom": "value"})

    def test_get_model_skips_extra_body_for_anthropic(self):
        from agentica.cli.runtime import get_model

        model = get_model(
            "anthropic",
            "claude-opus-4-8",
            api_key="k",
            extra_body={"some": "thing"},
        )
        self.assertFalse(hasattr(model, "extra_body") and model.extra_body)

    def test_get_model_passes_default_headers_for_anthropic(self):
        """default_headers reaches the Claude client (sticky routing on a proxy)."""
        from agentica.cli.runtime import get_model

        model = get_model(
            "anthropic",
            "claude-opus-4-8",
            api_key="k",
            default_headers={"X-Sticky-Routing": "token"},
        )
        self.assertEqual(model.default_headers, {"X-Sticky-Routing": "token"})

    def test_get_model_skips_default_headers_for_openai(self):
        """default_headers is an anthropic-path knob; openai ignores it."""
        from agentica.cli.runtime import get_model

        model = get_model(
            "openai",
            "gpt-5.2",
            api_key="k",
            default_headers={"X-Sticky-Routing": "token"},
        )
        self.assertIsNone(model.default_headers)

    def test_get_model_passes_cache_session_header_for_anthropic(self):
        """Regression: the wire_api gate used to drop this silently for anthropic.

        Load-balancing proxies need it — without sticky routing, unrouted
        requests hit a much higher rate of schema-validation 400s.
        """
        from agentica.cli.runtime import get_model

        model = get_model(
            "anthropic",
            "claude-opus-5",
            api_key="k",
            cache_control_session_header="X-Session-Id",
        )
        self.assertEqual(model.cache_control_session_header, "X-Session-Id")

    def test_get_model_passes_cache_session_header_for_openai(self):
        """The same knob still reaches OpenAIChat (no behaviour change there)."""
        from agentica.cli.runtime import get_model

        model = get_model(
            "openai",
            "gpt-5.2",
            api_key="k",
            cache_control_session_header="X-Session-Id",
        )
        self.assertEqual(model.cache_control_session_header, "X-Session-Id")

    def test_get_model_still_gates_openai_only_cache_knobs(self):
        """cache_control_messages / cache_keepalive remain OpenAI-only."""
        from agentica.cli.runtime import get_model

        model = get_model(
            "anthropic",
            "claude-opus-5",
            api_key="k",
            cache_control_messages=5,
            cache_keepalive=False,
        )
        self.assertFalse(hasattr(model, "cache_control_messages"))
        self.assertFalse(hasattr(model, "cache_keepalive"))

    def test_reasoning_effort_accepts_low_medium(self):
        import sys
        from agentica.cli.runtime import parse_args

        with patch.object(sys, "argv", ["agentica", "--reasoning_effort", "low"]):
            args = parse_args()
        self.assertEqual(args.reasoning_effort, "low")

    def test_compact_token_limit_flag(self):
        import sys
        from agentica.cli.runtime import parse_args

        with patch.object(sys, "argv", ["agentica", "--compact-token-limit", "300000"]):
            args = parse_args()
        self.assertEqual(args.compact_token_limit, 300000)

    def test_resolve_model_config_carries_profile_tuning_params(self):
        import argparse
        from agentica.cli.setup import resolve_model_config

        profile = {
            "model_provider": "deepseek",
            "model_name": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com",
            "api_key": "sk-x",
            "reasoning_effort": "high",
            "max_tokens": 4096,
            "context_window": 500000,
            "compact_token_limit": 300000,
            "temperature": 0.3,
            "top_p": 0.9,
            "default_headers": {"X-Sticky-Routing": "token"},
        }
        args = argparse.Namespace(
            model_provider=None,
            model_name=None,
            base_url=None,
            api_key=None,
            auxiliary_model_provider=None,
            auxiliary_model_name=None,
            auxiliary_base_url=None,
            auxiliary_api_key=None,
        )
        with patch("agentica.cli.setup.get_profile", return_value=profile):
            resolved = resolve_model_config(args, console=None)

        self.assertEqual(resolved["reasoning_effort"], "high")
        self.assertEqual(resolved["max_tokens"], 4096)
        self.assertEqual(resolved["context_window"], 500000)
        self.assertEqual(resolved["compact_token_limit"], 300000)
        self.assertEqual(resolved["temperature"], 0.3)
        self.assertEqual(resolved["top_p"], 0.9)
        self.assertEqual(resolved["default_headers"], {"X-Sticky-Routing": "token"})


class TestBuildSiblingModel(unittest.TestCase):
    """_build_sibling_model: same-provider inherits main base_url/api_key;
    cross-provider does NOT (would silently produce a broken client)."""

    def _cfg(self, **over):
        base = dict(
            model_provider="deepseek",
            model_name="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            api_key="sk-main",
            max_tokens=None,
            temperature=None,
            reasoning_effort=None,
            top_p=None,
            context_window=None,
        )
        base.update(over)
        return base

    def test_none_when_no_sibling_name(self):
        from agentica.cli.runtime import _build_sibling_model

        with patch("agentica.cli.runtime.get_model") as gm:
            self.assertIsNone(_build_sibling_model(self._cfg(), "auxiliary"))
            gm.assert_not_called()

    def test_same_provider_inherits_main_base_and_key(self):
        from agentica.cli.runtime import _build_sibling_model

        cfg = self._cfg(auxiliary_model_name="deepseek-chat")  # only name; same provider
        with patch("agentica.cli.runtime.get_model") as gm:
            _build_sibling_model(cfg, "auxiliary")
        _args, kw = gm.call_args
        self.assertEqual(kw["model_provider"], "deepseek")
        self.assertEqual(kw["model_name"], "deepseek-chat")
        self.assertEqual(kw["base_url"], "https://api.deepseek.com")
        self.assertEqual(kw["api_key"], "sk-main")

    def test_cross_provider_uses_sibling_base_and_key(self):
        from agentica.cli.runtime import _build_sibling_model

        cfg = self._cfg(
            auxiliary_model_provider="zhipuai",
            auxiliary_model_name="glm-4.7-flash",
            auxiliary_base_url="https://open.bigmodel.cn/api/paas/v4",
            auxiliary_api_key="sk-zhipu",
        )
        with patch("agentica.cli.runtime.get_model") as gm:
            _build_sibling_model(cfg, "auxiliary")
        _args, kw = gm.call_args
        self.assertEqual(kw["model_provider"], "zhipuai")
        self.assertEqual(kw["base_url"], "https://open.bigmodel.cn/api/paas/v4")
        self.assertEqual(kw["api_key"], "sk-zhipu")

    def test_cross_provider_missing_key_not_filled_with_main_key(self):
        from agentica.cli.runtime import _build_sibling_model

        cfg = self._cfg(
            auxiliary_model_provider="zhipuai",
            auxiliary_model_name="glm-4.7-flash",
            auxiliary_base_url="https://open.bigmodel.cn/api/paas/v4",
            auxiliary_api_key=None,  # no sibling key
        )
        with patch("agentica.cli.runtime.get_model") as gm:
            _build_sibling_model(cfg, "auxiliary")
        _args, kw = gm.call_args
        self.assertIsNone(kw["api_key"])  # must NOT fall back to sk-main
        self.assertEqual(kw["base_url"], "https://open.bigmodel.cn/api/paas/v4")

    def test_cross_provider_missing_base_not_filled_with_main_base(self):
        from agentica.cli.runtime import _build_sibling_model

        cfg = self._cfg(
            auxiliary_model_provider="zhipuai",
            auxiliary_model_name="glm-4.7-flash",
            auxiliary_base_url=None,  # no sibling base_url
            auxiliary_api_key="sk-zhipu",
        )
        with patch("agentica.cli.runtime.get_model") as gm:
            _build_sibling_model(cfg, "auxiliary")
        _args, kw = gm.call_args
        self.assertIsNone(kw["base_url"])  # must NOT fall back to deepseek base_url
        self.assertEqual(kw["api_key"], "sk-zhipu")

    def test_auxiliary_extra_body_passed_and_never_inherits_main(self):
        from agentica.cli.runtime import _build_sibling_model

        cfg = self._cfg(
            auxiliary_model_name="deepseek-chat",  # same provider as main
            extra_body={"main": True},  # main model's own extra_body
            auxiliary_extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}},
            auxiliary_extra_headers={"X-Aux": "1"},
        )
        with patch("agentica.cli.runtime.get_model") as gm:
            _build_sibling_model(cfg, "auxiliary")
        _args, kw = gm.call_args
        self.assertEqual(kw["extra_body"], {"chat_template_kwargs": {"reasoning_effort": "low"}})
        self.assertEqual(kw["extra_headers"], {"X-Aux": "1"})


class TestBuildFallbackModels(unittest.TestCase):
    """_build_fallback_models: turn resolved flat fallback dicts into Model instances."""

    def test_empty_when_no_fallback(self):
        from agentica.cli.runtime import _build_fallback_models

        with patch("agentica.cli.runtime.get_model") as gm:
            self.assertEqual(_build_fallback_models({}), [])
            self.assertEqual(_build_fallback_models({"fallback_models": []}), [])
            gm.assert_not_called()

    def test_builds_each_resolved_fallback(self):
        from agentica.cli.runtime import _build_fallback_models

        cfg = {
            "fallback_models": [
                {"model_provider": "zhipuai", "model_name": "glm-4.7-flash",
                 "base_url": "https://open.bigmodel.cn/api/paas/v4", "api_key": "sk-z"},
                {"model_provider": "openai", "model_name": "gpt-4o-mini", "api_key": "sk-o",
                 "reasoning_effort": "low", "extra_body": {"x": 1}},
            ],
        }
        fake_models = [Mock(), Mock()]
        with patch("agentica.cli.runtime.get_model", side_effect=fake_models) as gm:
            result = _build_fallback_models(cfg)
        self.assertEqual(result, fake_models)
        self.assertEqual(gm.call_count, 2)
        first_kw = gm.call_args_list[0].kwargs
        self.assertEqual(first_kw["model_provider"], "zhipuai")
        self.assertEqual(first_kw["model_name"], "glm-4.7-flash")
        self.assertEqual(first_kw["api_key"], "sk-z")
        second_kw = gm.call_args_list[1].kwargs
        self.assertEqual(second_kw["reasoning_effort"], "low")
        self.assertEqual(second_kw["extra_body"], {"x": 1})

    def test_skips_entries_missing_model_name(self):
        from agentica.cli.runtime import _build_fallback_models

        cfg = {
            "fallback_models": [
                {"model_provider": "zhipuai"},  # no model_name -> skipped
                {"model_provider": "openai", "model_name": "gpt-4o-mini", "api_key": "sk"},
            ],
        }
        with patch("agentica.cli.runtime.get_model") as gm:
            gm.return_value = Mock()
            result = _build_fallback_models(cfg)
        self.assertEqual(len(result), 1)
        self.assertEqual(gm.call_count, 1)
        self.assertEqual(gm.call_args.kwargs["model_name"], "gpt-4o-mini")


if __name__ == "__main__":
    unittest.main()
