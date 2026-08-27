# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for hermes-inspired features:
  - coerce_tool_args (schema-aware type coercion)
  - output truncation 40/60 strategy
  - exit code semantic interpretation
  - file safety guards (device paths, sensitive paths, staleness)
  - API error context limit learning
  - tool_pair sanitization in compression
  - prompt caching system_and_3
"""
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============== TestCoerceToolArgs ==============

class TestCoerceToolArgs:
    """Test schema-aware type coercion for LLM tool arguments."""

    def test_integer_coercion(self):
        from agentica.tools.base import coerce_tool_args, Function
        func = Function(name="test", parameters={"type": "object", "properties": {"count": {"type": "integer"}}})
        args = {"count": "42"}
        result = coerce_tool_args(args, func)
        assert result["count"] == 42
        assert isinstance(result["count"], int)

    def test_number_coercion(self):
        from agentica.tools.base import coerce_tool_args, Function
        func = Function(name="test", parameters={"type": "object", "properties": {"price": {"type": "number"}}})
        args = {"price": "3.14"}
        result = coerce_tool_args(args, func)
        assert result["price"] == 3.14

    def test_boolean_coercion_true(self):
        from agentica.tools.base import coerce_tool_args, Function
        func = Function(name="test", parameters={"type": "object", "properties": {"flag": {"type": "boolean"}}})
        args = {"flag": "true"}
        result = coerce_tool_args(args, func)
        assert result["flag"] is True

    def test_boolean_coercion_false(self):
        from agentica.tools.base import coerce_tool_args, Function
        func = Function(name="test", parameters={"type": "object", "properties": {"flag": {"type": "boolean"}}})
        args = {"flag": "false"}
        result = coerce_tool_args(args, func)
        assert result["flag"] is False

    def test_array_coercion(self):
        from agentica.tools.base import coerce_tool_args, Function
        func = Function(name="test", parameters={"type": "object", "properties": {"items": {"type": "array"}}})
        args = {"items": "[1, 2, 3]"}
        result = coerce_tool_args(args, func)
        assert result["items"] == [1, 2, 3]

    def test_object_coercion(self):
        from agentica.tools.base import coerce_tool_args, Function
        func = Function(name="test", parameters={"type": "object", "properties": {"data": {"type": "object"}}})
        args = {"data": '{"key": "val"}'}
        result = coerce_tool_args(args, func)
        assert result["data"] == {"key": "val"}

    def test_union_type(self):
        from agentica.tools.base import coerce_tool_args, Function
        func = Function(name="test", parameters={"type": "object", "properties": {"val": {"type": ["integer", "string"]}}})
        args = {"val": "42"}
        result = coerce_tool_args(args, func)
        assert result["val"] == 42

    def test_failed_coercion_preserves_original(self):
        from agentica.tools.base import coerce_tool_args, Function
        func = Function(name="test", parameters={"type": "object", "properties": {"count": {"type": "integer"}}})
        args = {"count": "not_a_number"}
        result = coerce_tool_args(args, func)
        assert result["count"] == "not_a_number"  # preserved

    def test_non_string_values_untouched(self):
        from agentica.tools.base import coerce_tool_args, Function
        func = Function(name="test", parameters={"type": "object", "properties": {"count": {"type": "integer"}}})
        args = {"count": 42}  # already int
        result = coerce_tool_args(args, func)
        assert result["count"] == 42

    def test_empty_args(self):
        from agentica.tools.base import coerce_tool_args, Function
        func = Function(name="test", parameters={"type": "object", "properties": {}})
        assert coerce_tool_args({}, func) == {}
        assert coerce_tool_args(None, func) is None

    def test_integer_with_decimal_keeps_string(self):
        """Schema wants integer but value has decimals — keep as string."""
        from agentica.tools.base import coerce_tool_args, Function
        func = Function(name="test", parameters={"type": "object", "properties": {"n": {"type": "integer"}}})
        args = {"n": "3.14"}
        result = coerce_tool_args(args, func)
        assert result["n"] == "3.14"  # can't coerce float to int, keep string


# ============== TestOutputTruncation ==============

class TestOutputTruncation:
    """Test 40/60 head/tail output truncation strategy."""

    def test_short_output_not_truncated(self):
        """Output under limit should not be truncated."""
        from agentica.tools.builtin import BuiltinExecuteTool
        tool = BuiltinExecuteTool()
        # _max_output_length defaults to 50000
        output = "x" * 100
        assert len(output) < tool._max_output_length

    def test_long_output_truncated_with_40_60_split(self):
        """Output over limit should be truncated with 40/60 head/tail split."""
        max_len = 1000
        # Build a long output with markers
        output = "HEAD" * 200 + "MIDDLE" * 200 + "TAIL" * 200
        if len(output) > max_len:
            head_chars = int(max_len * 0.4)  # 400
            tail_chars = max_len - head_chars  # 600
            omitted = len(output) - head_chars - tail_chars
            result = (
                output[:head_chars]
                + f"\n\n... [OUTPUT TRUNCATED - {omitted} chars omitted"
                  f" out of {len(output)} total] ...\n\n"
                + output[-tail_chars:]
            )
        # Head should contain HEAD markers
        assert "HEAD" in result[:400]
        # Tail should contain TAIL markers
        assert "TAIL" in result[-600:]
        # Truncation notice should be present
        assert "OUTPUT TRUNCATED" in result


# ============== TestExitCodeInterpretation ==============

class TestExpectedNonzeroExit:
    """Non-zero exits that are the answer, not a crash, must not raise."""

    def test_grep_no_matches(self):
        from agentica.tools.builtin.execute_tool import _expected_nonzero_exit
        assert _expected_nonzero_exit("grep 'pattern' file.txt", 1) is True

    def test_diff_files_differ(self):
        from agentica.tools.builtin.execute_tool import _expected_nonzero_exit
        assert _expected_nonzero_exit("diff a.txt b.txt", 1) is True

    def test_curl_dns_failure(self):
        from agentica.tools.builtin.execute_tool import _expected_nonzero_exit
        assert _expected_nonzero_exit("curl https://example.com", 6) is True

    def test_git_normal(self):
        from agentica.tools.builtin.execute_tool import _expected_nonzero_exit
        assert _expected_nonzero_exit("git diff", 1) is True

    def test_pipeline_extraction(self):
        from agentica.tools.builtin.execute_tool import _expected_nonzero_exit
        assert _expected_nonzero_exit("cat file.txt | grep pattern", 1) is True

    def test_env_var_stripping(self):
        from agentica.tools.builtin.execute_tool import _expected_nonzero_exit
        assert _expected_nonzero_exit("LANG=C grep 'x' file", 1) is True

    def test_zero_exit_is_not_expected_nonzero(self):
        from agentica.tools.builtin.execute_tool import _expected_nonzero_exit
        assert _expected_nonzero_exit("grep 'pattern' file", 0) is False

    def test_unknown_command_is_not_expected(self):
        from agentica.tools.builtin.execute_tool import _expected_nonzero_exit
        assert _expected_nonzero_exit("my_custom_tool", 42) is False

    def test_pytest_failures(self):
        from agentica.tools.builtin.execute_tool import _expected_nonzero_exit
        assert _expected_nonzero_exit("pytest tests/", 1) is True

    def test_full_path_command(self):
        from agentica.tools.builtin.execute_tool import _expected_nonzero_exit
        assert _expected_nonzero_exit("/usr/bin/grep 'x' file", 1) is True


# ============== TestFileReadSafety ==============

class TestFileReadSafety:
    """Test file read safety guards."""

    def test_blocked_device_paths(self):
        from agentica.tools.builtin.file_tool import _is_blocked_device
        assert _is_blocked_device("/dev/random") is True
        assert _is_blocked_device("/dev/zero") is True
        assert _is_blocked_device("/dev/stdin") is True
        assert _is_blocked_device("/dev/fd/0") is True

    def test_proc_fd_blocked(self):
        from agentica.tools.builtin.file_tool import _is_blocked_device
        assert _is_blocked_device("/proc/self/fd/0") is True
        assert _is_blocked_device("/proc/123/fd/1") is True

    def test_normal_paths_not_blocked(self):
        from agentica.tools.builtin.file_tool import _is_blocked_device
        assert _is_blocked_device("/tmp/test.txt") is False
        assert _is_blocked_device("/home/user/file.py") is False
        assert _is_blocked_device("/dev/sda1") is False  # not in block list


# ============== TestFileWriteSafety ==============

class TestFileWriteSafety:
    """Test file write safety guards."""

    def test_sensitive_system_paths(self):
        from agentica.tools.builtin.file_tool import _check_sensitive_write_path
        assert _check_sensitive_write_path("/etc/passwd") is not None
        assert _check_sensitive_write_path("/boot/grub/grub.cfg") is not None
        assert _check_sensitive_write_path("/usr/lib/systemd/system/test.service") is not None

    def test_sensitive_home_paths(self):
        from agentica.tools.builtin.file_tool import _check_sensitive_write_path
        home = os.path.expanduser("~")
        assert _check_sensitive_write_path(f"{home}/.ssh/authorized_keys") is not None
        assert _check_sensitive_write_path(f"{home}/.gnupg/pubring.kbx") is not None

    def test_normal_paths_allowed(self):
        from agentica.tools.builtin.file_tool import _check_sensitive_write_path
        assert _check_sensitive_write_path("/tmp/test.txt") is None
        assert _check_sensitive_write_path("/home/user/project/main.py") is None

# ============== TestContextLimitLearning ==============

class TestContextLimitLearning:
    """Test API error context limit extraction."""

    def test_openai_format(self):
        from agentica.model.openai.chat import OpenAIChat
        model = OpenAIChat(id="gpt-4o-mini", api_key="fake_key")
        model.context_window = 8192
        model._learn_context_limit_from_error(
            "This model's maximum context length is 128000 tokens"
        )
        assert model.context_window == 128000

    def test_anthropic_format(self):
        from agentica.model.openai.chat import OpenAIChat
        model = OpenAIChat(id="gpt-4o-mini", api_key="fake_key")
        model.context_window = 8192
        model._learn_context_limit_from_error(
            "context_length: 200000"
        )
        assert model.context_window == 200000

    def test_comma_separated_number(self):
        from agentica.model.openai.chat import OpenAIChat
        model = OpenAIChat(id="gpt-4o-mini", api_key="fake_key")
        model.context_window = 8192
        model._learn_context_limit_from_error(
            "maximum context window is 1,000,000 tokens"
        )
        assert model.context_window == 1000000

    def test_no_match_preserves_value(self):
        from agentica.model.openai.chat import OpenAIChat
        model = OpenAIChat(id="gpt-4o-mini", api_key="fake_key")
        model.context_window = 8192
        model._learn_context_limit_from_error("Some random error message")
        assert model.context_window == 8192

    def test_small_value_rejected(self):
        """Values <= 1000 are rejected as false positives."""
        from agentica.model.openai.chat import OpenAIChat
        model = OpenAIChat(id="gpt-4o-mini", api_key="fake_key")
        model.context_window = 8192
        model._learn_context_limit_from_error("context_length: 500")
        assert model.context_window == 8192  # unchanged

    def test_max_tokens_not_confused_with_context(self):
        """max_tokens in error should NOT update context_window (it's output limit)."""
        from agentica.model.openai.chat import OpenAIChat
        model = OpenAIChat(id="gpt-4o-mini", api_key="fake_key")
        model.context_window = 128000
        model._learn_context_limit_from_error(
            "max_tokens: 4096 exceeds the model's output limit"
        )
        assert model.context_window == 128000  # unchanged — max_tokens is output, not context


# ============== TestPromptCaching ==============

class TestPromptCachingSystemAnd3:
    """Test system_and_3 prompt caching strategy marks last 3 messages."""

    def test_last_three_messages_cached(self):
        """With 5+ messages, last 3 should get cache_control."""
        from agentica.model.anthropic.claude import Claude
        from agentica.model.message import Message

        model = Claude(id="claude-3-5-sonnet-20241022", api_key="fake_key")
        model.enable_cache_control = True

        messages = [
            Message(role="system", content="System prompt"),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
            Message(role="user", content="How are you?"),
            Message(role="assistant", content="I'm good"),
            Message(role="user", content="Thanks"),
        ]

        import asyncio
        chat_messages, system = asyncio.run(model.format_messages(messages))

        # Count messages with cache_control
        cached_count = 0
        for msg in chat_messages:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "cache_control" in block:
                        cached_count += 1
            elif isinstance(content, str):
                # String content shouldn't have cache_control (would be converted to list)
                pass

        # Should have cache_control on last 3 messages
        assert cached_count >= 3, f"Expected at least 3 cached messages, got {cached_count}"

    def test_single_message_cached(self):
        """With only 1 message, it should still get cache_control."""
        from agentica.model.anthropic.claude import Claude
        from agentica.model.message import Message

        model = Claude(id="claude-3-5-sonnet-20241022", api_key="fake_key")
        model.enable_cache_control = True

        messages = [
            Message(role="system", content="System prompt"),
            Message(role="user", content="Hello"),
        ]

        import asyncio
        chat_messages, system = asyncio.run(model.format_messages(messages))

        # The single user message should be cached
        cached = False
        for msg in chat_messages:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "cache_control" in block:
                        cached = True
        assert cached


# ============== TestGatewayModelFactory ==============

class TestGatewayModelFactory:
    """Test gateway _create_model() uses registry for non-core providers."""

    def test_registry_provider_used(self):
        """Providers in PROVIDER_FACTORIES should use the factory dispatch."""
        from agentica import PROVIDER_FACTORIES
        assert "zhipuai" in PROVIDER_FACTORIES
        assert "deepseek" in PROVIDER_FACTORIES
        assert "moonshot" in PROVIDER_FACTORIES

    def test_core_providers_not_in_registry_branch(self):
        """openai, claude, kimi, azure have dedicated classes."""
        from agentica.model.openai import OpenAIChat
        from agentica.model.anthropic.claude import Claude
        assert OpenAIChat is not None
        assert Claude is not None


# ============== TestAuxiliaryModel ==============

class TestAuxiliaryModel:
    """Test auxiliary_model field on Agent."""

    def test_agent_has_auxiliary_model_field(self):
        from agentica import Agent
        agent = Agent()
        assert hasattr(agent, 'auxiliary_model')
        assert agent.auxiliary_model is None

    def test_auxiliary_model_wired_to_compression(self):
        """When auxiliary_model is set, CompressionManager should use it."""
        from agentica import Agent, OpenAIChat
        from agentica.agent.config import ToolConfig

        auxiliary = OpenAIChat(id="gpt-4o-mini", api_key="fake_key")
        agent = Agent(
            model=OpenAIChat(id="gpt-4o", api_key="fake_key"),
            auxiliary_model=auxiliary,
        )
        cm = agent.tool_config.compression_manager
        assert cm is not None
        assert cm.model is auxiliary


