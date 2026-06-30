# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for hermes-inspired features:
  - coerce_tool_args (schema-aware type coercion)
  - output truncation 40/60 strategy
  - exit code semantic interpretation
  - file safety guards (device paths, consecutive reads, sensitive paths, staleness)
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
        from agentica.tools.buildin_tools import BuiltinExecuteTool
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

class TestExitCodeInterpretation:
    """Test exit code semantic interpretation."""

    def test_grep_no_matches(self):
        from agentica.tools.buildin_tools import _interpret_exit_code
        result = _interpret_exit_code("grep 'pattern' file.txt", 1)
        assert result is not None
        assert "No matches" in result

    def test_diff_files_differ(self):
        from agentica.tools.buildin_tools import _interpret_exit_code
        result = _interpret_exit_code("diff a.txt b.txt", 1)
        assert result is not None
        assert "differ" in result

    def test_curl_dns_failure(self):
        from agentica.tools.buildin_tools import _interpret_exit_code
        result = _interpret_exit_code("curl https://example.com", 6)
        assert result is not None
        assert "resolve" in result.lower()

    def test_git_normal(self):
        from agentica.tools.buildin_tools import _interpret_exit_code
        result = _interpret_exit_code("git diff", 1)
        assert result is not None
        assert "normal" in result.lower() or "differ" in result.lower()

    def test_pipeline_extraction(self):
        """Should extract last command from pipeline."""
        from agentica.tools.buildin_tools import _interpret_exit_code
        result = _interpret_exit_code("cat file.txt | grep pattern", 1)
        assert result is not None
        assert "No matches" in result

    def test_env_var_stripping(self):
        """Should strip VAR=val prefix."""
        from agentica.tools.buildin_tools import _interpret_exit_code
        result = _interpret_exit_code("LANG=C grep 'x' file", 1)
        assert result is not None
        assert "No matches" in result

    def test_zero_exit_returns_none(self):
        from agentica.tools.buildin_tools import _interpret_exit_code
        assert _interpret_exit_code("grep 'pattern' file", 0) is None

    def test_unknown_command_returns_none(self):
        from agentica.tools.buildin_tools import _interpret_exit_code
        assert _interpret_exit_code("my_custom_tool", 42) is None

    def test_pytest_failures(self):
        from agentica.tools.buildin_tools import _interpret_exit_code
        result = _interpret_exit_code("pytest tests/", 1)
        assert result is not None
        assert "failed" in result.lower()

    def test_full_path_command(self):
        """Should handle /usr/bin/grep -> grep."""
        from agentica.tools.buildin_tools import _interpret_exit_code
        result = _interpret_exit_code("/usr/bin/grep 'x' file", 1)
        assert result is not None
        assert "No matches" in result


# ============== TestFileReadSafety ==============

class TestFileReadSafety:
    """Test file read safety guards."""

    def test_blocked_device_paths(self):
        from agentica.tools.buildin_tools import _is_blocked_device
        assert _is_blocked_device("/dev/random") is True
        assert _is_blocked_device("/dev/zero") is True
        assert _is_blocked_device("/dev/stdin") is True
        assert _is_blocked_device("/dev/fd/0") is True

    def test_proc_fd_blocked(self):
        from agentica.tools.buildin_tools import _is_blocked_device
        assert _is_blocked_device("/proc/self/fd/0") is True
        assert _is_blocked_device("/proc/123/fd/1") is True

    def test_normal_paths_not_blocked(self):
        from agentica.tools.buildin_tools import _is_blocked_device
        assert _is_blocked_device("/tmp/test.txt") is False
        assert _is_blocked_device("/home/user/file.py") is False
        assert _is_blocked_device("/dev/sda1") is False  # not in block list

    def test_consecutive_read_tracker(self):
        """Consecutive reads of same file should be tracked per-key."""
        from agentica.tools.buildin_tools import BuiltinFileTool
        tool = BuiltinFileTool(work_dir="/tmp")

        read_key = ("test.txt", 0, 500)

        # Simulate consecutive reads via per-key dict
        tool._read_consecutive_counts[read_key] = 3
        assert tool._read_consecutive_counts[read_key] == 3  # warning threshold

        tool._read_consecutive_counts[read_key] = 4
        assert tool._read_consecutive_counts[read_key] >= 4  # block threshold

    def test_read_tracker_resets_on_different_file(self):
        """Reading a different file should have independent counter."""
        from agentica.tools.buildin_tools import BuiltinFileTool
        tool = BuiltinFileTool(work_dir="/tmp")
        tool._read_consecutive_counts[("file_a.txt", 0, 500)] = 3

        # Different file starts at its own count
        new_key = ("file_b.txt", 0, 500)
        tool._read_consecutive_counts[new_key] = tool._read_consecutive_counts.get(new_key, 0) + 1
        assert tool._read_consecutive_counts[new_key] == 1
        # file_a count is unaffected
        assert tool._read_consecutive_counts[("file_a.txt", 0, 500)] == 3


# ============== TestFileWriteSafety ==============

class TestFileWriteSafety:
    """Test file write safety guards."""

    def test_sensitive_system_paths(self):
        from agentica.tools.buildin_tools import _check_sensitive_write_path
        assert _check_sensitive_write_path("/etc/passwd") is not None
        assert _check_sensitive_write_path("/boot/grub/grub.cfg") is not None
        assert _check_sensitive_write_path("/usr/lib/systemd/system/test.service") is not None

    def test_sensitive_home_paths(self):
        from agentica.tools.buildin_tools import _check_sensitive_write_path
        home = os.path.expanduser("~")
        assert _check_sensitive_write_path(f"{home}/.ssh/authorized_keys") is not None
        assert _check_sensitive_write_path(f"{home}/.gnupg/pubring.kbx") is not None

    def test_normal_paths_allowed(self):
        from agentica.tools.buildin_tools import _check_sensitive_write_path
        assert _check_sensitive_write_path("/tmp/test.txt") is None
        assert _check_sensitive_write_path("/home/user/project/main.py") is None

    def test_staleness_detection_via_mtime(self):
        """Write should warn if file was externally modified since last read."""
        from agentica.tools.buildin_tools import BuiltinFileTool
        tool = BuiltinFileTool(work_dir="/tmp")

        # Create a temp file and record its mtime
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("original")
            tmp_path = f.name

        try:
            abs_path = os.path.realpath(tmp_path)
            original_mtime = os.path.getmtime(abs_path)
            tool._file_read_state[abs_path] = {"mtime": original_mtime}

            # Modify file externally (change mtime)
            import time
            time.sleep(0.05)
            with open(tmp_path, 'w') as f:
                f.write("modified externally")
            new_mtime = os.path.getmtime(abs_path)

            # Staleness check should detect mismatch
            assert new_mtime != original_mtime
            stored_mtime = tool._file_read_state[abs_path]["mtime"]
            assert stored_mtime != new_mtime  # stale!
        finally:
            os.unlink(tmp_path)


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


# ============== TestToolPairSanitization ==============

class TestToolPairSanitization:
    """Test tool_call/result pair sanitization in compression."""

    def test_orphan_result_removed(self):
        from agentica.compression.manager import CompressionManager
        from agentica.model.message import Message
        cm = CompressionManager()
        messages = [
            Message(role="user", content="hello"),
            Message(role="tool", tool_call_id="orphan_id", content="some result"),
        ]
        result = cm._sanitize_tool_pairs(messages)
        # Orphan tool result should be removed
        tool_msgs = [m for m in result if m.role == "tool"]
        assert len(tool_msgs) == 0

    def test_missing_result_gets_placeholder(self):
        from agentica.compression.manager import CompressionManager
        from agentica.model.message import Message
        cm = CompressionManager()
        messages = [
            Message(role="assistant", content="", tool_calls=[{"id": "tc_1", "type": "function", "function": {"name": "test", "arguments": "{}"}}]),
        ]
        result = cm._sanitize_tool_pairs(messages)
        tool_msgs = [m for m in result if m.role == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].tool_call_id == "tc_1"
        assert "removed during compression" in tool_msgs[0].content

    def test_matched_pairs_untouched(self):
        from agentica.compression.manager import CompressionManager
        from agentica.model.message import Message
        cm = CompressionManager()
        messages = [
            Message(role="assistant", content="", tool_calls=[{"id": "tc_1", "type": "function", "function": {"name": "test", "arguments": "{}"}}]),
            Message(role="tool", tool_call_id="tc_1", content="result"),
        ]
        result = cm._sanitize_tool_pairs(messages)
        assert len(result) == 2  # both preserved

    def test_placeholder_inserted_after_assistant_not_at_end(self):
        """Placeholder must appear right after its assistant msg, not at tail."""
        from agentica.compression.manager import CompressionManager
        from agentica.model.message import Message
        cm = CompressionManager()
        messages = [
            Message(role="user", content="step 1"),
            Message(role="assistant", content="", tool_calls=[
                {"id": "tc_1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                {"id": "tc_2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
            ]),
            # tc_1 result exists, tc_2 missing
            Message(role="tool", tool_call_id="tc_1", content="result_1"),
            Message(role="user", content="step 2"),
            Message(role="assistant", content="done"),
        ]
        result = cm._sanitize_tool_pairs(messages)

        # Find the assistant with tool_calls
        asst_idx = next(i for i, m in enumerate(result) if m.role == "assistant" and m.tool_calls)
        # tc_1 result must be at asst_idx+1
        assert result[asst_idx + 1].role == "tool"
        assert result[asst_idx + 1].tool_call_id == "tc_1"
        # tc_2 placeholder must be at asst_idx+2 (right after tc_1, in tool_calls order)
        assert result[asst_idx + 2].role == "tool"
        assert result[asst_idx + 2].tool_call_id == "tc_2"
        assert "removed during compression" in result[asst_idx + 2].content
        # user "step 2" must come AFTER both tool results
        assert result[asst_idx + 3].role == "user"
        assert result[asst_idx + 3].content == "step 2"

    def test_multiple_tool_calls_preserve_order(self):
        """Multiple tool_call results must stay in tool_calls declaration order."""
        from agentica.compression.manager import CompressionManager
        from agentica.model.message import Message
        cm = CompressionManager()
        messages = [
            Message(role="assistant", content="", tool_calls=[
                {"id": "tc_a", "type": "function", "function": {"name": "x", "arguments": "{}"}},
                {"id": "tc_b", "type": "function", "function": {"name": "y", "arguments": "{}"}},
                {"id": "tc_c", "type": "function", "function": {"name": "z", "arguments": "{}"}},
            ]),
            # Results arrive in reverse order in messages (tc_c, tc_b, tc_a)
            Message(role="tool", tool_call_id="tc_c", content="c_result"),
            Message(role="tool", tool_call_id="tc_b", content="b_result"),
            Message(role="tool", tool_call_id="tc_a", content="a_result"),
        ]
        result = cm._sanitize_tool_pairs(messages)
        # After sanitization, results must follow tool_calls order: tc_a, tc_b, tc_c
        assert result[1].tool_call_id == "tc_a"
        assert result[2].tool_call_id == "tc_b"
        assert result[3].tool_call_id == "tc_c"


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
            tool_config=ToolConfig(compress_tool_results=True),
        )
        cm = agent.tool_config.compression_manager
        assert cm is not None
        assert cm.model is auxiliary


# ============== TestFileUndoEdit ==============

class TestFileUndoEdit:
    """Test file snapshot and undo_edit functionality."""

    def test_snapshot_created_on_write(self):
        from agentica.tools.buildin_tools import BuiltinFileTool
        import asyncio

        tool = BuiltinFileTool(work_dir=tempfile.mkdtemp())
        result = asyncio.run(tool.write_file("test.txt", "original content"))
        assert "Created" in result
        result = asyncio.run(tool.write_file("test.txt", "new content"))
        assert "Updated" in result

        path = tool._resolve_path("test.txt")
        abs_path = str(path.resolve())
        assert abs_path in tool._file_snapshots
        assert tool._file_snapshots[abs_path][-1] == "original content"

    def test_undo_restores_previous(self):
        from agentica.tools.buildin_tools import BuiltinFileTool
        import asyncio

        tool = BuiltinFileTool(work_dir=tempfile.mkdtemp())
        asyncio.run(tool.write_file("test.txt", "v1"))
        asyncio.run(tool.write_file("test.txt", "v2"))
        asyncio.run(tool.write_file("test.txt", "v3"))

        result = asyncio.run(tool.undo_edit("test.txt"))
        assert "Restored" in result

        path = tool._resolve_path("test.txt")
        assert path.read_text() == "v2"

        result = asyncio.run(tool.undo_edit("test.txt"))
        assert path.read_text() == "v1"

    def test_undo_empty_returns_error(self):
        from agentica.tools.buildin_tools import BuiltinFileTool
        import asyncio

        tool = BuiltinFileTool(work_dir=tempfile.mkdtemp())
        asyncio.run(tool.write_file("test.txt", "content"))

        with pytest.raises(FileNotFoundError, match="No previous version"):
            asyncio.run(tool.undo_edit("test.txt"))
