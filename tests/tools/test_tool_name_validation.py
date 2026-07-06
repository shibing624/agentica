# -*- coding: utf-8 -*-
"""
Tests for tool-name validation and normalization.

OpenAI/Anthropic tool calling APIs require ``^[a-zA-Z0-9_-]{1,64}$``. We
catch bad names at construction time (``ValueError``) for explicit user
input, and auto-normalize for derived names (``as_tool()`` / ``Tool.register``).
"""
import os
import sys
import unittest

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from agentica.tools.base import (
    Function,
    Tool,
    normalize_tool_name,
    validate_tool_name,
)


class TestValidateToolName(unittest.TestCase):
    def test_accepts_legal_names(self):
        for name in ["foo", "foo_bar", "foo-bar", "FooBar123", "a" * 64]:
            validate_tool_name(name)

    def test_rejects_chinese_name(self):
        with self.assertRaises(ValueError):
            validate_tool_name("写代码")

    def test_rejects_spaces(self):
        with self.assertRaises(ValueError):
            validate_tool_name("hello world")

    def test_rejects_special_punctuation(self):
        with self.assertRaises(ValueError):
            validate_tool_name("foo!bar")

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            validate_tool_name("")

    def test_rejects_too_long(self):
        with self.assertRaises(ValueError):
            validate_tool_name("a" * 65)


class TestNormalizeToolName(unittest.TestCase):
    def test_replaces_spaces_and_punctuation(self):
        self.assertEqual(normalize_tool_name("Hello World!"), "hello_world")

    def test_collapses_consecutive_underscores(self):
        self.assertEqual(normalize_tool_name("a   b---c"), "a_b-c")

    def test_truncates_to_64(self):
        out = normalize_tool_name("x" * 200)
        self.assertEqual(len(out), 64)

    def test_chinese_name_becomes_legal(self):
        out = normalize_tool_name("写代码 Agent")
        validate_tool_name(out)
        self.assertTrue(out.endswith("agent") or "agent" in out)

    def test_empty_input_falls_back_to_default(self):
        self.assertEqual(normalize_tool_name(""), "tool")

    def test_idempotent(self):
        once = normalize_tool_name("Hello World!")
        twice = normalize_tool_name(once)
        self.assertEqual(once, twice)


class TestFunctionRejectsIllegalName(unittest.TestCase):
    def test_function_init_rejects_chinese(self):
        with self.assertRaises(Exception):  # pydantic wraps as ValidationError
            Function(name="写代码")

    def test_function_init_accepts_legal(self):
        fn = Function(name="run_query")
        self.assertEqual(fn.name, "run_query")


class TestAsToolGeneratesLegalName(unittest.TestCase):
    def test_chinese_agent_name_yields_legal_tool_name(self):
        from agentica.agent.base import Agent

        agent = Agent(name="写代码 Agent")
        tool_fn = agent.as_tool()
        validate_tool_name(tool_fn.name)

    def test_explicit_illegal_tool_name_raises(self):
        from agentica.agent.base import Agent

        agent = Agent(name="legal_agent")
        with self.assertRaises(Exception):
            agent.as_tool(tool_name="非法名字!")


class TestRegisterNormalizesName(unittest.TestCase):
    def test_register_long_function_name_is_truncated(self):
        long_name = "a" * 100

        def fn():
            return None

        fn.__name__ = long_name

        toolkit = Tool(name="t", description="d")
        toolkit.register(fn)

        self.assertEqual(len(list(toolkit.functions.keys())[0]), 64)


if __name__ == "__main__":
    unittest.main()
