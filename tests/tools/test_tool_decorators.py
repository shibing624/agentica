# -*- coding: utf-8 -*-
"""Tests for agentica.tools.decorators — @tool decorator + Function.from_callable integration."""
import unittest

from agentica.tools.decorators import tool
from agentica.tools.base import Function


class TestToolDecorator(unittest.TestCase):
    """@tool decorator attaches _tool_metadata to functions."""

    def test_metadata_attached(self):
        @tool(name="my_search", description="Search stuff")
        def search(query: str) -> str:
            """Docstring"""
            return query

        self.assertTrue(hasattr(search, "_tool_metadata"))
        meta = search._tool_metadata
        self.assertEqual(meta["name"], "my_search")
        self.assertEqual(meta["description"], "Search stuff")

    def test_defaults(self):
        @tool()
        def my_func():
            """My docstring"""
            pass

        meta = my_func._tool_metadata
        self.assertEqual(meta["name"], "my_func")
        self.assertEqual(meta["description"], "My docstring")
        self.assertFalse(meta["show_result"])
        self.assertTrue(meta["sanitize_arguments"])
        self.assertFalse(meta["stop_after_tool_call"])
        self.assertFalse(meta["concurrency_safe"])
        self.assertFalse(meta["is_read_only"])
        self.assertFalse(meta["is_destructive"])
        self.assertFalse(meta["deferred"])
        self.assertEqual(meta["interrupt_behavior"], "cancel")

    def test_concurrency_safe(self):
        @tool(concurrency_safe=True, is_read_only=True)
        def read_data(path: str) -> str:
            return ""

        meta = read_data._tool_metadata
        self.assertTrue(meta["concurrency_safe"])
        self.assertTrue(meta["is_read_only"])

    def test_destructive_flag(self):
        @tool(is_destructive=True, interrupt_behavior="block")
        def delete_file(path: str) -> str:
            return ""

        meta = delete_file._tool_metadata
        self.assertTrue(meta["is_destructive"])
        self.assertEqual(meta["interrupt_behavior"], "block")

    def test_deferred_flag(self):
        @tool(deferred=True)
        def heavy_tool() -> str:
            return ""

        self.assertTrue(heavy_tool._tool_metadata["deferred"])

    def test_available_when_metadata(self):
        @tool(name="conditional_tool", available_when=lambda: False)
        def conditional_tool() -> str:
            return ""

        self.assertIn("available_when", conditional_tool._tool_metadata)
        self.assertFalse(conditional_tool._tool_metadata["available_when"]())

    def test_wraps_preserves_function_name(self):
        @tool(name="custom_name")
        def original_func(x: int) -> int:
            """Original doc"""
            return x

        self.assertEqual(original_func.__name__, "original_func")
        self.assertEqual(original_func.__doc__, "Original doc")

    def test_decorated_function_still_callable(self):
        @tool()
        def add(a: int, b: int) -> int:
            return a + b

        self.assertEqual(add(1, 2), 3)


class TestFunctionFromCallableWithDecorator(unittest.TestCase):
    """Function.from_callable reads @tool metadata."""

    def test_reads_metadata(self):
        @tool(name="web_search", description="Search the web", concurrency_safe=True, is_read_only=True)
        def search(query: str, max_results: int = 5) -> str:
            """Search docstring"""
            return query

        func = Function.from_callable(search)
        self.assertEqual(func.name, "web_search")
        self.assertEqual(func.description, "Search the web")
        self.assertTrue(func.concurrency_safe)
        self.assertTrue(func.is_read_only)

    def test_reads_stop_after_tool_call(self):
        @tool(stop_after_tool_call=True)
        def done() -> str:
            return "done"

        func = Function.from_callable(done)
        self.assertTrue(func.stop_after_tool_call)

    def test_reads_destructive_and_interrupt(self):
        @tool(is_destructive=True, interrupt_behavior="block")
        def danger() -> str:
            return ""

        func = Function.from_callable(danger)
        self.assertTrue(func.is_destructive)
        self.assertEqual(func.interrupt_behavior, "block")

    def test_reads_deferred(self):
        @tool(deferred=True)
        def lazy() -> str:
            return ""

        func = Function.from_callable(lazy)
        self.assertTrue(func.deferred)

    def test_reads_available_when(self):
        @tool(available_when=lambda: False)
        def conditional() -> str:
            return ""

        func = Function.from_callable(conditional)
        self.assertFalse(func.is_available())

    def test_plain_function_without_decorator(self):
        def plain(x: int) -> int:
            """Plain docstring"""
            return x

        func = Function.from_callable(plain)
        self.assertEqual(func.name, "plain")
        self.assertFalse(func.concurrency_safe)
        self.assertFalse(func.is_read_only)
        self.assertFalse(func.is_destructive)


class TestFunctionSafetyMetadata(unittest.TestCase):
    """Function class safety metadata fields."""

    def test_default_safety_fields(self):
        func = Function(name="test")
        self.assertFalse(func.concurrency_safe)
        self.assertFalse(func.is_read_only)
        self.assertFalse(func.is_destructive)
        self.assertIsNone(func.max_result_size_chars)
        self.assertIsNone(func.timeout)
        self.assertFalse(func.manages_own_timeout)
        self.assertIsNone(func.validate_input)
        self.assertEqual(func.interrupt_behavior, "cancel")
        self.assertFalse(func.deferred)

    def test_agent_weakref(self):
        import weakref
        func = Function(name="test")
        mock_agent = type("Agent", (), {"name": "test"})()
        func._agent = mock_agent
        self.assertIs(func._agent, mock_agent)
        # Delete strong reference
        del mock_agent
        # Weakref may be collected
        # (just test that setting/getting doesn't crash)

    def test_agent_weakref_none(self):
        func = Function(name="test")
        self.assertIsNone(func._agent)
        func._agent = None
        self.assertIsNone(func._agent)


if __name__ == "__main__":
    unittest.main()
