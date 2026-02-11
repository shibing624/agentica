# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Unit tests for async tool support
"""
import asyncio
import pytest
from agentica.tools.base import Function, FunctionCall, Tool


# Async functions for testing
async def async_add(a: int, b: int) -> str:
    """Add two numbers asynchronously."""
    await asyncio.sleep(0.01)
    return str(a + b)


async def async_no_args() -> str:
    """Async function with no arguments."""
    await asyncio.sleep(0.01)
    return "async_no_args_result"


# Sync function for comparison
def sync_multiply(a: int, b: int) -> str:
    """Multiply two numbers synchronously."""
    return str(a * b)


class TestAsyncFunctionCall:
    """Tests for async function execution via FunctionCall."""

    def test_sync_function_execute(self):
        """Test that sync functions still work with execute()."""
        f = Function.from_callable(sync_multiply)
        fc = FunctionCall(function=f, arguments={'a': 3, 'b': 4})
        success = asyncio.run(fc.execute())
        assert success is True
        assert fc.result == "12"
        assert fc.error is None

    def test_async_function_execute(self):
        """Test that async functions work with execute() in sync context."""
        f = Function.from_callable(async_add)
        fc = FunctionCall(function=f, arguments={'a': 5, 'b': 7})
        success = asyncio.run(fc.execute())
        assert success is True
        assert fc.result == "12"
        assert fc.error is None

    def test_async_function_no_args_execute(self):
        """Test async function with no arguments via execute()."""
        f = Function.from_callable(async_no_args)
        fc = FunctionCall(function=f)
        success = asyncio.run(fc.execute())
        assert success is True
        assert fc.result == "async_no_args_result"

    def test_async_function_execute_with_args(self):
        """Test that async functions work with execute()."""
        f = Function.from_callable(async_add)
        fc = FunctionCall(function=f, arguments={'a': 10, 'b': 20})
        success = asyncio.run(fc.execute())
        assert success is True
        assert fc.result == "30"
        assert fc.error is None

    def test_sync_function_execute_with_args(self):
        """Test that sync functions work with execute()."""
        f = Function.from_callable(sync_multiply)
        fc = FunctionCall(function=f, arguments={'a': 5, 'b': 6})
        success = asyncio.run(fc.execute())
        assert success is True
        assert fc.result == "30"
        assert fc.error is None

    def test_async_function_no_args_execute_2(self):
        """Test async function with no arguments via execute()."""
        f = Function.from_callable(async_no_args)
        fc = FunctionCall(function=f)
        success = asyncio.run(fc.execute())
        assert success is True
        assert fc.result == "async_no_args_result"


class TestAsyncToolClass:
    """Tests for Tool class with async methods."""

    def test_async_tool_method_execute(self):
        """Test Tool class with async method via execute()."""
        class AsyncTool(Tool):
            def __init__(self):
                super().__init__(name="async_tool")
                self.register(self.async_method)

            async def async_method(self, x: int) -> str:
                """Async method."""
                await asyncio.sleep(0.01)
                return str(x * 2)

        tool = AsyncTool()
        assert "async_method" in tool.functions
        
        func = tool.functions["async_method"]
        fc = FunctionCall(function=func, arguments={'x': 25})
        success = asyncio.run(fc.execute())
        assert success is True
        assert fc.result == "50"

    def test_async_tool_method_execute_2(self):
        """Test Tool class with async method via execute()."""
        class AsyncTool(Tool):
            def __init__(self):
                super().__init__(name="async_tool")
                self.register(self.async_method)

            async def async_method(self, x: int) -> str:
                """Async method."""
                await asyncio.sleep(0.01)
                return str(x * 3)

        tool = AsyncTool()
        func = tool.functions["async_method"]
        fc = FunctionCall(function=func, arguments={'x': 10})
        success = asyncio.run(fc.execute())
        assert success is True
        assert fc.result == "30"


class TestAsyncHooks:
    """Tests for async pre/post hooks."""

    def test_async_pre_hook(self):
        """Test async pre-hook execution."""
        hook_called = []

        async def async_pre_hook():
            await asyncio.sleep(0.01)
            hook_called.append("pre")

        f = Function.from_callable(async_add)
        f.pre_hook = async_pre_hook
        
        fc = FunctionCall(function=f, arguments={'a': 1, 'b': 2})
        success = asyncio.run(fc.execute())
        
        assert success is True
        assert "pre" in hook_called

    def test_async_post_hook(self):
        """Test async post-hook execution."""
        hook_called = []

        async def async_post_hook():
            await asyncio.sleep(0.01)
            hook_called.append("post")

        f = Function.from_callable(async_add)
        f.post_hook = async_post_hook
        
        fc = FunctionCall(function=f, arguments={'a': 3, 'b': 4})
        success = asyncio.run(fc.execute())
        
        assert success is True
        assert "post" in hook_called

    def test_sync_pre_hook_with_async_func(self):
        """Test sync pre-hook with async function via execute()."""
        hook_called = []

        def sync_pre_hook():
            hook_called.append("sync_pre")

        f = Function.from_callable(async_add)
        f.pre_hook = sync_pre_hook
        
        fc = FunctionCall(function=f, arguments={'a': 5, 'b': 5})
        success = asyncio.run(fc.execute())
        
        assert success is True
        assert "sync_pre" in hook_called
        assert fc.result == "10"


class TestAsyncErrorHandling:
    """Tests for error handling in async functions."""

    def test_async_function_error_execute(self):
        """Test error handling in async function via execute()."""
        async def async_error_func() -> str:
            raise ValueError("Test error")

        f = Function.from_callable(async_error_func)
        fc = FunctionCall(function=f)
        success = asyncio.run(fc.execute())
        
        assert success is False
        assert fc.error is not None
        assert "Test error" in fc.error

    def test_async_function_error_execute_2(self):
        """Test error handling in async function via execute()."""
        async def async_error_func() -> str:
            raise ValueError("Test error async")

        f = Function.from_callable(async_error_func)
        fc = FunctionCall(function=f)
        success = asyncio.run(fc.execute())
        
        assert success is False
        assert fc.error is not None
        assert "Test error async" in fc.error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
