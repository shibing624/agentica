# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for v3 refactoring - provider factory, @tool decorator, Runner, guardrails core.

All tests mock LLM API keys to avoid authentication errors.
"""
import asyncio
import sys
import os
import unittest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===========================================================================
# Test 1: Provider Factory (Phase 1)
# ===========================================================================

class TestProviderFactory(unittest.TestCase):
    """Tests for top-level OpenAI-compatible provider factories."""

    def test_deepseek_factory(self):
        from agentica import DeepSeekChat
        from agentica import OpenAIChat
        model = DeepSeekChat(api_key="fake_key")
        self.assertIsInstance(model, OpenAIChat)
        self.assertIn("deepseek", model.base_url)
        self.assertEqual(model.id, "deepseek-v4-flash")
        self.assertEqual(model.context_window, 1_000_000)
        self.assertIsNone(model.reasoning_effort)
        self.assertIsNone(model.extra_body)

    def test_deepseek_factory_allows_overrides(self):
        from agentica import DeepSeekChat

        model = DeepSeekChat(
            id="deepseek-v4-flash",
            api_key="fake_key",
            reasoning_effort="low",
            extra_body={"thinking": {"type": "disabled"}},
        )

        self.assertEqual(model.id, "deepseek-v4-flash")
        self.assertEqual(model.reasoning_effort, "low")
        self.assertEqual(model.extra_body, {"thinking": {"type": "disabled"}})

    def test_qwen_factory(self):
        from agentica import QwenChat, OpenAIChat
        model = QwenChat(api_key="fake_key")
        self.assertIsInstance(model, OpenAIChat)

    def test_nvidia_factory_defaults(self):
        from agentica import NvidiaChat, OpenAIChat

        model = NvidiaChat(api_key="fake_key")

        self.assertIsInstance(model, OpenAIChat)
        self.assertEqual(model.id, "deepseek-ai/deepseek-v4-flash")
        self.assertEqual(model.base_url, "https://integrate.api.nvidia.com/v1")
        self.assertIsNone(model.temperature)
        self.assertIsNone(model.top_p)
        self.assertIsNone(model.extra_body)

    def test_provider_factories_registry(self):
        from agentica import PROVIDER_FACTORIES
        self.assertIn("deepseek", PROVIDER_FACTORIES)
        self.assertIn("qwen", PROVIDER_FACTORIES)
        self.assertIn("nvidia", PROVIDER_FACTORIES)

    def test_legacy_short_aliases(self):
        from agentica import DeepSeek, Qwen, ZhipuAI, DeepSeekChat
        self.assertIs(DeepSeek, DeepSeekChat)


# ===========================================================================
# Test 2: @tool Decorator (Phase 4)
# ===========================================================================

class TestToolDecorator(unittest.TestCase):
    """Tests for tools/decorators.py @tool decorator."""

    def test_tool_decorator_basic(self):
        from agentica.tools.decorators import tool

        @tool()
        def my_func(x: int, y: str = "hello") -> str:
            """A test function."""
            return f"{y}: {x}"

        self.assertTrue(hasattr(my_func, '_tool_metadata'))
        meta = my_func._tool_metadata
        self.assertEqual(meta['name'], 'my_func')
        self.assertEqual(meta['description'], 'A test function.')

    def test_tool_decorator_custom_name(self):
        from agentica.tools.decorators import tool

        @tool(name="custom_name", description="custom desc")
        def another_func():
            pass

        meta = another_func._tool_metadata
        self.assertEqual(meta['name'], 'custom_name')
        self.assertEqual(meta['description'], 'custom desc')

    def test_from_callable_with_tool_metadata(self):
        """Function.from_callable() should detect _tool_metadata."""
        from agentica.tools.decorators import tool
        from agentica.tools.base import Function

        @tool(name="add_numbers", description="Add two numbers")
        def add(a: int, b: int) -> int:
            """Original docstring."""
            return a + b

        func = Function.from_callable(add)
        self.assertEqual(func.name, "add_numbers")
        self.assertEqual(func.description, "Add two numbers")


# ===========================================================================
# Test 3: Global Tool Registry (Phase 4)
# ===========================================================================

class TestToolRegistry(unittest.TestCase):
    """Tests for tools/registry.py global registry."""

    def setUp(self):
        from agentica.tools.registry import clear_registry
        clear_registry()

    def tearDown(self):
        from agentica.tools.registry import clear_registry
        clear_registry()

    def test_register_and_get(self):
        from agentica.tools.registry import register_tool, get_tool

        def my_tool():
            pass

        register_tool("test_tool", my_tool)
        retrieved = get_tool("test_tool")
        self.assertIs(retrieved, my_tool)

    def test_list_tools(self):
        from agentica.tools.registry import register_tool, list_tools

        register_tool("tool_a", lambda: None)
        register_tool("tool_b", lambda: None)
        names = list_tools()
        self.assertIn("tool_a", names)
        self.assertIn("tool_b", names)

    def test_unregister(self):
        from agentica.tools.registry import register_tool, unregister_tool, get_tool

        register_tool("temp", lambda: None)
        unregister_tool("temp")
        with self.assertRaises(KeyError):
            get_tool("temp")


# ===========================================================================
# Test 4: Runner Independent Tests (Phase 5)
# ===========================================================================

class TestRunner(unittest.TestCase):
    """Tests for agentica/runner.py Runner class."""

    def test_runner_creation(self):
        """Runner should be created by Agent.__init__."""
        from agentica.agent import Agent
        agent = Agent()
        from agentica.runner import Runner
        self.assertIsInstance(agent._runner, Runner)
        self.assertIs(agent._runner.agent, agent)

    def test_runner_has_run_methods(self):
        """Runner should have all run methods."""
        from agentica.runner import Runner
        self.assertTrue(asyncio.iscoroutinefunction(Runner.run))
        self.assertTrue(hasattr(Runner, 'run_sync'))
        self.assertTrue(hasattr(Runner, 'run_stream'))
        self.assertTrue(hasattr(Runner, 'run_stream_sync'))

    def test_run_sync_delegates_to_runner(self):
        """agent.run_sync should delegate to agent._runner.run_sync."""
        from agentica.agent import Agent
        from agentica.run_response import RunResponse
        agent = Agent(model=Mock())
        agent._runner.run = AsyncMock(return_value=RunResponse(content="mocked"))
        resp = agent.run_sync("test")
        self.assertEqual(resp.content, "mocked")
        agent._runner.run.assert_called_once()

    def test_save_run_response_to_file(self):
        """Runner should have save_run_response_to_file method."""
        from agentica.runner import Runner
        self.assertTrue(hasattr(Runner, 'save_run_response_to_file'))


# ===========================================================================
# Test 5: Guardrails Core (Phase 6)
# ===========================================================================

class TestGuardrailsCore(unittest.TestCase):
    """Tests for guardrails/core.py unified abstraction."""

    def test_guardrail_output_allow(self):
        from agentica.guardrails.core import GuardrailOutput
        output = GuardrailOutput.allow(output_info="ok")
        self.assertFalse(output.tripwire_triggered)
        self.assertEqual(output.output_info, "ok")

    def test_guardrail_output_block(self):
        from agentica.guardrails.core import GuardrailOutput
        output = GuardrailOutput.block(output_info="bad")
        self.assertTrue(output.tripwire_triggered)

    def test_guardrail_triggered_exception(self):
        from agentica.guardrails.core import GuardrailTriggered, GuardrailOutput
        output = GuardrailOutput.block()
        exc = GuardrailTriggered("test_guard", output)
        self.assertEqual(exc.guardrail_name, "test_guard")
        self.assertIs(exc.output, output)

    def test_base_guardrail_invoke_sync(self):
        """BaseGuardrail._invoke should handle sync functions."""
        from agentica.guardrails.core import BaseGuardrail, GuardrailOutput

        def sync_func(data):
            return GuardrailOutput.allow()

        guard = BaseGuardrail(guardrail_function=sync_func)

        async def run():
            return await guard._invoke("data")

        result = asyncio.run(run())
        self.assertIsInstance(result, GuardrailOutput)
        self.assertFalse(result.tripwire_triggered)

    def test_base_guardrail_invoke_async(self):
        """BaseGuardrail._invoke should handle async functions."""
        from agentica.guardrails.core import BaseGuardrail, GuardrailOutput

        async def async_func(data):
            return GuardrailOutput.block()

        guard = BaseGuardrail(guardrail_function=async_func)

        async def run():
            return await guard._invoke("data")

        result = asyncio.run(run())
        self.assertTrue(result.tripwire_triggered)

    def test_run_guardrails_seq_all_pass(self):
        """run_guardrails_seq should return all results when none triggered."""
        from agentica.guardrails.core import run_guardrails_seq

        async def run_one(guard):
            return f"result_{guard}", False, guard, None

        async def run():
            return await run_guardrails_seq(["g1", "g2"], run_one)

        results = asyncio.run(run())
        self.assertEqual(results, ["result_g1", "result_g2"])

    def test_run_guardrails_seq_one_triggers(self):
        """run_guardrails_seq should raise when a guardrail triggers."""
        from agentica.guardrails.core import run_guardrails_seq, GuardrailTriggered

        async def run_one(guard):
            if guard == "bad":
                return "blocked", True, "bad", "bad_output"
            return "ok", False, guard, None

        async def run():
            return await run_guardrails_seq(["ok", "bad"], run_one)

        with self.assertRaises(GuardrailTriggered) as ctx:
            asyncio.run(run())
        self.assertEqual(ctx.exception.guardrail_name, "bad")

    def test_agent_guardrail_output_export(self):
        """GuardrailOutput should be exported directly."""
        import agentica.guardrails as guardrails
        from agentica.guardrails import GuardrailOutput as GuardrailOutputExport
        from agentica.guardrails.core import GuardrailOutput

        self.assertIs(GuardrailOutputExport, GuardrailOutput)
        self.assertFalse(hasattr(guardrails, "GuardrailFunctionOutput"))

    def test_exception_hierarchy(self):
        """InputGuardrailTripwireTriggered should inherit from GuardrailTriggered."""
        from agentica.guardrails import InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered
        from agentica.guardrails.core import GuardrailTriggered
        self.assertTrue(issubclass(InputGuardrailTripwireTriggered, GuardrailTriggered))
        self.assertTrue(issubclass(OutputGuardrailTripwireTriggered, GuardrailTriggered))

    def test_tool_exception_hierarchy(self):
        """Tool guardrail exceptions should inherit from GuardrailTriggered."""
        from agentica.guardrails import (
            ToolGuardrailTripwireTriggered,
            ToolInputGuardrailTripwireTriggered,
            ToolOutputGuardrailTripwireTriggered,
        )
        from agentica.guardrails.core import GuardrailTriggered
        self.assertTrue(issubclass(ToolGuardrailTripwireTriggered, GuardrailTriggered))
        self.assertTrue(issubclass(ToolInputGuardrailTripwireTriggered, ToolGuardrailTripwireTriggered))
        self.assertTrue(issubclass(ToolOutputGuardrailTripwireTriggered, ToolGuardrailTripwireTriggered))


# ===========================================================================
# Test 6: Model @dataclass Verification (Phase 2)
# ===========================================================================

class TestModelDataclass(unittest.TestCase):
    """Verify Model hierarchy uses @dataclass, not Pydantic BaseModel."""

    def test_model_base_is_dataclass(self):
        from agentica.model.base import Model
        import dataclasses
        self.assertTrue(dataclasses.is_dataclass(Model))

    def test_openai_chat_is_dataclass(self):
        from agentica.model.openai.chat import OpenAIChat
        import dataclasses
        self.assertTrue(dataclasses.is_dataclass(OpenAIChat))

    def test_model_has_to_dict(self):
        from agentica.model.base import Model
        self.assertTrue(hasattr(Model, 'to_dict'))

    def test_openai_chat_instantiation(self):
        """OpenAIChat should instantiate with api_key (no real API call)."""
        from agentica.model.openai.chat import OpenAIChat
        model = OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")
        self.assertEqual(model.id, "gpt-4o-mini")


# ===========================================================================
# Test 7: __init__.py Lazy Loading (Phase 7)
# ===========================================================================

class TestInitLazyLoading(unittest.TestCase):
    """Verify __init__.py lazy loading works correctly."""

    def test_eager_imports_available(self):
        """Core classes should be importable directly."""
        from agentica import Agent, OpenAIChat, Model, Message, RunResponse, Tool
        self.assertIsNotNone(Agent)
        self.assertIsNotNone(OpenAIChat)

    def test_lazy_guardrail_import(self):
        """Guardrails should be accessible via lazy loading."""
        from agentica import InputGuardrail, OutputGuardrail
        self.assertIsNotNone(InputGuardrail)

    def test_lazy_tool_import(self):
        """Tool classes should be accessible via lazy loading."""
        from agentica import ShellTool
        self.assertIsNotNone(ShellTool)

    def test_provider_alias_functions(self):
        """Provider alias functions should be callable."""
        from agentica import DeepSeekChat, QwenChat
        self.assertTrue(callable(DeepSeekChat))
        self.assertTrue(callable(QwenChat))


# ===========================================================================
# Test 8: Agent with Mocked Model (comprehensive integration)
# ===========================================================================

class TestAgentMockedModel(unittest.TestCase):
    """Integration tests with fully mocked model (no API key needed)."""

    def test_agent_with_mock_model_run_sync(self):
        """Agent.run_sync with completely mocked model."""
        from agentica.agent import Agent
        from agentica.run_response import RunResponse

        agent = Agent()
        agent._runner.run = AsyncMock(return_value=RunResponse(content="Hello!"))
        resp = agent.run_sync("Hi")
        self.assertEqual(resp.content, "Hello!")

    def test_agent_with_fake_openai_key(self):
        """Agent with OpenAIChat using fake key (mocked response)."""
        from agentica.agent import Agent
        from agentica.model.openai.chat import OpenAIChat
        from agentica.run_response import RunResponse

        model = OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")
        agent = Agent(model=model)
        agent._runner.run = AsyncMock(return_value=RunResponse(content="Mocked"))
        resp = agent.run_sync("test")
        self.assertEqual(resp.content, "Mocked")


if __name__ == '__main__':
    unittest.main()
