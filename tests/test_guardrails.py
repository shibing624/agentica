# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for Guardrails module.
"""
import asyncio
import sys
import unittest
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from agentica.guardrails import (
    # Agent-level guardrails
    GuardrailFunctionOutput,
    InputGuardrail,
    OutputGuardrail,
    InputGuardrailResult,
    OutputGuardrailResult,
    input_guardrail,
    output_guardrail,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    run_input_guardrails,
    run_output_guardrails,
    # Tool-level guardrails
    ToolGuardrailFunctionOutput,
    ToolInputGuardrail,
    ToolOutputGuardrail,
    ToolInputGuardrailData,
    ToolOutputGuardrailData,
    ToolContext,
    tool_input_guardrail,
    tool_output_guardrail,
    ToolInputGuardrailTripwireTriggered,
    ToolOutputGuardrailTripwireTriggered,
    run_tool_input_guardrails,
    run_tool_output_guardrails,
)


class TestGuardrailFunctionOutput(unittest.TestCase):
    """Test cases for GuardrailFunctionOutput."""

    def test_allow(self):
        """Test allow() class method."""
        output = GuardrailFunctionOutput.allow(output_info={"test": True})
        self.assertFalse(output.tripwire_triggered)
        self.assertEqual(output.output_info, {"test": True})

    def test_block(self):
        """Test block() class method."""
        output = GuardrailFunctionOutput.block(output_info={"reason": "blocked"})
        self.assertTrue(output.tripwire_triggered)
        self.assertEqual(output.output_info, {"reason": "blocked"})

    def test_default_values(self):
        """Test default values."""
        output = GuardrailFunctionOutput()
        self.assertFalse(output.tripwire_triggered)
        self.assertIsNone(output.output_info)


class TestToolGuardrailFunctionOutput(unittest.TestCase):
    """Test cases for ToolGuardrailFunctionOutput."""

    def test_allow(self):
        """Test allow() class method."""
        output = ToolGuardrailFunctionOutput.allow(output_info={"test": True})
        self.assertTrue(output.is_allow())
        self.assertFalse(output.is_reject_content())
        self.assertFalse(output.is_raise_exception())

    def test_reject_content(self):
        """Test reject_content() class method."""
        output = ToolGuardrailFunctionOutput.reject_content(
            message="Rejected!", output_info={"reason": "test"}
        )
        self.assertFalse(output.is_allow())
        self.assertTrue(output.is_reject_content())
        self.assertFalse(output.is_raise_exception())
        self.assertEqual(output.get_reject_message(), "Rejected!")

    def test_raise_exception(self):
        """Test raise_exception() class method."""
        output = ToolGuardrailFunctionOutput.raise_exception(output_info={"critical": True})
        self.assertFalse(output.is_allow())
        self.assertFalse(output.is_reject_content())
        self.assertTrue(output.is_raise_exception())


class TestInputGuardrailDecorator(unittest.TestCase):
    """Test cases for input_guardrail decorator."""

    def test_decorator_without_args(self):
        """Test decorator without arguments."""

        @input_guardrail
        def my_guardrail(ctx, agent, input_data):
            return GuardrailFunctionOutput.allow()

        self.assertIsInstance(my_guardrail, InputGuardrail)
        self.assertEqual(my_guardrail.get_name(), "my_guardrail")
        self.assertTrue(my_guardrail.run_in_parallel)

    def test_decorator_with_args(self):
        """Test decorator with arguments."""

        @input_guardrail(name="custom_name", run_in_parallel=False)
        def my_guardrail(ctx, agent, input_data):
            return GuardrailFunctionOutput.allow()

        self.assertIsInstance(my_guardrail, InputGuardrail)
        self.assertEqual(my_guardrail.get_name(), "custom_name")
        self.assertFalse(my_guardrail.run_in_parallel)

    def test_async_guardrail(self):
        """Test async guardrail function."""

        @input_guardrail
        async def async_guardrail(ctx, agent, input_data):
            return GuardrailFunctionOutput.allow()

        self.assertIsInstance(async_guardrail, InputGuardrail)


class TestOutputGuardrailDecorator(unittest.TestCase):
    """Test cases for output_guardrail decorator."""

    def test_decorator_without_args(self):
        """Test decorator without arguments."""

        @output_guardrail
        def my_guardrail(ctx, agent, output):
            return GuardrailFunctionOutput.allow()

        self.assertIsInstance(my_guardrail, OutputGuardrail)
        self.assertEqual(my_guardrail.get_name(), "my_guardrail")

    def test_decorator_with_args(self):
        """Test decorator with arguments."""

        @output_guardrail(name="custom_output_guardrail")
        def my_guardrail(ctx, agent, output):
            return GuardrailFunctionOutput.allow()

        self.assertIsInstance(my_guardrail, OutputGuardrail)
        self.assertEqual(my_guardrail.get_name(), "custom_output_guardrail")


class TestToolInputGuardrailDecorator(unittest.TestCase):
    """Test cases for tool_input_guardrail decorator."""

    def test_decorator_without_args(self):
        """Test decorator without arguments."""

        @tool_input_guardrail
        def my_guardrail(data: ToolInputGuardrailData):
            return ToolGuardrailFunctionOutput.allow()

        self.assertIsInstance(my_guardrail, ToolInputGuardrail)
        self.assertEqual(my_guardrail.get_name(), "my_guardrail")

    def test_decorator_with_args(self):
        """Test decorator with arguments."""

        @tool_input_guardrail(name="custom_tool_guardrail")
        def my_guardrail(data: ToolInputGuardrailData):
            return ToolGuardrailFunctionOutput.allow()

        self.assertIsInstance(my_guardrail, ToolInputGuardrail)
        self.assertEqual(my_guardrail.get_name(), "custom_tool_guardrail")


class TestToolOutputGuardrailDecorator(unittest.TestCase):
    """Test cases for tool_output_guardrail decorator."""

    def test_decorator_without_args(self):
        """Test decorator without arguments."""

        @tool_output_guardrail
        def my_guardrail(data: ToolOutputGuardrailData):
            return ToolGuardrailFunctionOutput.allow()

        self.assertIsInstance(my_guardrail, ToolOutputGuardrail)
        self.assertEqual(my_guardrail.get_name(), "my_guardrail")

    def test_decorator_with_args(self):
        """Test decorator with arguments."""

        @tool_output_guardrail(name="custom_tool_output_guardrail")
        def my_guardrail(data: ToolOutputGuardrailData):
            return ToolGuardrailFunctionOutput.allow()

        self.assertIsInstance(my_guardrail, ToolOutputGuardrail)
        self.assertEqual(my_guardrail.get_name(), "custom_tool_output_guardrail")


class TestGuardrailExecution(unittest.TestCase):
    """Test cases for guardrail execution."""

    def test_input_guardrail_run_allow(self):
        """Test running input guardrail that allows."""

        @input_guardrail
        def allow_guardrail(ctx, agent, input_data):
            return GuardrailFunctionOutput.allow(output_info="allowed")

        async def run_test():
            result = await allow_guardrail.run(None, "test input", None)
            self.assertIsInstance(result, InputGuardrailResult)
            self.assertFalse(result.output.tripwire_triggered)
            self.assertEqual(result.output.output_info, "allowed")

        asyncio.run(run_test())

    def test_input_guardrail_run_block(self):
        """Test running input guardrail that blocks."""

        @input_guardrail
        def block_guardrail(ctx, agent, input_data):
            return GuardrailFunctionOutput.block(output_info="blocked")

        async def run_test():
            result = await block_guardrail.run(None, "test input", None)
            self.assertTrue(result.output.tripwire_triggered)

        asyncio.run(run_test())

    def test_output_guardrail_run(self):
        """Test running output guardrail."""

        @output_guardrail
        def check_output(ctx, agent, output):
            if "bad" in str(output):
                return GuardrailFunctionOutput.block()
            return GuardrailFunctionOutput.allow()

        async def run_test():
            # Test allow
            result = await check_output.run(None, "good output", None)
            self.assertFalse(result.output.tripwire_triggered)

            # Test block
            result = await check_output.run(None, "bad output", None)
            self.assertTrue(result.output.tripwire_triggered)

        asyncio.run(run_test())

    def test_tool_input_guardrail_run(self):
        """Test running tool input guardrail."""

        @tool_input_guardrail
        def check_args(data: ToolInputGuardrailData):
            if "forbidden" in str(data.context.tool_arguments):
                return ToolGuardrailFunctionOutput.reject_content("Forbidden argument")
            return ToolGuardrailFunctionOutput.allow()

        async def run_test():
            context = ToolContext(tool_name="test_tool", tool_arguments='{"key": "value"}')
            data = ToolInputGuardrailData(context=context, agent=None)
            result = await check_args.run(data)
            self.assertTrue(result.output.is_allow())

            context = ToolContext(tool_name="test_tool", tool_arguments='{"key": "forbidden"}')
            data = ToolInputGuardrailData(context=context, agent=None)
            result = await check_args.run(data)
            self.assertTrue(result.output.is_reject_content())

        asyncio.run(run_test())

    def test_tool_output_guardrail_run(self):
        """Test running tool output guardrail."""

        @tool_output_guardrail
        def check_output(data: ToolOutputGuardrailData):
            if "secret" in str(data.output):
                return ToolGuardrailFunctionOutput.raise_exception(
                    output_info={"reason": "secret detected"}
                )
            return ToolGuardrailFunctionOutput.allow()

        async def run_test():
            context = ToolContext(tool_name="test_tool")
            data = ToolOutputGuardrailData(context=context, agent=None, output="normal data")
            result = await check_output.run(data)
            self.assertTrue(result.output.is_allow())

            data = ToolOutputGuardrailData(context=context, agent=None, output="secret data")
            result = await check_output.run(data)
            self.assertTrue(result.output.is_raise_exception())

        asyncio.run(run_test())


class TestRunGuardrails(unittest.TestCase):
    """Test cases for run_*_guardrails utility functions."""

    def test_run_input_guardrails_all_allow(self):
        """Test running multiple input guardrails that all allow."""

        @input_guardrail
        def guardrail1(ctx, agent, input_data):
            return GuardrailFunctionOutput.allow()

        @input_guardrail
        def guardrail2(ctx, agent, input_data):
            return GuardrailFunctionOutput.allow()

        async def run_test():
            results = await run_input_guardrails(None, "test", [guardrail1, guardrail2])
            self.assertEqual(len(results), 2)
            for result in results:
                self.assertFalse(result.output.tripwire_triggered)

        asyncio.run(run_test())

    def test_run_input_guardrails_one_blocks(self):
        """Test running input guardrails where one blocks."""

        @input_guardrail
        def allow_guardrail(ctx, agent, input_data):
            return GuardrailFunctionOutput.allow()

        @input_guardrail
        def block_guardrail(ctx, agent, input_data):
            return GuardrailFunctionOutput.block(output_info="blocked")

        async def run_test():
            with self.assertRaises(InputGuardrailTripwireTriggered) as context:
                await run_input_guardrails(None, "test", [allow_guardrail, block_guardrail])
            self.assertEqual(context.exception.guardrail_name, "block_guardrail")

        asyncio.run(run_test())

    def test_run_output_guardrails_all_allow(self):
        """Test running multiple output guardrails that all allow."""

        @output_guardrail
        def guardrail1(ctx, agent, output):
            return GuardrailFunctionOutput.allow()

        @output_guardrail
        def guardrail2(ctx, agent, output):
            return GuardrailFunctionOutput.allow()

        async def run_test():
            results = await run_output_guardrails(None, "test output", [guardrail1, guardrail2])
            self.assertEqual(len(results), 2)

        asyncio.run(run_test())

    def test_run_tool_input_guardrails_reject(self):
        """Test running tool input guardrails with rejection."""

        @tool_input_guardrail
        def reject_guardrail(data: ToolInputGuardrailData):
            return ToolGuardrailFunctionOutput.reject_content("Rejected!")

        async def run_test():
            context = ToolContext(tool_name="test")
            data = ToolInputGuardrailData(context=context, agent=None)
            result = await run_tool_input_guardrails(data, [reject_guardrail])
            self.assertTrue(result.is_reject_content())
            self.assertEqual(result.get_reject_message(), "Rejected!")

        asyncio.run(run_test())

    def test_run_tool_output_guardrails_exception(self):
        """Test running tool output guardrails with exception."""

        @tool_output_guardrail
        def exception_guardrail(data: ToolOutputGuardrailData):
            return ToolGuardrailFunctionOutput.raise_exception()

        async def run_test():
            context = ToolContext(tool_name="test")
            data = ToolOutputGuardrailData(context=context, agent=None, output="test")
            with self.assertRaises(ToolOutputGuardrailTripwireTriggered):
                await run_tool_output_guardrails(data, [exception_guardrail])

        asyncio.run(run_test())


class TestAsyncGuardrails(unittest.TestCase):
    """Test cases for async guardrail functions."""

    def test_async_input_guardrail(self):
        """Test async input guardrail."""

        @input_guardrail
        async def async_guardrail(ctx, agent, input_data):
            await asyncio.sleep(0.01)  # Simulate async operation
            return GuardrailFunctionOutput.allow(output_info="async completed")

        async def run_test():
            result = await async_guardrail.run(None, "test", None)
            self.assertEqual(result.output.output_info, "async completed")

        asyncio.run(run_test())

    def test_async_tool_guardrail(self):
        """Test async tool guardrail."""

        @tool_input_guardrail
        async def async_tool_guardrail(data: ToolInputGuardrailData):
            await asyncio.sleep(0.01)
            return ToolGuardrailFunctionOutput.allow()

        async def run_test():
            context = ToolContext(tool_name="test")
            data = ToolInputGuardrailData(context=context, agent=None)
            result = await async_tool_guardrail.run(data)
            self.assertTrue(result.output.is_allow())

        asyncio.run(run_test())


if __name__ == '__main__':
    unittest.main()
