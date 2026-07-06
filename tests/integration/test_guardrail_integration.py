# -*- coding: utf-8 -*-
"""Tests for guardrails integration into run_function_calls()."""
import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from agentica.agent import Agent
from agentica.model.openai import OpenAIChat
from agentica.guardrails.tool import (
    ToolInputGuardrail,
    ToolOutputGuardrail,
    ToolInputGuardrailData,
    ToolOutputGuardrailData,
    ToolGuardrailFunctionOutput,
)


def _make_agent(**kwargs):
    return Agent(
        model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
        **kwargs,
    )


class TestGuardrailFields(unittest.TestCase):
    """Agent carries guardrail lists."""

    def test_default_empty(self):
        agent = _make_agent()
        self.assertEqual(agent.tool_input_guardrails, [])
        self.assertEqual(agent.tool_output_guardrails, [])

    def test_set_guardrails(self):
        ig = ToolInputGuardrail(
            guardrail_function=lambda d: ToolGuardrailFunctionOutput.allow(),
            name="test_in",
        )
        og = ToolOutputGuardrail(
            guardrail_function=lambda d: ToolGuardrailFunctionOutput.allow(),
            name="test_out",
        )
        agent = _make_agent(
            tool_input_guardrails=[ig],
            tool_output_guardrails=[og],
        )
        self.assertEqual(len(agent.tool_input_guardrails), 1)
        self.assertEqual(len(agent.tool_output_guardrails), 1)


class TestInputGuardrailReject(unittest.TestCase):
    """Input guardrail that rejects should skip tool execution."""

    def test_input_guardrail_reject(self):
        async def blocking_guard(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
            return ToolGuardrailFunctionOutput.reject_content("Blocked by guardrail")

        ig = ToolInputGuardrail(guardrail_function=blocking_guard, name="blocker")
        agent = _make_agent(tool_input_guardrails=[ig])

        from agentica.guardrails.tool import (
            run_tool_input_guardrails,
            ToolContext,
        )

        async def _run():
            guard_data = ToolInputGuardrailData(
                context=ToolContext(
                    tool_name="test_tool",
                    tool_arguments=json.dumps({"arg": "val"}),
                    tool_call_id="tc_1",
                    agent=agent,
                ),
                agent=agent,
            )
            result = await run_tool_input_guardrails(guard_data, agent.tool_input_guardrails)
            return result

        result = asyncio.run(_run())
        self.assertTrue(result.is_reject_content())
        self.assertEqual(result.get_reject_message(), "Blocked by guardrail")


class TestInputGuardrailAllow(unittest.TestCase):
    """Input guardrail that allows should not block."""

    def test_input_guardrail_allow(self):
        async def allow_guard(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
            return ToolGuardrailFunctionOutput.allow()

        ig = ToolInputGuardrail(guardrail_function=allow_guard, name="allower")
        agent = _make_agent(tool_input_guardrails=[ig])

        from agentica.guardrails.tool import (
            run_tool_input_guardrails,
            ToolContext,
        )

        async def _run():
            guard_data = ToolInputGuardrailData(
                context=ToolContext(
                    tool_name="test_tool",
                    tool_arguments=None,
                    tool_call_id="tc_2",
                    agent=agent,
                ),
                agent=agent,
            )
            result = await run_tool_input_guardrails(guard_data, agent.tool_input_guardrails)
            return result

        result = asyncio.run(_run())
        self.assertTrue(result.is_allow())


class TestOutputGuardrailReject(unittest.TestCase):
    """Output guardrail that rejects should override tool result."""

    def test_output_guardrail_reject(self):
        async def blocking_out_guard(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
            return ToolGuardrailFunctionOutput.reject_content("Output blocked")

        og = ToolOutputGuardrail(guardrail_function=blocking_out_guard, name="out_blocker")
        agent = _make_agent(tool_output_guardrails=[og])

        from agentica.guardrails.tool import (
            run_tool_output_guardrails,
            ToolContext,
        )

        async def _run():
            guard_data = ToolOutputGuardrailData(
                context=ToolContext(
                    tool_name="test_tool",
                    tool_arguments=None,
                    tool_call_id="tc_3",
                    agent=agent,
                ),
                agent=agent,
                output="some tool output",
            )
            result = await run_tool_output_guardrails(guard_data, agent.tool_output_guardrails)
            return result

        result = asyncio.run(_run())
        self.assertTrue(result.is_reject_content())
        self.assertEqual(result.get_reject_message(), "Output blocked")


if __name__ == "__main__":
    unittest.main()
