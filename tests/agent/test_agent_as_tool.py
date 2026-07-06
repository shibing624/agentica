# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Unit tests for Agent.as_tool(), clone(), and when_to_use.
"""
import asyncio
import sys
import os
import json
import unittest
from unittest.mock import AsyncMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import Agent
from agentica.agent.config import PromptConfig
from agentica.agent.as_tool import _serialize_content
from agentica.tools.base import Function
from agentica.run_response import RunResponse


class TestClone(unittest.TestCase):
    """Test Agent.clone() resets all mutable runtime state."""

    def test_clone_basic(self):
        agent = Agent(name="Worker", instructions="Work hard")
        original_id = agent.agent_id
        clone = agent.clone()

        self.assertNotEqual(clone.agent_id, original_id)
        self.assertIsNone(clone.run_id)
        self.assertFalse(clone._running)
        self.assertIsNone(clone._run_hooks)
        self.assertIsNone(clone._enabled_tools)
        self.assertIsNone(clone._enabled_skills)

    def test_clone_resets_session_log(self):
        agent = Agent(name="Worker", instructions="Test", session_id="test-session")
        self.assertIsNotNone(agent._session_log)
        clone = agent.clone()
        self.assertIsNone(clone._session_log)

    def test_clone_resets_default_run_hooks(self):
        agent = Agent(name="Worker", instructions="Test")
        from agentica.hooks import ConversationArchiveHooks
        agent._default_run_hooks = ConversationArchiveHooks()
        clone = agent.clone()
        self.assertIsNone(clone._default_run_hooks)

    def test_clone_shares_config(self):
        """Clone shares heavy config (model def, tools, instructions)."""
        agent = Agent(name="Worker", instructions="Do stuff", tools=[])
        clone = agent.clone()
        self.assertEqual(clone.name, agent.name)
        self.assertEqual(clone.instructions, agent.instructions)
        # Fresh working memory
        self.assertIsNot(clone.working_memory, agent.working_memory)


class TestAsToolBasic(unittest.TestCase):
    """Test as_tool() basic behavior."""

    def test_as_tool_returns_function(self):
        agent = Agent(name="Test Agent", instructions="You are a test agent")
        tool = agent.as_tool()
        self.assertIsInstance(tool, Function)

    def test_as_tool_default_name_from_agent_name(self):
        agent = Agent(name="Chinese Translator", instructions="Translate to Chinese")
        tool = agent.as_tool()
        self.assertEqual(tool.name, "chinese_translator")

    def test_as_tool_custom_name(self):
        agent = Agent(name="Chinese Translator", instructions="Translate to Chinese")
        tool = agent.as_tool(tool_name="translate_zh")
        self.assertEqual(tool.name, "translate_zh")

    def test_as_tool_default_description_from_agent_description(self):
        agent = Agent(name="Translator", description="A professional translator agent", instructions="Translate text")
        tool = agent.as_tool()
        self.assertIn("A professional translator agent", tool.description)

    def test_as_tool_default_description_from_when_to_use(self):
        agent = Agent(
            name="Translator",
            description="A translator",
            when_to_use="Use when the user needs text translated to Chinese",
            instructions="Translate text",
        )
        tool = agent.as_tool()
        self.assertIn("Use when the user needs text translated to Chinese", tool.description)

    def test_as_tool_default_description_from_agent_role(self):
        agent = Agent(
            name="Translator",
            prompt_config=PromptConfig(role="Professional Chinese translator"),
            instructions="Translate text",
        )
        tool = agent.as_tool()
        self.assertIn("Professional Chinese translator", tool.description)

    def test_as_tool_custom_description(self):
        agent = Agent(name="Translator", instructions="Translate text")
        tool = agent.as_tool(tool_description="Custom description for translation")
        self.assertIn("Custom description for translation", tool.description)

    def test_as_tool_name_fallback_to_agent_id(self):
        agent = Agent(instructions="Test agent")
        tool = agent.as_tool()
        self.assertTrue(tool.name.startswith("agent_"))
        self.assertEqual(len(tool.name), 14)  # "agent_" + 8 chars

    def test_as_tool_has_entrypoint(self):
        agent = Agent(name="Test Agent", instructions="Test")
        tool = agent.as_tool()
        self.assertIsNotNone(tool.entrypoint)
        self.assertTrue(callable(tool.entrypoint))


class TestAsToolExecution(unittest.TestCase):
    """Test as_tool() execution via clone + run."""

    def test_as_tool_calls_agent_run(self):
        agent = Agent(name="Translator", instructions="Translate text")
        tool = agent.as_tool()

        mock_response = RunResponse(content="result text")
        with patch.object(Agent, 'run', new_callable=AsyncMock, return_value=mock_response):
            result = asyncio.run(tool.entrypoint("Hello world"))

        self.assertEqual(result, "result text")

    def test_as_tool_handles_none_content(self):
        agent = Agent(name="Translator", instructions="Translate text")
        tool = agent.as_tool()

        mock_response = RunResponse(content=None)
        with patch.object(Agent, 'run', new_callable=AsyncMock, return_value=mock_response):
            result = asyncio.run(tool.entrypoint("Hello"))

        self.assertEqual(result, "No response from agent.")

    def test_as_tool_handles_none_response(self):
        agent = Agent(name="Translator", instructions="Translate text")
        tool = agent.as_tool()

        with patch.object(Agent, 'run', new_callable=AsyncMock, return_value=None):
            result = asyncio.run(tool.entrypoint("Hello"))

        self.assertEqual(result, "No response from agent.")

    def test_as_tool_custom_output_extractor(self):
        def custom_extractor(response: RunResponse) -> str:
            return f"Extracted: {response.content} (run_id: {response.run_id})"

        agent = Agent(name="Translator", instructions="Translate text")
        tool = agent.as_tool(custom_output_extractor=custom_extractor)

        mock_response = RunResponse(content="raw output", run_id="test-123")
        with patch.object(Agent, 'run', new_callable=AsyncMock, return_value=mock_response):
            result = asyncio.run(tool.entrypoint("Hello"))

        self.assertEqual(result, "Extracted: raw output (run_id: test-123)")

    def test_as_tool_serializes_dict_content(self):
        agent = Agent(name="Analyzer", instructions="Analyze")
        tool = agent.as_tool()

        mock_response = RunResponse(content={"key": "value", "number": 42})
        with patch.object(Agent, 'run', new_callable=AsyncMock, return_value=mock_response):
            result = asyncio.run(tool.entrypoint("Analyze this"))

        parsed = json.loads(result)
        self.assertEqual(parsed["key"], "value")
        self.assertEqual(parsed["number"], 42)


class TestSerializeContent(unittest.TestCase):
    """Test _serialize_content helper."""

    def test_string_passthrough(self):
        self.assertEqual(_serialize_content("plain text"), "plain text")

    def test_dict_serialized(self):
        result = _serialize_content({"key": "value"})
        parsed = json.loads(result)
        self.assertEqual(parsed["key"], "value")

    def test_list_serialized(self):
        result = _serialize_content([1, 2, 3])
        parsed = json.loads(result)
        self.assertEqual(parsed, [1, 2, 3])


class TestWhenToUse(unittest.TestCase):
    """Test when_to_use routing hint."""

    def test_when_to_use_in_as_tool(self):
        agent = Agent(
            name="Coder",
            when_to_use="Use for code generation and debugging tasks",
            instructions="Generate code",
        )
        tool = agent.as_tool()
        self.assertEqual(tool.description, "Use for code generation and debugging tasks")


class TestIntegration(unittest.TestCase):
    """Integration tests for Agent as Tool pattern with mocked LLM."""

    def test_orchestrator_with_agent_tools(self):
        translator_agent = Agent(
            name="Chinese Translator",
            instructions="Translate to Chinese",
        )
        orchestrator = Agent(
            name="Orchestrator",
            instructions="Use translator when asked.",
            tools=[
                translator_agent.as_tool(
                    tool_name="translate_to_chinese",
                    tool_description="Translate text to Chinese",
                ),
            ],
        )
        self.assertIsNotNone(orchestrator.tools)
        self.assertEqual(len(orchestrator.tools), 1)
        tool = orchestrator.tools[0]
        self.assertIsInstance(tool, Function)
        self.assertEqual(tool.name, "translate_to_chinese")

    def test_multiple_agent_tools(self):
        translator = Agent(name="Translator", instructions="Translate")
        summarizer = Agent(name="Summarizer", instructions="Summarize")
        analyzer = Agent(name="Analyzer", instructions="Analyze")

        orchestrator = Agent(
            name="Orchestrator",
            instructions="Coordinate",
            tools=[
                translator.as_tool(tool_name="translate", tool_description="Translate"),
                summarizer.as_tool(tool_name="summarize", tool_description="Summarize"),
                analyzer.as_tool(tool_name="analyze", tool_description="Analyze"),
            ],
        )
        self.assertEqual(len(orchestrator.tools), 3)
        tool_names = [t.name for t in orchestrator.tools]
        self.assertIn("translate", tool_names)
        self.assertIn("summarize", tool_names)
        self.assertIn("analyze", tool_names)

    def test_agent_tool_execution_chain(self):
        summarizer = Agent(name="Summarizer", instructions="Summarize")
        translator = Agent(name="Translator", instructions="Translate")

        summarize_tool = summarizer.as_tool(tool_name="summarize")
        translate_tool = translator.as_tool(tool_name="translate")

        call_count = 0

        async def _mock_run(msg, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return RunResponse(content="AI is intelligence demonstrated by machines.")
            else:
                return RunResponse(content="AI is machine intelligence. (Chinese)")

        with patch.object(Agent, 'run', new_callable=AsyncMock, side_effect=_mock_run):
            summary = asyncio.run(summarize_tool.entrypoint("Long text about AI..."))
            self.assertEqual(summary, "AI is intelligence demonstrated by machines.")

            translation = asyncio.run(translate_tool.entrypoint(summary))
            self.assertEqual(translation, "AI is machine intelligence. (Chinese)")

        self.assertEqual(call_count, 2)


if __name__ == "__main__":
    unittest.main()
