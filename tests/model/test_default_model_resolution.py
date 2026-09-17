# -*- coding: utf-8 -*-
import os
import unittest
from unittest.mock import patch


class TestDefaultModelResolution(unittest.TestCase):
    def test_default_model_preserves_openai_priority_when_configured(self):
        from agentica.model.defaults import create_default_model

        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "fake_openai_key",
                "DEEPSEEK_API_KEY": "fake_deepseek_key",
            },
            clear=True,
        ):
            model = create_default_model()

        self.assertEqual(model.provider, "OpenAI")
        self.assertEqual(model.id, "gpt-4o")
        self.assertEqual(model.api_key, "fake_openai_key")

    def test_default_model_uses_anthropic_when_configured(self):
        from agentica.model.defaults import create_default_model

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake_anthropic_key"}, clear=True):
            model = create_default_model()

        self.assertEqual(model.provider, "Anthropic")
        self.assertEqual(model.id, "claude-3-5-sonnet-20241022")
        self.assertEqual(model.api_key, "fake_anthropic_key")

    def test_agent_default_model_uses_configured_provider_key(self):
        from agentica.agent import Agent

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fake_deepseek_key"}, clear=True):
            agent = Agent()
            agent.update_model()

        self.assertEqual(agent.model.provider, "DeepSeek")
        self.assertEqual(agent.model.id, "deepseek-v4-flash")
        self.assertEqual(agent.model.api_key, "fake_deepseek_key")

    def test_deep_agent_default_model_uses_configured_provider_key(self):
        from agentica.agent.deep import DeepAgent

        with patch.dict(os.environ, {"ZHIPUAI_API_KEY": "fake_zhipu_key"}, clear=True):
            agent = DeepAgent(include_web_search=False, include_fetch_url=False)

        self.assertEqual(agent.model.provider, "ZhipuAI")
        self.assertEqual(agent.model.id, "glm-4.7-flash")
        self.assertIs(agent.auxiliary_model, agent.model)

    def test_vision_tool_receives_the_agent_model(self):
        """``analyze_image`` routes on the agent's own model, so it must get it.

        Without this wiring the tool cannot tell a vision-capable agent (hand it
        the pixels) from a text-only one (describe or OCR instead).
        """
        from agentica import DeepSeekChat
        from agentica.agent import Agent
        from agentica.tools.builtin.vision_tool import BuiltinVisionTool

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fake_deepseek_key"}, clear=True):
            model = DeepSeekChat()
            tool = BuiltinVisionTool()
            Agent(model=model, tools=[tool])

        self.assertIs(tool._agent_model, model)


if __name__ == "__main__":
    unittest.main()
