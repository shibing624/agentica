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

    def test_image_analysis_tool_reuses_agent_model(self):
        from agentica import DeepSeekChat
        from agentica.agent import Agent
        from agentica.tools.image_analysis_tool import ImageAnalysisTool

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fake_deepseek_key"}, clear=True):
            model = DeepSeekChat()
            tool = ImageAnalysisTool()
            Agent(model=model, tools=[tool])
            tool.update_llm()

        self.assertIs(tool.llm, model)


if __name__ == "__main__":
    unittest.main()
