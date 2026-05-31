# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for AuxiliaryModelRouter + Agent.resolve_auxiliary_model.
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

from agentica.auxiliary_router import AuxiliaryModelRouter, COMPRESSION, GOAL_JUDGE


class _M:
    """Lightweight stand-in for a Model."""
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"M({self.name})"


class TestAuxiliaryModelRouter(unittest.TestCase):
    def setUp(self):
        self.main = _M("main")
        self.aux = _M("aux")
        self.cheap = _M("cheap")

    def test_resolve_falls_back_to_main(self):
        r = AuxiliaryModelRouter(self.main)
        self.assertIs(r.resolve("compression"), self.main)

    def test_resolve_uses_auxiliary(self):
        r = AuxiliaryModelRouter(self.main, self.aux)
        self.assertIs(r.resolve("anything"), self.aux)

    def test_task_override_wins(self):
        r = AuxiliaryModelRouter(self.main, self.aux, {COMPRESSION: self.cheap})
        self.assertIs(r.resolve(COMPRESSION), self.cheap)
        self.assertIs(r.resolve(GOAL_JUDGE), self.aux)  # falls back to aux

    def test_register(self):
        r = AuxiliaryModelRouter(self.main, self.aux)
        r.register("title", self.cheap)
        self.assertIs(r.resolve("title"), self.cheap)


class TestAgentIntegration(unittest.TestCase):
    def _make_agent(self, **kwargs):
        from agentica import Agent
        from agentica.model.openai import OpenAIChat
        main = OpenAIChat(id="main-model", api_key="fake_openai_key")
        return Agent(model=main, **kwargs), main

    def test_resolve_defaults_to_main(self):
        agent, main = self._make_agent()
        self.assertIs(agent.resolve_auxiliary_model("compression"), main)

    def test_resolve_uses_auxiliary_model(self):
        from agentica.model.openai import OpenAIChat
        aux = OpenAIChat(id="aux-model", api_key="fake_openai_key")
        agent, main = self._make_agent(auxiliary_model=aux)
        self.assertIs(agent.resolve_auxiliary_model("compression"), aux)

    def test_per_task_override(self):
        from agentica.model.openai import OpenAIChat
        aux = OpenAIChat(id="aux-model", api_key="fake_openai_key")
        judge = OpenAIChat(id="judge-model", api_key="fake_openai_key")
        agent, main = self._make_agent(
            auxiliary_model=aux,
            auxiliary_task_models={"goal_judge": judge},
        )
        self.assertIs(agent.resolve_auxiliary_model("goal_judge"), judge)
        self.assertIs(agent.resolve_auxiliary_model("compression"), aux)


if __name__ == "__main__":
    unittest.main()
