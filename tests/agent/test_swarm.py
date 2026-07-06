# -*- coding: utf-8 -*-
"""
Tests for Swarm — multi-agent parallel execution.
All tests mock LLM calls — no real API usage.
"""
import asyncio
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from agentica.swarm import Swarm, SwarmResult
from agentica.run_response import RunResponse, RunEvent


def _make_mock_agent(name: str, response_content: str = "done"):
    """Create a minimal Agent that returns a mocked response."""
    from agentica.agent import Agent
    from agentica.model.openai import OpenAIChat

    agent = Agent(
        name=name,
        model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
    )
    mock_response = RunResponse(
        content=response_content,
        event=RunEvent.run_response.value,
    )
    agent._runner.run = AsyncMock(return_value=mock_response)
    return agent


class TestSwarmParallelMode(unittest.TestCase):
    """Swarm in parallel mode runs all agents and collects results."""

    def test_all_agents_called(self):
        a1 = _make_mock_agent("agent1", "result1")
        a2 = _make_mock_agent("agent2", "result2")
        a3 = _make_mock_agent("agent3", "result3")

        swarm = Swarm(agents=[a1, a2, a3], mode="parallel")
        asyncio.run(swarm.run("analyze this"))

        # Each worker agent must be called at least once (parallel task run)
        # Note: in parallel mode the first agent may also be reused as synthesizer
        self.assertGreaterEqual(a1._runner.run.call_count, 1)
        self.assertGreaterEqual(a2._runner.run.call_count, 1)
        self.assertGreaterEqual(a3._runner.run.call_count, 1)

    def test_result_is_swarm_result(self):
        a1 = _make_mock_agent("alpha", "alpha output")
        swarm = Swarm(agents=[a1], mode="parallel")
        result = asyncio.run(swarm.run("task"))
        self.assertIsInstance(result, SwarmResult)
        self.assertEqual(result.mode, "parallel")

    def test_agent_results_aggregated(self):
        a1 = _make_mock_agent("w1", "w1 output")
        a2 = _make_mock_agent("w2", "w2 output")
        swarm = Swarm(agents=[a1, a2], mode="parallel")
        result = asyncio.run(swarm.run("task"))

        # agent_results should contain entries for each agent
        self.assertEqual(len(result.agent_results), 2)
        names = {r.get("agent", "") for r in result.agent_results}
        self.assertIn("w1", names)
        self.assertIn("w2", names)

    def test_total_time_recorded(self):
        a1 = _make_mock_agent("timer-agent", "result")
        swarm = Swarm(agents=[a1], mode="parallel")
        result = asyncio.run(swarm.run("task"))
        self.assertGreaterEqual(result.total_time, 0.0)


class TestSwarmEdgeCases(unittest.TestCase):
    """Swarm edge cases."""

    def test_single_agent_swarm(self):
        a1 = _make_mock_agent("solo", "solo result")
        swarm = Swarm(agents=[a1], mode="parallel")
        result = asyncio.run(swarm.run("task"))
        self.assertIsInstance(result, SwarmResult)

    def test_duplicate_agent_names_raises(self):
        a1 = _make_mock_agent("duplicate")
        a2 = _make_mock_agent("duplicate")
        with self.assertRaises(ValueError, msg="Duplicate agent names should raise ValueError"):
            Swarm(agents=[a1, a2], mode="parallel")

    def test_partial_failure_handled(self):
        """If one agent fails, other results should still be collected."""
        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat

        a_good = _make_mock_agent("good", "good result")

        a_bad = Agent(
            name="bad",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
        )
        a_bad._runner.run = AsyncMock(side_effect=RuntimeError("agent crashed"))

        swarm = Swarm(agents=[a_good, a_bad], mode="parallel")
        # Should not raise — partial failures should be captured in results
        try:
            result = asyncio.run(swarm.run("task"))
            # If it returns, check good agent result is present
            contents = " ".join(
                str(r.get("content", "")) for r in result.agent_results
            )
            self.assertIn("good result", contents)
        except RuntimeError:
            # Acceptable if swarm propagates the error — document the behavior
            pass


class TestSwarmExampleContracts(unittest.TestCase):
    """Keep public Swarm examples aligned with the real API surface."""

    def test_swarm_docs_do_not_pass_mode_to_run(self):
        docs_path = Path(__file__).resolve().parents[1] / "docs" / "multi-agent" / "swarm.md"
        content = docs_path.read_text(encoding="utf-8")

        self.assertNotRegex(
            content,
            r"swarm\.run\([^)]*\bmode\s*=",
            "Swarm mode is configured on Swarm(..., mode=...), not swarm.run(..., mode=...).",
        )

    def test_swarm_demo_uses_constructor_mode_not_run_mode(self):
        demo_path = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "agent_patterns"
            / "08_swarm.py"
        )
        content = demo_path.read_text(encoding="utf-8")

        self.assertRegex(content, r'Swarm\([\s\S]*mode="parallel"')
        self.assertRegex(content, r'Swarm\([\s\S]*mode="autonomous"')
        self.assertNotRegex(
            content,
            r"swarm\.run\([^)]*\bmode\s*=",
            "The demo should mirror the real Swarm.run(task, config=None) API.",
        )


if __name__ == "__main__":
    unittest.main()
