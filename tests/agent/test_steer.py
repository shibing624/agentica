# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for mid-run steering (agent.steer + post-tool injection).
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

from agentica import Agent
from agentica.model.message import Message


class TestSteerBuffer(unittest.TestCase):
    def test_steer_buffers_and_drains(self):
        agent = Agent()
        self.assertTrue(agent.steer("keep it compatible"))
        self.assertEqual(agent._drain_steer(), ["keep it compatible"])
        # Drained -> empty on next call.
        self.assertEqual(agent._drain_steer(), [])

    def test_empty_steer_ignored(self):
        agent = Agent()
        self.assertFalse(agent.steer(""))
        self.assertFalse(agent.steer("   "))
        self.assertEqual(agent._drain_steer(), [])

    def test_multiple_steers_preserve_order(self):
        agent = Agent()
        agent.steer("first")
        agent.steer("second")
        self.assertEqual(agent._drain_steer(), ["first", "second"])


class TestSteerInjection(unittest.TestCase):
    """The Runner flushes pending steering right before each model inference."""

    def test_inject_steering_appends_guidance(self):
        from agentica.runner import Runner
        agent = Agent()
        agent.steer("don't change the API")
        messages = [Message(role="user", content="do the task")]
        Runner._inject_steering(messages, agent)
        injected = [m for m in messages if m.role == "user" and "don't change the API" in (m.content or "")]
        self.assertEqual(len(injected), 1)
        self.assertIn("[User guidance received while you were working]", injected[0].content)

    def test_inject_noop_without_steer(self):
        from agentica.runner import Runner
        agent = Agent()
        messages = [Message(role="user", content="do the task")]
        Runner._inject_steering(messages, agent)
        self.assertEqual(len(messages), 1)  # nothing buffered -> no-op

    def test_steer_consumed_once(self):
        from agentica.runner import Runner
        agent = Agent()
        agent.steer("guidance")
        messages = []
        Runner._inject_steering(messages, agent)
        Runner._inject_steering(messages, agent)  # second inference: nothing new
        injected = [m for m in messages if "guidance" in (m.content or "")]
        self.assertEqual(len(injected), 1)

    def test_leftover_survives_to_next_run(self):
        # If a run ends before flush, guidance stays buffered for the next run.
        from agentica.runner import Runner
        agent = Agent()
        agent.steer("later")
        # No inference happened (buffer untouched) -> still there.
        messages = []
        Runner._inject_steering(messages, agent)
        self.assertEqual(len([m for m in messages if "later" in (m.content or "")]), 1)

    def test_steer_folds_into_trailing_tool_result(self):
        # When the loop is mid-task (last message is a tool result), steering is
        # folded into that tool result instead of appending a new user message,
        # so role alternation stays intact (no double user turn).
        from agentica.runner import Runner
        agent = Agent()
        agent.steer("focus on the edge cases")
        messages = [
            Message(role="user", content="run the task"),
            Message(role="assistant", content="", tool_calls=[{"id": "c1"}]),
            Message(role="tool", content="tool output", tool_call_id="c1", tool_name="search"),
        ]
        Runner._inject_steering(messages, agent)
        # No new message appended; folded into the tool result.
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[-1].role, "tool")
        self.assertIn("tool output", messages[-1].content)
        self.assertIn("focus on the edge cases", messages[-1].content)
        self.assertIn("[User guidance received while you were working]", messages[-1].content)

    def test_steer_no_double_user_turn_after_tool(self):
        # Regression: must not create two consecutive user-role turns after a
        # tool result (breaks Anthropic alternation on live call and on replay).
        from agentica.runner import Runner
        agent = Agent()
        agent.steer("g")
        messages = [
            Message(role="tool", content="out", tool_call_id="c1", tool_name="t"),
        ]
        Runner._inject_steering(messages, agent)
        self.assertTrue(all(m.role != "user" for m in messages))

    def test_multiple_steers_folded_together(self):
        from agentica.runner import Runner
        agent = Agent()
        agent.steer("first")
        agent.steer("second")
        messages = [Message(role="tool", content="out", tool_call_id="c1", tool_name="t")]
        Runner._inject_steering(messages, agent)
        self.assertEqual(len(messages), 1)
        self.assertIn("first", messages[-1].content)
        self.assertIn("second", messages[-1].content)


if __name__ == "__main__":
    unittest.main()
