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


if __name__ == "__main__":
    unittest.main()
