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
    # steer() only accepts guidance while a run is active (the Runner opens the
    # window via _begin_steer_window and closes it via _end_steer_window). This
    # gate prevents stale guidance from leaking into an unrelated later run.
    # Tests open the window explicitly to exercise the in-run path.
    def test_steer_buffers_and_drains(self):
        agent = Agent()
        agent._begin_steer_window()
        self.assertTrue(agent.steer("keep it compatible"))
        self.assertEqual(agent._drain_steer(), ["keep it compatible"])
        # Drained -> empty on next call.
        self.assertEqual(agent._drain_steer(), [])

    def test_empty_steer_ignored(self):
        agent = Agent()
        agent._begin_steer_window()
        self.assertFalse(agent.steer(""))
        self.assertFalse(agent.steer("   "))
        self.assertEqual(agent._drain_steer(), [])

    def test_steer_rejected_when_not_running(self):
        # Outside a run window steer() must return False (caller falls back to
        # queuing a fresh turn) and must NOT buffer anything.
        agent = Agent()
        self.assertFalse(agent.steer("no active run"))
        self.assertEqual(agent._drain_steer(), [])

    def test_multiple_steers_preserve_order(self):
        agent = Agent()
        agent._begin_steer_window()
        agent.steer("first")
        agent.steer("second")
        self.assertEqual(agent._drain_steer(), ["first", "second"])


class TestUndeliveredSteer(unittest.TestCase):
    """Steering accepted during a run's final inference is never dropped: the
    run ends before another inference can drain it, so _end_steer_window parks
    it and the interactive caller re-queues it as the next turn."""

    def test_end_window_parks_undrained_steer(self):
        agent = Agent()
        agent._begin_steer_window()
        agent.steer("typed during the final inference")
        agent._end_steer_window()
        self.assertEqual(
            agent.pop_undelivered_steer(),
            [("typed during the final inference", False)],
        )

    def test_pop_drains_once(self):
        agent = Agent()
        agent._begin_steer_window()
        agent.steer("a")
        agent.steer("b")
        agent._end_steer_window()
        self.assertEqual(agent.pop_undelivered_steer(), [("a", False), ("b", False)])
        self.assertEqual(agent.pop_undelivered_steer(), [])

    def test_parked_steer_keeps_relayed_provenance(self):
        # A peer/bg line parked mid-flight must go back tagged, not as plain
        # input — otherwise it could regain slash-command dispatch.
        agent = Agent()
        agent._begin_steer_window()
        agent.steer("typed by the user")
        agent.steer("#3 (term_2) finished: ok", relayed=True)
        agent._end_steer_window()
        self.assertEqual(
            agent.pop_undelivered_steer(),
            [("typed by the user", False), ("#3 (term_2) finished: ok", True)],
        )

    def test_drained_steer_is_not_parked(self):
        # Guidance that DID reach an inference was delivered — it must not
        # resurface as a queued next turn.
        agent = Agent()
        agent._begin_steer_window()
        agent.steer("delivered")
        agent._drain_steer()  # what the Runner does before each inference
        agent._end_steer_window()
        self.assertEqual(agent.pop_undelivered_steer(), [])

    def test_parked_steer_never_leaks_into_the_next_run(self):
        from agentica.runner import Runner

        agent = Agent()
        agent._begin_steer_window()
        agent.steer("too late")
        agent._end_steer_window()
        # A new run opens: the parked text must not be injected into its
        # messages — only the CLI's explicit re-queue may deliver it.
        agent._begin_steer_window()
        messages = [Message(role="user", content="next task")]
        Runner._inject_steering(messages, agent)
        self.assertFalse(any("too late" in (m.content or "") for m in messages))
        # ...and it is still waiting for the caller to pop it.
        self.assertEqual(agent.pop_undelivered_steer(), [("too late", False)])


class TestSteerInjection(unittest.TestCase):
    """The Runner flushes pending steering right before each model inference."""

    def test_inject_steering_appends_guidance(self):
        from agentica.runner import Runner

        agent = Agent()
        agent._begin_steer_window()
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
        agent._begin_steer_window()
        agent.steer("guidance")
        messages = []
        Runner._inject_steering(messages, agent)
        Runner._inject_steering(messages, agent)  # second inference: nothing new
        injected = [m for m in messages if "guidance" in (m.content or "")]
        self.assertEqual(len(injected), 1)

    def test_leftover_survives_until_injected(self):
        # Guidance steered during a run stays buffered until an inference drains
        # it. (Under the run-window model, _begin_steer_window clears stale
        # guidance at run start, so "leftover" only means "not yet injected in
        # THIS run" — it is drained on the next inference, not carried across
        # runs.)
        from agentica.runner import Runner

        agent = Agent()
        agent._begin_steer_window()
        agent.steer("later")
        # No inference happened yet (buffer untouched) -> still there.
        messages = []
        Runner._inject_steering(messages, agent)
        self.assertEqual(len([m for m in messages if "later" in (m.content or "")]), 1)

    def test_steer_folds_into_trailing_tool_result(self):
        # When the loop is mid-task (last message is a tool result), steering is
        # folded into that tool result instead of appending a new user message,
        # so role alternation stays intact (no double user turn).
        from agentica.runner import Runner

        agent = Agent()
        agent._begin_steer_window()
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
        agent._begin_steer_window()
        agent.steer("g")
        messages = [
            Message(role="tool", content="out", tool_call_id="c1", tool_name="t"),
        ]
        Runner._inject_steering(messages, agent)
        self.assertTrue(all(m.role != "user" for m in messages))

    def test_multiple_steers_folded_together(self):
        from agentica.runner import Runner

        agent = Agent()
        agent._begin_steer_window()
        agent.steer("first")
        agent.steer("second")
        messages = [Message(role="tool", content="out", tool_call_id="c1", tool_name="t")]
        Runner._inject_steering(messages, agent)
        self.assertEqual(len(messages), 1)
        self.assertIn("first", messages[-1].content)
        self.assertIn("second", messages[-1].content)


if __name__ == "__main__":
    unittest.main()
