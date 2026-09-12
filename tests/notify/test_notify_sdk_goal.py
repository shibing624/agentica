# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: The SDK's ``Agent.run_goal()`` must report the held completion too.

The CLI has its own goal hook, and that is where the release was first put.
But ``run_goal()`` in ``agent/goal_mixin.py`` drives its own ``while`` loop and
never goes through that hook — so an SDK process that installs a sink and calls
``run_goal`` held the completion and never released it. Same silent shape as the
regression before it: nothing errors, the sink keeps working, and the desktop
app just never learns the work is over.

These drive the real ``run_goal`` loop (not a hand-rolled dispatch sequence),
because the whole bug was that the release was wired to the wrong driver.
"""

from __future__ import annotations

import asyncio
import pathlib
import tempfile
import time
from unittest.mock import AsyncMock, patch

import pytest

from agentica.agent import Agent
from agentica.goals import GoalRunResult
from agentica.model.base import ModelResponse
from agentica.model.openai import OpenAIChat
from agentica.notify import install_sink, reset_sink_for_tests
from agentica.notify.config import NotifyConfig
from agentica.run_response import RunResponse

from tests.notify.test_notify_sink import _FakeDesktop


@pytest.fixture(autouse=True)
def _clean():
    reset_sink_for_tests()
    yield
    reset_sink_for_tests()


def _judge(content: str):
    model = AsyncMock()
    model.response = AsyncMock(return_value=ModelResponse(content=content))
    return model


def _completed_bodies(desktop):
    """The payloads of the ``run.completed`` events that reached the app."""
    return [
        r["json"]["payload"]
        for r in desktop.requests
        if r["json"] and r["json"].get("event") == "run.completed"
    ]


def _bare_agent(tmp_path, session_id: str, judge: str) -> Agent:
    """An Agent built by hand: no session log, no real model, no tools."""
    agent = Agent.__new__(Agent)
    agent.model = None
    agent.auxiliary_model = _judge(judge)
    agent.auxiliary_task_models = {}
    agent.session_id = session_id
    agent.session_base_dir = str(tmp_path)
    agent._session_log = None
    agent.goal_manager = None
    agent.work_dir = str(tmp_path)
    agent.tools = None
    agent.user_id = None
    agent.task_anchor = None
    agent._anchor_session_id = None
    agent._tool_runtime_configs = {}
    agent._skill_runtime_configs = {}
    return agent


def _run_goal(tmp_path, session_id, desktop, *, judge, laps=3):
    """Run the real ``run_goal`` loop with the sink installed."""
    install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))
    agent = _bare_agent(tmp_path, session_id, judge)

    response = RunResponse(content="did some work")

    async def _sync_run(self, *_a, **_k):
        # Each lap is a real run as far as the sink is concerned: the runner
        # emits run.completed while the goal is still active, which is exactly
        # why the release cannot be decided at that moment.
        #
        # ``self`` is the agent that actually runs the lap — the clone, not the
        # outer instance ``run_goal`` was called on. Dispatching against the
        # outer one would record the held completion somewhere the release
        # never looks, and the test would then pass only because a single-lap
        # goal reports one completion either way.
        from agentica.notify.sink import notify_sink_dispatch
        from agentica.run_events import RunEventRecord, RunEventType

        notify_sink_dispatch(
            RunEventRecord(
                run_id="r",
                event_type=RunEventType.run_completed,
                agent_id="a",
                payload={"agent_name": "P", "duration_seconds": 1.5, "had_response": True},
            ),
            agent=self,
        )
        return response

    # A plain function, not an AsyncMock: only a real function is a descriptor,
    # so only then is the running agent bound to ``self``.
    with patch.object(Agent, "run", new=_sync_run):
        result = asyncio.run(agent.run_goal("ship it", turn_budget=laps, auto_judge=True))

    return result


class TestTheSdkGoalLoopReleasesTheHeldCompletion:
    def test_a_goal_that_completes_reports_exactly_one_completion(self, tmp_path):
        """The regression: held by every lap, released when the loop ends."""
        desktop = _FakeDesktop()
        try:
            result = _run_goal(
                tmp_path, "sdk-goal-1", desktop, judge='{"done": true, "reason": "shipped"}'
            )
            assert isinstance(result, GoalRunResult)
            assert result.status == "complete"

            time.sleep(0.5)
            bodies = _completed_bodies(desktop)
            assert len(bodies) == 1, f"expected exactly one completion, got {len(bodies)}"
        finally:
            desktop.close()

    def test_a_goal_that_hits_the_turn_budget_also_releases(self, tmp_path):
        """Running out of turns is 'no next lap' too — the app still waits."""
        desktop = _FakeDesktop()
        try:
            result = _run_goal(
                tmp_path,
                "sdk-goal-budget",
                desktop,
                judge='{"done": false, "reason": "keep going"}',
                laps=2,
            )
            assert result.status in {"paused", "budget_limited", "complete"}

            time.sleep(0.5)
            bodies = _completed_bodies(desktop)
            assert len(bodies) == 1, f"expected exactly one completion, got {len(bodies)}"
        finally:
            desktop.close()

    def test_the_released_completion_carries_the_run_payload(self, tmp_path):
        """A held-then-released completion must look like any other one.

        Two producers of the same event name must not hand the consumer
        different shapes — the desktop app cannot explain why ``duration_seconds``
        is there sometimes and not others, and that is the field it wants most
        (how long was I away and how long did it run).
        """
        desktop = _FakeDesktop()
        try:
            _run_goal(
                tmp_path, "sdk-goal-payload", desktop, judge='{"done": true, "reason": "ok"}'
            )
            time.sleep(0.5)
            bodies = _completed_bodies(desktop)
            assert len(bodies) == 1
            payload = bodies[0]
            assert payload["agent_name"] == "P"
            assert "duration_seconds" in payload
            assert payload["had_response"] is True
            assert payload.get("title")
        finally:
            desktop.close()

    def test_an_agent_that_never_ran_a_goal_reports_nothing_extra(self, tmp_path):
        """The release must not invent a completion when none was held."""
        desktop = _FakeDesktop()
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))
            from agentica.notify import goal_finished

            agent = _bare_agent(tmp_path, "sdk-goal-none", '{"done": true}')
            goal_finished(agent, session_id="sdk-goal-none")
            time.sleep(0.3)
            assert _completed_bodies(desktop) == []
        finally:
            desktop.close()


class TestTheCloneTrap:
    def test_the_lap_really_holds_before_it_releases(self, tmp_path):
        """Prove the hold happens at all — otherwise these tests are vacuous.

        Two ways this file could pass while proving nothing, both worth pinning:

        1. ``_goal_is_driving`` reads the goal out of ``agent._session_log``. If
           the lap's agent had no log, holding would never trigger and every
           assertion here would be testing "something that was never held got
           released once" instead.
        2. A single-lap goal reports one completion whether or not holding
           works, so only a multi-lap goal can tell the two apart.

        So: run a lap and assert the completion was actually held, then run the
        multi-lap case where a broken hold would report N completions.
        """
        from agentica.notify.sink import _goal_is_driving

        desktop = _FakeDesktop()
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))
            agent = _bare_agent(tmp_path, "sdk-goal-hold", '{"done": false, "reason": "go"}')

            lap_agents = []
            driving = []

            async def _run(self, *_a, **_k):
                from agentica.notify.sink import notify_sink_dispatch
                from agentica.run_events import RunEventRecord, RunEventType

                lap_agents.append(self)
                # Sampled while the lap is in flight: this is the exact call the
                # sink makes to decide whether to hold the completion.
                driving.append(_goal_is_driving(self))
                notify_sink_dispatch(
                    RunEventRecord(
                        run_id="r",
                        event_type=RunEventType.run_completed,
                        agent_id="a",
                        payload={"agent_name": "P", "duration_seconds": 1.0, "had_response": True},
                    ),
                    agent=self,
                )
                return RunResponse(content="working")

            with patch.object(Agent, "run", new=_run):
                asyncio.run(agent.run_goal("ship it", turn_budget=1, auto_judge=True))

            # The lap runs on a clone, and that clone must be able to see the
            # goal — that is what makes the hold fire.
            assert lap_agents, "the goal loop never ran a lap"
            assert lap_agents[0] is not agent, "run_goal should clone by default"
            assert all(driving), (
                "the lap's agent could not see the active goal, so nothing was "
                "held and the assertions in this file would be vacuous"
            )

            time.sleep(0.4)
            # One lap, one completion: held by the sink, released by the loop.
            assert len(_completed_bodies(desktop)) == 1
        finally:
            desktop.close()

    def test_the_release_lands_even_though_run_goal_clones(self, tmp_path):
        """``run_goal`` runs a clone by default; the flag is on the clone.

        Recording the held completion on one object and releasing from another
        would silently report nothing at all, so this pins that both ends agree.
        """
        desktop = _FakeDesktop()
        try:
            _run_goal(
                tmp_path, "sdk-goal-clone", desktop, judge='{"done": true, "reason": "ok"}'
            )
            time.sleep(0.5)
            assert len(_completed_bodies(desktop)) == 1
        finally:
            desktop.close()
