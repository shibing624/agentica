# -*- coding: utf-8 -*-
"""AuxSession tests — bounded, isolated history for auxiliary LLM tasks."""
import asyncio

from agentica.aux_session import AuxSession
from agentica.goals import GoalManager, judge_goal
from agentica.memory.session_log import SessionLog


class _FakeJudgeModel:
    """Records every request it receives; answers with a valid verdict."""

    id = "fake-judge"

    def __init__(self, payload):
        self.requests = []
        self._payload = payload

    async def response(self, messages):
        self.requests.append(messages)
        return type("Resp", (), {"content": self._payload})()


VALID_VERDICT = '{"done": false, "reason": "more work needed"}'
VALID_VERDICT_DONE = '{"done": true, "reason": "evidence found"}'


class TestAuxSessionBasics:
    def test_build_request_orders_system_history_user(self):
        s = AuxSession("judge")
        s.commit("u1", "a1")
        msgs = s.build_request("SYS", "u2")
        assert [m.role for m in msgs] == ["system", "user", "assistant", "user"]
        assert msgs[0].content == "SYS"
        assert msgs[-1].content == "u2"

    def test_bounded_by_exchanges(self):
        s = AuxSession("judge", max_exchanges=3)
        for i in range(5):
            s.commit(f"u{i}", f"a{i}")
        assert len(s) == 3
        contents = [m.content for m in s.context_messages()]
        assert "u0" not in contents and "a0" not in contents
        assert "u4" in contents

    def test_oldest_pair_trimmed_first(self):
        s = AuxSession("judge", max_exchanges=2)
        s.commit("old-u", "old-a")
        s.commit("mid-u", "mid-a")
        s.commit("new-u", "new-a")
        contents = [m.content for m in s.context_messages()]
        assert contents == ["mid-u", "mid-a", "new-u", "new-a"]

    def test_reset_clears(self):
        s = AuxSession("judge")
        s.commit("u", "a")
        s.reset()
        assert len(s) == 0
        assert s.context_messages() == []


class TestJudgeGoalSession:
    def test_judge_accumulates_history_across_turns(self):
        model = _FakeJudgeModel(VALID_VERDICT)
        session = AuxSession("goal_judge")
        asyncio.run(judge_goal(model, "obj", "resp-1", session=session))
        asyncio.run(judge_goal(model, "obj", "resp-2", session=session))

        assert len(session) == 2
        second_request = model.requests[1]
        # system + first exchange + current user turn
        assert [m.role for m in second_request] == ["system", "user", "assistant", "user"]
        assert "resp-1" in second_request[1].content
        assert second_request[-1].content.endswith("(empty)") is False

    def test_unparseable_verdict_not_committed(self):
        model = _FakeJudgeModel("not json at all")
        session = AuxSession("goal_judge")
        verdict, _reason, parse_failed = asyncio.run(
            judge_goal(model, "obj", "resp", session=session)
        )
        assert parse_failed or verdict == "continue"
        assert len(session) == 0

    def test_no_session_keeps_stateless_behavior(self):
        model = _FakeJudgeModel(VALID_VERDICT)
        asyncio.run(judge_goal(model, "obj", "resp"))
        assert [m.role for m in model.requests[0]] == ["system", "user"]


class TestGoalManagerJudgeSession:
    def _manager(self, tmp_path, model):
        log = SessionLog("s-judge", base_dir=str(tmp_path))
        return GoalManager(log, judge_model=model, auto_judge=True)

    def test_manager_uses_persistent_judge_session(self, tmp_path):
        model = _FakeJudgeModel(VALID_VERDICT)
        mgr = self._manager(tmp_path, model)
        mgr.set("build a thing")
        asyncio.run(mgr.evaluate_after_turn("working on it", token_delta=100, elapsed_sec=1.0))
        asyncio.run(mgr.evaluate_after_turn("still working", token_delta=100, elapsed_sec=1.0))
        assert len(mgr._judge_session) == 2
        # second judge call carried the first exchange → stable growing prefix
        assert [m.role for m in model.requests[1]] == ["system", "user", "assistant", "user"]

    def test_new_goal_resets_judge_session(self, tmp_path):
        model = _FakeJudgeModel(VALID_VERDICT)
        mgr = self._manager(tmp_path, model)
        mgr.set("goal one")
        asyncio.run(mgr.evaluate_after_turn("progress", token_delta=100, elapsed_sec=1.0))
        assert len(mgr._judge_session) == 1
        mgr.set("goal two")
        assert len(mgr._judge_session) == 0
        asyncio.run(mgr.evaluate_after_turn("fresh turn", token_delta=100, elapsed_sec=1.0))
        # judge request after reset starts clean again
        last_request = model.requests[-1]
        assert [m.role for m in last_request] == ["system", "user"]


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-v"]))
