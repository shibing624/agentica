# -*- coding: utf-8 -*-
"""CLI tool-approval UX: Codex y/p/esc routing, prompt copy, non-interactive deny."""
import asyncio
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")

from agentica.agent.approvals import PendingApproval
from agentica.cli.approvals import (
    approval_decision_from_key,
    build_interactive_approve,
    build_noninteractive_approve,
    format_approval_prompt,
    format_approval_record,
    interrupt_approvals,
    is_approval_request,
)
from agentica.cli.interactive.session_state import SessionState, _InputRequest
from agentica.cli.interactive.tui import (
    _ASK_KEY_HINT,
    _ask_prompt_lines,
    _input_prompt_fragments,
)
from agentica.tools.base import Function, FunctionCall


def _fc(name, arguments=None, *, call_id="c1"):
    fn = Function(name=name)
    fn.entrypoint = lambda **kwargs: "ran"
    fn.is_destructive = name in ("execute", "write_file")
    return FunctionCall(function=fn, arguments=arguments or {}, call_id=call_id)


def _agent(*, mode="ask", work_dir="/tmp/work", cancelled=False):
    return SimpleNamespace(
        tool_config=SimpleNamespace(permission_mode=mode),
        work_dir=work_dir,
        tools=[],
        user_id="default",
        _cancelled=cancelled,
        session_id="test-session",
        run_context=None,
        model=None,
        session_log=None,
    )


class TestApprovalKeyRouting:
    def test_y_p_esc_and_numbers(self):
        assert approval_decision_from_key("y") == "allow"
        assert approval_decision_from_key("Y") == "allow"
        assert approval_decision_from_key("1") == "allow"
        assert approval_decision_from_key("p") == "allow_prefix"
        assert approval_decision_from_key("2") == "allow_prefix"
        assert approval_decision_from_key("esc") == "deny"
        assert approval_decision_from_key("escape") == "deny"
        assert approval_decision_from_key("3") == "deny"
        assert approval_decision_from_key("n") == "deny"
        assert approval_decision_from_key("4") == "deny_prefix"
        assert approval_decision_from_key("x") == "deny_prefix"
        assert approval_decision_from_key("X") == "deny_prefix"
        assert approval_decision_from_key("hello") is None
        assert approval_decision_from_key("") is None

    def test_number_keys_follow_visible_options(self):
        two = ("allow", "deny")
        assert approval_decision_from_key("1", two) == "allow"
        assert approval_decision_from_key("2", two) == "deny"
        assert approval_decision_from_key("3", two) is None
        assert approval_decision_from_key("p", two) is None
        assert approval_decision_from_key("x", two) is None
        assert approval_decision_from_key("n", two) == "deny"
        assert approval_decision_from_key("esc", two) == "deny"

    def test_kind_distinguishes_approval_from_ask(self):
        ask = _InputRequest(prompt="?")
        card = _InputRequest(prompt="Would you like", kind="approval")
        assert not is_approval_request(ask)
        assert is_approval_request(card)
        assert not is_approval_request(None)


class TestApprovalPromptCopy:
    def test_execute_matches_codex_cli(self):
        pending = PendingApproval(
            tool_call_id="t1",
            name="execute",
            arguments={"command": "rm -f /Users/xuming/Documents/temp/1.ini"},
            question="q",
            preview="rm -f /Users/xuming/Documents/temp/1.ini",
        )
        text = format_approval_prompt(pending)
        assert "Would you like to run the following command?" in text
        assert "Environment: local" in text
        assert "$ rm -f /Users/xuming/Documents/temp/1.ini" in text
        assert "1. Yes, proceed (y)" in text
        assert "don't ask again for `rm -f` commands (p)" in text
        assert "No, deny (esc/n)" in text
        assert "don't ask again for `rm -f` commands (x)" in text

    def test_file_and_network_option_two(self):
        file_pending = PendingApproval(
            tool_call_id="t2",
            name="write_file",
            arguments={"file_path": "/tmp/out.txt"},
            question="q",
            preview="/tmp/out.txt",
        )
        net_pending = PendingApproval(
            tool_call_id="t3",
            name="web_search",
            arguments={"queries": ["news"]},
            question="q",
            preview="news",
        )
        file_text = format_approval_prompt(file_pending)
        net_text = format_approval_prompt(net_pending)
        assert "don't ask again for this class of path" in file_text
        assert "don't ask again for this class of network tool" in net_text
        assert "Would you like to allow this file operation?" in file_text
        assert "Would you like to allow this network request?" in net_text

    def test_compound_execute_hides_prefix_option(self):
        pending = PendingApproval(
            tool_call_id="t4",
            name="execute",
            arguments={"command": "echo hi && rm -f /tmp/x"},
            question="q",
            preview="echo hi && rm -f /tmp/x",
            options=("allow", "deny"),
        )
        text = format_approval_prompt(pending)
        assert "1. Yes, proceed (y)" in text
        assert "2. No, deny (esc/n)" in text
        assert "(p)" not in text
        assert "(x)" not in text
        assert "don't ask again" not in text
        assert "3." not in text

    def test_bare_rm_path_is_allow_and_deny_only(self):
        pending = PendingApproval(
            tool_call_id="t5",
            name="execute",
            arguments={
                "command": "rm /Users/xuming/Documents/Codes/SimpleMem/tests/test_vector_store.py",
            },
            question="q",
            preview="rm /Users/xuming/Documents/Codes/SimpleMem/tests/test_vector_store.py",
            options=("allow", "deny"),
        )
        text = format_approval_prompt(pending)
        assert "1. Yes, proceed (y)" in text
        assert "2. No, deny (esc/n)" in text
        assert "don't ask again" not in text

    def test_ask_prompt_lines_keep_deny_on_two_option_card(self):
        req = _InputRequest(
            prompt=format_approval_prompt(
                PendingApproval(
                    tool_call_id="t6",
                    name="execute",
                    arguments={"command": "rm /tmp/x"},
                    question="q",
                    preview="rm /tmp/x",
                    options=("allow", "deny"),
                )
            ),
            kind="approval",
        )
        lines = _ask_prompt_lines(req)
        assert any(line.startswith("2. No, deny") for line in lines)
        blob = "".join(text for _, text in _input_prompt_fragments(req))
        assert blob.startswith("Would you like to run the following command?")
        assert "2. No, deny (esc/n)" in blob

    def test_approval_record_keeps_execute_command(self):
        pending = PendingApproval(
            tool_call_id="t8",
            name="execute",
            arguments={"command": "rm /tmp/x"},
            question="q",
            preview="rm /tmp/x",
            options=("allow", "deny"),
        )
        allowed = format_approval_record(pending, "allow")
        denied = format_approval_record(pending, "deny")
        assert "$ rm /tmp/x" in allowed
        assert "✓ allowed" in allowed
        assert "$ rm /tmp/x" in denied
        assert "✗ denied" in denied
        assert "tell the agent" not in denied

    def test_long_command_is_dumped_in_full(self):
        command = "rm " + "x" * 80
        req = _InputRequest(
            prompt=format_approval_prompt(
                PendingApproval(
                    tool_call_id="t7",
                    name="execute",
                    arguments={"command": command},
                    question="q",
                    preview=command,
                    options=("allow", "deny"),
                )
            ),
            kind="approval",
        )
        assert command in "\n".join(_ask_prompt_lines(req))

    def test_ask_prompt_lines_skip_typed_answer_hint(self):
        req = _InputRequest(
            prompt=format_approval_prompt(
                PendingApproval(
                    tool_call_id="t1",
                    name="execute",
                    arguments={"command": "rm -f x"},
                    question="q",
                    preview="rm -f x",
                )
            ),
            kind="approval",
        )
        lines = _ask_prompt_lines(req)
        blob = "\n".join(lines)
        assert "Would you like to run the following command?" in blob
        assert _ASK_KEY_HINT not in blob


class TestNoninteractiveApprove:
    @pytest.mark.asyncio
    async def test_print_mode_denies_instead_of_parking(self):
        agent = _agent(mode="ask")
        approve = build_noninteractive_approve(agent)
        fc = _fc("execute", {"command": "rm -f /tmp/x"})
        assert await approve(fc) == "deny"

    @pytest.mark.asyncio
    async def test_allow_all_still_allows(self):
        agent = _agent(mode="allow-all")
        approve = build_noninteractive_approve(agent)
        fc = _fc("execute", {"command": "rm -f /tmp/x"})
        assert await approve(fc) == "allow"


class TestInteractiveApprove:
    async def _wait_for_prompt(self, state, timeout=1.0):
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            if is_approval_request(state.input_request):
                return state.input_request
            await asyncio.sleep(0)
        raise AssertionError("approval prompt did not appear")

    @pytest.mark.asyncio
    async def test_y_allows_once(self):
        from agentica.cli.approvals import complete_approval

        state = SessionState()
        state.current_agent = _agent()
        approve = build_interactive_approve(state, {})
        fc = _fc("execute", {"command": "rm -f /tmp/x"}, call_id="a1")
        task = asyncio.create_task(approve(fc))
        req = await self._wait_for_prompt(state)
        assert "Would you like to run the following command?" in req.prompt
        assert complete_approval(state, "allow")
        assert await asyncio.wait_for(task, 2.0) == "allow"

    @pytest.mark.asyncio
    async def test_esc_denies(self):
        from agentica.cli.approvals import complete_approval

        state = SessionState()
        state.current_agent = _agent()
        approve = build_interactive_approve(state, {})
        fc = _fc("execute", {"command": "rm -f /tmp/x"}, call_id="a2")
        task = asyncio.create_task(approve(fc))
        await self._wait_for_prompt(state)
        complete_approval(state, "deny")
        assert await asyncio.wait_for(task, 2.0) == "deny"

    @pytest.mark.asyncio
    async def test_bare_rm_prompt_numbers_deny_second(self):
        from agentica.cli.approvals import complete_approval

        state = SessionState()
        state.current_agent = _agent()
        approve = build_interactive_approve(state, {})
        fc = _fc(
            "execute",
            {"command": "rm /Users/xuming/Documents/Codes/SimpleMem/tests/test_vector_store.py"},
            call_id="a5",
        )
        task = asyncio.create_task(approve(fc))
        req = await self._wait_for_prompt(state)
        assert "1. Yes, proceed (y)" in req.prompt
        assert "2. No, deny (esc/n)" in req.prompt
        assert "(p)" not in req.prompt
        assert req.options == ["allow", "deny"]
        complete_approval(state, "deny")
        assert await asyncio.wait_for(task, 2.0) == "deny"

    @pytest.mark.asyncio
    async def test_decision_prints_command_into_transcript(self, monkeypatch):
        printed: list[str] = []
        monkeypatch.setattr(
            "agentica.cli.interactive.console_io._cprint",
            lambda text: printed.append(text),
        )
        from agentica.cli.approvals import complete_approval

        state = SessionState()
        state.current_agent = _agent()
        approve = build_interactive_approve(state, {})
        fc = _fc("execute", {"command": "rm /tmp/x"}, call_id="a6")
        task = asyncio.create_task(approve(fc))
        await self._wait_for_prompt(state)
        complete_approval(state, "allow")
        assert await asyncio.wait_for(task, 2.0) == "allow"
        blob = "\n".join(printed)
        assert "$ rm /tmp/x" in blob
        assert "✓ allowed" in blob

    @pytest.mark.asyncio
    async def test_ctrl_c_deny_all(self):
        state = SessionState()
        state.current_agent = _agent()
        approve = build_interactive_approve(state, {})
        fc = _fc("execute", {"command": "rm -f /tmp/x"}, call_id="a3")
        task = asyncio.create_task(approve(fc))
        await self._wait_for_prompt(state)
        interrupt_approvals(state)
        assert await asyncio.wait_for(task, 2.0) == "deny"

    @pytest.mark.asyncio
    async def test_parallel_prompts_are_serialized(self):
        from agentica.cli.approvals import complete_approval

        state = SessionState()
        state.current_agent = _agent()
        approve = build_interactive_approve(state, {})
        fc1 = _fc("execute", {"command": "rm -f /tmp/a"}, call_id="p1")
        fc2 = _fc("execute", {"command": "rm -f /tmp/b"}, call_id="p2")
        t1 = asyncio.create_task(approve(fc1))
        t2 = asyncio.create_task(approve(fc2))
        first = await self._wait_for_prompt(state)
        first_id = first.approval_id
        complete_approval(state, "allow")
        second = await self._wait_for_prompt(state)
        assert second.approval_id != first_id
        complete_approval(state, "deny")
        assert await asyncio.wait_for(t1, 2.0) == "allow"
        assert await asyncio.wait_for(t2, 2.0) == "deny"

    @pytest.mark.asyncio
    async def test_cancelled_agent_denies_without_prompt(self):
        state = SessionState()
        state.current_agent = _agent(cancelled=True)
        approve = build_interactive_approve(state, {})
        fc = _fc("execute", {"command": "rm -f /tmp/x"}, call_id="a4")
        assert await approve(fc) == "deny"
        assert state.input_request is None


class TestApprovalTimeout:
    async def _wait_for_prompt(self, state, timeout=1.0):
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            if is_approval_request(state.input_request):
                return state.input_request
            await asyncio.sleep(0)
        raise AssertionError("approval prompt did not appear")

    @pytest.mark.asyncio
    async def test_unanswered_card_is_not_run_and_not_recorded_as_refusal(self, monkeypatch):
        import agentica.cli.approvals as cli_approvals
        from agentica.agent.approvals import DENIED_TOOL_RESULT

        monkeypatch.setattr(cli_approvals, "human_wait_timeout", lambda: 0.05)
        printed = []
        monkeypatch.setattr(cli_approvals, "_print_approval_record",
                            lambda pending, decision, expired_after=None: printed.append(
                                format_approval_record(pending, decision,
                                                       expired_after=expired_after)))
        state = SessionState()
        state.current_agent = _agent()
        approve = build_interactive_approve(state, {})
        fc = _fc("execute", {"command": "git push"}, call_id="t1")
        task = asyncio.create_task(approve(fc))
        req = await self._wait_for_prompt(state)

        assert await asyncio.wait_for(task, 5.0) == "deny"
        assert fc.denied_result is not None
        assert fc.denied_result != DENIED_TOOL_RESULT
        assert "nobody answered the approval prompt within 0s" in fc.denied_result
        assert fc.approval_trace["reason"] == "timeout"
        assert state.approval_grants.deny_command_prefixes == []
        assert state.input_request is None
        assert printed == ["$ git push\n✗ not run (no answer within 0s)"]
        # A keypress arriving after expiry has nothing left to decide.
        assert cli_approvals.submit_approval_decision(req, "allow") is False

    @pytest.mark.asyncio
    async def test_answer_before_deadline_wins_and_expiry_is_a_no_op(self, monkeypatch):
        import agentica.cli.approvals as cli_approvals
        from agentica.cli.approvals import complete_approval

        monkeypatch.setattr(cli_approvals, "human_wait_timeout", lambda: 0.5)
        state = SessionState()
        state.current_agent = _agent()
        approve = build_interactive_approve(state, {})
        fc = _fc("execute", {"command": "rm -f /tmp/x"}, call_id="t2")
        task = asyncio.create_task(approve(fc))
        req = await self._wait_for_prompt(state)
        assert complete_approval(state, "allow")
        assert await asyncio.wait_for(task, 2.0) == "allow"
        await asyncio.sleep(1.2)
        assert state.approval_registry.pop_expired("t2") is None
        assert fc.denied_result is None

    @pytest.mark.asyncio
    async def test_zero_waits_forever(self, monkeypatch):
        import agentica.cli.approvals as cli_approvals

        monkeypatch.setattr(cli_approvals, "human_wait_timeout", lambda: 0)
        state = SessionState()
        state.current_agent = _agent()
        approve = build_interactive_approve(state, {})
        task = asyncio.create_task(approve(_fc("execute", {"command": "rm x"}, call_id="t3")))
        await self._wait_for_prompt(state)
        await asyncio.sleep(0.2)
        assert not task.done()
        interrupt_approvals(state)
        assert await asyncio.wait_for(task, 2.0) == "deny"

    @pytest.mark.asyncio
    async def test_expiry_loses_to_a_keypress_whose_decide_is_still_queued(self, monkeypatch):
        """The TUI thread submits, then posts ``decide`` to the loop. A timer
        firing in between must not override the user's answer."""
        import agentica.cli.approvals as cli_approvals

        monkeypatch.setattr(cli_approvals, "human_wait_timeout", lambda: 0)
        state = SessionState()
        state.current_agent = _agent()
        approve = build_interactive_approve(state, {})
        fc = _fc("execute", {"command": "rm -f /tmp/x"}, call_id="t4")
        task = asyncio.create_task(approve(fc))
        req = await self._wait_for_prompt(state)
        assert req.submit("allow")  # keypress landed; its decide not yet run
        cli_approvals._expire_approval(state, req, 300, {})
        state.approval_registry.decide("t4", "allow")
        assert await asyncio.wait_for(task, 2.0) == "allow"
        assert fc.denied_result is None


class TestIdleCountdown:
    def test_keypress_restarts_the_idle_clock(self):
        req = _InputRequest(prompt="q", idle_timeout=300)
        req.last_activity -= 290
        assert req.idle_remaining() < 11
        req.touch()
        assert req.idle_remaining() > 299

    def test_no_timeout_means_no_clock(self):
        assert _InputRequest(prompt="q").idle_remaining() is None

    def test_countdown_appears_only_in_the_last_stretch(self):
        from agentica.cli.interactive.tui import IDLE_COUNTDOWN_SECONDS

        req = _InputRequest(prompt="Pick one", options=["A", "B"], idle_timeout=300)
        blob = "".join(t for _, t in _input_prompt_fragments(req))
        assert "any key to stay" not in blob
        req.last_activity -= 300 - IDLE_COUNTDOWN_SECONDS + 5
        blob = "".join(t for _, t in _input_prompt_fragments(req))
        assert blob.endswith("continuing without an answer in 15s · any key to stay")

    def test_approval_card_countdown_says_the_call_is_skipped(self):
        req = _InputRequest(prompt="Run it?", kind="approval", idle_timeout=300)
        req.last_activity -= 295
        blob = "".join(t for _, t in _input_prompt_fragments(req))
        assert blob.startswith("Run it?")
        assert blob.endswith("skipping this call in 5s · any key to stay")


def test_a_keypress_in_the_tui_restarts_the_pending_requests_clock(tmp_path):
    from unittest.mock import patch

    from agentica.cli.commands.context import PendingQueue
    from agentica.cli.interactive.tui import _setup_tui

    state = SessionState()
    with patch("agentica.cli.interactive.tui.history_file",
               return_value=str(tmp_path / "history")):
        app = _setup_tui(state, skills_registry=None, tui_state={},
                         pending_queue=PendingQueue(), image_counter_ref=[0])
    req = _InputRequest(prompt="q", idle_timeout=300)
    req.last_activity -= 290
    state.input_request = req
    app.key_processor.before_key_press.fire()
    assert req.idle_remaining() > 299
