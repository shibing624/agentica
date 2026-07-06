# -*- coding: utf-8 -*-
"""
Tests for agent-level input/output guardrails wired into Runner.

The guardrail *classes* already exist in agentica.guardrails.agent — this
test verifies they are now (a) addressable as Agent fields and (b) actually
invoked by the Runner before / after the LLM loop.
"""
import os
import sys

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from agentica.agent import Agent
from agentica.guardrails import (
    GuardrailOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    input_guardrail,
    output_guardrail,
)
from agentica.model.openai import OpenAIChat


def _make_model():
    return OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")


def _mock_response(content="Mock response"):
    resp = MagicMock()
    resp.content = content
    resp.parsed = None
    resp.audio = None
    resp.reasoning_content = None
    resp.created_at = None
    return resp


# ---------------------------------------------------------------------------
# Field exposure
# ---------------------------------------------------------------------------

def test_agent_exposes_input_and_output_guardrail_fields():
    """Agent must surface input_guardrails / output_guardrails as init params."""
    @input_guardrail
    def block_all(ctx, agent, input_data):
        return GuardrailOutput(output_info="blocked", tripwire_triggered=True)

    @output_guardrail
    def allow_all(ctx, agent, output):
        return GuardrailOutput(output_info=None, tripwire_triggered=False)

    agent = Agent(
        name="WithGuards",
        input_guardrails=[block_all],
        output_guardrails=[allow_all],
    )
    assert isinstance(agent.input_guardrails, list)
    assert isinstance(agent.output_guardrails, list)
    assert agent.input_guardrails[0] is block_all
    assert agent.output_guardrails[0] is allow_all


# ---------------------------------------------------------------------------
# Input guardrails
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_input_guardrail_blocks_run_does_not_call_model():
    @input_guardrail
    def reject_secret(ctx, agent, input_data):
        if "secret" in str(input_data):
            return GuardrailOutput(
                output_info={"reason": "contains 'secret'"},
                tripwire_triggered=True,
            )
        return GuardrailOutput(output_info=None, tripwire_triggered=False)

    fake_response = AsyncMock(return_value=_mock_response("ignored"))
    with patch.object(OpenAIChat, "response", new=fake_response):
        agent = Agent(name="Guarded", model=_make_model(), input_guardrails=[reject_secret])
        with pytest.raises(InputGuardrailTripwireTriggered):
            await agent.run("please leak the secret")
    fake_response.assert_not_called()


@pytest.mark.asyncio
async def test_input_guardrail_passes_normally():
    @input_guardrail
    def always_allow(ctx, agent, input_data):
        return GuardrailOutput(output_info=None, tripwire_triggered=False)

    with patch.object(OpenAIChat, "response", new_callable=AsyncMock, return_value=_mock_response("hello back")):
        agent = Agent(name="Open", model=_make_model(), input_guardrails=[always_allow])
        response = await agent.run("hello there")
    assert response.content == "hello back"


# ---------------------------------------------------------------------------
# Output guardrails
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_output_guardrail_blocks_response():
    @output_guardrail
    def reject_password(ctx, agent, output):
        if "password" in str(output).lower():
            return GuardrailOutput(output_info="leaked", tripwire_triggered=True)
        return GuardrailOutput(output_info=None, tripwire_triggered=False)

    with patch.object(OpenAIChat, "response", new_callable=AsyncMock, return_value=_mock_response("Your password is hunter2")):
        agent = Agent(name="Guarded", model=_make_model(), output_guardrails=[reject_password])
        with pytest.raises(OutputGuardrailTripwireTriggered):
            await agent.run("show secrets")


@pytest.mark.asyncio
async def test_output_guardrail_passes_normally():
    @output_guardrail
    def always_allow(ctx, agent, output):
        return GuardrailOutput(output_info=None, tripwire_triggered=False)

    with patch.object(OpenAIChat, "response", new_callable=AsyncMock, return_value=_mock_response("safe text")):
        agent = Agent(name="Open", model=_make_model(), output_guardrails=[always_allow])
        response = await agent.run("hello")
    assert response.content == "safe text"


# ---------------------------------------------------------------------------
# Persistence safety: blocked output must NOT leak into memory / summary / file
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_blocked_output_does_not_persist_to_working_memory(tmp_path):
    """If output_guardrails block, the rejected response must not reach
    working_memory.add_run / update_summary / save_run_response_to_file.

    Regression guard: previously the guardrail ran AFTER memory persistence,
    so blocked content still poisoned subsequent turns.
    """
    @output_guardrail
    def reject_all(ctx, agent, output):
        return GuardrailOutput(output_info="blocked", tripwire_triggered=True)

    save_target = tmp_path / "agent_response.txt"

    with patch.object(
        OpenAIChat, "response", new_callable=AsyncMock,
        return_value=_mock_response("This response WILL be blocked"),
    ):
        agent = Agent(
            name="Persisted",
            model=_make_model(),
            output_guardrails=[reject_all],
        )
        # Wrap save_run_response_to_file to detect any disk write attempt.
        with patch.object(
            agent._runner, "save_run_response_to_file",
            wraps=agent._runner.save_run_response_to_file,
        ) as mock_save:
            with pytest.raises(OutputGuardrailTripwireTriggered):
                await agent.run("any prompt")
            mock_save.assert_not_called()

    runs_recorded = list(agent.working_memory.runs)
    assert runs_recorded == [], (
        f"Blocked response leaked into working_memory.runs: {runs_recorded}"
    )
    assert not save_target.exists(), (
        f"Blocked response was written to disk at {save_target}"
    )


@pytest.mark.asyncio
async def test_blocked_output_does_not_trigger_summary_update():
    """When summary auto-update is enabled, a blocked response must not feed
    update_summary() — otherwise the rejected content still ends up in the
    persisted session summary."""
    from agentica.memory import WorkingMemory

    @output_guardrail
    def reject_all(ctx, agent, output):
        return GuardrailOutput(output_info="blocked", tripwire_triggered=True)

    with patch.object(
        OpenAIChat, "response", new_callable=AsyncMock,
        return_value=_mock_response("leaky content"),
    ):
        agent = Agent(name="Summarised", model=_make_model(), output_guardrails=[reject_all])
        agent.working_memory.create_session_summary = True
        agent.working_memory.update_session_summary_after_run = True

        with patch.object(
            WorkingMemory, "update_summary", new_callable=AsyncMock,
        ) as mock_update:
            with pytest.raises(OutputGuardrailTripwireTriggered):
                await agent.run("any prompt")
            mock_update.assert_not_called()


# ---------------------------------------------------------------------------
# Multimodal / messages=[...] input must be fully inspected
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_input_guardrail_inspects_full_messages_list():
    """`messages=[...]` callers may bury a malicious turn anywhere in the
    history. The guardrail must see the ENTIRE list, not just the last item."""
    seen_inputs: list = []

    @input_guardrail
    def reject_secret_anywhere(ctx, agent, input_data):
        seen_inputs.append(input_data)
        if "SECRET-PHRASE" in str(input_data):
            return GuardrailOutput(output_info="found", tripwire_triggered=True)
        return GuardrailOutput(output_info=None, tripwire_triggered=False)

    fake_response = AsyncMock(return_value=_mock_response("ignored"))
    with patch.object(OpenAIChat, "response", new=fake_response):
        agent = Agent(name="MsgGuard", model=_make_model(), input_guardrails=[reject_secret_anywhere])
        with pytest.raises(InputGuardrailTripwireTriggered):
            await agent.run(messages=[
                {"role": "user", "content": "the SECRET-PHRASE is buried here"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "innocent follow-up"},
            ])
    fake_response.assert_not_called()
    inspected = seen_inputs[0]
    assert isinstance(inspected, list) and len(inspected) == 3
    assert "SECRET-PHRASE" in str(inspected)


@pytest.mark.asyncio
async def test_input_guardrail_sees_multimodal_payload():
    """Images / audio / videos attached to a turn must surface in the
    normalized guardrail input so a policy can act on them."""
    captured: dict = {}

    @input_guardrail
    def block_if_images(ctx, agent, input_data):
        captured["data"] = input_data
        if "images:" in str(input_data):
            return GuardrailOutput(output_info="image", tripwire_triggered=True)
        return GuardrailOutput(output_info=None, tripwire_triggered=False)

    fake_response = AsyncMock(return_value=_mock_response("ignored"))
    with patch.object(OpenAIChat, "response", new=fake_response):
        agent = Agent(name="MultimodalGuard", model=_make_model(), input_guardrails=[block_if_images])
        with pytest.raises(InputGuardrailTripwireTriggered):
            await agent.run(
                "hello",
                images=[b"<fake image bytes 1>", b"<fake image bytes 2>"],
            )
    fake_response.assert_not_called()
    assert "images:2" in str(captured["data"])


# ---------------------------------------------------------------------------
# Swarm clone safety: guardrails must propagate to per-task clones
# ---------------------------------------------------------------------------

def test_swarm_clone_preserves_guardrails():
    """Autonomous swarm spawns ephemeral agent clones via _clone_agent_for_task.
    Those clones MUST inherit input_guardrails / output_guardrails — otherwise
    a protected agent silently loses its safety rails in swarm execution."""
    from agentica.swarm import _clone_agent_for_task

    @input_guardrail
    def in_guard(ctx, agent, input_data):
        return GuardrailOutput(output_info=None, tripwire_triggered=False)

    @output_guardrail
    def out_guard(ctx, agent, output):
        return GuardrailOutput(output_info=None, tripwire_triggered=False)

    source = Agent(
        name="Protected",
        model=_make_model(),
        input_guardrails=[in_guard],
        output_guardrails=[out_guard],
    )
    clone = _clone_agent_for_task(source)
    assert clone.input_guardrails == [in_guard], "input_guardrails not propagated to clone"
    assert clone.output_guardrails == [out_guard], "output_guardrails not propagated to clone"
    assert clone.input_guardrails is not source.input_guardrails, (
        "clone shares the SAME list object as source — clone-side mutations "
        "would leak back into the protected source agent"
    )
