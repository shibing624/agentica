# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for Agent-as-Session pattern (V2).

V2 changes:
- SessionMixin removed from Agent inheritance
- Agent itself IS the session via WorkingMemory
- session_id, db, read_from_storage etc. moved to SessionManager (external)
- Core session capability: memory.runs, add_history_to_messages, history_window
"""
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock
from agentica.agent import Agent
from agentica.model.openai import OpenAIChat
from agentica.memory import WorkingMemory, AgentRun
from agentica.model.message import Message
from agentica.run_response import RunResponse


# ===========================================================================
# TestSessionPersistence -> V2: WorkingMemory run tracking
# ===========================================================================


class TestSessionPersistence:
    """V2: Agent session persistence is handled by WorkingMemory.runs.

    Each agent.run() call records an AgentRun to memory.runs.
    No external db required for basic session tracking.
    """

    def test_memory_initialized_by_default(self):
        """Agent should have WorkingMemory by default, no db needed."""
        agent = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini", api_key="fake_openai_key"))
        assert isinstance(agent.working_memory, WorkingMemory)
        assert len(agent.working_memory.runs) == 0
        assert len(agent.working_memory.messages) == 0

    def test_memory_add_run(self):
        """WorkingMemory.add_run stores AgentRun entries."""
        agent = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini", api_key="fake_openai_key"))
        run = AgentRun(
            message=Message(role="user", content="hello"),
            response=RunResponse(content="hi"),
        )
        agent.working_memory.add_run(run)
        assert len(agent.working_memory.runs) == 1
        assert agent.working_memory.runs[0].message.content == "hello"

    def test_memory_add_messages(self):
        """WorkingMemory.add_messages stores message history."""
        agent = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini", api_key="fake_openai_key"))
        agent.working_memory.add_message(Message(role="user", content="test"))
        agent.working_memory.add_message(Message(role="assistant", content="reply"))
        assert len(agent.working_memory.messages) == 2

    def test_get_messages_from_runs(self):
        """get_messages_from_last_n_runs retrieves history from recorded runs."""
        memory = WorkingMemory()
        # Simulate 3 runs with response messages
        for i in range(3):
            run = AgentRun(
                response=RunResponse(
                    content=f"response_{i}",
                    messages=[
                        Message(role="user", content=f"q{i}"),
                        Message(role="assistant", content=f"a{i}"),
                    ],
                )
            )
            memory.add_run(run)

        # Get last 2 runs
        msgs = memory.get_messages_from_last_n_runs(last_n=2)
        contents = [m.content for m in msgs if m.role == "user"]
        assert "q1" in contents
        assert "q2" in contents
        assert "q0" not in contents

    def test_get_messages_from_all_runs(self):
        """get_messages_from_last_n_runs(last_n=None) returns all history."""
        memory = WorkingMemory()
        for i in range(5):
            run = AgentRun(
                response=RunResponse(
                    content=f"r{i}",
                    messages=[
                        Message(role="user", content=f"q{i}"),
                        Message(role="assistant", content=f"a{i}"),
                    ],
                )
            )
            memory.add_run(run)

        msgs = memory.get_messages_from_last_n_runs(last_n=None)
        user_msgs = [m.content for m in msgs if m.role == "user"]
        assert len(user_msgs) == 5

    def test_add_history_to_messages_flag(self):
        """Agent respects add_history_to_messages configuration."""
        agent_no_hist = Agent(
            name="A",
            model=OpenAIChat(model="gpt-4o-mini", api_key="fake_openai_key"),
            add_history_to_messages=False,
        )
        assert agent_no_hist.add_history_to_messages is False

        agent_with_hist = Agent(
            name="B",
            model=OpenAIChat(model="gpt-4o-mini", api_key="fake_openai_key"),
            add_history_to_messages=True,
            history_window=5,
        )
        assert agent_with_hist.add_history_to_messages is True
        assert agent_with_hist.history_window == 5


# ===========================================================================
# TestSessionState -> V2: Agent memory state management
# ===========================================================================


class TestSessionState:
    """V2: Session state is managed through WorkingMemory.

    - memory.runs tracks all run history
    - memory.messages tracks all messages
    - No session_id on Agent; use Agent instance identity
    """

    def test_memory_state_accessible(self):
        """Agent memory should be accessible and properly typed."""
        agent = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini", api_key="fake_openai_key"))
        assert agent.working_memory is not None
        assert isinstance(agent.working_memory, WorkingMemory)
        assert isinstance(agent.working_memory.runs, list)
        assert isinstance(agent.working_memory.messages, list)

    def test_custom_memory(self):
        """Agent accepts custom WorkingMemory instance via direct assignment."""
        custom_memory = WorkingMemory()
        custom_memory.add_run(AgentRun(
            response=RunResponse(
                content="pre-loaded",
                messages=[Message(role="assistant", content="pre-loaded")],
            )
        ))

        agent = Agent(
            name="A",
            model=OpenAIChat(model="gpt-4o-mini", api_key="fake_openai_key"),
        )
        agent.working_memory = custom_memory
        assert agent.working_memory is custom_memory
        assert len(agent.working_memory.runs) == 1

    def test_shared_memory_between_agents(self):
        """Two agents can share the same WorkingMemory (session handoff)."""
        shared = WorkingMemory()
        agent_a = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini", api_key="fake_openai_key"))
        agent_a.working_memory = shared
        agent_b = Agent(name="B", model=OpenAIChat(model="gpt-4o-mini", api_key="fake_openai_key"))
        agent_b.working_memory = shared

        # Agent A records a run
        shared.add_run(AgentRun(
            response=RunResponse(
                content="from_a",
                messages=[
                    Message(role="user", content="hello from a"),
                    Message(role="assistant", content="from_a"),
                ],
            )
        ))

        # Agent B sees it
        msgs = agent_b.working_memory.get_messages_from_last_n_runs(last_n=1)
        assert any("from a" in str(m.content) for m in msgs)

    def test_clear_memory(self):
        """Clearing memory resets session state."""
        agent = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini", api_key="fake_openai_key"))
        agent.working_memory.add_message(Message(role="user", content="test"))
        agent.working_memory.add_run(AgentRun(response=RunResponse(content="r")))
        assert len(agent.working_memory.messages) == 1
        assert len(agent.working_memory.runs) == 1

        # Reset by creating new memory
        agent.working_memory = WorkingMemory()
        assert len(agent.working_memory.messages) == 0
        assert len(agent.working_memory.runs) == 0

    def test_independent_sessions_via_separate_agents(self):
        """Each Agent instance has independent session state."""
        agent_a = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini", api_key="fake_openai_key"))
        agent_b = Agent(name="B", model=OpenAIChat(model="gpt-4o-mini", api_key="fake_openai_key"))

        agent_a.working_memory.add_message(Message(role="user", content="a_msg"))
        agent_b.working_memory.add_message(Message(role="user", content="b_msg"))

        assert len(agent_a.working_memory.messages) == 1
        assert len(agent_b.working_memory.messages) == 1
        assert agent_a.working_memory.messages[0].content == "a_msg"
        assert agent_b.working_memory.messages[0].content == "b_msg"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
