# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for SessionMixin async persistence.
"""
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agentica.agent import Agent
from agentica.model.openai import OpenAIChat


# ===========================================================================
# TestSessionPersistence
# ===========================================================================


class TestSessionPersistence:
    """Tests for SessionMixin async storage methods."""

    @pytest.mark.asyncio
    async def test_read_from_storage_no_db_noop(self):
        """No db configured — read_from_storage should not raise."""
        agent = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini"))
        assert agent.db is None
        result = await agent.read_from_storage()
        assert result is None

    @pytest.mark.asyncio
    async def test_write_to_storage_no_db_noop(self):
        """No db configured — write_to_storage should not raise."""
        agent = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini"))
        result = await agent.write_to_storage()
        assert result is None

    @pytest.mark.asyncio
    async def test_read_from_storage_is_async(self):
        assert asyncio.iscoroutinefunction(Agent.read_from_storage)

    @pytest.mark.asyncio
    async def test_write_to_storage_is_async(self):
        assert asyncio.iscoroutinefunction(Agent.write_to_storage)

    @pytest.mark.asyncio
    async def test_load_session_is_async(self):
        assert asyncio.iscoroutinefunction(Agent.load_session)

    @pytest.mark.asyncio
    async def test_generate_session_name_is_async(self):
        assert asyncio.iscoroutinefunction(Agent.generate_session_name)


# ===========================================================================
# TestSessionState
# ===========================================================================


class TestSessionState:
    """Tests for session_state across runs."""

    @pytest.mark.asyncio
    async def test_session_state_accessible(self):
        agent = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini"))
        assert agent.session_state is not None or agent.session_state is None
        # Just ensure no attribute error

    @pytest.mark.asyncio
    async def test_session_id_generation(self):
        agent = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini"))
        assert agent.session_id is not None
        assert len(agent.session_id) > 0

    @pytest.mark.asyncio
    async def test_custom_session_id(self):
        agent = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini"), session_id="custom-123")
        assert agent.session_id == "custom-123"

    @pytest.mark.asyncio
    async def test_new_session_clears_memory(self):
        agent = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini"))
        old_session = agent.session_id
        agent.new_session()
        assert agent.session_id != old_session
        assert len(agent.memory.messages) == 0

    @pytest.mark.asyncio
    async def test_reset_clears_state(self):
        agent = Agent(name="A", model=OpenAIChat(model="gpt-4o-mini"))
        agent.memory.add_message(MagicMock(role="user"))
        agent.reset()
        assert len(agent.memory.messages) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
