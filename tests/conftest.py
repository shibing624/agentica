# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Shared fixtures for all test modules.
"""
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import AsyncMock, MagicMock
from agentica.agent import Agent
from agentica.model.base import Model
from agentica.model.response import ModelResponse


# ---------------------------------------------------------------------------
# Simple tool helpers (usable as Agent tools)
# ---------------------------------------------------------------------------

def sync_add(a: int, b: int) -> str:
    """Add two numbers synchronously."""
    return str(a + b)


def sync_multiply(a: int, b: int) -> str:
    """Multiply two numbers synchronously."""
    return str(a * b)


async def async_search(query: str) -> str:
    """Search for something asynchronously."""
    await asyncio.sleep(0.01)
    return f"Result for: {query}"


async def async_slow_tool(seconds: float = 0.1) -> str:
    """A slow async tool for timing tests."""
    await asyncio.sleep(seconds)
    return f"Completed after {seconds}s"


def failing_tool() -> str:
    """A tool that always raises."""
    raise ValueError("Tool intentionally failed")


async def async_failing_tool() -> str:
    """An async tool that always raises."""
    raise RuntimeError("Async tool intentionally failed")


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model_response():
    """A simple ModelResponse with text content."""
    return ModelResponse(content="Hello from mock model!")


@pytest.fixture
def mock_model(mock_model_response):
    """A mock Model that returns a simple text response via async response()."""
    model = MagicMock(spec=Model)
    model.id = "mock-model"
    model.name = "MockModel"
    model.provider = "mock"
    model.tools = None
    model.functions = {}
    model.function_call_stack = None
    model.run_tools = True
    model.tool_call_limit = None
    model.system_prompt = None
    model.instructions = None
    model.structured_outputs = None
    model.supports_structured_outputs = False
    model.context_window = 128000
    model.max_output_tokens = None
    model.metrics = {}
    model.response_format = None
    model.session_id = None
    model.user_id = None
    model.agent_name = None
    model.tool_choice = None

    model.response = AsyncMock(return_value=mock_model_response)
    model.response_stream = AsyncMock(return_value=_empty_async_iter())
    model.get_tools_for_api = MagicMock(return_value=None)
    model.add_tool = MagicMock()
    model.sanitize_messages = Model.sanitize_messages
    model.to_dict = MagicMock(return_value={"id": "mock-model"})
    model.deactivate_function_calls = MagicMock()
    return model


@pytest.fixture
def simple_agent(mock_model):
    """An Agent with a mock model and no tools (for basic run tests)."""
    agent = Agent(name="TestAgent", model=mock_model)
    return agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _empty_async_iter():
    """An empty async iterator."""
    return
    yield  # noqa – makes this an async generator


async def async_iter_from_list(items):
    """Turn a list into an async iterator."""
    for item in items:
        yield item
