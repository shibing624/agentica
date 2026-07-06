# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This module contains tests for the LLM class.
"""

import os
import asyncio
from unittest.mock import AsyncMock, patch
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import Message


@pytest.mark.asyncio
async def test_respond():
    """Tests the async response method of the LLM class."""
    mock_llm_instance = AsyncMock()
    mock_llm_instance.response.return_value = "Yes, I am here!"

    messages = [Message(role="user", content="This is a test. Are you there?")]
    res = await mock_llm_instance.response(messages)
    print('res:', res)
    assert res == "Yes, I am here!", "Response is not as expected"
    assert res is not None, "Response is None"
    mock_llm_instance.response.assert_called_once_with(messages)
