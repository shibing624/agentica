# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This module contains tests for the LLM class.
"""

import os
from unittest.mock import MagicMock, patch

from agentica.message import Message


@patch("agentica.llm.openai_llm.OpenAILLM")
def test_respond(mock_llm_class):
    """
    Tests the response method of the LLM class.

    The LLM class and its response method are mocked.
    """
    mock_llm_instance = MagicMock()
    mock_llm_instance.response.return_value = "Yes, I am here!"
    mock_llm_class.return_value = mock_llm_instance

    llm = mock_llm_class()
    user_message = Message(role="user", content="This is a test. Are you there?")
    messages = [user_message]
    res = llm.response(messages)
    print('res:', res)
    assert res == "Yes, I am here!", "Response is not as expected"
    assert res is not None, "Response is None"
    mock_llm_instance.response.assert_called_once_with(messages)
