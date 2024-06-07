# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This module contains tests for the LLM class.
"""

import os
from unittest.mock import MagicMock, patch

from actionflow.llm import Settings


def test_settings(monkeypatch):
    """
    Tests that key settings defaults are properly set.
    """

    # Test that we're properly setting the default model
    if "DEFAULT_MODEL" in os.environ:
        monkeypatch.delenv("DEFAULT_MODEL")
    settings = Settings()
    assert len(settings.model) > 0


@patch("actionflow.llm.LLM")
def test_respond(mock_llm_class):
    """
    Tests the respond method of the LLM class.

    The LLM class and its respond method are mocked. The test checks if the respond method
    is called correctly and if it returns a non-None value when called with a settings object
    and a list of messages.
    """
    mock_llm_instance = MagicMock()
    mock_llm_instance.respond.return_value = {
        "role": "assistant",
        "content": "Yes, I am here!",
    }
    mock_llm_class.return_value = mock_llm_instance

    llm = mock_llm_class()
    settings = Settings()
    messages = [{"role": "user", "content": "This is a test. Are you there?"}]
    response = llm.respond(settings, messages)
    assert response is not None, "Response is None"
    mock_llm_instance.respond.assert_called_once_with(settings, messages)
