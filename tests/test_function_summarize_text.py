# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import shutil
from unittest.mock import patch

import pytest

from actionflow.output import Output
from actionflow.tools.summarize_text import SummarizeText


@pytest.fixture
def output():
    flow = "test_summarize_text"
    output = Output(flow)
    yield output
    shutil.rmtree(output.data_dir)


def test_get_definition(output):
    """
    Tests the get_definition method of the SummarizeText class. It checks that the definition of the function is correct.
    """
    summarizer = SummarizeText(output)
    definition = summarizer.get_definition()
    assert definition["name"] == "summarize_text"
    assert "text_to_summarize" in definition["parameters"]["properties"]


def test_truncate_text(output):
    """
    Tests the _truncate_text method of the SummarizeText class. It checks that the text is truncated correctly.
    """
    summarizer = SummarizeText(output)
    text = "a" * (summarizer.max_input_chars + 10)
    truncated_text = summarizer._truncate_text(text)
    assert len(truncated_text) == summarizer.max_input_chars


def test_prepare_messages(output):
    """
    Tests the _prepare_messages method of the SummarizeText class. It checks that the messages are prepared correctly.
    """
    summarizer = SummarizeText(output)
    messages = summarizer._prepare_messages("text", "instructions")
    assert messages[0]["content"] == "You are an AI summarizer. instructions"
    assert messages[1]["content"] == "Text to summarize: text"


@patch("actionflow.tools.summarize_text.LLM")
def test_execute(MockLLM, output):
    """
    Tests the execute method of the SummarizeText class.
    """
    mock_summary = "This is the summary."
    MockLLM.return_value.respond.return_value.content = mock_summary
    summarizer = SummarizeText(output)

    summary = summarizer.execute("Text to summarize.", "Instruction summary.")
    assert summary == mock_summary
