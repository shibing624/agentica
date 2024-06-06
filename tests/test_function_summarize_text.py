# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import shutil
from unittest.mock import patch

import pytest

from actionflow.tools.summarize_text import SummarizeText
from actionflow.output import Output


@pytest.fixture
def output():
    flow = "test_summarize_text"
    output = Output(flow)
    yield output
    shutil.rmtree(output.output_dir)


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
    text = "a" * (summarizer.max_tokens * summarizer.chars_per_token + 10)
    truncated_text = summarizer._truncate_text(text)
    assert len(truncated_text) == summarizer.max_tokens * summarizer.chars_per_token


def test_prepare_messages(output):
    """
    Tests the _prepare_messages method of the SummarizeText class. It checks that the messages are prepared correctly.
    """
    summarizer = SummarizeText(output)
    messages = summarizer._prepare_messages("text", "instructions")
    assert messages[0]["content"] == "You are an AI summarizer. instructions"
    assert messages[1]["content"] == "Text to summarize: text"


def test_select_model_base(output):
    """
    Tests the _select_model method of the SummarizeText class. It checks that the correct model is selected for 4000 tokens or less.
    """
    summarizer = SummarizeText(output)
    # Constructing a message size that fits the base model
    messages = [{"content": "a" * (1000 * summarizer.chars_per_token)}]
    model, _ = summarizer._select_model(messages)
    assert model == "gpt-3.5-turbo"


def test_select_model_16k(output):
    """
    Tests the _select_model method of the SummarizeText class. It checks that the correct model is selected for more than 4000 tokens.
    """
    summarizer = SummarizeText(output)
    # Constructing a message size that requires the larger 16k model
    messages = [{"content": "a" * (5000 * summarizer.chars_per_token)}]
    model, _ = summarizer._select_model(messages)
    assert model == "gpt-3.5-turbo-16k"


def test_calculate_tokens(output):
    """
    Tests the _calculate_tokens method of the SummarizeText class. It checks that the number of tokens is calculated correctly.
    """
    summarizer = SummarizeText(output)
    tokens = summarizer._calculate_tokens([{"content": "a" * 1000}])
    assert tokens == 254


@patch("actionflow.functions.summarize_text.LLM")
def test_execute(MockLLM, output):
    """
    Tests the execute method of the SummarizeText class.
    """
    mock_summary = "This is the summary."
    MockLLM.return_value.respond.return_value.content = mock_summary
    summarizer = SummarizeText(output)

    summary = summarizer.execute("Text to summarize.", "Instruction summary.")
    assert summary == mock_summary
