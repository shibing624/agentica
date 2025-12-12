# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import json
from unittest.mock import patch, Mock, MagicMock

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.tools.jina_tool import JinaTool


@pytest.fixture
def jina_tool():
    """Fixture to create a JinaTool instance for testing."""
    return JinaTool(api_key="test_api_key", work_dir="./test_work_dir")


@patch("agentica.tools.jina_tool.requests.post")
def test_read_url_error(mock_post):
    mock_post.side_effect = Exception("Test error")

    tools = JinaTool(api_key="test_key")
    result = tools.jina_url_reader("https://example.com")

    # jina_url_reader returns JSON format
    result_dict = json.loads(result)
    assert result_dict["url"] == "https://example.com"
    assert result_dict["content"] == ""
    assert "Error reading URL" in result_dict["error"]


@patch("agentica.tools.jina_tool.requests.get")
def test_search_query_error(mock_get):
    mock_get.side_effect = Exception("Test error")

    tools = JinaTool(api_key="test_key")
    result = tools.jina_search("test query")

    # jina_search returns JSON format: {"query": "result"}
    result_dict = json.loads(result)
    assert "test query" in result_dict
    assert "Error performing search: Test error" in result_dict["test query"]


@patch('agentica.tools.jina_tool.requests.post')
def test_jina_url_reader(mock_post):
    """Test the jina_url_reader method."""
    # Mock response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = "This is a test content from the URL."
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response

    jina_tool = JinaTool(api_key="test_key")
    url = "https://abc.com/test-url"
    result = jina_tool.jina_url_reader(url)

    # jina_url_reader returns JSON format
    result_dict = json.loads(result)
    assert result_dict["url"] == url
    assert "This is a test content from the URL." in result_dict["content"]
    assert "error" not in result_dict


@patch('agentica.tools.jina_tool.requests.get')
def test_jina_search(mock_get):
    """Test the jina_search method."""
    # Mock response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = "Search results for the query."
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    query = "苹果的最新产品是啥？"
    jina_tool = JinaTool(api_key="test_key")
    result = jina_tool.jina_search(query)

    # jina_search returns JSON format: {"query": "result"}
    result_dict = json.loads(result)
    assert query in result_dict
    assert result_dict[query] == "Search results for the query."


@patch('agentica.tools.jina_tool.OpenAIChat')
@patch('agentica.tools.jina_tool.requests.post')
def test_jina_url_reader_by_goal(mock_post, mock_openai_chat):
    """Test the jina_url_reader_by_goal method."""
    # Mock requests.post response for jina_url_reader
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = "This is test webpage content about AI."
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response

    # Mock OpenAIChat and its client
    mock_client = MagicMock()
    mock_chat_response = MagicMock()
    mock_chat_response.choices = [MagicMock()]
    mock_chat_response.choices[0].message.content = '{"rational": "test", "evidence": "AI evidence", "summary": "AI summary"}'
    mock_client.chat.completions.create.return_value = mock_chat_response
    mock_openai_chat_instance = MagicMock()
    mock_openai_chat_instance.get_client.return_value = mock_client
    mock_openai_chat.return_value = mock_openai_chat_instance

    jina_tool = JinaTool(api_key="test_key")
    url = "https://example.com/ai-article"
    goal = "Learn about AI"
    result = jina_tool.jina_url_reader_by_goal(url, goal)

    # jina_url_reader_by_goal returns JSON format
    result_dict = json.loads(result)
    assert result_dict["goal"] == goal
    assert url in result_dict["urls"]
    assert len(result_dict["results"]) == 1
    assert result_dict["results"][0]["url"] == url
    assert result_dict["results"][0]["evidence"] == "AI evidence"
    assert result_dict["results"][0]["summary"] == "AI summary"
