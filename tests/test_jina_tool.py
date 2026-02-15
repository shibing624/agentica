# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import asyncio
import json
from unittest.mock import patch, MagicMock

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.tools.jina_tool import JinaTool


@pytest.fixture
def jina_tool():
    """Fixture to create a JinaTool instance for testing."""
    return JinaTool(api_key="test_api_key", work_dir="./tmp")


def test_read_url_error():
    async def mock_post(self, url, **kwargs):
        raise Exception("Test error")

    with patch('httpx.AsyncClient.post', new=mock_post):
        tools = JinaTool(api_key="test_key", work_dir="./tmp")
        result = asyncio.run(tools.jina_url_reader("https://example.com"))

        result_dict = json.loads(result)
        assert result_dict["url"] == "https://example.com"
        assert result_dict["content"] == ""
        assert "Error reading URL" in result_dict["error"]


def test_search_query_error():
    async def mock_get(self, url, **kwargs):
        raise Exception("Test error")

    with patch('httpx.AsyncClient.get', new=mock_get):
        tools = JinaTool(api_key="test_key", work_dir="./tmp", jina_search=True)
        result = asyncio.run(tools.jina_search("test query"))

        result_dict = json.loads(result)
        assert "test query" in result_dict
        assert "Error performing search: Test error" in result_dict["test query"]


def test_jina_url_reader():
    """Test the jina_url_reader method."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "This is a test content from the URL."
    mock_response.raise_for_status = MagicMock()

    async def mock_post(self, url, **kwargs):
        return mock_response

    with patch('httpx.AsyncClient.post', new=mock_post):
        jina_tool = JinaTool(api_key="test_key", work_dir="./tmp")
        url = "https://abc.com/test-url"
        result = asyncio.run(jina_tool.jina_url_reader(url))

        result_dict = json.loads(result)
        assert result_dict["url"] == url
        assert "This is a test content from the URL." in result_dict["content"]
        assert "error" not in result_dict


def test_jina_search():
    """Test the jina_search method."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "Search results for the query."
    mock_response.raise_for_status = MagicMock()

    async def mock_get(self, url, **kwargs):
        return mock_response

    with patch('httpx.AsyncClient.get', new=mock_get):
        query = "苹果的最新产品是啥？"
        jina_tool = JinaTool(api_key="test_key", work_dir="./tmp", jina_search=True)
        result = asyncio.run(jina_tool.jina_search(query))

        result_dict = json.loads(result)
        assert query in result_dict
        assert result_dict[query] == "Search results for the query."


@patch('agentica.tools.jina_tool.OpenAIChat')
def test_jina_url_reader_by_goal(mock_openai_chat):
    """Test the jina_url_reader_by_goal method."""
    # Mock httpx.AsyncClient.post response for jina_url_reader
    mock_http_response = MagicMock()
    mock_http_response.status_code = 200
    mock_http_response.text = "This is test webpage content about AI."
    mock_http_response.raise_for_status = MagicMock()

    async def mock_post(self, url, **kwargs):
        return mock_http_response

    # Mock OpenAIChat and its client
    mock_client = MagicMock()
    mock_chat_response = MagicMock()
    mock_chat_response.choices = [MagicMock()]
    mock_chat_response.choices[0].message.content = '{"rational": "test", "evidence": "AI evidence", "summary": "AI summary"}'
    mock_client.chat.completions.create.return_value = mock_chat_response
    mock_openai_chat_instance = MagicMock()
    mock_openai_chat_instance.get_client.return_value = mock_client
    mock_openai_chat.return_value = mock_openai_chat_instance

    with patch('httpx.AsyncClient.post', new=mock_post):
        jina_tool = JinaTool(api_key="test_key", work_dir="./tmp")
        url = "https://example.com/ai-article"
        goal = "Learn about AI"
        result = asyncio.run(jina_tool.jina_url_reader_by_goal(url, goal))

        result_dict = json.loads(result)
        assert result_dict["goal"] == goal
        assert url in result_dict["urls"]
        assert len(result_dict["results"]) == 1
        assert result_dict["results"][0]["url"] == url
        assert result_dict["results"][0]["evidence"] == "AI evidence"
        assert result_dict["results"][0]["summary"] == "AI summary"
