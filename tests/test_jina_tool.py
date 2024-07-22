# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
from unittest.mock import patch, Mock

import pytest

from agentica.tools.jina import JinaTool


@pytest.fixture
def jina_tool():
    """Fixture to create a JinaTool instance for testing."""
    return JinaTool(api_key="test_api_key", work_dir="./test_work_dir")


@patch("requests.get")
def test_read_url_error(mock_get):
    mock_get.side_effect = Exception("Test error")

    tools = JinaTool(api_key="test_key")
    result = tools.jina_url_reader("https://example.com")

    assert result == "Error reading URL: Test error"


@patch("requests.get")
def test_search_query_error(mock_get):
    mock_get.side_effect = Exception("Test error")

    tools = JinaTool(api_key="test_key")
    result = tools.jina_search("test query")

    assert result == "Error performing search: Test error"


@patch('requests.get')
def test_jina_url_reader(mock_get):
    """Test the jina_url_reader method."""
    # Mock response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = "This is a test content from the URL."
    mock_get.return_value = mock_response

    jina_tool = JinaTool(api_key="test_key")
    url = "https://example.com/test-url"
    result = jina_tool.jina_url_reader(url)

    # Assertions
    assert result == "This is a test content from the URL."
    mock_get.assert_called_once_with('https://r.jina.ai/https://example.com/test-url', headers=jina_tool._get_headers())
    saved_file_path = os.path.join(jina_tool.work_dir, jina_tool._generate_file_name_from_url(url))
    with open(saved_file_path, 'r', encoding='utf-8') as f:
        saved_content = f.read()
    assert saved_content == mock_response.text

    # Clean up
    os.remove(saved_file_path)


@patch('requests.get')
def test_jina_search(mock_get):
    """Test the jina_search method."""
    # Mock response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = "Search results for the query."
    mock_get.return_value = mock_response

    query = "苹果的最新产品是啥？"
    jina_tool = JinaTool(api_key="test_key")
    result = jina_tool.jina_search(query)

    # Assertions
    assert result == "Search results for the query."
    mock_get.assert_called_once_with('https://s.jina.ai/苹果的最新产品是啥？', headers=jina_tool._get_headers())
    saved_file_path = os.path.join(jina_tool.work_dir, jina_tool._generate_file_name_from_url('https://s.jina.ai/苹果的最新产品是啥？'))
    with open(saved_file_path, 'r', encoding='utf-8') as f:
        saved_content = f.read()
    assert saved_content == mock_response.text

    # Clean up
    os.remove(saved_file_path)
