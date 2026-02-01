# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import os
from unittest.mock import patch, Mock
import pytest
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.tools.url_crawler_tool import UrlCrawlerTool


@pytest.fixture
def url_crawler_tool():
    """Fixture to create a UrlCrawlerTool instance for testing."""
    return UrlCrawlerTool(base_dir="./test_work_dir")


@patch('agentica.tools.url_crawler_tool.requests.get')
def test_crawl_url_to_file_html(mock_get):
    """Test the crawl_url_to_file method for HTML content."""
    # Mock response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/html; charset=utf-8"}
    html_content = "<html><head><title>Test</title></head><body><h1>Test Content</h1></body></html>"
    mock_response.text = html_content
    mock_response.content = html_content.encode('utf-8')
    mock_response.apparent_encoding = "utf-8"
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    url = "https://example.com/test-url"
    url_crawler_tool = UrlCrawlerTool(base_dir="./test_work_dir")
    res = url_crawler_tool.url_crawl(url)
    print('content:', res)
    # Assertions
    assert res is not None
    import json
    result_dict = json.loads(res)
    assert result_dict["url"] == url
    assert "Test Content" in result_dict["content"]


@patch('agentica.tools.url_crawler_tool.requests.get')
def test_crawl_url_to_file_non_html(mock_get):
    """Test the crawl_url_to_file method for non-HTML content."""
    # Mock response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/pdf"}
    mock_response.text = "%PDF-1.4 ..."
    mock_response.content = b"%PDF-1.4 ..."
    mock_response.apparent_encoding = "utf-8"
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    url = "https://example.com/test-file.pdf"
    url_crawler_tool = UrlCrawlerTool(base_dir="./test_work_dir")
    res = url_crawler_tool.url_crawl(url)
    print('content:', res)
    # Assertions
    assert res is not None


@patch('agentica.tools.url_crawler_tool.requests.get')
def test_url_crawl(mock_get):
    """Test the url_crawl method."""
    # Mock response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/html; charset=utf-8"}
    html_content = "<html><head><title>Test</title></head><body><h1>Test</h1></body></html>"
    mock_response.text = html_content
    mock_response.content = html_content.encode('utf-8')
    mock_response.apparent_encoding = "utf-8"
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    url = "https://example.com/test-url"
    url_crawler_tool = UrlCrawlerTool(base_dir="./test_work_dir")
    result = url_crawler_tool.url_crawl(url)
    print('result:', result)
    # Assertions
    assert result is not None
    import json
    result_dict = json.loads(result)
    assert result_dict["url"] == url


@patch('agentica.tools.url_crawler_tool.requests.get')
def test_crawl_url_to_file_error(mock_get):
    """Test the crawl_url_to_file method with an error response."""
    # Mock an error response
    mock_get.side_effect = Exception("404 Not Found")

    url = "https://example.com/test-url"
    url_crawler_tool = UrlCrawlerTool(base_dir="./test_work_dir")
    content = url_crawler_tool.url_crawl(url)
    print('content:', content)
    # Assertions
    assert content is not None
    import json
    result_dict = json.loads(content)
    assert result_dict["url"] == url
    assert result_dict["content"] == ""  # Error case returns empty content


@patch('agentica.tools.url_crawler_tool.requests.get')
def test_url_crawl_error(mock_get):
    """Test the url_crawl method with an error response."""
    # Mock an error response
    mock_get.side_effect = Exception("404 Not Found")

    url = "https://example.com/test-url"
    url_crawler_tool = UrlCrawlerTool(base_dir="./test_work_dir")
    result = url_crawler_tool.url_crawl(url)
    print('result:', result)
    # Assertions
    assert result is not None
    import json
    result_dict = json.loads(result)
    assert result_dict["url"] == url


if __name__ == '__main__':
    pytest.main()
