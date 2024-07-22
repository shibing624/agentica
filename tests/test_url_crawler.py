# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import os
from unittest.mock import patch, Mock
import shutil
import pytest

from agentica.tools.url_crawler import UrlCrawlerTool


@pytest.fixture
def url_crawler_tool():
    """Fixture to create a UrlCrawlerTool instance for testing."""
    return UrlCrawlerTool(work_dir="./test_work_dir")


@patch('requests.get')
def test_crawl_url_to_file_html(mock_get):
    """Test the crawl_url_to_file method for HTML content."""
    # Mock response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/html"}
    mock_response.iter_content.return_value = "test"
    mock_get.return_value = mock_response

    url = "https://example.com/test-url"
    url_crawler_tool = UrlCrawlerTool(work_dir="./test_work_dir")
    content, save_path = url_crawler_tool.crawl_url_to_file(url)
    print('content:', content, '\tsave_path:', save_path)
    # Assertions
    assert save_path is not None

    # Clean up
    os.remove(save_path)


@patch('requests.get')
def test_crawl_url_to_file_non_html(mock_get):
    """Test the crawl_url_to_file method for non-HTML content."""
    # Mock response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/pdf"}
    mock_response.iter_content.return_value = b"%PDF-1.4 ..."
    mock_get.return_value = mock_response

    url = "https://example.com/test-file.pdf"
    url_crawler_tool = UrlCrawlerTool(work_dir="./test_work_dir")
    content, save_path = url_crawler_tool.crawl_url_to_file(url)
    print('content:', content, '\tsave_path:', save_path)
    # Assertions
    assert save_path is not None

    # Clean up
    os.remove(save_path)


@patch('requests.get')
def test_url_crawl(mock_get):
    """Test the url_crawl method."""
    # Mock response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/html"}
    mock_response.iter_content.return_value = "<html><body><h1>Test</h1></body></html>"
    mock_get.return_value = mock_response

    url = "https://example.com/test-url"
    url_crawler_tool = UrlCrawlerTool(work_dir="./test_work_dir")
    result = url_crawler_tool.url_crawl(url)
    print('result:', result)
    # Assertions
    assert "Test" in result

    # Clean up
    shutil.rmtree(url_crawler_tool.work_dir)


@patch('requests.get')
def test_crawl_url_to_file_error(mock_get):
    """Test the crawl_url_to_file method with an error response."""
    # Mock an error response
    mock_response = Mock()
    mock_response.status_code = 404
    mock_get.side_effect = Exception("404 Not Found")

    url = "https://example.com/test-url"
    url_crawler_tool = UrlCrawlerTool(work_dir="./test_work_dir")
    content, save_path = url_crawler_tool.crawl_url_to_file(url)
    print('content:', content, '\tsave_path:', save_path)
    # Assertions
    assert content == ""
    shutil.rmtree(url_crawler_tool.work_dir)


@patch('requests.get')
def test_url_crawl_error(mock_get):
    """Test the url_crawl method with an error response."""
    # Mock an error response
    mock_get.side_effect = Exception("404 Not Found")

    url = "https://example.com/test-url"
    url_crawler_tool = UrlCrawlerTool(work_dir="./test_work_dir")
    result = url_crawler_tool.url_crawl(url)

    # Assertions
    assert result is not None
    shutil.rmtree(url_crawler_tool.work_dir)


if __name__ == '__main__':
    pytest.main()
