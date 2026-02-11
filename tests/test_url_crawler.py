# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

import asyncio
import json
import os
from unittest.mock import patch, AsyncMock, MagicMock
import pytest
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.tools.url_crawler_tool import UrlCrawlerTool


@pytest.fixture
def url_crawler_tool():
    """Fixture to create a UrlCrawlerTool instance for testing."""
    return UrlCrawlerTool(base_dir="./tmp")


def _make_mock_response(status_code, headers, content_bytes, text=None):
    """Helper to create a mock httpx.Response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.headers = headers
    mock_resp.content = content_bytes
    mock_resp.text = text or content_bytes.decode('utf-8', errors='replace')
    mock_resp.encoding = 'utf-8'
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def test_crawl_url_to_file_html():
    """Test the crawl_url_to_file method for HTML content."""
    html_content = "<html><head><title>Test</title></head><body><h1>Test Content</h1></body></html>"
    mock_resp = _make_mock_response(200, {"content-type": "text/html; charset=utf-8"}, html_content.encode('utf-8'), html_content)

    async def mock_get(self, url, **kwargs):
        return mock_resp

    with patch('httpx.AsyncClient.get', new=mock_get):
        url = "https://example.com/test-url"
        tool = UrlCrawlerTool(base_dir="./tmp")
        res = asyncio.run(tool.url_crawl(url))
        print('content:', res)
        assert res is not None
        result_dict = json.loads(res)
        assert result_dict["url"] == url
        assert "Test Content" in result_dict["content"]


def test_crawl_url_to_file_non_html():
    """Test the crawl_url_to_file method for non-HTML content."""
    mock_resp = _make_mock_response(200, {"content-type": "application/pdf"}, b"%PDF-1.4 ...", "%PDF-1.4 ...")

    async def mock_get(self, url, **kwargs):
        return mock_resp

    with patch('httpx.AsyncClient.get', new=mock_get):
        url = "https://example.com/test-file.pdf"
        tool = UrlCrawlerTool(base_dir="./tmp")
        res = asyncio.run(tool.url_crawl(url))
        print('content:', res)
        assert res is not None


def test_url_crawl():
    """Test the url_crawl method."""
    html_content = "<html><head><title>Test</title></head><body><h1>Test</h1></body></html>"
    mock_resp = _make_mock_response(200, {"content-type": "text/html; charset=utf-8"}, html_content.encode('utf-8'), html_content)

    async def mock_get(self, url, **kwargs):
        return mock_resp

    with patch('httpx.AsyncClient.get', new=mock_get):
        url = "https://example.com/test-url"
        tool = UrlCrawlerTool(base_dir="./tmp")
        result = asyncio.run(tool.url_crawl(url))
        print('result:', result)
        assert result is not None
        result_dict = json.loads(result)
        assert result_dict["url"] == url


def test_crawl_url_to_file_error():
    """Test the crawl_url_to_file method with an error response."""
    async def mock_get(self, url, **kwargs):
        raise Exception("404 Not Found")

    with patch('httpx.AsyncClient.get', new=mock_get):
        url = "https://example.com/test-url"
        tool = UrlCrawlerTool(base_dir="./tmp")
        content = asyncio.run(tool.url_crawl(url))
        print('content:', content)
        assert content is not None
        result_dict = json.loads(content)
        assert result_dict["url"] == url
        assert result_dict["content"] == ""  # Error case returns empty content


def test_url_crawl_error():
    """Test the url_crawl method with an error response."""
    async def mock_get(self, url, **kwargs):
        raise Exception("404 Not Found")

    with patch('httpx.AsyncClient.get', new=mock_get):
        url = "https://example.com/test-url"
        tool = UrlCrawlerTool(base_dir="./tmp")
        result = asyncio.run(tool.url_crawl(url))
        print('result:', result)
        assert result is not None
        result_dict = json.loads(result)
        assert result_dict["url"] == url


if __name__ == '__main__':
    pytest.main()
