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
    return UrlCrawlerTool(work_dir="./tmp")


def _make_mock_response(status_code, headers, content_bytes, text=None):
    """Helper to create a mock httpx.Response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.headers = headers
    mock_resp.content = content_bytes
    mock_resp.text = text or content_bytes.decode("utf-8", errors="replace")
    mock_resp.encoding = "utf-8"
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def test_crawl_url_to_file_html():
    """Test the crawl_url_to_file method for HTML content."""
    html_content = "<html><head><title>Test</title></head><body><h1>Test Content</h1></body></html>"
    mock_resp = _make_mock_response(
        200, {"content-type": "text/html; charset=utf-8"}, html_content.encode("utf-8"), html_content
    )

    async def mock_get(self, url, **kwargs):
        return mock_resp

    with patch("httpx.AsyncClient.get", new=mock_get):
        url = "https://example.com/test-url"
        tool = UrlCrawlerTool(work_dir="./tmp")
        res = asyncio.run(tool.url_crawl(url))
        print("content:", res)
        assert res is not None
        result_dict = json.loads(res)
        assert result_dict["url"] == url
        assert "Test Content" in result_dict["content"]


def test_crawl_caps_on_disk_cache_size():
    """A huge page must NOT write an unbounded file to the cache: the on-disk
    copy is capped at MAX_CACHE_FILE_CHARS while the returned content stays
    trimmed to max_content_length. Guards against a model read_file-ing a
    multi-MB dump and blowing the token budget (regression)."""
    big = "<html><body>" + ("word " * 300000) + "</body></html>"
    mock_resp = _make_mock_response(200, {"content-type": "text/html; charset=utf-8"}, big.encode("utf-8"), big)

    async def mock_get(self, url, **kwargs):
        return mock_resp

    with patch("httpx.AsyncClient.get", new=mock_get):
        tool = UrlCrawlerTool(work_dir="./tmp", max_content_length=16000)
        res = asyncio.run(tool.url_crawl("https://example.com/huge"))
        result_dict = json.loads(res)
        # Returned content trimmed for the model
        assert len(result_dict["content"]) <= 16000 + 50
        # On-disk cache bounded
        size = os.path.getsize(result_dict["save_path"])
        assert size <= UrlCrawlerTool.MAX_CACHE_FILE_CHARS + 200, size


def test_crawl_url_to_file_non_html():
    """Test the crawl_url_to_file method for non-HTML content."""
    mock_resp = _make_mock_response(200, {"content-type": "application/pdf"}, b"%PDF-1.4 ...", "%PDF-1.4 ...")

    async def mock_get(self, url, **kwargs):
        return mock_resp

    with patch("httpx.AsyncClient.get", new=mock_get):
        url = "https://example.com/test-file.pdf"
        tool = UrlCrawlerTool(work_dir="./tmp")
        res = asyncio.run(tool.url_crawl(url))
        print("content:", res)
        assert res is not None


def test_url_crawl():
    """Test the url_crawl method."""
    html_content = "<html><head><title>Test</title></head><body><h1>Test</h1></body></html>"
    mock_resp = _make_mock_response(
        200, {"content-type": "text/html; charset=utf-8"}, html_content.encode("utf-8"), html_content
    )

    async def mock_get(self, url, **kwargs):
        return mock_resp

    with patch("httpx.AsyncClient.get", new=mock_get):
        url = "https://example.com/test-url"
        tool = UrlCrawlerTool(work_dir="./tmp")
        result = asyncio.run(tool.url_crawl(url))
        print("result:", result)
        assert result is not None
        result_dict = json.loads(result)
        assert result_dict["url"] == url


def test_crawl_url_to_file_error():
    """Test the crawl_url_to_file method with an error response."""

    async def mock_get(self, url, **kwargs):
        raise Exception("404 Not Found")

    with patch("httpx.AsyncClient.get", new=mock_get):
        url = "https://example.com/test-url"
        tool = UrlCrawlerTool(work_dir="./tmp")
        content = asyncio.run(tool.url_crawl(url))
        print("content:", content)
        assert content is not None
        result_dict = json.loads(content)
        assert result_dict["url"] == url
        assert result_dict["content"] == ""  # Error case returns empty content
        # Failure must surface an `error` field — silently returning empty
        # content tricks the model into thinking the page is genuinely empty
        # and re-fetching the same URL.
        assert "error" in result_dict, "url_crawl must include an 'error' key on failure"
        assert result_dict["error"], "'error' must be a non-empty description"


def test_url_crawl_error():
    """Test the url_crawl method with an error response."""

    async def mock_get(self, url, **kwargs):
        raise Exception("404 Not Found")

    with patch("httpx.AsyncClient.get", new=mock_get):
        url = "https://example.com/test-url"
        tool = UrlCrawlerTool(work_dir="./tmp")
        result = asyncio.run(tool.url_crawl(url))
        print("result:", result)
        assert result is not None
        result_dict = json.loads(result)
        assert result_dict["url"] == url
        assert "error" in result_dict
        assert "404 Not Found" in result_dict["error"]


def test_url_crawl_success_has_no_error_key():
    """On success, the result must NOT contain an 'error' key — that key is
    the model's signal that something went wrong."""
    html_content = "<html><body><h1>OK</h1></body></html>"
    mock_resp = _make_mock_response(
        200,
        {"content-type": "text/html; charset=utf-8"},
        html_content.encode("utf-8"),
        html_content,
    )

    async def mock_get(self, url, **kwargs):
        return mock_resp

    with patch("httpx.AsyncClient.get", new=mock_get):
        tool = UrlCrawlerTool(work_dir="./tmp")
        result = asyncio.run(tool.url_crawl("https://example.com/ok"))
        result_dict = json.loads(result)
        assert "error" not in result_dict
        assert "OK" in result_dict["content"]


def test_url_crawl_http_status_error_includes_status_code():
    """HTTP 4xx/5xx must surface the status code so the model can decide
    whether to retry, rewrite the URL, or give up — not a generic 'failed'."""
    import httpx as _httpx

    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.reason_phrase = "Not Found"

    def _raise():
        raise _httpx.HTTPStatusError("404", request=MagicMock(), response=mock_resp)

    mock_resp.raise_for_status = _raise

    async def mock_get(self, url, **kwargs):
        return mock_resp

    with patch("httpx.AsyncClient.get", new=mock_get):
        tool = UrlCrawlerTool(work_dir="./tmp")
        result = asyncio.run(tool.url_crawl("https://example.com/missing"))
        result_dict = json.loads(result)
        assert "error" in result_dict
        assert "404" in result_dict["error"]


if __name__ == "__main__":
    pytest.main()
