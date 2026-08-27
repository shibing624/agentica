# -*- coding: utf-8 -*-
"""write_html: wrap a report, default path, factory/DeepAgent registration."""
import asyncio
import os
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-real")

from agentica.tools.builtin import BuiltinFileTool, get_builtin_tools
from agentica.tools.builtin.file_tool import (
    _MAX_HTML_CHARS,
    is_full_html_document,
    slugify_html_title,
    wrap_report_html,
)


def test_slugify_keeps_cjk_and_replaces_spaces():
    assert slugify_html_title("竞品分析") == "竞品分析"
    assert slugify_html_title("Q3 竞品 / 对比") == "Q3-竞品-对比"
    assert slugify_html_title("...") == "report"


def test_is_full_html_document():
    assert is_full_html_document("<!DOCTYPE html><html></html>")
    assert is_full_html_document("  <html lang='zh'>")
    assert not is_full_html_document("<h1>fragment</h1>")


def test_wrap_report_html_injects_title_and_css():
    page = wrap_report_html("竞品分析", "<h2>A</h2><p>body</p>")
    assert page.startswith("<!DOCTYPE html>")
    assert "<title>竞品分析</title>" in page
    assert "<h1>竞品分析</h1>" in page
    assert "<h2>A</h2><p>body</p>" in page
    assert "<style>" in page and "max-width: 880px" in page


def test_wrap_plaintext_escapes():
    page = wrap_report_html("Note", "a < b & c")
    assert "<pre>a &lt; b &amp; c</pre>" in page


def test_factory_default_does_not_register_write_html():
    file_tool = next(
        t for t in get_builtin_tools(work_dir="/tmp")
        if type(t).__name__ == "BuiltinFileTool"
    )
    assert "write_html" not in file_tool.functions
    assert "write_file" in file_tool.functions


def test_factory_include_html_report_registers_write_html():
    file_tool = next(
        t for t in get_builtin_tools(work_dir="/tmp", include_html_report=True)
        if type(t).__name__ == "BuiltinFileTool"
    )
    assert "write_html" in file_tool.functions


def test_no_file_tools_means_no_write_html():
    names = {type(t).__name__ for t in get_builtin_tools(
        work_dir="/tmp", include_file_tools=False, include_html_report=True,
    )}
    assert "BuiltinFileTool" not in names


def test_write_html_default_path_and_overwrite(tmp_path):
    tool = BuiltinFileTool(work_dir=str(tmp_path), include_html_report=True)
    first = asyncio.run(tool.write_html(
        "竞品分析", "<h2>v1</h2><p>alpha</p>",
    ))
    dest = tmp_path / "tmp" / "reports" / "竞品分析.html"
    assert dest.is_file()
    assert f"Wrote HTML report: {dest.resolve()}" in first
    assert "Open: file://" in first
    assert "html=" not in first
    assert "<h2>v1</h2>" in dest.read_text(encoding="utf-8")

    second = asyncio.run(tool.write_html(
        "竞品分析", "<h2>v2</h2><p>beta</p>",
    ))
    assert dest.is_file()
    body = dest.read_text(encoding="utf-8")
    assert "<h2>v2</h2>" in body
    assert "v1" not in body
    assert str(dest.resolve()) in second


def test_write_html_full_document_passthrough(tmp_path):
    tool = BuiltinFileTool(work_dir=str(tmp_path), include_html_report=True)
    raw = "<!DOCTYPE html><html><body><p>raw</p></body></html>"
    asyncio.run(tool.write_html("Raw", raw, file_path="out/custom"))
    dest = tmp_path / "out" / "custom.html"
    assert dest.read_text(encoding="utf-8") == raw


def test_write_html_rejects_empty_and_oversized(tmp_path):
    tool = BuiltinFileTool(work_dir=str(tmp_path), include_html_report=True)
    assert "title is empty" in asyncio.run(tool.write_html("  ", "<p>x</p>"))
    assert "html is empty" in asyncio.run(tool.write_html("T", "  "))
    huge = "x" * (_MAX_HTML_CHARS + 1)
    msg = asyncio.run(tool.write_html("T", huge))
    assert "Nothing written" in msg
    assert not list(Path(tmp_path).rglob("*.html"))
