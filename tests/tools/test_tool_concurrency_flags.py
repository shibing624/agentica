# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Read-only tools must declare ``concurrency_safe``.

``Model.run_function_calls`` gathers only the calls whose function is
``concurrency_safe`` and runs the rest in a serial loop, so a read-only tool
that forgets the flag quietly turns "three searches in one turn" into three
round trips. These tests pin the flag on the tools where a single turn
routinely issues several independent calls.
"""
import importlib

import pytest

# (module, class, ctor kwargs, functions that must be parallelizable)
READ_ONLY_TOOLS = [
    ("agentica.tools.search_serper_tool", "SearchSerperTool", {"api_key": "fake"}, ["search_google"]),
    ("agentica.tools.search_exa_tool", "SearchExaTool", {"api_key": "fake"}, ["search_exa"]),
    ("agentica.tools.search_bocha_tool", "SearchBochaTool", {"api_key": "fake"}, ["search_bocha"]),
    ("agentica.tools.baidu_search_tool", "BaiduSearchTool", {}, ["baidu_search"]),
    ("agentica.tools.duckduckgo_tool", "DuckDuckGoTool", {}, ["duckduckgo_search"]),
    ("agentica.tools.zhipu_web_search_tool", "ZhipuWebSearchTool", {"api_key": "fake"}, ["zhipu_web_search"]),
    ("agentica.tools.wikipedia_tool", "WikipediaTool", {}, ["search_wikipedia"]),
    ("agentica.tools.dblp_tool", "DblpTool", {}, ["search_dblp_and_return_articles"]),
    ("agentica.tools.hackernews_tool", "HackerNewsTool", {},
     ["get_top_hackernews_stories", "get_user_details"]),
    ("agentica.tools.weather_tool", "WeatherTool", {}, ["get_weather"]),
    ("agentica.tools.arxiv_tool", "ArxivTool", {}, ["search_arxiv_and_return_articles"]),
    ("agentica.tools.yfinance_tool", "YFinanceTool", {}, ["get_current_stock_price"]),
    ("agentica.tools.code_tool", "CodeTool", {},
     ["analyze_code", "lint_code", "find_symbols", "get_code_outline"]),
]


def _build(module_name, class_name, kwargs):
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:  # optional third-party dependency
        pytest.skip(f"{module_name} unavailable: {exc}")
    return getattr(module, class_name)(**kwargs)


@pytest.mark.parametrize(
    "module_name,class_name,kwargs,function_names",
    READ_ONLY_TOOLS,
    ids=[entry[1] for entry in READ_ONLY_TOOLS],
)
def test_read_only_tools_can_run_in_parallel(module_name, class_name, kwargs, function_names):
    tool = _build(module_name, class_name, kwargs)
    for name in function_names:
        function = tool.functions[name]
        assert function.concurrency_safe is True, f"{class_name}.{name} would serialize"
        assert function.is_read_only is True, f"{class_name}.{name} is not marked read-only"


def test_side_effecting_siblings_stay_serial():
    """The flag is per function, not per tool: the ones that write must not
    inherit it from the read-only functions registered beside them."""
    arxiv = _build("agentica.tools.arxiv_tool", "ArxivTool", {})
    # read_arxiv_papers downloads PDFs into a shared download dir.
    assert arxiv.functions["read_arxiv_papers"].concurrency_safe is False

    code = _build("agentica.tools.code_tool", "CodeTool", {})
    # format_code rewrites the file in place.
    assert code.functions["format_code"].concurrency_safe is False


def test_sql_schema_reads_are_parallel_but_queries_are_not():
    sql = _build("agentica.tools.sql_tool", "SQLTool", {"db_url": "sqlite:///:memory:"})
    assert sql.functions["list_tables"].concurrency_safe is True
    assert sql.functions["describe_table"].concurrency_safe is True
    # run_sql_query accepts arbitrary SQL, including INSERT/UPDATE/DELETE.
    assert sql.functions["run_sql_query"].concurrency_safe is False


def test_lsp_queries_are_parallel_and_the_write_path_is_not():
    lsp = _build("agentica.tools.lsp_tool", "LspTool", {})
    for name in ("goto_definition", "find_references", "hover_info"):
        assert lsp.functions[name].concurrency_safe is True

    formatting = _build("agentica.tools.lsp_tool", "LspTool", {"enable_formatting": True})
    # format_document writes the formatted buffer back to disk.
    assert formatting.functions["format_document"].concurrency_safe is False


def test_lsp_writes_one_message_at_a_time():
    """Parallel LSP queries share one stdin pipe; interleaving a header with
    another message's body desyncs the framing for everything after it."""
    import threading

    from agentica.tools.lsp_tool import JsonRpcClient

    class _FakeStdin:
        def __init__(self):
            self.writes = []

        def write(self, data):
            self.writes.append(data)

        def flush(self):
            pass

    class _FakeProcess:
        def __init__(self):
            self.stdin = _FakeStdin()
            self.stdout = None

    client = JsonRpcClient.__new__(JsonRpcClient)
    client._process = _FakeProcess()

    holder = threading.Lock()
    holder.acquire()
    client._lock = holder
    blocked = threading.Thread(
        target=client._send_message, args=({"jsonrpc": "2.0", "method": "x"},), daemon=True,
    )
    blocked.start()
    blocked.join(timeout=0.2)

    assert blocked.is_alive(), "_send_message wrote to stdin without holding the lock"
    holder.release()
    blocked.join(timeout=1)
    assert client._process.stdin.writes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
