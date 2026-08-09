# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Which tool calls the executor is allowed to overlap.

``Model.run_function_calls`` gathers only the concurrency-safe calls and runs
the rest in a serial loop, so a read-only tool that forgets the flag quietly
turns "three searches in one turn" into three round trips. These tests pin the
flag on the tools where a single turn routinely issues several independent
calls.

``execute`` is the exception that has no tool-level answer — the same tool runs
`pytest` and `git commit` — so it declares safety per call via
``parallel_safe``. Those tests assert the schedule (how many calls were in
flight) rather than the flag, since reading the flag back would pass even if
the executor ignored it.
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


def _shell_calls(tool, *commands, **shared):
    """A batch of execute() calls the way one assistant message issues them."""
    from agentica.tools.base import FunctionCall

    return [
        FunctionCall(
            function=tool.functions["execute"],
            arguments={"command": command, **shared},
            call_id=f"c{i}",
        )
        for i, command in enumerate(commands)
    ]


def _drive(calls):
    """Run one batch through the real executor and return the results list."""
    import asyncio

    from agentica.model.openai import OpenAIChat

    model = OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")
    model.metrics = {}
    model.function_call_stack = None
    model.tool_call_limit = None

    async def _run():
        results = []
        async for _ in model.run_function_calls(calls, results):
            pass
        return results

    return asyncio.run(_run())


def _peak_tracking_tool():
    """A BuiltinExecuteTool whose execute() records how many calls overlap."""
    from agentica.tools.builtin.execute_tool import BuiltinExecuteTool
    import asyncio

    tool = BuiltinExecuteTool()
    state = {"in_flight": 0, "peak": 0}
    real = tool.functions["execute"]

    async def _fake(command, timeout=None, background=False, parallel_safe=False):
        state["in_flight"] += 1
        state["peak"] = max(state["peak"], state["in_flight"])
        try:
            await asyncio.sleep(0.05)
        finally:
            state["in_flight"] -= 1
        return f"ran {command}"

    real.entrypoint = _fake
    return tool, state


def test_parallel_safe_shell_calls_actually_overlap():
    """Assert the schedule, not the flag: three calls must be in flight at once.

    ``execute`` cannot answer "is this parallel-safe" at registration time — the
    same tool runs `pytest` and `git commit` — so the answer travels per call.
    A test that reads the flag back would pass even if the executor ignored it.
    """
    tool, state = _peak_tracking_tool()
    calls = _shell_calls(
        tool, "pytest tests/a -q", "pytest tests/b -q", "pytest tests/c -q",
        parallel_safe=True,
    )

    _drive(calls)

    assert state["peak"] == 3, f"ran {state['peak']}-at-a-time; must overlap"


def test_shell_calls_are_serial_unless_the_caller_says_otherwise():
    """The default has to stay serial: `git add` then `git commit` is the same
    shape as two independent test runs, and only the caller can tell them
    apart. Guessing "parallel" here corrupts a working tree."""
    tool, state = _peak_tracking_tool()

    _drive(_shell_calls(tool, "git add -A", "git commit -m x", "git push"))

    assert state["peak"] == 1, f"ran {state['peak']}-at-a-time; must be serial"


def test_a_parallel_shell_failure_still_cancels_the_serial_remainder():
    """Sibling-abort is a property of the batch, not of the branch that ran the
    failure. Without this, opting one call into the parallel phase would let a
    later dependent command run against the state a failed one left behind."""
    from agentica.tools.base import ToolCallException
    from agentica.tools.builtin.execute_tool import BuiltinExecuteTool

    tool = BuiltinExecuteTool()

    async def _fake(command, timeout=None, background=False, parallel_safe=False):
        if "boom" in command:
            raise ToolCallException("command blew up")
        return f"ran {command}"

    tool.functions["execute"].entrypoint = _fake
    calls = _shell_calls(tool, "boom", parallel_safe=True) + _shell_calls(tool, "git commit -m x")
    calls[1].call_id = "c1"

    results = _drive(calls)

    assert "Cancelled" in str(results[1].content)


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
