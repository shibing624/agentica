# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: concurrency_safe 工具并发分流 demo

演示 Optimization 1: Tool concurrency_safe 标记驱动的并发分流策略

核心机制（来自 CC StreamingToolExecutor）：
  - concurrency_safe=True  只读工具 → asyncio.gather 并行执行
  - concurrency_safe=False 写入工具 → 串行执行
  - execute/bash 失败      → 取消后续串行工具（sibling-error 模式）

内置工具中已自动标记 concurrency_safe：
  True：  ls, read_file, glob, grep, web_search, fetch_url
  False： execute, write_file, apply_patch（默认）

用法：
    python 08_concurrency_safe_tools.py
"""
import sys
import os
import asyncio
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica.tools.decorators import tool
from agentica.tools.base import Function, FunctionCall


# ============================================================================
# Demo 1: concurrency_safe 标记对比（不需要 LLM）
# ============================================================================

async def _slow_read(name: str, delay: float = 0.2) -> str:
    await asyncio.sleep(delay)
    return f"content of {name}"

async def _slow_write(name: str, delay: float = 0.3) -> str:
    await asyncio.sleep(delay)
    return f"wrote {name}"


async def demo_concurrency_split():
    """用 run_function_calls 直接演示分流执行时序差异。"""
    from agentica.model.openai import OpenAIChat
    from agentica.tools.base import FunctionCall, Function

    print("=" * 60)
    print("Demo 1: 只读工具 vs 写入工具的并发策略对比")
    print("=" * 60)

    m = OpenAIChat(id="gpt-4o-mini", api_key="fake")
    m.function_call_stack = None
    m.tool_call_limit = None

    # 构造 3 个"只读"工具 Function（concurrency_safe=True）
    async def read_a(path: str = "a") -> str:
        """Read file a."""
        await asyncio.sleep(0.2)
        return f"<content:{path}>"

    async def read_b(path: str = "b") -> str:
        """Read file b."""
        await asyncio.sleep(0.2)
        return f"<content:{path}>"

    async def read_c(path: str = "c") -> str:
        """Read file c."""
        await asyncio.sleep(0.2)
        return f"<content:{path}>"

    # 构造 2 个"写入"工具 Function（concurrency_safe=False，默认）
    async def write_x(path: str = "x") -> str:
        """Write file x."""
        await asyncio.sleep(0.2)
        return f"wrote {path}"

    async def write_y(path: str = "y") -> str:
        """Write file y."""
        await asyncio.sleep(0.2)
        return f"wrote {path}"

    def _make_fc(fn, safe: bool) -> FunctionCall:
        f = Function.from_callable(fn)
        f.concurrency_safe = safe
        return FunctionCall(function=f, arguments={}, call_id=fn.__name__)

    # Case 1: 3 个只读工具（并行） — 预期约 0.2s
    safe_fcs = [_make_fc(read_a, True), _make_fc(read_b, True), _make_fc(read_c, True)]
    results = []
    t0 = time.perf_counter()
    async for _ in m.run_function_calls(safe_fcs, results):
        pass
    safe_time = time.perf_counter() - t0
    print(f"\n[只读工具 × 3, concurrency_safe=True]")
    print(f"  耗时: {safe_time:.2f}s  (预期 ≈ 0.2s，并行)")
    for r in results:
        print(f"  {r.tool_name}: {r.content}")

    # Case 2: 2 个写入工具（串行） — 预期约 0.4s
    unsafe_fcs = [_make_fc(write_x, False), _make_fc(write_y, False)]
    results2 = []
    t0 = time.perf_counter()
    async for _ in m.run_function_calls(unsafe_fcs, results2):
        pass
    unsafe_time = time.perf_counter() - t0
    print(f"\n[写入工具 × 2, concurrency_safe=False]")
    print(f"  耗时: {unsafe_time:.2f}s  (预期 ≈ 0.4s，串行)")

    print(f"\n  并行加速比: {unsafe_time / safe_time:.1f}x  "
          f"({safe_time:.2f}s vs {unsafe_time:.2f}s)")


# ============================================================================
# Demo 2: @tool(concurrency_safe=True) 装饰器用法（需要 LLM API）
# ============================================================================

@tool(concurrency_safe=True, description="Read a config file (safe, read-only)")
async def read_config(filename: str) -> str:
    """Read a config file.

    Args:
        filename: Config filename to read.
    """
    await asyncio.sleep(0.1)
    configs = {
        "db.yaml":     "host: localhost\nport: 5432",
        "app.yaml":    "debug: true\nport: 8080",
        "cache.yaml":  "backend: redis\nttl: 3600",
    }
    return configs.get(filename, f"# {filename} not found")


@tool(concurrency_safe=True, description="Fetch API status (safe, read-only)")
async def fetch_api_status(service: str) -> str:
    """Fetch service health status.

    Args:
        service: Service name to check.
    """
    await asyncio.sleep(0.1)
    return f"{service}: OK (latency 12ms)"


@tool(concurrency_safe=False, description="Write a config file (unsafe, write)")
async def write_config(filename: str, content: str) -> str:
    """Write a config file.

    Args:
        filename: Target filename.
        content:  File content.
    """
    await asyncio.sleep(0.1)
    return f"Written {len(content)} chars to {filename}"


async def demo_agent_concurrency_safe():
    """Agent 使用混合只读/写入工具，框架自动分流并发。"""
    from agentica import Agent, OpenAIChat

    print("\n" + "=" * 60)
    print("Demo 2: Agent 混合工具 — 框架自动并发分流")
    print("=" * 60)
    print("read_config × 3 → concurrency_safe=True  → 并行执行")
    print("write_config × 1 → concurrency_safe=False → 串行执行\n")

    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[read_config, fetch_api_status, write_config],
        instructions=[
            "你是一个配置管理助手。",
            "读取多个配置文件时，请一次性发起所有读取操作。",
        ],
    )

    t0 = time.perf_counter()
    response = await agent.run(
        "请同时读取 db.yaml、app.yaml、cache.yaml 三个配置文件，并检查 api-gateway 服务状态。"
        "最后把汇总信息写入 summary.yaml。"
    )
    elapsed = time.perf_counter() - t0

    print(response.content)
    print(f"\n  Wall-clock: {elapsed:.2f}s")

    if response.tool_calls:
        for t in response.tool_calls:
            safe = getattr(t, 'concurrency_safe', '?')
            print(f"  [{t.tool_name}] {t.elapsed:.2f}s")


# ============================================================================
# Main
# ============================================================================

async def main():
    await demo_concurrency_split()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        await demo_agent_concurrency_safe()
    else:
        print("\n[Demo 2 skipped — set OPENAI_API_KEY to run Agent demo]")


if __name__ == "__main__":
    asyncio.run(main())
