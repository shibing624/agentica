# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 上下文压缩 + Agent Loop 状态管理 demo

演示 Optimization 3, 4, 5:
  - 工具结果淘汰   — 上下文吃紧时淘汰最旧的 tool_result（零成本）
  - Agent Loop 状态管理 — max_tokens 恢复 / API 错误重试 / 循环安全阀
  - Reactive compact — context_length_exceeded 时紧急压缩后重试

运行方式：
    # 基础演示（无需 API Key）
    python 12_compression_and_loop.py

    # 完整演示（含 Agent 运行）
    export OPENAI_API_KEY=sk-xxx
    python 12_compression_and_loop.py
"""
import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ============================================================================
# Demo 1: evict_tool_results 单元演示（无需 LLM）
# ============================================================================

def demo_evict_tool_results():
    """展示工具结果淘汰：有压力才动手，按最旧优先淘汰到目标为止。"""
    from agentica.compression.evict import (
        evict_tool_results,
        EVICT_THRESHOLD_RATIO,
        EVICT_TARGET_RATIO,
    )
    from agentica.model.message import Message

    print("=" * 60)
    print("Demo 1: 工具结果淘汰 — 有压力才清，清到目标就停")
    print("=" * 60)
    print(f"  策略: 占用超过窗口 {EVICT_THRESHOLD_RATIO:.0%} 才动手，淘汰最旧的直到降回 {EVICT_TARGET_RATIO:.0%}")
    print("        没有「保留最近 N 条」这种参数：固定条数必然输给 N+1 个并行调用\n")

    window = 20_000

    def build_conversation(rounds):
        messages = [Message(role="user", content="请帮我分析这些文件")]
        for i in range(rounds):
            messages.append(Message(
                role="assistant",
                content=f"正在读取文件 {i}.py",
                tool_calls=[{"id": f"call_{i}", "function": {"name": "read_file"}}],
            ))
            messages.append(Message(
                role="tool",
                tool_call_id=f"call_{i}",
                tool_name="read_file",
                tool_args={"file_path": f"{i}.py"},
                content=" ".join(f"line{i}-token{w}" for w in range(200)),
            ))
        return messages

    # 场景 1：窗口还宽裕 —— 清掉只会逼模型重跑工具，所以一条都不动
    roomy = build_conversation(8)
    n_roomy = evict_tool_results(roomy, context_tokens=4_000, context_window=window)
    print(f"  窗口宽裕 (4k/20k):  淘汰 {n_roomy} 条  ← 保留全部，避免重读循环")

    # 场景 2：窗口吃紧 —— 淘汰最旧的，最新那一批（模型还没看过）永不淘汰
    tight = build_conversation(8)
    before = sum(len(str(m.content or "")) for m in tight if m.role == "tool")
    n_tight = evict_tool_results(tight, context_tokens=18_000, context_window=window)
    after = sum(len(str(m.content or "")) for m in tight if m.role == "tool")
    saved = before - after
    print(f"  窗口吃紧 (18k/20k): 淘汰 {n_tight} 条  |  节省字符: {saved:,} ({saved / before * 100:.0f}%)")

    tool_msgs = [m for m in tight if m.role == "tool"]
    assert n_roomy == 0, "窗口宽裕时不应淘汰"
    assert n_tight > 0, "窗口吃紧时应淘汰旧结果"
    assert tool_msgs[0]._evicted, "最旧结果应被淘汰"
    assert not tool_msgs[-1]._evicted, "模型还没看过的最新一批应原样保留"
    assert not all(m._evicted for m in tool_msgs), "清到目标就停，不应全部淘汰"
    print("\n  验证通过 ✓")
    print(f"  tool_msg[0]:  '{tool_msgs[0].content[:72]}'  ← 已淘汰(占位符写明调用，可重发)")
    print(f"  tool_msg[-1]: '{tool_msgs[-1].content[:40]}...'  ← 保留\n")


# ============================================================================
# Demo 2: CompressionManager.auto_compact 演示（无需 LLM）
# ============================================================================

def demo_auto_compact_config():
    """展示 CompressionManager 的三层压缩配置方式。"""
    from agentica.compression.manager import CompressionManager

    print("=" * 60)
    print("Demo 2: CompressionManager 三层压缩配置")
    print("=" * 60)

    # 配置 1: 默认（规则截断，无 LLM）
    cm1 = CompressionManager(
        compress_token_limit=80_000,
        compress_target_token_limit=40_000,
    )
    print(f"  配置 1 (规则截断):")
    print(f"    触发阈值: {cm1.compress_token_limit:,} tokens")
    print(f"    目标阈值: {cm1.compress_target_token_limit:,} tokens")
    print(f"    LLM 压缩: {cm1.use_llm_compression}")

    # 配置 2: 启用 LLM 压缩（用轻量模型）
    cm2 = CompressionManager(
        compress_token_limit=60_000,
        use_llm_compression=True,
    )
    print(f"\n  配置 2 (LLM 压缩):")
    print(f"    触发阈值: {cm2.compress_token_limit:,} tokens")
    print(f"    LLM 压缩: {cm2.use_llm_compression}")

    # 显示 auto_compact circuit-breaker 状态
    print(f"\n  Auto-compact circuit-breaker:")
    print(f"    最大连续失败次数: {cm1._max_auto_compact_failures}")
    print(f"    预留 buffer tokens: {cm1._auto_compact_buffer_tokens:,}")
    print(f"    (等同于 CC 的 AUTOCOMPACT_BUFFER_TOKENS = 13,000)\n")


# ============================================================================
# Demo 3: Agent Loop 安全阀演示（需要 LLM API）
# ============================================================================

async def demo_loop_state_management():
    """展示 Agent Loop 状态管理：安全阀 + 重试计数器。"""
    from agentica import Agent, OpenAIChat

    print("=" * 60)
    print("Demo 3: Agent Loop 状态管理")
    print("=" * 60)

    call_count = [0]

    async def always_needs_more(step: int = 1) -> str:
        """A tool that always says there's more work to do.

        Args:
            step: Current step number.
        """
        call_count[0] += 1
        # 模拟：前几次调用返回"继续"指令
        if call_count[0] <= 3:
            return f"Step {step} done. Need to continue to step {step + 1}."
        return f"All {step} steps completed successfully!"

    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[always_needs_more],
        instructions=[
            "You are a step-by-step task executor.",
            "Always call always_needs_more tool and follow its instructions.",
        ],
    )

    response = await agent.run("Execute a 3-step process.")
    print(f"  Tool calls: {call_count[0]}")
    print(f"  Response: {response.content[:200]}")
    print(f"  Cost: ${response.total_cost_usd:.6f}\n")


# ============================================================================
# Demo 4: CompressionManager 集成到 Agent（需要 LLM API）
# ============================================================================

async def demo_compression_with_agent():
    """展示 Agent 使用 CompressionManager 自动管理上下文。"""
    from agentica import Agent, OpenAIChat
    from agentica.agent.config import ToolConfig
    from agentica.compression.manager import CompressionManager

    print("=" * 60)
    print("Demo 4: Agent + CompressionManager 自动三层压缩")
    print("=" * 60)

    file_contents = {f"file_{i}.py": f"# File {i}\n" + "x = " + str(i) * 100 for i in range(10)}

    async def read_source_file(filename: str) -> str:
        """Read a Python source file.

        Args:
            filename: Name of the file to read.
        """
        import asyncio
        await asyncio.sleep(0.02)
        return file_contents.get(filename, f"# {filename} not found")

    # CompressionManager: low token threshold for demo
    cm = CompressionManager(
        compress_token_limit=2000,      # low threshold to trigger compression in demo
        truncate_head_chars=50,         # keep max 50 chars per old tool result
    )

    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[read_source_file],
        tool_config=ToolConfig(
            compress_tool_results=True,
            compression_manager=cm,
        ),
        instructions=["You are a code analyzer. Read files and summarize their purpose."],
    )

    response = await agent.run(
        "Read file_0.py, file_1.py, file_2.py, file_3.py, file_4.py "
        "and tell me what each one does."
    )

    print(response.content[:400])
    print(f"\n  Compression stats: {cm.get_stats()}")
    print(f"  Token usage: {response.usage.total_tokens if response.usage else 'N/A'} tokens")
    print(f"  Cost: ${response.total_cost_usd:.6f}")
    print(f"\n  cost_summary:\n{chr(10).join('    ' + l for l in response.cost_summary.splitlines())}\n")


# ============================================================================
# Main
# ============================================================================

async def main():
    # Demo 1 & 2 无需 API Key，始终运行
    demo_evict_tool_results()
    demo_auto_compact_config()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        await demo_loop_state_management()
        await demo_compression_with_agent()
    else:
        print("=" * 60)
        print("Demo 3 & 4: Skipped (set OPENAI_API_KEY to run)")
        print("  运行: export OPENAI_API_KEY=sk-xxx && python 12_compression_and_loop.py")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
