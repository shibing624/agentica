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
# Demo 1: prune_tool_results 单元演示（无需 LLM）
# ============================================================================

def demo_prune_tool_results():
    """展示工具结果修剪：只在换窗线上动手，保留头尾，够不到目标就整体放弃。"""
    from agentica.compression.evict import (
        prune_tool_results,
        PRUNE_TARGET_RATIO,
        PRUNE_HEAD_CHARS,
        PRUNE_TAIL_CHARS,
    )
    from agentica.model.message import Message

    print("=" * 60)
    print("Demo 1: 工具结果修剪 — 一个触发点，修剪或换窗二选一")
    print("=" * 60)
    print(f"  策略: 到换窗线才动手；旧结果保留前 {PRUNE_HEAD_CHARS} + 后 {PRUNE_TAIL_CHARS} 字符，")
    print(f"        修剪后能降到工作窗口 {PRUNE_TARGET_RATIO:.0%} 以下才应用，否则直接换窗\n")

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
                content=" ".join(f"line{i}-token{w}" for w in range(2000)),
            ))
        return messages

    # 场景 1：修剪够得到目标 —— 应用，窗口保住，只断一次缓存
    reach = build_conversation(8)
    r1 = prune_tool_results(reach, context_tokens=60_000, target_tokens=50_000)
    # 场景 2：修剪够不到目标 —— 一个字都不改，交给换窗
    short = build_conversation(8)
    before = [m.content for m in short]
    r2 = prune_tool_results(short, context_tokens=60_000, target_tokens=1_000)

    tool_msgs = [m for m in reach if m.role == "tool"]
    print(f"  够得到目标: 修剪 {r1.pruned} 条，释放约 {r1.freed_tokens:,} tokens")
    print(f"  够不到目标: 修剪 {r2.pruned} 条  ← 历史原样不动，直接换窗")
    assert r1.pruned == 7, "除了模型还没看过的最新一批，其余全部修剪"
    assert "pruned from the middle" not in tool_msgs[-1].content, "最新一批原样保留"
    assert r2.pruned == 0 and [m.content for m in short] == before
    print("\n  验证通过 ✓")
    print(f"  tool_msg[0] 中段标记: '{tool_msgs[0].content[PRUNE_HEAD_CHARS:PRUNE_HEAD_CHARS + 70].strip()}'\n")


# ============================================================================
# Demo 2: CompressionManager.auto_compact 演示（空窗换窗，无需 LLM）
# ============================================================================

def demo_auto_compact_config():
    """展示 CompressionManager 的空窗换窗配置。"""
    from agentica.compression.manager import CompressionManager

    print("=" * 60)
    print("Demo 2: Layer 2 — CompressionManager（空窗换窗）")
    print("=" * 60)
    print("  淘汰兜不住时才走这层：丢掉活动窗旧轮次，装上 <context_window> + notes")
    print("  不调用 LLM，也不走 /responses/compact。原文留在 session JSONL。\n")

    cm = CompressionManager(compact_token_limit=80_000)
    print(f"  工作阈值 compact_token_limit: {cm.compact_token_limit:,} tokens")

    auto = CompressionManager()
    print(f"  零配置 compact_token_limit: {auto.compact_token_limit} → 运行时按窗口×0.95 换窗")

    from agentica.compression.manager import AUTO_COMPACT_THRESHOLD_RATIO
    print(f"  触发比例: context_window 的 {AUTO_COMPACT_THRESHOLD_RATIO:.0%}（距窗顶 5% 余量）\n")


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
        compress_token_limit=2000,      # low threshold to trigger compaction in demo
    )

    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[read_source_file],
        tool_config=ToolConfig(compression_manager=cm),
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
    demo_prune_tool_results()
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
