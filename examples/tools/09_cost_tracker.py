# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Cost Tracker & RunResponse.cost_summary demo

演示 Optimization 2 & 6:
  - CostTracker — 追踪每次 API 调用的 token 用量和 USD 成本
  - RunResponse.cost_summary  — 直接打印格式化成本摘要
  - RunResponse.total_cost_usd — 直接读取数值

运行方式：
    # 需要 LLM API Key（支持 OpenAI / DeepSeek / ZhipuAI 等）
    export OPENAI_API_KEY=sk-xxx
    python 09_cost_tracker.py

    # 也可以测试纯 CostTracker（无需 API Key）
    python 09_cost_tracker.py --unit-test
"""
import sys
import os
import asyncio
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ============================================================================
# Demo 1: CostTracker 单元测试（无需 LLM，验证定价逻辑）
# ============================================================================

def demo_cost_tracker_unit():
    """直接调用 CostTracker 验证定价计算。"""
    from agentica.model.cost import CostTracker, MODEL_PRICING

    print("=" * 60)
    print("Demo 1: CostTracker 单元测试（无需 API Key）")
    print("=" * 60)

    ct = CostTracker()

    # 模拟 3 次 API 调用
    calls = [
        ("gpt-4o",      1000,  200, 0,   0),
        ("gpt-4o",       500,  100, 200, 0),   # 第 2 次有 cache_read
        ("gpt-4o-mini",  800,  150, 0,   0),
    ]
    for model_id, inp, out, cr, cw in calls:
        cost = ct.record(model_id, inp, out, cr, cw)
        pricing = MODEL_PRICING.get(model_id, {})
        expected = (
            inp * pricing.get("input", 0) / 1_000_000
            + out * pricing.get("output", 0) / 1_000_000
            + cr  * pricing.get("cache_read", 0) / 1_000_000
        )
        print(f"  [{model_id}] in={inp}, out={out}, cache_read={cr}"
              f"  → ${cost:.6f}  (expected=${expected:.6f})")
        assert abs(cost - expected) < 1e-9, f"Mismatch! {cost} != {expected}"

    print(f"\n{ct.summary()}")

    # 验证未知模型被标记
    ct2 = CostTracker()
    ct2.record("some-unknown-model-xyz", 1000, 100)
    assert ct2.has_unknown_model
    print("\n  Unknown model flag: ✓")
    print("\nAll assertions passed ✓\n")


# ============================================================================
# Demo 2: RunResponse.cost_summary（需要 LLM API Key）
# ============================================================================

async def demo_cost_in_response():
    """通过 Agent.run() 返回的 RunResponse 访问成本摘要。"""
    from agentica import Agent, OpenAIChat

    print("=" * 60)
    print("Demo 2: RunResponse.cost_summary（需要 API Key）")
    print("=" * 60)

    async def fetch_price(symbol: str) -> str:
        """Fetch stock price for symbol.

        Args:
            symbol: Stock ticker symbol.
        """
        import asyncio
        await asyncio.sleep(0.05)
        prices = {"AAPL": "$198.50", "MSFT": "$420.10", "GOOGL": "$175.30"}
        return prices.get(symbol.upper(), "$100.00")

    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[fetch_price],
        instructions=["You are a financial assistant."],
    )

    queries = [
        "What is the current price of AAPL?",
        "Compare MSFT and GOOGL stock prices.",
    ]

    total_cost = 0.0
    for i, q in enumerate(queries, 1):
        print(f"\n--- Query {i}: {q} ---")
        response = await agent.run(q)
        print(response.content)

        # ── 访问成本字段 ──────────────────────────────────
        print(f"\n  cost_summary:\n    {chr(10).join('    ' + l for l in response.cost_summary.splitlines())}")
        print(f"  total_cost_usd: ${response.total_cost_usd:.6f}")

        total_cost += response.total_cost_usd

    print(f"\n{'=' * 60}")
    print(f"  2 次查询累计费用: ${total_cost:.6f}")


# ============================================================================
# Demo 3: 多模型对比（展示 per-model 成本分解）
# ============================================================================

def demo_multi_model_pricing():
    """对比不同模型相同 token 量下的成本差异。"""
    from agentica.model.cost import CostTracker, MODEL_PRICING

    print("=" * 60)
    print("Demo 3: 不同模型相同 token 量成本对比")
    print("=" * 60)
    print(f"  场景：1000 input + 500 output tokens\n")

    inp, out = 1000, 500
    results = []
    for model_id, pricing in MODEL_PRICING.items():
        cost = (
            inp * pricing["input"]  / 1_000_000
            + out * pricing["output"] / 1_000_000
        )
        results.append((cost, model_id))

    results.sort()
    for cost, model_id in results:
        bar = "█" * int(cost * 500_000) if cost > 0 else "·"
        print(f"  {model_id:<35} ${cost:.6f}  {bar}")
    print()


# ============================================================================
# Main
# ============================================================================

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-test", action="store_true",
                        help="Only run unit tests (no API key needed)")
    args, _ = parser.parse_known_args()

    # Demo 1 always runs
    demo_cost_tracker_unit()
    demo_multi_model_pricing()

    if args.unit_test:
        return

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        await demo_cost_in_response()
    else:
        print("=" * 60)
        print("Demo 2: Skipped (set OPENAI_API_KEY to run)")
        print("=" * 60)
        print("Tip: python 09_cost_tracker.py --unit-test  (runs Demos 1 & 3 only)")


if __name__ == "__main__":
    asyncio.run(main())
