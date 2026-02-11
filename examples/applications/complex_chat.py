# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 更复杂的 Agent 应用示例 - 多工具协作与多轮场景演示

使用方法:
    python complex_chat.py

功能演示:
1. 多工具协作（计算、汇率换算、知识搜索、行程规划、RAG 检索）
2. 多轮对话与上下文记忆
3. 自动工具调用（Function Calling）
4. 流式输出（Markdown 格式）
5. 演示一套完整的场景工作流
"""
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat


# ============================================================================
# 工具定义
# ============================================================================

def calculate(expression: str) -> str:
    """计算数学表达式（安全字符白名单）。"""
    try:
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含非法字符"
        result = eval(expression)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"


def search_knowledge(query: str) -> str:
    """搜索知识库获取信息。"""
    knowledge = {
        "python": "Python 是一种高级编程语言，以简洁易读著称。最新稳定版本是 3.12。",
        "ai agent": "AI Agent 能够自主执行任务，常见架构包含感知、规划、执行、反思。",
        "function calling": "Function Calling 允许 LLM 调用外部函数，实现与真实世界交互与计算。",
    }
    results = []
    for key, value in knowledge.items():
        if query.lower() in key.lower():
            results.append(f"- {key}: {value}")
    return "\n".join(results) if results else f"未找到关于 '{query}' 的相关信息"


def currency_convert(amount: float, from_currency: str, to_currency: str) -> str:
    """简单的汇率换算（静态汇率示例）。"""
    rates_to_cny = {"CNY": 1.0, "USD": 7.2, "EUR": 7.8, "JPY": 0.05}
    # 统一换算为 CNY 再到目标
    if from_currency not in rates_to_cny or to_currency not in rates_to_cny:
        return "错误：暂不支持该币种"
    amount_cny = amount * (rates_to_cny[from_currency] if from_currency == "CNY" else rates_to_cny[from_currency]
                           )
    # 以 CNY 为基准反推目标币种（示例简化）
    if to_currency == "CNY":
        converted = amount_cny
    else:
        converted = amount_cny / rates_to_cny[to_currency]
    return f"{amount} {from_currency} ≈ {converted:.2f} {to_currency}"


def recommend_places(city: str, outdoor_preference: bool = True) -> str:
    """推荐城市景点。"""
    places = {
        "北京": [
            ("故宫", "历史文化"),
            ("天安门广场", "地标建筑"),
            ("颐和园", "园林/户外"),
            ("南锣鼓巷", "街区漫步"),
            ("长城(八达岭)", "户外/爬山"),
        ],
        "上海": [
            ("外滩", "城市风光"),
            ("迪士尼", "主题乐园"),
            ("豫园", "古典园林"),
            ("徐家汇公园", "户外"),
        ],
    }
    selected = places.get(city, [])
    if outdoor_preference:
        selected = [p for p in selected if "户外" in p[1] or "园林" in p[1] or "爬山" in p[1]] or selected
    return "\n".join([f"- {name}（{tag}）" for name, tag in selected]) or f"未找到 {city} 的景点信息"


def plan_day_trip(city: str, budget_cny: float, people: int = 1, preferences: str = "") -> str:
    """根据预算与偏好生成一日游规划（示例规则）。"""
    per_person = budget_cny / max(people, 1)
    recs = recommend_places(city, outdoor_preference=("户外" in preferences or "outdoor" in preferences.lower()))
    plan = f"""
# {city}一日游计划
- 人数：{people} 人
- 总预算：{budget_cny} 元（人均约 {per_person:.0f} 元）
- 天气：请出发前查看天气预报并准备相应装备（防晒/雨具）

建议路线：
{recs}

餐饮建议：选择本地特色餐馆，人均 60-100 元。
交通建议：地铁/公交优先，市内出行人均 20-40 元。

注意事项：根据天气携带防晒/雨具，建议提前预约热门景点。
"""
    return plan.strip()


# ---------------------------
# 简易 RAG（检索增强生成）工具
# ---------------------------

DOCS = [
    {
        "title": "Agentica 框架简介",
        "content": (
            "Agentica 是一个用于构建 AI Agent 应用的框架，支持创建 Agent、注册工具(Function Calling)、"
            "流式输出、对话历史记忆，以及与 OpenAI 等模型的集成。你可以通过提供 instructions、tools、"
            "和模型配置来快速搭建具备任务执行能力的助手。"
        ),
    },
    {
        "title": "如何创建一个简单的 Agent",
        "content": (
            "在 Agentica 中，创建 Agent 的核心步骤包括：1) 选择模型(如 OpenAIChat)，2) 编写系统指令，"
            "3) 注册工具函数，4) 启用对话历史或 Markdown 渲染(可选)。然后使用 run 或 run_stream 调用即可。"
        ),
    },
    {
        "title": "RAG 工作原理",
        "content": (
            "RAG（检索增强生成）通过先检索相关文档，再让大模型基于检索到的上下文生成回答。常见流程包含："
            "查询改写、向量检索/关键词检索、Top-K 片段拼接、答案生成与引用标注。"
        ),
    },
]


def rag_search(query: str, top_k: int = 3) -> str:
    """基于内置小型语料的简易检索，返回最相关的文档片段（含来源）。"""
    def tokenize(text: str):
        return [t for t in ''.join(c if c.isalnum() else ' ' for c in text.lower()).split() if t]

    q_tokens = set(tokenize(query))
    scored = []
    for doc in DOCS:
        d_tokens = set(tokenize(doc["content"])) | set(tokenize(doc["title"]))
        score = len(q_tokens & d_tokens)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [d for s, d in scored[: max(1, min(top_k, len(scored)))] if s > 0] or [scored[0][1]]

    lines = ["# RAG 检索结果"]
    for i, d in enumerate(top, 1):
        lines.append(f"{i}. 《{d['title']}》\n   摘要：{d['content']}")
    return "\n".join(lines)


# ---------------------------
# BM25 关键词检索工具（Okapi BM25）
# ---------------------------

def rag_search_bm25(query: str, top_k: int = 3, k1: float = 1.5, b: float = 0.75) -> str:
    """使用 Okapi BM25 对内置语料进行关键词检索并返回 Top-K 文档片段。"""
    import math

    def tokenize(text: str):
        return [t for t in ''.join(c if c.isalnum() else ' ' for c in text.lower()).split() if t]

    # 构建语料
    corpus = [tokenize(d["title"] + " " + d["content"]) for d in DOCS]
    N = len(corpus)
    doc_lens = [len(doc) for doc in corpus]
    avgdl = (sum(doc_lens) / N) if N > 0 else 0.0

    # 文档频率与 IDF
    df = {}
    for doc in corpus:
        for t in set(doc):
            df[t] = df.get(t, 0) + 1

    idf = {}
    for t, n in df.items():
        # 采用常见的 BM25 IDF 变体，保证数值稳定且非负
        idf[t] = math.log((N - n + 0.5) / (n + 0.5) + 1)

    # 查询词
    q_tokens = tokenize(query)

    # 计算每个文档的 BM25 分数
    scores = []
    for i, doc_tokens in enumerate(corpus):
        dl = doc_lens[i]
        tf = {}
        for t in doc_tokens:
            tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for t in q_tokens:
            if t in tf:
                term_idf = idf.get(t, math.log((N - 0 + 0.5) / (0 + 0.5) + 1))
                denom = tf[t] + k1 * (1 - b + b * (dl / avgdl)) if avgdl > 0 else tf[t] + k1
                score += term_idf * (tf[t] * (k1 + 1)) / denom
        scores.append((score, i))

    scores.sort(key=lambda x: x[0], reverse=True)
    top_indices = [i for s, i in scores[: max(1, min(top_k, N))] if s > 0]
    if not top_indices and scores:
        top_indices = [scores[0][1]]

    lines = ["# BM25 检索结果"]
    for rank, idx in enumerate(top_indices, 1):
        d = DOCS[idx]
        lines.append(f"{rank}. 《{d['title']}》\n   摘要：{d['content']}")
    return "\n".join(lines)


# ============================================================================
# Agent 配置
# ============================================================================

SYSTEM_INSTRUCTIONS = """
你是一个专业的旅行与技术助手，能够：
1. 做预算计算与简单汇率换算
2. 检索知识点并做简要解释
3. 结合用户偏好规划一日游路线
4. 在需要时使用 RAG 从知识库检索并引用来源
5. 输出尽量使用 Markdown，条理清晰，列表与代码块格式化
6. 当用户提到“BM25”或“关键词检索”时，优先调用 BM25 检索工具(rag_search_bm25)
"""


async def create_advanced_agent(
    model_id: str = "gpt-4o-mini",
    debug_mode: bool = False,
) -> Agent:
    return Agent(
        name="MultiDomainAssistant",
        model=OpenAIChat(id=model_id),
        description="旅行规划 + 技术问答的多工具助手（含简易 RAG 与 BM25 关键词检索）",
        instructions=SYSTEM_INSTRUCTIONS,
        tools=[
            calculate,
            search_knowledge,
            currency_convert,
            recommend_places,
            plan_day_trip,
            rag_search,
            rag_search_bm25,
        ],
        read_chat_history=True,
        add_history_to_messages=True,
        num_history_responses=5,
        markdown=True,
        debug_mode=debug_mode,
    )


# ============================================================================
# 演示模式（更复杂场景）
# ============================================================================

async def complex_demo():
    print("=" * 60)
    print("多工具助手 - 复杂场景演示")
    print("=" * 60)

    agent = await create_advanced_agent(debug_mode=False)

    examples = [
        "帮我制定一个北京一日游计划，预算 300 元，优先户外活动。",
        "把预算换算成美元，并重新计算每人费用（3 人）。",
        "查一下 AI Agent 和 Function Calling 的知识点，并总结要点。",
        "给个 Python 例子：读取 CSV 并按列求平均值。",
        "请用列表形式总结上述计划的关键步骤。",
        "使用 RAG：回答‘Agentica 框架能做什么？如何创建一个简单的 Agent？’，并给出引用来源。",
        "使用 BM25：请进行关键词检索，查询‘Agentica 框架’与‘创建 Agent 步骤’相关内容，并引用来源。",
    ]

    for i, question in enumerate(examples, 1):
        print(f"\n{'=' * 60}")
        print(f"示例 {i}: {question}")
        print("-" * 60)

        print("\n助手: ", end="", flush=True)
        full_response = ""
        for delta in agent.run_stream_sync(question):
            full_response += delta.content
            print(delta.content, end="", flush=True)
        print("\n")


# ============================================================================
# 主函数
# ============================================================================

async def main():
    await complex_demo()


if __name__ == "__main__":
    asyncio.run(main())