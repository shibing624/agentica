# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 简化的 Agent 应用示例 - 演示核心功能组合

使用方法:
    python simple_chat.py

功能演示:
1. 自定义提示词
2. 多轮对话
3. 工具调用（自动）
4. 流式输出
5. 演示模式（自动运行示例问题）
"""
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat


# ============================================================================
# 工具定义
# ============================================================================

def get_weather(location: str) -> str:
    """获取指定城市的天气信息。

    Args:
        location: 城市名称，如 "北京"、"上海"

    Returns:
        天气信息的字符串描述
    """
    weather_data = {
        "北京": {"temp": 15, "condition": "晴天", "humidity": 45},
        "上海": {"temp": 18, "condition": "多云", "humidity": 65},
        "深圳": {"temp": 25, "condition": "小雨", "humidity": 80},
    }
    data = weather_data.get(location, {"temp": 20, "condition": "晴天", "humidity": 50})
    return f"{location}天气：{data['condition']}，温度 {data['temp']}°C，湿度 {data['humidity']}%"


def calculate(expression: str) -> str:
    """计算数学表达式。

    Args:
        expression: 数学表达式，如 "2 + 3 * 4"

    Returns:
        计算结果
    """
    try:
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return f"错误：表达式包含非法字符"
        result = eval(expression)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"


def search_knowledge(query: str) -> str:
    """搜索知识库获取信息。

    Args:
        query: 搜索查询关键词

    Returns:
        搜索结果
    """
    knowledge = {
        "python": "Python 是一种高级编程语言，以简洁易读著称。最新稳定版本是 3.12。",
        "ai agent": "AI Agent 是能够自主执行任务的智能系统，通常包含感知、决策、执行三个核心模块。",
        "function calling": "Function Calling 允许 LLM 调用外部函数，实现与真实世界的交互。",
    }
    results = []
    for key, value in knowledge.items():
        if query.lower() in key.lower():
            results.append(f"- {key}: {value}")
    if results:
        return "\n".join(results)
    return f"未找到关于 '{query}' 的相关信息"


# ============================================================================
# Agent 配置
# ============================================================================

# 系统提示词 - 定义 Agent 行为
SYSTEM_INSTRUCTIONS = """
你是一个专业的 Python 技术助手。
你的职责是：
1. 回答 Python 编程相关问题
2. 提供代码示例
3. 解释技术概念
4. 帮助调试代码

请保持回答简洁、准确、有帮助。
"""


async def create_assistant_agent(
    model_id: str = "gpt-4o-mini",
    debug_mode: bool = False,
) -> Agent:
    """创建助手 Agent

    Args:
        model_id: 模型 ID
        debug_mode: 是否开启调试模式

    Returns:
        Agent 实例
    """
    return Agent(
        name="PythonAssistant",
        model=OpenAIChat(id=model_id),
        description="专业的 Python 技术助手",
        instructions=SYSTEM_INSTRUCTIONS,
        # 启用工具调用
        tools=[get_weather, calculate, search_knowledge],
        # 启用多轮对话
        read_chat_history=True,
        add_history_to_messages=True,
        num_history_responses=3,
        # 启用流式输出
        markdown=True,
        debug_mode=debug_mode,
    )


# ============================================================================
# 演示模式
# ============================================================================

async def demo_mode():
    """演示模式 - 自动运行示例问题"""
    print("=" * 60)
    print("Python 技术助手 - 演示模式")
    print("=" * 60)

    agent = await create_assistant_agent(debug_mode=False)

    # 示例问题列表
    examples = [
        "如何用 Python 读取 CSV 文件？",
        "解释一下 Python 的装饰器是什么",
        "给我写一个快速排序的代码",
        "什么是异步编程？",
        "查询北京的天气",
        "计算 2 + 3 * 4",
        "搜索关于 AI Agent 的信息",
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
    """主函数"""
    await demo_mode()


if __name__ == "__main__":
    asyncio.run(main())