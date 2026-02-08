# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Function Calling with Auto Demo

演示两种 Agent Loop 实现方式的对比：

1. **手动 Loop** (传统方式)
   - 循环在你的代码中
   - 完全可控
   - 需要处理更多细节

2. **Agentica Agent** (框架封装)
   - 框架自动处理工具调用循环
   - 通过 ToolCallStarted/ToolCallCompleted 事件输出执行过程
   - 支持流式和非流式两种模式

运行方式:
    python examples/basic/07_function_calling_auto_demo.py
"""
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
    for key, value in knowledge.items():
        if key in query.lower():
            return f"找到相关信息：{value}"
    return f"未找到关于 '{query}' 的信息，建议使用更具体的关键词搜索。"


# 工具定义（OpenAI 格式，用于手动 Loop）
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名称，如 '北京'、'上海'"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 '2 + 3 * 4'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "搜索知识库获取信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询关键词"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_knowledge": search_knowledge,
}


# ============================================================================
# 方式 1：手动 Loop（传统方式，完全可控）
# ============================================================================

def demo_manual_loop():
    """
    手动实现 agentic loop。

    特点：
    - 完全控制循环逻辑
    - 可以自定义终止条件、重试策略、错误处理
    - 适合需要深度定制的场景
    """
    print("\n" + "=" * 70)
    print("方式 1: 手动 Agentic Loop (完全可控)")
    print("=" * 70)

    try:
        from openai import OpenAI
        client = OpenAI()

        query = "查询北京和上海的天气，然后计算两地温差"
        print(f"\n用户查询: {query}")
        print("-" * 70)

        messages = [{"role": "user", "content": query}]
        max_iterations = 10

        for iteration in range(1, max_iterations + 1):
            print(f"\n第 {iteration} 轮对话...")

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )

            assistant_message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            if finish_reason == "tool_calls" and assistant_message.tool_calls:
                print(f"   模型请求调用 {len(assistant_message.tool_calls)} 个工具")

                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
                })

                for tool_call in assistant_message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)

                    print(f"   执行工具: {func_name}({func_args})")

                    if func_name in TOOL_FUNCTIONS:
                        result = TOOL_FUNCTIONS[func_name](**func_args)
                    else:
                        result = f"未知工具: {func_name}"

                    print(f"   工具结果: {result}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
            else:
                print(f"\n最终响应 (第 {iteration} 轮):")
                print(f"   {assistant_message.content}")
                break
        else:
            print(f"\n达到最大迭代次数 ({max_iterations})，强制终止")

    except ImportError:
        print("请安装 openai: pip install openai")
    except Exception as e:
        print(f"调用失败: {e}")


# ============================================================================
# 方式 2：使用 Agentica Agent（框架级封装）
# ============================================================================

def demo_agentica_agent():
    """
    使用 Agentica 框架的 Agent，内置了完整的 agentic loop。

    特点：
    - 通过 ToolCallStarted/ToolCallCompleted 事件输出工具调用过程和结果
    - 支持流式和非流式输出
    - 无需手动处理事件类型
    """
    print("\n" + "=" * 70)
    print("方式 2: Agentica Agent (框架自动输出执行过程)")
    print("=" * 70)

    try:
        from agentica import Agent, OpenAIChat

        # 创建 Agent
        agent = Agent(
            model=OpenAIChat(id='gpt-4o-mini'),
            tools=[get_weather, calculate, search_knowledge],
        )

        query = "查询北京和上海的天气，然后计算两地温差"
        print(f"\n用户查询: {query}")
        print("-" * 70)

        # 非流式调用 - 最简单的方式
        print("\n【非流式输出】")
        response = agent.run(query)
        print(response.content)

        # 流式调用 - 边执行边输出
        print("\n" + "-" * 70)
        print("\n【流式输出】")
        for chunk in agent.run("查询深圳天气", stream=True):
            if chunk.content:
                print(chunk.content, end="", flush=True)
        print()

    except Exception as e:
        import traceback
        print(f"Agentica Agent 调用失败: {e}")
        traceback.print_exc()


# ============================================================================
# 对比总结
# ============================================================================

def print_comparison():
    """打印两种方式的对比总结"""
    print("\n" + "=" * 70)
    print("两种实现方式对比")
    print("=" * 70)

    comparison = """
┌─────────────────┬─────────────────────┬─────────────────────┐
│      特性        │     手动 Loop        │   Agentica Agent    │
├─────────────────┼─────────────────────┼─────────────────────┤
│ 循环位置         │ 你的代码中           │ 框架内部             │
│ 代码复杂度       │ 高（需要处理细节）    │ 低（一行代码）       │
│ 可控性          │ 完全可控             │ 可配置扩展           │
│ 工具执行输出     │ 需要自己实现         │ 事件驱动             │
│ 流式输出        │ 可自定义             │ stream=True          │
│ 错误处理        │ 需要自己实现         │ 框架处理             │
│ 多提供商支持     │ 需要适配             │ 20+ 提供商           │
│ 适用场景        │ 深度定制             │ 生产环境             │
└─────────────────┴─────────────────────┴─────────────────────┘

选择建议：
   - 需要完全控制 → 手动 Loop
   - 快速开发     → Agentica Agent (stream_intermediate_steps=True)
"""
    print(comparison)


# ============================================================================
# Main
# ============================================================================

def main():
    """运行所有演示"""
    print("Function Calling with Auto - 两种实现方式对比演示")
    print("=" * 70)

    # 方式 1: 手动 Loop
    demo_manual_loop()

    # 方式 2: Agentica Agent
    demo_agentica_agent()

    # 对比总结
    print_comparison()


if __name__ == "__main__":
    main()
