# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Web search demo - Demonstrates using search tools for web queries

This example shows how to use different search tools:
2. BaiduSearchTool (Baidu search)
3. DuckDuckGoTool (DuckDuckGo search)
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat, WeatherTool
from agentica.tools.baidu_search_tool import BaiduSearchTool


def main():
    print("=" * 60)
    print("Example 1: Web Search with Baidu Search(free)")
    print("=" * 60)
    
    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=[BaiduSearchTool(), WeatherTool()],
        add_datetime_to_instructions=True,
        read_chat_history=True,
    )
    
    # Simple search
    response = agent.run("一句话介绍林黛玉")
    print(response)
    
    # Weather-related search
    response = agent.run("上海今天适合穿什么衣服")
    print(response)
    
    # Context-aware follow-up
    response = agent.run("总结前面的问答")
    print(response)

    # Example 2: Research task
    print("\n" + "=" * 60)
    print("Example 2: Research Task")
    print("=" * 60)
    
    agent.print_response(
        "搜索最新的人工智能发展趋势，总结3个关键点",
        stream=True
    )


if __name__ == "__main__":
    main()
