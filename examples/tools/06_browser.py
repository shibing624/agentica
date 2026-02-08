# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Browser tool demo - Demonstrates web browsing capabilities

This example shows how to use browser tools for web interaction:
1. BrowserTool - Web page navigation and interaction
2. UrlCrawlerTool - URL content extraction
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat
from agentica.tools.browser_tool import BrowserTool


def main():
    # Create agent with browser tool
    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=[BrowserTool()],
        add_datetime_to_instructions=True,
    )

    # Example 1: Visit a webpage
    print("=" * 60)
    print("Example 1: Visit Webpage")
    print("=" * 60)
    agent.print_response(
        "访问 https://github.com/shibing624/agentica 并总结这个项目的主要功能"
    )

    # Example 2: Search and browse
    print("\n" + "=" * 60)
    print("Example 2: Search and Browse")
    print("=" * 60)
    agent.print_response(
        "搜索Python最新版本的发布说明，并总结主要更新内容"
    )


if __name__ == "__main__":
    main()
