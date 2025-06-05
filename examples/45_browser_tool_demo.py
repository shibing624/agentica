import sys
from pathlib import Path

sys.path.append('..')

from agentica import Agent, OpenAIChat
from agentica.tools.browser_tool import BrowserTool


def main():
    # 创建Agent实例
    agent = Agent(
        model=OpenAIChat(),
        tools=[BrowserTool()],
        output_language="zh"  # 设置输出语言为中文
    )

    # 示例 1：访问新闻网站并获取头条
    print("\n=== 示例: 新浪新闻头条 ===")
    agent.print_response(
        "访问新浪新闻首页(https://news.sina.com.cn)，告诉我现在的头条新闻是什么"
    )

    # 示例 2：获取天气信息
    print("\n=== 示例: 天气信息查询 ===")
    agent.print_response(
        "访问中国天气网(http://www.weather.com.cn)，查看北京今天的天气情况，包括温度、空气质量等信息"
    )


if __name__ == "__main__":
    main()
