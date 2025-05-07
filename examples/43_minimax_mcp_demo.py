# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: minimax-mcp demo

pip install minimax-mcp weather-forecast-server
"""

import asyncio
import sys

sys.path.append('..')
from agentica import Agent, OpenAIChat
from agentica.tools.mcp_tool import McpTool
from agentica import ShellTool


async def mcp_toolkit_with_agent_demo():
    """
    Demonstrates how to use the MCPToolkit with an agent.
    
    This is a more advanced example that shows how to integrate MCP tools with an agent.
    The agent part is commented out as it depends on your agent implementation.
    """
    try:
        async with McpTool.from_config() as mcp_tool1:
            m = Agent(
                model=OpenAIChat(model="gpt-4o"),
                tools=[ShellTool(), mcp_tool1],
                show_tool_calls=True,
                add_datetime_to_instructions=True,
                debug=True,
            )
            r = m.get_tools()
            print(r)
            for i in r:
                print(i.name, i.functions)

            await m.aprint_response("调天气工具 get_weather 查询合肥市天气咋样")
            await m.aprint_response("我爱中国，转为英文，并保存音频")
            await m.aprint_response("现在几点了")
    except Exception as e:
        print(f"Error in MCPToolkit with agent demo: {e}")
        sys.exit(1)


async def main():
    await mcp_toolkit_with_agent_demo()


if __name__ == "__main__":
    asyncio.run(main())
