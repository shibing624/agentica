#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: MCP SSE client demo

使用说明:
1. 首先启动服务器:
    python examples/42_mcp_sse_server.py
2. 然后运行此客户端:
    python examples/42_mcp_sse_client.py
"""

import asyncio
import sys

sys.path.append('..')
from agentica import Agent, OpenAIChat, logger, ShellTool
from agentica.mcp.server import MCPServerSse
from agentica.mcp.client import MCPClient
from agentica.tools.mcp_tool import McpTool


async def sse_server_demo() -> None:
    """
    演示如何使用基于SSE的MCP服务器
    """
    print("\n=== 测试基于SSE的MCP服务器(直接连接) ===")

    async with MCPServerSse(
            name="SSE Python Server",
            params={
                "url": "http://localhost:8081/sse",
            }
    ) as server:
        try:
            async with MCPClient(server=server) as client:
                # 列出可用工具
                tools = await client.list_tools()
                tool_names = [tool.name for tool in tools]
                logger.debug(f"可用工具: {tool_names}")

                city = "安陆市"
                result = await client.call_tool("get_current_weather", {"city": city})
                weather_result = client.extract_result_text(result)
                logger.info(f"{city}天气 = {weather_result}")
        except Exception as e:
            logger.error(f"SSE服务器演示中出错: {e}")
            sys.exit(1)


async def mcp_toolkit_with_agent_demo() -> None:
    """
    演示如何将MCPToolkit与代理一起使用(通过SSE)
    """
    print("\n=== 测试MCPToolkit与SSE和代理的结合 ===")
    try:
        # 使用直接的sse_server_url参数而不是环境变量
        # 这展示了McpTool类中的SSE支持
        mcp_tool = McpTool(
            url="http://localhost:8081/sse",  # 直接使用参数而非环境变量
            sse_timeout=5.0,  # HTTP请求超时(秒)
            sse_read_timeout=300.0  # SSE连接超时(秒)
        )

        async with mcp_tool:
            agent = Agent(
                model=OpenAIChat(model="gpt-4o"),
                tools=[ShellTool(), mcp_tool],
                show_tool_calls=False,
                add_datetime_to_instructions=True,
                debug_mode=False
            )

            # 显示代理可用的工具
            print("代理可用的工具:")
            for tool in agent.get_tools():
                print(f" - {tool.name}: {list(tool.functions.keys())}")

            # 测试代理使用工具
            print("\n测试代理使用天气工具:")
            await agent.aprint_response("查询北京市今天的气温，并用温度的值乘以 314159.14=？")
            # 预期结果类似: 北京市今天的气温为12°C。将这个温度值乘以314159.14得到的结果是：3769909.68。
    except Exception as e:
        logger.error(f"MCPToolkit与代理演示中出错: {e}")
        sys.exit(1)


async def main() -> None:
    """
    运行所有示例的主函数
    """
    print("MCP SSE 客户端演示")
    print("===================")

    await sse_server_demo()
    await mcp_toolkit_with_agent_demo()


if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main())
