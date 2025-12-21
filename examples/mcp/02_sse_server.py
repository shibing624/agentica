# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: MCP SSE server demo - Creates an MCP server with SSE transport

Run this server first, then use 02_sse_client.py to connect.

Usage:
    python 02_sse_server.py
"""
import requests
from loguru import logger
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Echo Server", host="0.0.0.0", port=8081)


@mcp.tool()
def get_current_weather(city: str) -> str:
    """Get current weather for a specified location.

    Parameters:
        city: Location name, e.g., "Beijing", "New York", "Tokyo"
    Returns:
        str: Current weather information in plain text format.
    """
    logger.info(f"get_current_weather({city})")
    try:
        endpoint = "https://wttr.in"
        response = requests.get(f"{endpoint}/{city}")
        res = response.text
    except Exception as e:
        res = f"city: {city}, error: {str(e)}"
    return res


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    logger.info(f"multiply({a}, {b})")
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b


if __name__ == "__main__":
    logger.info("Starting MCP Echo Server with SSE transport on http://localhost:8081/sse")
    mcp.run(transport="sse")
