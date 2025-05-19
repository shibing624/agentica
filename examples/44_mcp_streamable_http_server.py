#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: MCP StreamableHttp server demo
"""
import requests
from loguru import logger
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Echo Server", host="0.0.0.0", port=8000)


@mcp.tool()
def get_current_weather(city: str) -> str:
    """
    Get current weather for a specified location using wttr.in service.

    Parameters:
        city: Location name, e.g., "Beijing", "New York", "Tokyo", "武汉"
        If None, it will return the weather for the current location.
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
    logger.info(f"get_current_weather({city}) response: {res}")
    return res


@mcp.tool()
def add(a: float, b: float) -> float:
    """
    Add two numbers.
    """
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """
    Subtract two numbers.
    """
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers.
    """
    logger.info(f"multiply({a}, {b})")
    r = a * b
    logger.info(f"multiply({a}, {b}) = {r}")
    return r


@mcp.tool()
def divide(a: float, b: float) -> float:
    """
    Divide two numbers.
    """
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b


if __name__ == "__main__":
    logger.info("Starting MCP Echo Server with StreamableHttp transport on http://localhost:8000/mcp")
    mcp.run(transport="streamable-http")
