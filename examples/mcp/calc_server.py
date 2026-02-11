# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Self-contained MCP server with math, string and datetime tools.

No external API keys required. Used by MCP demo scripts.

Can be run standalone:
    python calc_server.py                    # stdio (default)
    python calc_server.py --transport sse    # SSE on port 8081
    python calc_server.py --transport http   # StreamableHttp on port 8000
"""
import argparse
import math
from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("CalcServer")


# ---------------------------------------------------------------------------
# Math tools
# ---------------------------------------------------------------------------

@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract b from a and return the result."""
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the result."""
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide a by b. Raises error if b is zero."""
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b


@mcp.tool()
def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent."""
    return math.pow(base, exponent)


@mcp.tool()
def sqrt(number: float) -> float:
    """Calculate the square root of a number."""
    if number < 0:
        raise ValueError("Cannot compute square root of a negative number.")
    return math.sqrt(number)


# ---------------------------------------------------------------------------
# String tools
# ---------------------------------------------------------------------------

@mcp.tool()
def string_length(text: str) -> int:
    """Return the length of a string."""
    return len(text)


@mcp.tool()
def string_reverse(text: str) -> str:
    """Reverse a string."""
    return text[::-1]


@mcp.tool()
def string_upper(text: str) -> str:
    """Convert a string to uppercase."""
    return text.upper()


@mcp.tool()
def word_count(text: str) -> int:
    """Count the number of words in a string."""
    return len(text.split())


# ---------------------------------------------------------------------------
# Datetime tools
# ---------------------------------------------------------------------------

@mcp.tool()
def current_time() -> str:
    """Return the current date and time in ISO 8601 format (UTC)."""
    return datetime.now(timezone.utc).isoformat()


@mcp.tool()
def timestamp_to_date(timestamp: float) -> str:
    """Convert a Unix timestamp to a human-readable date string (UTC)."""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CalcServer MCP Server")
    parser.add_argument(
        "--transport", choices=["stdio", "sse", "http"], default="stdio",
        help="Transport type (default: stdio)",
    )
    args = parser.parse_args()

    if args.transport == "sse":
        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = 8081
        print("Starting CalcServer on http://localhost:8081/sse")
        mcp.run(transport="sse")
    elif args.transport == "http":
        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = 8000
        print("Starting CalcServer on http://localhost:8000/mcp")
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")
