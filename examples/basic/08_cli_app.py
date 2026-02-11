# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: CLI App demo - Demonstrates using agent.cli_app() for quick interactive testing

This example shows how to use the simple built-in CLI for quick agent testing.

For full-featured CLI with tools, skills, and session management, use:
    from agentica.cli import main

Usage:
    python 08_cli_app.py

Type 'exit', 'quit', or 'bye' to exit.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat


def main():
    # Create an agent
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        description="You are a helpful assistant.",
    )

    # Start interactive CLI (streaming by default for better experience)
    print("=" * 60)
    print("Agent CLI - Type your message and press Enter")
    print("Commands: exit, quit, bye - to exit")
    print("=" * 60)
    print()

    # Start CLI with streaming (default)
    agent.cli_app()


if __name__ == "__main__":
    main()
