# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: CLI demo - Demonstrates various ways to use the agentica CLI

This demo shows:
1. Running CLI interactively (default mode)
2. Running CLI with a single query (non-interactive)
3. Running CLI with extra tools
4. Running CLI with custom model settings

Usage Examples:
    # Interactive mode (default)
    python 01_cli_demo.py

    # Single query mode
    python 01_cli_demo.py --query "What is 2+2?"

    # With extra tools
    python 01_cli_demo.py --tools calculator wikipedia --verbose 1

    # With custom model
    python 01_cli_demo.py --model_provider deepseek --model_name deepseek-chat

    # With working directory
    python 01_cli_demo.py --work_dir /path/to/project

Interactive Features:
    Enter           Submit your message
    Alt+Enter       Insert newline for multi-line
    Ctrl+J          Insert newline (alternative)
    @filename       Type @ to auto-complete files
    /command        Type / to see available commands

Interactive Commands:
    /help           Show available commands
    /clear          Clear screen and reset conversation
    /tools          List available additional tools
    /exit, /quit    Exit the CLI

DeepAgent Built-in Tools:
    - ls: List directory contents
    - read_file: Read file content
    - write_file: Write file content
    - edit_file: Edit file
    - glob: File pattern matching
    - grep: Search file content
    - execute: Execute Python code
    - web_search: Web search
    - fetch_url: Fetch URL content
    - write_todos: Create task list
    - read_todos: Read task list
    - task: Launch subagent

Available Extra Tools (use --tools to enable):
    airflow, arxiv, baidu_search, calculator, code, dalle,
    duckduckgo, file, hackernews, jina, mcp, newspaper,
    ocr, run_python_code, search_serper, shell, skill,
    weather, wikipedia, yfinance
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica.cli import main

if __name__ == "__main__":
    main()
