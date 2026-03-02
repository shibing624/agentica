# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Kimi for Coding model demo with extended thinking

Kimi for Coding uses Anthropic Claude API protocol.
Set KIMI_API_KEY environment variable before running.

Usage:
    export KIMI_API_KEY=your_kimi_api_key
    python 06_kimi_coding.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, KimiChat

# Part 1: Basic streaming output
print("=" * 60)
print("Part 1: Basic streaming output")
print("=" * 60)

agent = Agent(
    model=KimiChat(id="k2p5"),
)
agent.print_response_stream_sync("write a quick sort in Python")

# Part 2: Enable extended thinking (Anthropic-compatible thinking protocol)
print("\n" + "=" * 60)
print("Part 2: Extended thinking enabled")
print("=" * 60)

agent_thinking = Agent(
    model=KimiChat(
        id="k2p5",
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": 8000},
    ),
)
agent_thinking.print_response_stream_sync("Prove that there are infinitely many prime numbers")