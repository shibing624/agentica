# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Kimi for Coding model demo

Kimi for Coding uses Anthropic Claude API protocol.
Set KIMI_API_KEY environment variable before running.

Usage:
    export KIMI_API_KEY=your_kimi_api_key
    python kimi_coding.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, KimiChat

agent = Agent(
    model=KimiChat(id="k2p5"), # name for kimi-for-coding model
)

# Streaming output
agent.print_response_stream_sync("write a quick sort in Python")
