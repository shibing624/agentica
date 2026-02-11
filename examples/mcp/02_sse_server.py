# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: MCP SSE server demo - Starts the CalcServer with SSE transport

Run this server first, then use 02_sse_client.py to connect.

Usage:
    python 02_sse_server.py
"""
import subprocess
import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SERVER_SCRIPT = os.path.join(_SCRIPT_DIR, "calc_server.py")

if __name__ == "__main__":
    print("Starting CalcServer with SSE transport on http://localhost:8081/sse")
    print("Press Ctrl+C to stop.\n")
    subprocess.run([sys.executable, _SERVER_SCRIPT, "--transport", "sse"])
