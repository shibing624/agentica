# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: MCP StreamableHttp server demo - Starts CalcServer with HTTP streaming

Run this server first, then use 03_http_client.py to connect.

Usage:
    python 03_http_server.py
"""
import subprocess
import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SERVER_SCRIPT = os.path.join(_SCRIPT_DIR, "calc_server.py")

if __name__ == "__main__":
    print("Starting CalcServer with StreamableHttp transport on http://localhost:8000/mcp")
    print("Press Ctrl+C to stop.\n")
    subprocess.run([sys.executable, _SERVER_SCRIPT, "--transport", "http"])
