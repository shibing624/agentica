# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tools module exports

Available tools:
- Base: Tool base class
- CodeTool: Code analysis, formatting, and linting
- LspTool: LSP-based code navigation (goto definition, find references)
- PatchTool: Apply diff/patch files (V4A and unified diff formats)
"""
from agentica.tools.base import Tool, Function, FunctionCall
from agentica.tools.code_tool import CodeTool
from agentica.tools.lsp_tool import LspTool
from agentica.tools.patch_tool import PatchTool

__all__ = [
    # Base classes
    "Tool",
    "Function",
    "FunctionCall",
    # Gain layer tools (built on top of deep_tools.py)
    "CodeTool",      # Code analysis, formatting, linting
    "LspTool",       # LSP-based code navigation
    "PatchTool",     # Apply diff/patch files
]
