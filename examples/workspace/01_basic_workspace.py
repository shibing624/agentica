# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Basic workspace usage example

This example shows how to:
1. Create and initialize a workspace
2. Create an agent from a workspace
3. Use workspace memory
"""
import os
import sys
import asyncio
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, ZhipuAI
from agentica.workspace import Workspace


async def main():
    # Create a temporary workspace for demo
    temp_dir = tempfile.mkdtemp()
    workspace_path = Path(temp_dir) / "my_workspace"

    print(f"Creating workspace at: {workspace_path}")

    # Method 1: Create workspace manually
    workspace = Workspace(workspace_path)
    workspace.initialize()

    # Customize workspace files
    workspace.write_file("USER.md", """# User Profile

## Preferences
- Language: 中文
- Style: Concise and technical
- Focus: Python programming

## Context
Software developer working on AI projects.
""")

    workspace.write_file("AGENT.md", """# Agent Instructions

You are a helpful AI assistant specialized in Python programming.

## Guidelines
1. Provide concise, accurate answers
2. Include code examples when helpful
3. Use Chinese for explanations
4. Store important user preferences in memory
""")

    # Create agent from workspace
    agent = Agent(
        model=ZhipuAI(model="glm-4-flash"),
        workspace=workspace,
        load_workspace_context=True,
        load_workspace_memory=True,
    )

    # Run agent
    response = await agent.run("用 Python 写一个快速排序算法")
    print("\n=== Response ===")
    print(response.content)

    # Save memory to workspace
    await agent.save_memory("User asked about quicksort algorithm in Python")

    # Verify memory was saved
    print("\n=== Workspace Memory ===")
    memory_content = await workspace.get_memory_prompt(days=1)
    print(memory_content)

    # Method 2: Create agent from workspace using factory method
    print("\n=== Creating agent from workspace (factory method) ===")
    agent2 = Agent.from_workspace(
        workspace_path=str(workspace_path),
        model=ZhipuAI(model="glm-4-flash"),
    )
    response2 = await agent2.run("你能帮我做什么？")
    print(response2.content)

    # Clean up
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
