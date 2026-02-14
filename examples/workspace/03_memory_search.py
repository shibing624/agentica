# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Workspace memory search example

This example shows how to:
1. Index workspace memory files
2. Search through memories
3. Get relevant context for queries
"""
import asyncio
import os
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica.workspace import Workspace
from agentica.memory import WorkspaceMemorySearch


async def main():
    # Create a temporary workspace
    temp_dir = tempfile.mkdtemp()
    workspace_path = Path(temp_dir) / "workspace"

    print(f"Creating workspace at: {workspace_path}")

    # Initialize workspace
    workspace = Workspace(workspace_path)
    workspace.initialize()

    # Add some memories
    await workspace.write_memory("""# Project Notes

## Python Project Ideas
- Build a web scraper using Beautiful Soup
- Create a REST API with FastAPI
- Implement a chatbot using LLM

## Learning Goals
- Master async programming in Python
- Learn about vector databases
- Study machine learning fundamentals
""", to_daily=False)

    await workspace.write_memory("""# Daily Notes

## Today's Progress
- Finished the FastAPI tutorial
- Started learning about embeddings
- Had a meeting about the new project deadline
""", to_daily=True)

    # Create memory search
    search = WorkspaceMemorySearch(workspace_path=str(workspace_path))

    # Index the memories
    print("\n=== Indexing Memories ===")
    num_chunks = search.index()
    print(f"Indexed {num_chunks} chunks")

    # Search memories
    print("\n=== Searching for 'Python project' ===")
    results = search.search("Python project", limit=3)
    for i, result in enumerate(results):
        print(f"\nResult {i+1} (score: {result.score:.2f}):")
        print(f"  File: {result.file_path}")
        print(f"  Lines: {result.start_line}-{result.end_line}")
        print(f"  Content preview: {result.content[:100]}...")

    # Search for another topic
    print("\n=== Searching for 'FastAPI' ===")
    results = search.search("FastAPI learning", limit=3)
    for i, result in enumerate(results):
        print(f"\nResult {i+1} (score: {result.score:.2f}):")
        print(f"  File: {result.file_path}")
        print(f"  Content preview: {result.content[:100]}...")

    # Get context for a query
    print("\n=== Getting Context for 'machine learning' ===")
    context = search.get_context("machine learning", max_chars=1000)
    print(context)

    # Clean up
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
