# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: DeepAgent File Operations Demo

This example demonstrates DeepAgent's built-in file system capabilities:
- ls: List directory contents
- read_file: Read file contents
- write_file: Write to files
- edit_file: Edit existing files
- glob: Find files by pattern
- grep: Search file contents
"""
import sys
import os
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import DeepAgent, OpenAIChat


def file_listing_demo():
    """Demo: List files and directories."""
    print("=" * 60)
    print("Demo 1: File Listing (ls)")
    print("=" * 60)

    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        name="FileExplorer",
        # debug_mode=True,
    )

    response = agent.run("List all Python files in the current directory")
    print(f"\nResponse:\n{response.content}")


def file_read_write_demo():
    """Demo: Read and write files."""
    print("\n" + "=" * 60)
    print("Demo 2: Read and Write Files")
    print("=" * 60)

    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        name="FileManager",
        # debug_mode=True,
    )

    # Create a temporary directory for demo
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.txt")

        # Write a file
        response = agent.run(
            f"Create a file at {test_file} with the content: 'Hello, DeepAgent!'"
        )
        print(f"\nWrite Response:\n{response.content}")

        # Read the file
        response = agent.run(f"Read the contents of {test_file}")
        print(f"\nRead Response:\n{response.content}")


def file_search_demo():
    """Demo: Search files with glob and grep."""
    print("\n" + "=" * 60)
    print("Demo 3: File Search (glob and grep)")
    print("=" * 60)

    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        name="FileSearcher",
        # debug_mode=True,
    )

    # Find Python files
    response = agent.run(
        "Find all Python files in the current directory that contain the word 'import'"
    )
    print(f"\nSearch Response:\n{response.content}")


def file_edit_demo():
    """Demo: Edit existing files."""
    print("\n" + "=" * 60)
    print("Demo 4: Edit Files")
    print("=" * 60)

    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        name="FileEditor",
        # debug_mode=True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "config.py")

        # Create initial file
        with open(test_file, "w") as f:
            f.write("""# Configuration file
DEBUG = False
VERSION = "1.0.0"
MAX_RETRIES = 3
""")

        print(f"Created test file: {test_file}")

        # Edit the file
        response = agent.run(
            f"Edit the file {test_file} to change DEBUG from False to True"
        )
        print(f"\nEdit Response:\n{response.content}")

        # Verify the change
        with open(test_file, "r") as f:
            print(f"\nFile contents after edit:\n{f.read()}")


if __name__ == "__main__":
    file_listing_demo()
    file_read_write_demo()
    file_search_demo()
    file_edit_demo()
