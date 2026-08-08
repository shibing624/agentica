# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: SandboxConfig demo - Best-effort execution isolation

Demonstrates how SandboxConfig restricts file operations and command execution:
1. Blocked paths: prevent access to sensitive directories (.ssh, .aws, etc.)
2. Writable dirs whitelist: restrict file writes to specific directories
3. Blocked commands: intercept dangerous shell commands (rm -rf /, mkfs, etc.)
4. Execution timeout: limit command execution time

NOTE: SandboxConfig is best-effort, NOT a true security sandbox.
For untrusted code, use OS-level sandboxing (Docker, seccomp, etc.).

Usage:
    python examples/tools/10_sandbox_config.py
"""
import sys
import os
import asyncio
import tempfile
import shutil
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat, SandboxConfig
from agentica.tools.builtin import BuiltinFileTool, BuiltinExecuteTool


# ---------------------------------------------------------------------------
# 1. Direct tool-level sandbox demo (no LLM needed)
# ---------------------------------------------------------------------------

async def demo_sandbox_file_tool():
    """Demonstrate SandboxConfig on BuiltinFileTool: path blocking and write restriction."""
    print("=" * 60)
    print("Demo 1: SandboxConfig + BuiltinFileTool")
    print("=" * 60)

    work_dir = tempfile.mkdtemp()
    allowed_dir = os.path.join(work_dir, "allowed")
    os.makedirs(allowed_dir, exist_ok=True)

    sandbox = SandboxConfig(
        enabled=True,
        writable_dirs=[allowed_dir],
        blocked_paths=[".ssh", ".aws", "id_rsa"],
    )

    file_tool = BuiltinFileTool(work_dir=work_dir, sandbox_config=sandbox)

    # Test 1: Write to allowed directory
    print("\n[Test 1] Write to allowed directory:")
    try:
        result = await file_tool.write_file(
            file_path=os.path.join(allowed_dir, "hello.txt"),
            content="Hello from sandbox!",
        )
        print(f"  OK: {result}")
    except PermissionError as e:
        print(f"  BLOCKED: {e}")

    # Test 2: Write outside allowed directory
    print("\n[Test 2] Write outside allowed directory:")
    try:
        result = await file_tool.write_file(
            file_path=os.path.join(work_dir, "outside.txt"),
            content="Should be blocked",
        )
        print(f"  OK (unexpected): {result}")
    except PermissionError as e:
        print(f"  BLOCKED (expected): {e}")

    # Test 3: Access blocked path (.ssh)
    print("\n[Test 3] Access blocked path (.ssh):")
    try:
        result = await file_tool.read_file(file_path=os.path.expanduser("~/.ssh/config"))
        print(f"  OK (unexpected): read {len(result)} chars")
    except PermissionError as e:
        print(f"  BLOCKED (expected): {e}")
    except FileNotFoundError:
        print(f"  File not found (path validation passed, file doesn't exist)")

    # Test 4: Read from work_dir (allowed)
    print("\n[Test 4] Read from allowed directory:")
    try:
        result = await file_tool.read_file(
            file_path=os.path.join(allowed_dir, "hello.txt"),
        )
        print(f"  OK: {result[:50]}")
    except Exception as e:
        print(f"  Error: {e}")

    shutil.rmtree(work_dir, ignore_errors=True)


async def demo_sandbox_execute_tool():
    """Demonstrate SandboxConfig on BuiltinExecuteTool: command blocking."""
    print("\n" + "=" * 60)
    print("Demo 2: SandboxConfig + BuiltinExecuteTool")
    print("=" * 60)

    sandbox = SandboxConfig(
        enabled=True,
        blocked_commands=["rm -rf /", "mkfs", "dd if=", ":(){ :|:& };:"],
        max_execution_time=10,
    )

    exec_tool = BuiltinExecuteTool(work_dir=tempfile.mkdtemp(), sandbox_config=sandbox)

    # Test 1: Safe command
    print("\n[Test 1] Safe command (echo):")
    result = await exec_tool.execute(command="echo 'Hello from sandbox!'")
    print(f"  Output: {result[:100]}")

    # Test 2: Blocked command
    print("\n[Test 2] Blocked command (rm -rf /):")
    result = await exec_tool.execute(command="rm -rf /")
    print(f"  Result: {result[:200]}")

    # Test 3: Another safe command
    print("\n[Test 3] Safe command (ls):")
    result = await exec_tool.execute(command="ls -la /tmp | head -5")
    print(f"  Output: {result[:200]}")

    # Test 4: Timeout enforcement
    print(f"\n[Test 4] Execution timeout: {sandbox.max_execution_time}s")
    print(f"  (timeout is enforced but we skip the slow test for demo speed)")


async def demo_sandbox_allowed_commands():
    """Demonstrate allowed_commands whitelist (new feature).

    When allowed_commands is set, ONLY commands whose first token matches
    one of the allowed prefixes are permitted. Everything else is blocked.
    This is useful for AI coding agents where you want to restrict execution
    to a known safe set (e.g. python, pytest, git).
    """
    print("\n" + "=" * 60)
    print("Demo 2b: SandboxConfig.allowed_commands whitelist")
    print("=" * 60)

    sandbox = SandboxConfig(
        enabled=True,
        allowed_commands=["python", "pytest", "echo"],  # only these prefixes allowed
    )

    exec_tool = BuiltinExecuteTool(work_dir=tempfile.mkdtemp(), sandbox_config=sandbox)

    # Test 1: Allowed command — python
    print("\n[Test 1] Allowed command (python3 --version):")
    result = await exec_tool.execute(command="python3 --version")
    print(f"  Output: {result.strip()}")

    # Test 2: Allowed command — echo
    print("\n[Test 2] Allowed command (echo):")
    result = await exec_tool.execute(command="echo hello")
    print(f"  Output: {result.strip()}")

    # Test 3: Blocked by whitelist — ls
    print("\n[Test 3] Blocked command (ls — not in whitelist):")
    result = await exec_tool.execute(command="ls /tmp")
    print(f"  Result: {result[:200]}")

    # Test 4: Blocked by whitelist — curl
    print("\n[Test 4] Blocked command (curl — not in whitelist):")
    result = await exec_tool.execute(command="curl https://example.com")
    print(f"  Result: {result[:200]}")

    # Test 5: Absolute path is normalized — /usr/bin/python3 → python3
    print("\n[Test 5] Absolute path normalized (/usr/bin/python3 → python3, allowed):")
    result = await exec_tool.execute(command="/usr/bin/python3 -c 'print(1+1)'")
    print(f"  Output: {result.strip()}")

    print(f"\n  allowed_commands={sandbox.allowed_commands}")
    print("  None = no whitelist restriction (default behavior)")
    print("  list = only commands with matching first-token prefix are permitted")


# ---------------------------------------------------------------------------
# 2. Agent-level sandbox demo (with LLM)
# ---------------------------------------------------------------------------

async def demo_agent_with_sandbox():
    """Demonstrate Agent with SandboxConfig: Agent passes sandbox to its tools."""
    print("\n" + "=" * 60)
    print("Demo 3: Agent with SandboxConfig")
    print("=" * 60)

    work_dir = tempfile.mkdtemp()
    sandbox = SandboxConfig(
        enabled=True,
        writable_dirs=[work_dir],
        blocked_paths=[".ssh", ".aws", ".gnupg"],
        blocked_commands=["rm -rf /", "mkfs"],
        max_execution_time=30,
    )

    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        sandbox_config=sandbox,
    )

    print(f"  Agent sandbox enabled: {agent.sandbox_config.enabled}")
    print(f"  Writable dirs: {agent.sandbox_config.writable_dirs}")
    print(f"  Blocked paths: {agent.sandbox_config.blocked_paths[:3]}...")
    print(f"  Blocked commands: {agent.sandbox_config.blocked_commands[:2]}...")
    print(f"  Max execution time: {agent.sandbox_config.max_execution_time}s")

    # The agent will use sandbox-restricted tools when executing tasks
    print(f"\n  Agent is configured with sandbox restrictions.")
    print(f"  When the agent calls file/execute tools, sandbox rules are enforced.")

    shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 3. Custom sandbox config
# ---------------------------------------------------------------------------

def demo_custom_sandbox():
    """Show how to create a custom SandboxConfig for different use cases."""
    print("\n" + "=" * 60)
    print("Demo 4: Custom SandboxConfig Examples")
    print("=" * 60)

    # Restrictive sandbox for untrusted agents
    restrictive = SandboxConfig(
        enabled=True,
        writable_dirs=["/tmp/agent_output"],
        blocked_paths=[".ssh", ".gnupg", ".aws", ".azure", ".config", ".env", ".netrc"],
        blocked_commands=[
            "rm -rf /", "rm -rf /*", "mkfs", "dd if=",
            ":(){ :|:& };:", "chmod -R 777 /",
            "curl|sh", "wget|sh",
        ],
        max_execution_time=30,
    )
    print(f"\n  Restrictive sandbox:")
    print(f"    Writable: {restrictive.writable_dirs}")
    print(f"    Blocked paths: {len(restrictive.blocked_paths)} patterns")
    print(f"    Blocked commands: {len(restrictive.blocked_commands)} patterns")
    print(f"    Timeout: {restrictive.max_execution_time}s")

    # Relaxed sandbox for trusted internal agents
    relaxed = SandboxConfig(
        enabled=True,
        writable_dirs=[os.path.expanduser("~/projects")],
        blocked_paths=[".ssh", "id_rsa"],
        blocked_commands=["rm -rf /", "mkfs"],
        max_execution_time=300,
    )
    print(f"\n  Relaxed sandbox:")
    print(f"    Writable: {relaxed.writable_dirs}")
    print(f"    Blocked paths: {len(relaxed.blocked_paths)} patterns")
    print(f"    Timeout: {relaxed.max_execution_time}s")

    # Disabled sandbox (default)
    default = SandboxConfig(enabled=False)
    print(f"\n  Default (disabled): enabled={default.enabled}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    print("Agentica SandboxConfig: Best-Effort Execution Isolation\n")

    await demo_sandbox_file_tool()
    await demo_sandbox_execute_tool()
    await demo_sandbox_allowed_commands()

    if os.getenv("OPENAI_API_KEY"):
        await demo_agent_with_sandbox()
    else:
        print("\n[INFO] Set OPENAI_API_KEY to run Agent sandbox demo")

    demo_custom_sandbox()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
    SandboxConfig provides best-effort isolation:
    - blocked_paths: Path component matching (not substring) to prevent access
    - writable_dirs: Whitelist for file writes
    - blocked_commands: Regex boundary matching to block dangerous commands
    - allowed_commands: Optional whitelist — only commands matching these prefixes run
      (None = no restriction, list = only listed prefixes allowed)
    - max_execution_time: Timeout for command execution

    Usage:
      agent = Agent(sandbox_config=SandboxConfig(enabled=True, ...))
      # Coding agent: only allow python/pytest/git
      sandbox = SandboxConfig(enabled=True, allowed_commands=["python", "pytest", "git"])

    WARNING: This is NOT a true security sandbox. Use Docker/seccomp for untrusted code.
    """)


if __name__ == "__main__":
    asyncio.run(main())
