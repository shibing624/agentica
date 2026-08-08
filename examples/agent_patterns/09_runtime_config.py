# -*- coding: utf-8 -*-
"""
Runtime Config Demo - Tool/Skill enable/disable control.

Demonstrates three levels of tool/skill control:
1. Agent-level: enable_tool/disable_tool/enable_skill/disable_skill
2. Workspace-level: .agentica/runtime_config.yaml
3. Query-level: run(config=RunConfig(enabled_tools=[...]))
"""
from agentica import Agent, RunConfig
from agentica.tools.builtin import get_builtin_tools


def demo_agent_level_control():
    """Demo 1: Agent-level tool enable/disable."""
    print("=" * 60)
    print("Demo 1: Agent-level tool enable/disable")
    print("=" * 60)

    agent = Agent(
        name="demo-agent",
        tools=get_builtin_tools(),
    )

    # Check all function names
    all_tools = agent.get_tools()
    print(f"\nTotal tools: {len(all_tools)}")

    # Disable execute tool
    agent.disable_tool("execute")
    print(f"After disable_tool('execute'): execute enabled = {agent._is_tool_enabled('execute')}")
    print(f"  read_file enabled = {agent._is_tool_enabled('read_file')}")

    # Re-enable
    agent.enable_tool("execute")
    print(f"After enable_tool('execute'): execute enabled = {agent._is_tool_enabled('execute')}")
    print()


def demo_query_level_control():
    """Demo 2: Query-level tool whitelist via RunConfig."""
    print("=" * 60)
    print("Demo 2: Query-level tool whitelist via RunConfig")
    print("=" * 60)

    agent = Agent(
        name="demo-agent",
        tools=get_builtin_tools(),
    )

    # When enabled_tools is set, only listed tools are available
    config = RunConfig(enabled_tools=["web_search", "read_file"])
    print(f"\nRunConfig(enabled_tools={config.enabled_tools})")
    print("  - web_search: allowed")
    print("  - read_file: allowed")
    print("  - execute: blocked (not in whitelist)")
    print("  - write_file: blocked (not in whitelist)")

    # This simulates what happens inside _run_impl
    agent._enabled_tools = config.enabled_tools
    print(f"\n  agent._is_tool_enabled('web_search') = {agent._is_tool_enabled('web_search')}")
    print(f"  agent._is_tool_enabled('read_file') = {agent._is_tool_enabled('read_file')}")
    print(f"  agent._is_tool_enabled('execute') = {agent._is_tool_enabled('execute')}")
    print(f"  agent._is_tool_enabled('write_file') = {agent._is_tool_enabled('write_file')}")

    # Clean up
    agent._enabled_tools = None
    print()


def demo_priority():
    """Demo 3: Priority - query > agent > default."""
    print("=" * 60)
    print("Demo 3: Priority - query-level > agent-level > default")
    print("=" * 60)

    agent = Agent(
        name="demo-agent",
        tools=get_builtin_tools(),
    )

    # Default: all enabled
    print(f"\n[Default] execute = {agent._is_tool_enabled('execute')}")

    # Agent-level: disable execute
    agent.disable_tool("execute")
    print(f"[Agent-level disable] execute = {agent._is_tool_enabled('execute')}")

    # Query-level: whitelist includes execute -> overrides agent-level
    agent._enabled_tools = ["execute", "read_file"]
    print(f"[Query-level whitelist] execute = {agent._is_tool_enabled('execute')}")
    print(f"[Query-level whitelist] write_file = {agent._is_tool_enabled('write_file')}")

    # Clean up
    agent._enabled_tools = None
    print()


def demo_yaml_config():
    """Demo 4: Workspace YAML config — create file, load, and verify."""
    import tempfile
    import os

    print("=" * 60)
    print("Demo 4: Workspace YAML config — create, load & verify")
    print("=" * 60)

    yaml_content = """\
tools:
  execute:
    enabled: false
  write_file:
    enabled: true
  web_search:
    enabled: true

skills:
  iwiki-doc:
    enabled: false
  paper-digest:
    enabled: true
"""

    # Create a temp directory as workspace, write YAML config
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = os.path.join(tmpdir, ".agentica")
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, "runtime_config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        print(f"\nCreated YAML config at: {config_path}")
        print(f"Content:\n{yaml_content}")

        # Save and change cwd so Agent._load_runtime_config() finds the file
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            agent = Agent(
                name="yaml-demo-agent",
                tools=get_builtin_tools(),
            )

            # Verify tool configs loaded from YAML
            print("Loaded _tool_runtime_configs:")
            for name, cfg in agent._tool_runtime_configs.items():
                print(f"  {name}: enabled={cfg.enabled}")

            print("\nLoaded _skill_runtime_configs:")
            for name, cfg in agent._skill_runtime_configs.items():
                print(f"  {name}: enabled={cfg.enabled}")

            # Verify _is_tool_enabled reflects YAML settings
            print("\nVerify _is_tool_enabled():")
            print(f"  execute  = {agent._is_tool_enabled('execute')}  (expected: False)")
            print(f"  write_file = {agent._is_tool_enabled('write_file')}  (expected: True)")
            print(f"  web_search = {agent._is_tool_enabled('web_search')}  (expected: True)")
            print(f"  read_file  = {agent._is_tool_enabled('read_file')}  (expected: True, default)")

            # Verify _is_skill_enabled reflects YAML settings
            print("\nVerify _is_skill_enabled():")
            print(f"  iwiki-doc    = {agent._is_skill_enabled('iwiki-doc')}  (expected: False)")
            print(f"  paper-digest = {agent._is_skill_enabled('paper-digest')}  (expected: True)")
            print(f"  unknown-skill = {agent._is_skill_enabled('unknown-skill')}  (expected: True, default)")

            # Verify assertions
            assert agent._is_tool_enabled('execute') is False, "execute should be disabled"
            assert agent._is_tool_enabled('write_file') is True, "write_file should be enabled"
            assert agent._is_tool_enabled('read_file') is True, "read_file should default to True"
            assert agent._is_skill_enabled('iwiki-doc') is False, "iwiki-doc should be disabled"
            assert agent._is_skill_enabled('paper-digest') is True, "paper-digest should be enabled"
            print("\n✅ All YAML config assertions passed!")
        finally:
            os.chdir(original_cwd)
    print()


if __name__ == "__main__":
    demo_agent_level_control()
    demo_query_level_control()
    demo_priority()
    demo_yaml_config()
