# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for Workspace module
"""
import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
import os
import sys
from unittest.mock import patch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agentica.workspace import Workspace, WorkspaceConfig


class TestWorkspaceConfig:
    """Test WorkspaceConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WorkspaceConfig()
        assert config.agent_md == "AGENTS.md"
        assert config.memory_md == "MEMORY.md"
        assert config.memory_dir == "memory"
        assert config.skills_dir == "skills"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = WorkspaceConfig(
            agent_md="CUSTOM_AGENT.md",
            memory_dir="memories",
        )
        assert config.agent_md == "CUSTOM_AGENT.md"
        assert config.memory_dir == "memories"

    def test_default_user_agent_scaffold_is_empty_of_rules(self):
        """User AGENTS scaffold carries no behavioural rules — empty by design."""
        assert Workspace._is_empty_template(
            Workspace.DEFAULT_USER_AGENT_MD.format(user_id="default")
        ), "a fresh user AGENTS.md must not inject its own scaffold"


class TestWorkspace:
    """Test Workspace class."""

    @pytest.fixture
    def temp_workspace_path(self):
        """Create a temporary directory for workspace testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_workspace_init(self, temp_workspace_path):
        """Test workspace initialization."""
        workspace = Workspace(temp_workspace_path)
        # Use resolve() on both sides to handle macOS /var vs /private/var symlinks
        assert workspace.path.resolve() == temp_workspace_path.resolve()
        assert workspace.config is not None

    def test_workspace_with_custom_config(self, temp_workspace_path):
        """Test workspace with custom config."""
        config = WorkspaceConfig(agent_md="CUSTOM_AGENT.md")
        workspace = Workspace(temp_workspace_path, config=config)
        assert workspace.config.agent_md == "CUSTOM_AGENT.md"

    def test_workspace_initialize(self, temp_workspace_path):
        """Test workspace initialization creates default files."""
        workspace = Workspace(temp_workspace_path)
        result = workspace.initialize()

        assert result is True
        # No workspace-root AGENTS.md — standing rules live per user
        assert not (temp_workspace_path / "AGENTS.md").exists()
        assert (temp_workspace_path / "skills").is_dir()
        # This user's own instructions live with the rest of their data
        user_path = temp_workspace_path / "users" / "default"
        assert (user_path / "AGENTS.md").exists()
        assert (user_path / "memory").is_dir()

    def test_workspace_exists(self, temp_workspace_path):
        """Test workspace exists check."""
        workspace = Workspace(temp_workspace_path)

        # Before initialization
        assert workspace.exists() is False

        # After initialization
        workspace.initialize()
        assert workspace.exists() is True

    def test_read_write_file(self, temp_workspace_path):
        """Test reading and writing files."""
        workspace = Workspace(temp_workspace_path)
        workspace.initialize()

        # Write content
        test_content = "# Test Content\n\nThis is a test."
        workspace.write_file("test.md", test_content)

        # Read content
        read_content = workspace.read_file("test.md")
        assert read_content == test_content

    def test_read_nonexistent_file(self, temp_workspace_path):
        """Test reading a file that doesn't exist."""
        workspace = Workspace(temp_workspace_path)
        workspace.initialize()

        content = workspace.read_file("nonexistent.md")
        assert content is None

    def test_append_file(self, temp_workspace_path):
        """Test appending to a file."""
        workspace = Workspace(temp_workspace_path)
        workspace.initialize()

        # Write initial content
        workspace.write_file("test.md", "First line")

        # Append content
        workspace.append_file("test.md", "Second line")

        # Read and verify
        content = workspace.read_file("test.md")
        assert "First line" in content
        assert "Second line" in content

    def test_get_context_prompt(self, temp_workspace_path):
        """Test getting context prompt from workspace files.

        Pins the AGENTS.md discovery to the temp workspace by chdir-ing into
        it and pointing AGENTICA_HOME at a temp dir. Previously this test
        relied on the host having ``~/.agentica/AGENTS.md`` or an ancestor
        ``AGENTS.md`` in cwd, which made it pass locally but fail in clean CI
        environments. The test now controls all three discovery sources
        (global home / cwd chain / workspace) so the outcome is deterministic.
        """
        # Seed this user's AGENTS.md so _load_agent_md_chain has non-empty
        # content regardless of host filesystem state.
        workspace = Workspace(temp_workspace_path)
        workspace.initialize()
        workspace.user_agent_md_path().write_text(
            "# Project Agent\nProject-specific agent instructions go here.\n",
            encoding="utf-8",
        )

        # Isolate cwd + AGENTICA_HOME so host pollution can't influence merge.
        prev_cwd = os.getcwd()
        empty_home = temp_workspace_path / "_empty_home"
        empty_home.mkdir()
        try:
            os.chdir(temp_workspace_path)
            with patch("agentica.workspace.AGENTICA_HOME", str(empty_home)):
                context = asyncio.run(workspace.get_context_prompt())
        finally:
            os.chdir(prev_cwd)

        assert "AGENTS.md" in context or "Project Agent" in context
        assert len(context) > 0

    def test_write_memory_daily(self, temp_workspace_path):
        """Test writing daily memory (now delegates to write_memory_entry)."""
        workspace = Workspace(temp_workspace_path)
        workspace.initialize()

        # write_memory now delegates to write_memory_entry (indexed storage)
        asyncio.run(workspace.write_memory("Today I learned about Python.", to_daily=True))

        # Check that a memory entry file was created in memory/ dir
        memory_dir = temp_workspace_path / "users" / "default" / "memory"
        md_files = list(memory_dir.glob("*.md"))
        assert len(md_files) >= 1

        # Check content is in one of the files
        found = False
        for f in md_files:
            if "Today I learned about Python." in f.read_text():
                found = True
                break
        assert found

    def test_write_memory_long_term(self, temp_workspace_path):
        """Test writing long-term memory (now delegates to write_memory_entry)."""
        workspace = Workspace(temp_workspace_path)
        workspace.initialize()

        # write_memory now delegates to write_memory_entry (indexed storage)
        asyncio.run(workspace.write_memory("User prefers concise answers.", to_daily=False))

        # Check MEMORY.md index was updated
        memory_index = temp_workspace_path / "users" / "default" / "MEMORY.md"
        assert memory_index.exists()

        # Check that content is in a memory entry file
        memory_dir = temp_workspace_path / "users" / "default" / "memory"
        found = False
        for f in memory_dir.glob("*.md"):
            if "User prefers concise answers." in f.read_text():
                found = True
                break
        assert found

    def test_get_memory_prompt(self, temp_workspace_path):
        """Test getting relevant memories (replaces old get_memory_prompt)."""
        workspace = Workspace(temp_workspace_path)
        workspace.initialize()

        # Write memories via write_memory_entry (structured, indexed)
        asyncio.run(workspace.write_memory_entry(
            title="Python preference",
            content="User prefers concise Python code.",
            memory_type="feedback",
            description="python concise coding style",
        ))
        asyncio.run(workspace.write_memory_entry(
            title="Daily note",
            content="Worked on memory system refactor.",
            memory_type="project",
            description="memory system refactor project",
        ))

        # get_relevant_memories without query returns top entries
        memory_prompt = asyncio.run(workspace.get_relevant_memories())
        assert len(memory_prompt) > 0

        # get_relevant_memories with a matching query returns relevant entry
        memory_prompt_python = asyncio.run(workspace.get_relevant_memories(query="python coding"))
        assert "Python preference" in memory_prompt_python or len(memory_prompt_python) > 0

    def test_get_context_prompt_prioritizes_user_agents_with_budget(self, temp_workspace_path):
        """User AGENTS is first in budget; workspace-root leftovers are ignored."""
        ws_root = temp_workspace_path / "ws"
        repo_root = temp_workspace_path / "repo"
        cwd = repo_root / "nested"
        cwd.mkdir(parents=True)
        ws_root.mkdir()
        (repo_root / ".git").mkdir()
        (repo_root / "AGENTS.md").write_text("# Project\n" + ("B" * 2000), encoding="utf-8")
        (cwd / "AGENTS.md").write_text("# Nested\n" + ("C" * 2000), encoding="utf-8")
        (ws_root / "AGENTS.md").write_text("# WorkspaceRoot\nSHOULD_NOT_APPEAR\n", encoding="utf-8")

        workspace = Workspace(ws_root)
        workspace.initialize()
        workspace.user_agent_md_path().write_text(
            "# User\n" + ("A" * 2000),
            encoding="utf-8",
        )

        previous_cwd = os.getcwd()
        try:
            os.chdir(cwd)
            context = asyncio.run(workspace.get_context_prompt())
        finally:
            os.chdir(previous_cwd)

        assert "# User" in context
        assert "# Nested" in context
        assert "# Project" in context
        assert "SHOULD_NOT_APPEAR" not in context

        # When the user's own rules alone fill the budget, they are what
        # survives: standing rules are the reason this block exists, and the
        # project chain is one `read_file` away in the repo.
        workspace.user_agent_md_path().write_text(
            "# User\n" + ("A" * Workspace.MAX_MEMORY_CHARACTER_COUNT),
            encoding="utf-8",
        )
        try:
            os.chdir(cwd)
            tight = asyncio.run(workspace.get_context_prompt())
        finally:
            os.chdir(previous_cwd)

        assert "# User" in tight
        assert "# Nested" not in tight
        assert "# Project" not in tight
        assert len(tight) <= Workspace.MAX_MEMORY_CHARACTER_COUNT + 80

    def test_default_user_gets_home_agents_symlink_to_canonical_file(self, temp_workspace_path):
        """The default user keeps one real file plus a mainstream-agent alias."""
        home = temp_workspace_path / "home"
        home.mkdir()
        with patch("agentica.workspace.AGENTICA_HOME", str(home)), \
                patch("agentica.workspace.AGENTICA_WORKSPACE_DIR", str(temp_workspace_path)):
            default_user = Workspace(temp_workspace_path).user_agent_md_path()
            tenant = Workspace(temp_workspace_path, user_id="tenant-a").user_agent_md_path()

        root = temp_workspace_path.resolve()
        assert default_user == root / "users" / "default" / "AGENTS.md"
        assert tenant == root / "users" / "tenant-a" / "AGENTS.md"
        assert (home / "AGENTS.md").is_symlink()
        assert (home / "AGENTS.md").resolve() == default_user

        (home / "AGENTS.md").write_text("mainstream path update", encoding="utf-8")
        assert default_user.read_text(encoding="utf-8") == "mainstream path update"

    def test_existing_home_rules_are_moved_then_replaced_with_symlink(self, temp_workspace_path):
        """A legacy regular file is preserved before the home path becomes an alias."""
        home = temp_workspace_path / "home"
        home.mkdir()
        legacy = home / "AGENTS.md"
        legacy.write_text("Always write a CHANGELOG entry.", encoding="utf-8")

        with patch("agentica.workspace.AGENTICA_HOME", str(home)), \
                patch("agentica.workspace.AGENTICA_WORKSPACE_DIR", str(temp_workspace_path)):
            workspace = Workspace(temp_workspace_path)
            target = workspace.user_agent_md_path()

        assert target.read_text(encoding="utf-8") == "Always write a CHANGELOG entry."
        assert legacy.is_symlink()
        assert legacy.resolve() == target

    def test_home_alias_never_crosses_users_and_existing_target_wins(self, temp_workspace_path):
        home = temp_workspace_path / "home"
        home.mkdir()
        (home / "AGENTS.md").write_text("home rules", encoding="utf-8")

        with patch("agentica.workspace.AGENTICA_HOME", str(home)), \
                patch("agentica.workspace.AGENTICA_WORKSPACE_DIR", str(temp_workspace_path)):
            tenant = Workspace(temp_workspace_path, user_id="tenant-a")
            assert not tenant.user_agent_md_path().exists()
            assert (home / "AGENTS.md").is_file(), "another user's session must not move it"

            default_user = Workspace(temp_workspace_path)
            target = default_user._get_user_path() / "AGENTS.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("already mine", encoding="utf-8")

            assert default_user.user_agent_md_path() == target
            assert target.read_text(encoding="utf-8") == "already mine\n\nhome rules\n"
            assert (home / "AGENTS.md").is_symlink()
            assert (home / "AGENTS.md").resolve() == target.resolve()

    def test_one_users_rules_never_reach_another(self, temp_workspace_path):
        """The whole point of keeping rules per user rather than in HOME."""
        outside = temp_workspace_path / "cwd"
        outside.mkdir()
        alice = Workspace(temp_workspace_path, user_id="alice")
        alice.initialize()
        alice.user_agent_md_path().write_text("Alice ships on Fridays.", encoding="utf-8")
        bob = Workspace(temp_workspace_path, user_id="bob")
        bob.initialize()

        previous_cwd = os.getcwd()
        try:
            os.chdir(outside)
            alice_context = asyncio.run(alice.get_context_prompt())
            bob_context = asyncio.run(bob.get_context_prompt())
        finally:
            os.chdir(previous_cwd)

        assert "Alice ships on Fridays." in alice_context
        assert "Alice" not in bob_context

    def test_get_skills_dir(self, temp_workspace_path):
        """Test getting skills directory."""
        workspace = Workspace(temp_workspace_path)
        workspace.initialize()

        skills_dir = workspace.get_skills_dir()
        # Use resolve() on both sides to handle macOS /var vs /private/var symlinks
        assert skills_dir.resolve() == (temp_workspace_path / "skills").resolve()

    def test_list_files(self, temp_workspace_path):
        """Test listing workspace files."""
        workspace = Workspace(temp_workspace_path)
        workspace.initialize()

        files = workspace.list_files()

        # list_files reports this user's AGENTS.md
        assert files == {"AGENTS.md": True}

    def test_search_memory(self, temp_workspace_path):
        """Test searching memory."""
        workspace = Workspace(temp_workspace_path)
        workspace.initialize()

        # Write some memories (async)
        asyncio.run(workspace.write_memory("Python is a great programming language.", to_daily=False))
        asyncio.run(workspace.write_memory("I love coding in JavaScript too.", to_daily=True))

        # Search for Python (sync method)
        results = workspace.search_memory("Python programming", limit=5)

        assert len(results) > 0
        assert any("Python" in r["content"] for r in results)

    def test_clear_daily_memory(self, temp_workspace_path):
        """Test clearing old daily memory."""
        workspace = Workspace(temp_workspace_path)
        workspace.initialize()

        # Create some memory files under users/default/memory/
        memory_dir = temp_workspace_path / "users" / "default" / "memory"
        for i in range(10):
            (memory_dir / f"2024-01-{i+1:02d}.md").write_text(f"Memory {i}")

        # Clear, keeping only 3 days
        workspace.clear_daily_memory(keep_days=3)

        # Check remaining files
        remaining = list(memory_dir.glob("*.md"))
        assert len(remaining) == 3

    def test_workspace_repr(self, temp_workspace_path):
        """Test workspace string representation."""
        workspace = Workspace(temp_workspace_path)

        repr_str = repr(workspace)
        assert "Workspace" in repr_str
        assert str(temp_workspace_path) in repr_str

    def test_workspace_str(self, temp_workspace_path):
        """Test workspace string conversion."""
        workspace = Workspace(temp_workspace_path)

        str_value = str(workspace)
        # Use resolve() to handle macOS /var vs /private/var symlinks
        assert str(temp_workspace_path.resolve()) == str_value


class TestWorkspaceExpansion:
    """Test workspace path expansion."""

    def test_home_expansion(self):
        """Test that ~ is expanded in workspace path."""
        workspace = Workspace("~/test_workspace")
        assert "~" not in str(workspace.path)
        assert workspace.path.is_absolute()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
