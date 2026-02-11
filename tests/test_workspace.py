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
from datetime import date
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agentica.workspace import Workspace, WorkspaceConfig


class TestWorkspaceConfig:
    """Test WorkspaceConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WorkspaceConfig()
        assert config.agent_md == "AGENT.md"
        assert config.persona_md == "PERSONA.md"
        assert config.tools_md == "TOOLS.md"
        assert config.user_md == "USER.md"
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
        # Global shared files
        assert (temp_workspace_path / "AGENT.md").exists()
        assert (temp_workspace_path / "PERSONA.md").exists()
        assert (temp_workspace_path / "TOOLS.md").exists()
        assert (temp_workspace_path / "skills").is_dir()
        # User-specific files under users/default/
        user_path = temp_workspace_path / "users" / "default"
        assert (user_path / "USER.md").exists()
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
        """Test getting context prompt from workspace files."""
        workspace = Workspace(temp_workspace_path)
        workspace.initialize()

        context = asyncio.run(workspace.get_context_prompt())

        # Should include content from default files
        assert "AGENT.md" in context or "Agent" in context
        assert len(context) > 0

    def test_write_memory_daily(self, temp_workspace_path):
        """Test writing daily memory."""
        workspace = Workspace(temp_workspace_path)
        workspace.initialize()

        # Write daily memory (async)
        asyncio.run(workspace.write_memory("Today I learned about Python.", to_daily=True))

        # Check memory file exists under users/default/memory/
        today = date.today().isoformat()
        memory_file = temp_workspace_path / "users" / "default" / "memory" / f"{today}.md"
        assert memory_file.exists()

        content = memory_file.read_text()
        assert "Today I learned about Python." in content

    def test_write_memory_long_term(self, temp_workspace_path):
        """Test writing long-term memory."""
        workspace = Workspace(temp_workspace_path)
        workspace.initialize()

        # Write long-term memory (async)
        asyncio.run(workspace.write_memory("User prefers concise answers.", to_daily=False))

        # Check MEMORY.md file under users/default/
        memory_file = temp_workspace_path / "users" / "default" / "MEMORY.md"
        assert memory_file.exists()

        content = memory_file.read_text()
        assert "User prefers concise answers." in content

    def test_get_memory_prompt(self, temp_workspace_path):
        """Test getting memory prompt."""
        workspace = Workspace(temp_workspace_path)
        workspace.initialize()

        # Write some memories (async)
        asyncio.run(workspace.write_memory("Long-term preference", to_daily=False))
        asyncio.run(workspace.write_memory("Daily note", to_daily=True))

        # Get memory prompt (async)
        memory_prompt = asyncio.run(workspace.get_memory_prompt(days=2))

        assert len(memory_prompt) > 0

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

        # list_files only returns globally shared files
        assert "AGENT.md" in files
        assert files["AGENT.md"] is True
        assert "PERSONA.md" in files
        assert "TOOLS.md" in files

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
