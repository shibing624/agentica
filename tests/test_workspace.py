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

    def test_default_global_templates_are_minimal_scaffolds(self):
        """Default scaffolds carry no behavioural rules — empty by design.

        Previous templates injected ~1KB of "Friendly and professional" /
        "Run lint then typecheck then tests" boilerplate into every system
        prompt with zero project-specific signal. The new defaults are
        deliberately minimal so the prompt only grows when the user adds
        real rules to AGENTS.md.
        """
        agents_md = Workspace.DEFAULT_GLOBAL_FILES["AGENTS.md"]
        assert "Use shell tool to run" not in agents_md
        assert "Friendly and professional" not in agents_md
        # Marker comments are fine — they're stripped by _is_empty_template.
        assert Workspace._is_empty_template(agents_md), (
            "default AGENTS.md should look empty to the prompt assembler"
        )
        assert Workspace._is_empty_template(Workspace.DEFAULT_GLOBAL_FILES["PERSONA.md"])
        assert Workspace._is_empty_template(Workspace.DEFAULT_GLOBAL_FILES["TOOLS.md"])


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
        assert (temp_workspace_path / "AGENTS.md").exists()
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
        assert "AGENTS.md" in context or "Agent" in context
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

    def test_write_memory_entry_syncs_feedback_to_global_agent_md(self, temp_workspace_path):
        """Confirmed user/feedback memories can be compiled into ~/.agentica/AGENTS.md."""
        workspace = Workspace(temp_workspace_path)
        workspace.initialize()

        global_home = temp_workspace_path / "global-home"
        global_home.mkdir()

        with patch("agentica.workspace.AGENTICA_HOME", str(global_home)):
            asyncio.run(
                workspace.write_memory_entry(
                    title="Python Style",
                    content="Prefer concise, typed Python. Avoid unnecessary getattr.",
                    memory_type="feedback",
                    description="python style concise typed",
                    sync_to_global_agent_md=True,
                )
            )

        global_agent_md = global_home / "AGENTS.md"
        assert global_agent_md.exists()
        content = global_agent_md.read_text(encoding="utf-8")
        assert "Learned Preferences" in content
        assert "Python Style" in content
        assert "Avoid unnecessary getattr" in content

    def test_write_memory_entry_sync_skips_non_durable_feedback(self, temp_workspace_path):
        """Global AGENTS sync should keep durable rules and skip task-specific notes."""
        workspace = Workspace(temp_workspace_path)
        workspace.initialize()

        global_home = temp_workspace_path / "global-home"
        global_home.mkdir()

        with patch("agentica.workspace.AGENTICA_HOME", str(global_home)):
            asyncio.run(
                workspace.write_memory_entry(
                    title="Python Style",
                    content="Prefer concise, typed Python. Avoid unnecessary getattr.",
                    memory_type="feedback",
                    description="durable python coding preference",
                    sync_to_global_agent_md=True,
                )
            )
            asyncio.run(
                workspace.write_memory_entry(
                    title="RAG Oracle Flow",
                    content="RAG pipeline: inspect prediction samples first, then compare MRR / P@3 / R@3 / F1 before tuning.",
                    memory_type="feedback",
                    description="oracle style rag debugging note",
                    sync_to_global_agent_md=True,
                )
            )

        global_agent_md = global_home / "AGENTS.md"
        content = global_agent_md.read_text(encoding="utf-8")
        assert "Python Style" in content
        assert "Avoid unnecessary getattr" in content
        assert "RAG Oracle Flow" not in content
        assert "MRR / P@3 / R@3 / F1" not in content

    def test_get_context_prompt_prioritizes_high_priority_agents_with_budget(self, temp_workspace_path):
        """AGENTS context should cap at 40K chars and preserve higher-priority files."""
        repo_root = temp_workspace_path / "repo"
        cwd = repo_root / "nested"
        cwd.mkdir(parents=True)
        (repo_root / ".git").mkdir()
        (repo_root / "AGENTS.md").write_text("# Project\n" + ("B" * 19000), encoding="utf-8")
        (cwd / "AGENTS.md").write_text("# Nested\n" + ("C" * 19000), encoding="utf-8")

        global_home = temp_workspace_path / "global-home"
        global_home.mkdir()
        (global_home / "AGENTS.md").write_text("# Global\n" + ("A" * 30000), encoding="utf-8")

        workspace = Workspace(repo_root)
        previous_cwd = os.getcwd()
        try:
            os.chdir(cwd)
            with patch("agentica.workspace.AGENTICA_HOME", str(global_home)):
                context = asyncio.run(workspace.get_context_prompt())
        finally:
            os.chdir(previous_cwd)

        assert "# Nested" in context
        assert "# Project" in context
        assert "# Global" not in context
        assert "C" * 500 in context
        assert "B" * 500 in context

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
        assert "AGENTS.md" in files
        assert files["AGENTS.md"] is True
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
