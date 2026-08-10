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
        """Test getting context prompt from workspace files.

        Pins the AGENTS.md discovery to the temp workspace by chdir-ing into
        it and pointing AGENTICA_HOME at a temp dir. Previously this test
        relied on the host having ``~/.agentica/AGENTS.md`` or an ancestor
        ``AGENTS.md`` in cwd, which made it pass locally but fail in clean CI
        environments. The test now controls all three discovery sources
        (global home / cwd chain / workspace) so the outcome is deterministic.
        """
        # Seed a real AGENTS.md in the workspace so _load_agent_md_chain has
        # non-empty content regardless of host filesystem state.
        (temp_workspace_path / "AGENTS.md").write_text(
            "# Project Agent\nProject-specific agent instructions go here.\n",
            encoding="utf-8",
        )

        workspace = Workspace(temp_workspace_path)
        workspace.initialize()

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


class TestGitContext:
    """``get_git_context`` runs on the per-turn system-prompt path."""

    @pytest.fixture
    def temp_workspace_path(self):
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_a_plain_directory_has_no_git_context(self, temp_workspace_path):
        workspace = Workspace(temp_workspace_path)
        assert asyncio.run(workspace.get_git_context()) is None

    def test_a_repo_reports_branch_and_uncommitted_changes(self, temp_workspace_path):
        import subprocess

        def git(*args):
            subprocess.run(
                ["git", *args], cwd=temp_workspace_path,
                capture_output=True, check=True,
            )

        try:
            git("init", "-b", "main")
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("git is unavailable")
        git("config", "user.email", "t@example.com")
        git("config", "user.name", "t")
        (temp_workspace_path / "a.txt").write_text("hello")
        git("add", "a.txt")
        git("commit", "-m", "first commit")
        (temp_workspace_path / "b.txt").write_text("pending")

        context = asyncio.run(Workspace(temp_workspace_path).get_git_context())

        assert "Git branch: main" in context
        assert "b.txt" in context
        assert "first commit" in context

    def test_the_three_reads_run_together_behind_the_repo_check(self, temp_workspace_path):
        """Four blocking 5s-timeout git calls on the prompt path can stall the
        whole event loop; the repo check gates the rest, which then overlap."""
        workspace = Workspace(temp_workspace_path)
        order = []
        in_flight = 0
        peak = 0

        async def fake_git(_self, *args, timeout=5.0):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            order.append(args[0])
            try:
                await asyncio.sleep(0.05)
            finally:
                in_flight -= 1
            return "main" if args[0] == "branch" else "x"

        with patch.object(Workspace, "_git", fake_git):
            asyncio.run(workspace.get_git_context())

        assert order[0] == "rev-parse", "the repo check must settle first"
        assert peak == 3, f"the three reads ran {peak}-at-a-time"

    def test_a_non_repo_is_probed_once_per_workspace(self, temp_workspace_path):
        """A service workspace (~/.agentica/workspace) will not become a repo
        mid-process, so re-spawning `git rev-parse` every turn buys nothing."""
        workspace = Workspace(temp_workspace_path)
        calls = []

        async def fake_git(_self, *args, timeout=5.0):
            calls.append(args[0])
            return None

        async def main():
            with patch.object(Workspace, "_git", fake_git):
                return [await workspace.get_git_context() for _ in range(3)]

        assert asyncio.run(main()) == [None, None, None]
        assert calls == ["rev-parse"], f"probed {len(calls)} times"

    def test_a_repo_is_re_read_every_turn(self, temp_workspace_path):
        """Branch, status and commits are exactly what changes between turns."""
        workspace = Workspace(temp_workspace_path)
        calls = []

        async def fake_git(_self, *args, timeout=5.0):
            calls.append(args[0])
            return "main" if args[0] == "branch" else "x"

        async def main():
            with patch.object(Workspace, "_git", fake_git):
                for _ in range(2):
                    await workspace.get_git_context()

        asyncio.run(main())
        assert calls.count("status") == 2
        assert calls.count("rev-parse") == 2

    def test_it_does_not_block_the_event_loop(self, temp_workspace_path):
        """A concurrent coroutine must keep getting scheduled while git runs."""
        workspace = Workspace(temp_workspace_path)
        ticks = 0

        async def fake_git(_self, *args, timeout=5.0):
            await asyncio.sleep(0.05)
            return "x"

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.005)
                ticks += 1

        async def main():
            spinner = asyncio.create_task(ticker())
            with patch.object(Workspace, "_git", fake_git):
                await workspace.get_git_context()
            spinner.cancel()

        asyncio.run(main())
        assert ticks > 5, f"event loop only advanced {ticks} times during git reads"


class TestWorkspaceExpansion:
    """Test workspace path expansion."""

    def test_home_expansion(self):
        """Test that ~ is expanded in workspace path."""
        workspace = Workspace("~/test_workspace")
        assert "~" not in str(workspace.path)
        assert workspace.path.is_absolute()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
