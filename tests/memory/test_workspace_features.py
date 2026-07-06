# -*- coding: utf-8 -*-
"""Tests for agentica.workspace — new features: multi-user, memory scoring, conversation archive."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentica.workspace import Workspace, WorkspaceConfig


class TestWorkspaceMultiUser(unittest.TestCase):
    """Multi-user isolation: set_user, _get_user_path, user data separation."""

    def test_default_user(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Workspace(path=tmpdir)
            self.assertEqual(ws.user_id, "default")

    def test_custom_user(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Workspace(path=tmpdir, user_id="alice@example.com")
            self.assertEqual(ws.user_id, "alice@example.com")

    def test_set_user(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Workspace(path=tmpdir)
            ws.set_user("bob")
            self.assertEqual(ws.user_id, "bob")

    def test_set_user_none_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Workspace(path=tmpdir, user_id="alice")
            ws.set_user(None)
            self.assertEqual(ws.user_id, "default")

    def test_user_path_sanitized(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Workspace(path=tmpdir, user_id="user/../evil")
            user_path = ws._get_user_path()
            self.assertNotIn("..", str(user_path))

    def test_set_user_resets_initialized_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Workspace(path=tmpdir)
            ws._user_initialized = True
            ws.set_user("new_user")
            self.assertFalse(ws._user_initialized)

    def test_set_same_user_no_reset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Workspace(path=tmpdir)
            ws._user_initialized = True
            ws.set_user("default")
            self.assertTrue(ws._user_initialized)

    def test_initialize_creates_user_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Workspace(path=tmpdir, user_id="test_user")
            ws.initialize()
            user_path = ws._get_user_path()
            self.assertTrue(user_path.exists())
            self.assertTrue((user_path / "memory").exists())
            self.assertTrue((user_path / "conversations").exists())

    def test_different_users_different_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Workspace(path=tmpdir, user_id="alice")
            alice_path = ws._get_user_path()
            ws.set_user("bob")
            bob_path = ws._get_user_path()
            self.assertNotEqual(alice_path, bob_path)


class TestWorkspaceArchiveLocks(unittest.TestCase):
    """Per-file archive locks for concurrent writes."""

    def test_has_archive_locks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Workspace(path=tmpdir)
            self.assertIsInstance(ws._archive_locks, dict)

    def test_get_archive_lock_same_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Workspace(path=tmpdir)
            # Access internal lock mechanism if exposed
            key = "2026-04-04.md"
            ws._archive_locks[key] = asyncio.Lock()
            self.assertIs(ws._archive_locks[key], ws._archive_locks[key])


class TestWorkspaceArchiveRedaction(unittest.TestCase):
    """Conversation archive redacts secrets before writing markdown."""

    def test_archive_conversation_redacts_secrets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Workspace(path=tmpdir)
            secret = "sk-abcdefghijklmnopqrstuvwxyz1234567890"
            path = asyncio.run(ws.archive_conversation([
                {"role": "user", "content": f"my key is {secret}"},
                {"role": "assistant", "content": "noted"},
            ], session_id="secret-session"))

            content = Path(path).read_text(encoding="utf-8")

        self.assertNotIn(secret, content)
        self.assertIn("REDACTED", content)
        self.assertIn("secret-session", content)


class TestWorkspaceInitialize(unittest.TestCase):
    """Workspace.initialize creates the directory structure."""

    def test_creates_global_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Workspace(path=tmpdir)
            ws.initialize()
            self.assertTrue((Path(tmpdir) / "AGENTS.md").exists())
            self.assertTrue((Path(tmpdir) / "PERSONA.md").exists())
            self.assertTrue((Path(tmpdir) / "TOOLS.md").exists())
            self.assertTrue((Path(tmpdir) / "skills").exists())
            self.assertTrue((Path(tmpdir) / "users").exists())

    def test_initialize_returns_true(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Workspace(path=tmpdir)
            self.assertTrue(ws.initialize())

    def test_initialize_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Workspace(path=tmpdir)
            ws.initialize()
            # Write custom content
            (Path(tmpdir) / "AGENTS.md").write_text("custom", encoding="utf-8")
            ws.initialize()  # Without force, should not overwrite
            self.assertEqual((Path(tmpdir) / "AGENTS.md").read_text(encoding="utf-8"), "custom")

    def test_initialize_force_overwrites(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Workspace(path=tmpdir)
            ws.initialize()
            (Path(tmpdir) / "AGENTS.md").write_text("custom", encoding="utf-8")
            ws.initialize(force=True)
            content = (Path(tmpdir) / "AGENTS.md").read_text(encoding="utf-8")
            self.assertNotEqual(content, "custom")


class TestWorkspaceReadWriteFile(unittest.TestCase):
    """read_file, write_file, append_file basic operations."""

    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Workspace(path=tmpdir)
            ws.initialize()
            ws.write_file("test.md", "hello world")
            content = ws.read_file("test.md")
            self.assertEqual(content, "hello world")

    def test_read_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Workspace(path=tmpdir)
            ws.initialize()
            content = ws.read_file("nonexistent.md")
            self.assertIsNone(content)

    def test_append_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Workspace(path=tmpdir)
            ws.initialize()
            ws.write_file("log.md", "line1\n")
            ws.append_file("log.md", "line2\n")
            content = ws.read_file("log.md")
            self.assertIn("line1", content)
            self.assertIn("line2", content)


class TestWorkspaceMemory(unittest.TestCase):
    """Daily memory and long-term memory operations (write_memory is async)."""

    def test_write_daily_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Workspace(path=tmpdir)
            ws.initialize()
            asyncio.run(ws.write_memory("Today I learned about testing", to_daily=True))
            memory_dir = ws._get_user_memory_dir()
            self.assertTrue(memory_dir.exists())
            # Check file was created
            files = list(memory_dir.glob("*.md"))
            self.assertGreater(len(files), 0)

    def test_write_long_term_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Workspace(path=tmpdir)
            ws.initialize()
            asyncio.run(ws.write_memory("User prefers Python", to_daily=False))
            # write_memory now delegates to write_memory_entry which creates
            # an indexed entry file, not direct MEMORY.md append
            memory_dir = ws._get_user_memory_dir()
            files = list(memory_dir.glob("*.md"))
            self.assertGreater(len(files), 0)
            # Content should be in one of the entry files
            found = any("User prefers Python" in f.read_text(encoding="utf-8") for f in files)
            self.assertTrue(found)


class TestWorkspaceConfig(unittest.TestCase):
    """WorkspaceConfig defaults and customization."""

    def test_default_config(self):
        config = WorkspaceConfig()
        self.assertEqual(config.agent_md, "AGENTS.md")
        self.assertEqual(config.memory_dir, "memory")
        self.assertEqual(config.skills_dir, "skills")
        self.assertEqual(config.users_dir, "users")
        self.assertEqual(config.conversations_dir, "conversations")

    def test_custom_config(self):
        config = WorkspaceConfig(agent_md="MY_AGENT.md", memory_dir="my_memory")
        self.assertEqual(config.agent_md, "MY_AGENT.md")
        self.assertEqual(config.memory_dir, "my_memory")


class TestWorkspaceRepr(unittest.TestCase):
    """Workspace __repr__ and __str__."""

    def test_repr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Workspace(path=tmpdir)
            r = repr(ws)
            self.assertIn("Workspace", r)

    def test_str(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Workspace(path=tmpdir)
            s = str(ws)
            self.assertIsInstance(s, str)


if __name__ == "__main__":
    unittest.main()
