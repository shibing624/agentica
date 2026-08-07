# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for buildin_tools.py built-in tools (async-first).

All tools in buildin_tools.py are async. Tests use asyncio.run() to drive them.
LLM-dependent tools (BuiltinTaskTool) are tested with mocked Agent/Model.
"""

import pytest
import asyncio
import json
import os
import queue
import shlex
import tempfile
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch
import os
import sys

# bs4 / lxml / markdownify / requests are in agentica core deps (since v1.3.6),
# so no skip needed for builtin web search / fetch url tools.

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from agentica.tools.buildin_tools import (
    BuiltinFileTool,
    BuiltinExecuteTool,
    BuiltinWebSearchTool,
    BuiltinFetchUrlTool,
    BuiltinTodoTool,
    BuiltinTaskTool,
    get_builtin_tools,
)
from agentica.tools.background_processes import BackgroundProcess, BackgroundProcessRegistry
from agentica.tools.builtin.web_tools import (
    BuiltinFetchUrlTool as CanonicalBuiltinFetchUrlTool,
    BuiltinWebSearchTool as CanonicalBuiltinWebSearchTool,
)
from agentica.tools.builtin.task_state_tools import (
    BuiltinTodoTool as CanonicalBuiltinTodoTool,
)
from agentica.model.message import Message
from agentica.tools.shell_tool import ShellTool


class BlockingSubprocess:
    """Minimal subprocess double whose first communicate call blocks."""

    def __init__(self):
        self.started = asyncio.Event()

    async def communicate(self):
        self.started.set()
        await asyncio.Future()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    """Create a temporary directory for file operation tests."""
    d = tempfile.mkdtemp(prefix="test_buildin_tools_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def file_tool(tmp_dir):
    """BuiltinFileTool scoped to a temp directory."""
    return BuiltinFileTool(work_dir=tmp_dir)


@pytest.fixture
def execute_tool(tmp_dir):
    """BuiltinExecuteTool scoped to a temp directory."""
    return BuiltinExecuteTool(work_dir=tmp_dir)


@pytest.fixture
def todo_tool():
    return BuiltinTodoTool()


# ===========================================================================
# BuiltinFileTool tests
# ===========================================================================

class TestBuiltinFileToolLs:
    def test_ls_empty_dir(self, file_tool, tmp_dir):
        result = asyncio.run(file_tool.ls(tmp_dir))
        items = json.loads(result)
        assert isinstance(items, list)
        assert len(items) == 0

    def test_ls_with_files(self, file_tool, tmp_dir):
        # Create a file and a subdirectory
        Path(tmp_dir, "hello.txt").write_text("hello")
        Path(tmp_dir, "subdir").mkdir()

        result = asyncio.run(file_tool.ls(tmp_dir))
        items = json.loads(result)
        names = {i["name"] for i in items}
        assert "hello.txt" in names
        assert "subdir" in names
        # Check types
        types = {i["name"]: i["type"] for i in items}
        assert types["hello.txt"] == "file"
        assert types["subdir"] == "dir"

    def test_ls_nonexistent_dir(self, file_tool):
        with pytest.raises(FileNotFoundError):
            asyncio.run(file_tool.ls("/nonexistent_dir_abc123"))

    def test_ls_file_not_dir(self, file_tool, tmp_dir):
        f = Path(tmp_dir, "afile.txt")
        f.write_text("x")
        with pytest.raises(NotADirectoryError):
            asyncio.run(file_tool.ls(str(f)))


class TestBuiltinFileToolReadFile:
    def test_path_tool_descriptions_require_grounded_paths(self):
        for fn in (
            BuiltinFileTool.read_file,
            BuiltinFileTool.grep,
            BuiltinFileTool.glob,
            BuiltinFileTool.edit_file,
            BuiltinFileTool.apply_patch,
        ):
            description = " ".join((fn.__doc__ or "").split()).lower()
            assert "ground" in description
            assert "speculative absolute path" in description

    def test_empty_file_returns_reminder(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "empty.txt")
        Path(fp).touch()
        result = asyncio.run(file_tool.read_file(fp))
        assert "<system-reminder>" in result
        assert "0 bytes" in result

    def test_read_simple_file(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "test.txt")
        p.write_text("line1\nline2\nline3\n")
        result = asyncio.run(file_tool.read_file(str(p)))
        assert "line1" in result
        assert "line2" in result
        assert "line3" in result
        assert "[File metadata:" not in result

    def test_read_file_with_offset_limit(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "lines.txt")
        p.write_text("\n".join(f"line{i}" for i in range(1, 21)))
        # Read lines 5-9 (offset=4, limit=5)
        result = asyncio.run(file_tool.read_file(str(p), offset=4, limit=5))
        assert "line5" in result
        assert "line9" in result
        # line1 should not be present since offset skips it
        assert "line1\t" not in result  # Use tab to avoid matching "line10"

    def test_read_nonexistent_file_returns_path_state(self, file_tool, tmp_dir):
        existing = Path(tmp_dir, "src")
        existing.mkdir()
        candidate = existing / "usage.py"
        candidate.write_text("x = 1\n")

        with pytest.raises(FileNotFoundError) as exc:
            asyncio.run(file_tool.read_file(str(existing / "usage_tracking.py")))

        msg = str(exc.value)
        assert "File not found:" in msg
        assert "Resolved path:" in msg
        assert f"Nearest existing parent: {existing}" in msg
        assert "Candidate paths:" not in msg
        assert str(candidate) not in msg
        assert "do not retry speculative absolute paths" in msg

    def test_read_long_lines_truncated(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "long.txt")
        p.write_text("A" * 5000 + "\n")
        result = asyncio.run(file_tool.read_file(str(p)))
        # Default max_line_length is 2000, line should be truncated with "..."
        assert "..." in result

    def test_read_file_shows_line_numbers(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "numbered.txt")
        p.write_text("alpha\nbeta\ngamma\n")
        result = asyncio.run(file_tool.read_file(str(p)))
        # Should contain line numbers (cat -n format)
        assert "\t" in result  # tab separator between line number and content


class TestMissingPathErrors:
    """Missing-path errors expose path state without guessing candidates."""

    def test_grep_missing_path_does_not_suggest_candidates(self, file_tool, tmp_dir):
        existing = Path(tmp_dir, "src")
        existing.mkdir()
        (existing / "config.py").write_text("X = 1\n")

        with pytest.raises(FileNotFoundError) as exc:
            asyncio.run(file_tool.grep("class .*Config", path="missing-dir/config.py"))

        msg = str(exc.value)
        assert "Path not found:" in msg
        assert "Resolved path:" in msg
        assert "Nearest existing parent:" in msg
        assert "Did you mean:" not in msg
        assert "src/config.py" not in msg
        assert "do not retry speculative absolute paths" in msg

    def test_glob_missing_path_does_not_suggest_candidates(self, file_tool, tmp_dir):
        Path(tmp_dir, "src").mkdir()
        Path(tmp_dir, "src", "config.py").write_text("X = 1\n")

        with pytest.raises(FileNotFoundError) as exc:
            asyncio.run(file_tool.glob("*.py", path="missing-dir/config.py"))

        msg = str(exc.value)
        assert "Directory not found:" in msg
        assert "Did you mean:" not in msg
        assert "run glob '**/config.py'" not in msg

    def test_edit_file_missing_path_does_not_suggest_candidates(self, file_tool, tmp_dir):
        Path(tmp_dir, "src").mkdir()
        Path(tmp_dir, "src", "config.py").write_text("X = 1\n")

        with pytest.raises(FileNotFoundError) as exc:
            asyncio.run(file_tool.edit_file("missing-dir/config.py", "a", "b"))

        msg = str(exc.value)
        assert "File not found:" in msg
        assert "Did you mean:" not in msg
        assert "src/config.py" not in msg


class TestBuiltinFileToolReadCorrectness:
    """read_file always reflects current disk content (no memoization)."""

    def test_second_read_returns_same_content(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "cached.txt")
        p.write_text("\n".join(f"line{i}" for i in range(1, 11)))
        r1 = asyncio.run(file_tool.read_file(str(p)))
        r2 = asyncio.run(file_tool.read_file(str(p)))
        assert r1 == r2  # no cache: each call re-reads, content identical

    def test_external_size_change_reflected(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "ext_size.txt")
        p.write_text("v1")
        asyncio.run(file_tool.read_file(str(p)))
        p.write_text("v2 with more bytes")
        r2 = asyncio.run(file_tool.read_file(str(p)))
        assert "v2 with more bytes" in r2

    def test_external_mtime_only_change_reflected(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "ext_mtime.txt")
        p.write_text("same-size-a")
        asyncio.run(file_tool.read_file(str(p)))
        p.write_text("same-size-b")  # identical size, different content
        st = p.stat()
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
        r2 = asyncio.run(file_tool.read_file(str(p)))
        assert "same-size-b" in r2

    def test_edit_file_reflected_on_next_read(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "edit.txt")
        p.write_text("hello world")
        asyncio.run(file_tool.read_file(str(p)))
        asyncio.run(file_tool.edit_file(str(p), old_string="world", new_string="agentica"))
        r = asyncio.run(file_tool.read_file(str(p)))
        assert "hello agentica" in r
        assert "hello world" not in r

    def test_write_file_reflected_on_next_read(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "w.txt")
        p.write_text("old content")
        asyncio.run(file_tool.read_file(str(p)))
        asyncio.run(file_tool.write_file(str(p), "new content"))
        r = asyncio.run(file_tool.read_file(str(p)))
        assert "new content" in r
        assert "old content" not in r

    def test_offset_and_limit_select_distinct_windows(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "pages.txt")
        p.write_text("\n".join(f"line{i}" for i in range(1, 21)))
        r1 = asyncio.run(file_tool.read_file(str(p), offset=0, limit=5))
        r2 = asyncio.run(file_tool.read_file(str(p), offset=5, limit=5))
        assert r1 != r2
        assert "line6" in r2 and "line6" not in r1


class TestBuiltinFileToolWriteFile:
    def test_write_new_file(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "new.txt")
        result = asyncio.run(file_tool.write_file(fp, "hello world"))
        assert "Created" in result
        assert Path(fp).read_text() == "hello world"

    def test_write_overwrite_file(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "existing.txt")
        Path(fp).write_text("old")
        result = asyncio.run(file_tool.write_file(fp, "new content"))
        assert "Updated" in result
        assert Path(fp).read_text() == "new content"

    def test_write_creates_parent_dirs(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "a", "b", "c.txt")
        result = asyncio.run(file_tool.write_file(fp, "nested"))
        assert "Created" in result
        assert Path(fp).read_text() == "nested"

    def test_write_returns_absolute_path(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "abs.txt")
        result = asyncio.run(file_tool.write_file(fp, "test"))
        # Result should contain the absolute path
        assert tmp_dir in result


class TestBuiltinFileToolRequestPathAccess:
    """Sandbox / sensitive-path escalation: request_path_access(path, reason)."""

    def _sandboxed_tool(self, tmp_dir, consent_callback=None):
        from agentica.agent.config import SandboxConfig

        sandbox = SandboxConfig(enabled=True, writable_dirs=[tmp_dir])
        return BuiltinFileTool(
            work_dir=tmp_dir, sandbox_config=sandbox, consent_callback=consent_callback
        )

    # -- sandboxed ("auto"/"ask") writable_dirs escalation --------------

    def test_write_outside_writable_dirs_is_blocked(self, tmp_dir):
        with tempfile.TemporaryDirectory() as other_dir:
            tool = self._sandboxed_tool(tmp_dir)
            fp = os.path.join(other_dir, "outside.txt")
            with pytest.raises(PermissionError, match="request_path_access"):
                asyncio.run(tool.write_file(fp, "content"))

    def test_request_path_access_denied_without_consent_callback(self, tmp_dir):
        with tempfile.TemporaryDirectory() as other_dir:
            tool = self._sandboxed_tool(tmp_dir, consent_callback=None)
            result = tool.request_path_access(other_dir, reason="need to edit a config file there")
            data = json.loads(result)
            assert data["granted"] is False

    def test_request_path_access_granted_extends_whitelist_and_unblocks_write(self, tmp_dir):
        with tempfile.TemporaryDirectory() as other_dir:
            consent = MagicMock(return_value="yes")
            tool = self._sandboxed_tool(tmp_dir, consent_callback=consent)

            result = tool.request_path_access(other_dir, reason="need to edit a config file there")
            data = json.loads(result)
            assert data["granted"] is True
            consent.assert_called_once()

            fp = os.path.join(other_dir, "now_allowed.txt")
            write_result = asyncio.run(tool.write_file(fp, "content"))
            assert "Created" in write_result
            assert Path(fp).read_text() == "content"

    def test_request_path_access_denied_by_user(self, tmp_dir):
        with tempfile.TemporaryDirectory() as other_dir:
            consent = MagicMock(return_value="no")
            tool = self._sandboxed_tool(tmp_dir, consent_callback=consent)

            result = tool.request_path_access(other_dir, reason="need to edit a config file there")
            data = json.loads(result)
            assert data["granted"] is False

            fp = os.path.join(other_dir, "still_blocked.txt")
            with pytest.raises(PermissionError):
                asyncio.run(tool.write_file(fp, "content"))

    def test_request_path_access_noop_when_sandbox_disabled(self, tmp_dir):
        tool = BuiltinFileTool(work_dir=tmp_dir, sandbox_config=None)
        result = tool.request_path_access("/some/other/path", reason="anything")
        data = json.loads(result)
        assert data["granted"] is True

    def test_request_path_access_noop_when_already_writable(self, tmp_dir):
        tool = self._sandboxed_tool(tmp_dir)
        result = tool.request_path_access(tmp_dir, reason="already inside work_dir")
        data = json.loads(result)
        assert data["granted"] is True

    # -- sensitive-path escalation (applies in ANY permission mode) -----

    def test_sensitive_write_blocked_even_in_allow_all_mode(self, tmp_dir):
        """allow-all disables sandbox scoping, but sensitive system/credentials
        paths still require explicit escalation — the model shouldn't silently
        touch ~/.ssh or /etc even when the sandbox is fully open."""
        tool = BuiltinFileTool(work_dir=tmp_dir, sandbox_config=None)
        with pytest.raises(PermissionError, match="request_path_access"):
            asyncio.run(tool.write_file("/etc/hosts", "content"))

    def test_sensitive_write_granted_after_user_approval(self, tmp_dir):
        home = str(Path.home())
        ssh_dir = os.path.join(home, ".ssh")
        target = os.path.join(ssh_dir, "authorized_keys")
        consent = MagicMock(return_value="yes")
        tool = BuiltinFileTool(work_dir=tmp_dir, sandbox_config=None, consent_callback=consent)

        result = tool.request_path_access(target, reason="user asked to add a public key")
        data = json.loads(result)
        assert data["granted"] is True
        consent.assert_called_once()
        assert tool._sensitive_write_guard(target) is None

    def test_sensitive_write_stays_blocked_when_denied(self, tmp_dir):
        consent = MagicMock(return_value="no")
        tool = BuiltinFileTool(work_dir=tmp_dir, sandbox_config=None, consent_callback=consent)
        result = tool.request_path_access("/etc/hosts", reason="testing")
        data = json.loads(result)
        assert data["granted"] is False
        with pytest.raises(PermissionError):
            asyncio.run(tool.write_file("/etc/hosts", "content"))

    # -- read-side blocked_paths escalation (sandboxed modes only) ------

    def test_read_blocked_path_component_is_escalatable(self, tmp_dir):
        blocked_dir = os.path.join(tmp_dir, ".ssh")
        os.makedirs(blocked_dir, exist_ok=True)
        target = os.path.join(blocked_dir, "id_rsa")
        Path(target).write_text("fake-key")

        consent = MagicMock(return_value="yes")
        tool = self._sandboxed_tool(tmp_dir, consent_callback=consent)

        with pytest.raises(PermissionError, match="request_path_access"):
            asyncio.run(tool.read_file(target))

        result = tool.request_path_access(target, reason="user asked to inspect this key")
        assert json.loads(result)["granted"] is True

        content = asyncio.run(tool.read_file(target))
        assert "fake-key" in content


class TestBuiltinFileToolApplyPatch:
    def test_registered_as_serial_destructive_raw_string_tool(self, file_tool):
        function = file_tool.functions["apply_patch"]
        function.process_entrypoint(strict=False)

        assert function.is_destructive is True
        assert function.concurrency_safe is False
        assert function.sanitize_arguments is False
        assert function.parameters["required"] == ["patch"]
        assert function.parameters["properties"]["patch"]["type"] == "string"

    def test_description_requires_read_file_before_updates_and_deletes(self, file_tool):
        function = file_tool.functions["apply_patch"]
        function.process_entrypoint(strict=False)

        assert "MUST call read_file before every Update or Delete" in function.description

    def test_applies_update_add_and_delete_in_one_call(self, file_tool, tmp_dir):
        Path(tmp_dir, "app.py").write_text("VALUE = 1\nKEEP = True\n")
        Path(tmp_dir, "obsolete.py").write_text("remove me\n")
        patch_text = """*** Begin Patch
*** Update File: app.py
@@
-VALUE = 1
+VALUE = 2
 KEEP = True
*** Add File: tests/test_app.py
+def test_value():
+    assert True
*** Delete File: obsolete.py
*** End Patch"""

        result = asyncio.run(file_tool.apply_patch(patch_text))

        assert "Successfully applied patch to 3 files" in result
        assert "M app.py (+1 -1)" in result
        assert "A tests/test_app.py (+2 -0)" in result
        assert "D obsolete.py (+0 -1)" in result
        assert Path(tmp_dir, "app.py").read_text() == "VALUE = 2\nKEEP = True\n"
        assert Path(tmp_dir, "tests/test_app.py").read_text() == (
            "def test_value():\n    assert True"
        )
        assert not Path(tmp_dir, "obsolete.py").exists()

    def test_failed_later_hunk_writes_nothing(self, file_tool, tmp_dir):
        first = Path(tmp_dir, "first.py")
        second = Path(tmp_dir, "second.py")
        first.write_text("FIRST = 1\n")
        second.write_text("SECOND = 2\n")
        patch_text = """*** Begin Patch
*** Update File: first.py
@@
-FIRST = 1
+FIRST = 10
*** Update File: second.py
@@
-STALE = 2
+SECOND = 20
*** End Patch"""

        with pytest.raises(ValueError, match="second.py"):
            asyncio.run(file_tool.apply_patch(patch_text))

        assert first.read_text() == "FIRST = 1\n"
        assert second.read_text() == "SECOND = 2\n"

    def test_add_existing_file_writes_nothing(self, file_tool, tmp_dir):
        existing = Path(tmp_dir, "existing.py")
        existing.write_text("keep\n")
        patch_text = """*** Begin Patch
*** Add File: existing.py
+replace
*** End Patch"""

        with pytest.raises(FileExistsError, match="existing.py"):
            asyncio.run(file_tool.apply_patch(patch_text))

        assert existing.read_text() == "keep\n"

    def test_reports_all_file_and_hunk_preflight_failures(self, file_tool, tmp_dir):
        first = Path(tmp_dir, "first.py")
        second = Path(tmp_dir, "second.py")
        existing = Path(tmp_dir, "existing.py")
        first.write_text("FIRST = 1\n")
        second.write_text("SECOND = 2\n")
        existing.write_text("keep\n")
        patch_text = """*** Begin Patch
*** Update File: first.py
@@
-STALE_FIRST = 1
+FIRST = 10
@@
-STALE_FIRST_AGAIN = 1
+FIRST = 11
*** Update File: second.py
@@
-STALE_SECOND = 2
+SECOND = 20
*** Add File: existing.py
+replace
*** End Patch"""

        with pytest.raises(ValueError) as exc:
            asyncio.run(file_tool.apply_patch(patch_text))

        message = str(exc.value)
        assert "Patch preflight failed for 3 files" in message
        assert "- first.py:" in message
        assert "Hunk 1: context not found" in message
        assert "Hunk 2: context not found" in message
        assert "- second.py:" in message
        assert "- existing.py:" in message
        assert "short unique context" in message
        assert first.read_text() == "FIRST = 1\n"
        assert second.read_text() == "SECOND = 2\n"
        assert existing.read_text() == "keep\n"

    def test_context_failure_shows_actual_content(self, file_tool, tmp_dir):
        """Stale-context hunks show the actual current lines next to the expected ones."""
        target = Path(tmp_dir, "hello.txt")
        target.write_text("alpha\nbeta-current\ngamma\n")
        patch_text = """*** Begin Patch
*** Update File: hello.txt
@@
 alpha
-beta
 gamma
*** End Patch"""

        with pytest.raises(ValueError) as exc:
            asyncio.run(file_tool.apply_patch(patch_text))

        message = str(exc.value)
        assert "Expected context:" in message
        assert "Actual from line 1:" in message
        assert "  beta-current" in message
        assert "Read or re-read" in message
        assert "read_file" in message

    def test_absolute_patch_path_is_reported_relative_to_work_dir(self, file_tool, tmp_dir):
        target = Path(tmp_dir, "pkg", "app.py")
        target.parent.mkdir()
        target.write_text("VALUE = 1\n")
        patch_text = f"""*** Begin Patch
*** Update File: {target}
@@
-VALUE = 1
+VALUE = 2
*** End Patch"""

        result = asyncio.run(file_tool.apply_patch(patch_text))

        assert "M pkg/app.py (+1 -1)" in result
        assert tmp_dir not in result

    def test_symlink_patch_path_keeps_lexical_work_dir_path(self, file_tool, tmp_dir):
        with tempfile.TemporaryDirectory() as outside_dir:
            outside = Path(outside_dir)
            target = outside / "app.py"
            target.write_text("VALUE = 1\n")
            link = Path(tmp_dir, "linked")
            link.symlink_to(outside, target_is_directory=True)
            patch_text = """*** Begin Patch
*** Update File: linked/app.py
@@
-VALUE = 1
+VALUE = 2
*** End Patch"""

            result = asyncio.run(file_tool.apply_patch(patch_text))

            assert "M linked/app.py (+1 -1)" in result
            assert outside_dir not in result
            assert target.read_text() == "VALUE = 2\n"

    @pytest.mark.parametrize(
        ("patch_path", "exception_type"),
        (("missing.py", FileNotFoundError), ("directory", IsADirectoryError)),
    )
    def test_single_preflight_keeps_filesystem_exception_type(
            self, file_tool, tmp_dir, patch_path, exception_type
    ):
        Path(tmp_dir, "directory").mkdir()
        patch_text = f"""*** Begin Patch
*** Update File: {patch_path}
@@
-OLD = 1
+NEW = 1
*** End Patch"""

        with pytest.raises(exception_type, match=patch_path):
            asyncio.run(file_tool.apply_patch(patch_text))

    def test_sandbox_blocks_any_target_before_writing(self, tmp_dir):
        from agentica.agent.config import SandboxConfig

        with tempfile.TemporaryDirectory() as outside_dir:
            tool = BuiltinFileTool(
                work_dir=tmp_dir,
                sandbox_config=SandboxConfig(enabled=True, writable_dirs=[tmp_dir]),
            )
            outside = Path(outside_dir, "outside.py")
            patch_text = f"""*** Begin Patch
*** Add File: {outside}
+blocked = True
*** End Patch"""

            with pytest.raises(PermissionError, match="not allowed"):
                asyncio.run(tool.apply_patch(patch_text))

            assert not outside.exists()


class TestBuiltinFileToolEditFile:
    @staticmethod
    def _read(file_tool, file_path):
        asyncio.run(file_tool.read_file(file_path))

    def test_edit_without_read_succeeds(self, file_tool, tmp_dir):
        """No freshness machinery: edit works without a prior read, no tips."""
        fp = os.path.join(tmp_dir, "unread.txt")
        Path(fp).write_text("before")
        result = asyncio.run(file_tool.edit_file(fp, "before", "after"))
        assert "Successfully" in result
        assert "Tip:" not in result
        assert Path(fp).read_text() == "after"

    def test_edit_after_external_change_succeeds(self, file_tool, tmp_dir):
        """External on-disk change is invisible to edit_file — no tip, plain edit."""
        fp = os.path.join(tmp_dir, "external.txt")
        Path(fp).write_text("before")
        self._read(file_tool, fp)
        Path(fp).write_text("after!")
        result = asyncio.run(file_tool.edit_file(fp, "after!", "final!"))
        assert "Successfully" in result
        assert "changed on disk" not in result
        assert Path(fp).read_text() == "final!"

    def test_edit_failure_hints_read_or_reread(self, file_tool, tmp_dir):
        """A failed exact-string edit gives one stateless recovery action."""
        fp = os.path.join(tmp_dir, "not_found.txt")
        Path(fp).write_text("before")
        with pytest.raises(ValueError) as exc:
            asyncio.run(file_tool.edit_file(fp, "missing", "after"))
        msg = str(exc.value)
        assert "String not found" in msg
        assert "Current file state:" not in msg
        assert "Read or re-read the relevant region" in msg
        assert "read_file" in msg
        assert "old_string" in msg

    def test_edit_string_not_found_does_not_show_fuzzy_region(self, file_tool, tmp_dir):
        """A stale-guess old_string stays a stateless exact-match failure."""
        fp = os.path.join(tmp_dir, "test_writer.py")
        Path(fp).write_text(
            "def test_writer():\n"
            '    assert search_calls == ["saved preference"]\n'
            "    assert documents == []\n"
        )
        with pytest.raises(ValueError) as exc:
            asyncio.run(
                file_tool.edit_file(
                    fp,
                    'assert search_calls == ["new preference"]\nassert documents == []',
                    'assert search_calls == ["x"]\nassert documents == []',
                )
            )
        msg = str(exc.value)
        assert "String not found" in msg
        assert "Most similar region" not in msg
        assert 'assert search_calls == ["saved preference"]' not in msg
        assert "Read or re-read the relevant region" in msg

    def test_single_edit(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "edit.txt")
        Path(fp).write_text("hello world")
        self._read(file_tool, fp)
        result = asyncio.run(file_tool.edit_file(fp, "world", "python"))
        assert "Successfully" in result
        assert Path(fp).read_text() == "hello python"

    def test_multiple_edits_via_separate_calls(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "multi.txt")
        Path(fp).write_text("aaa bbb ccc")
        self._read(file_tool, fp)
        result1 = asyncio.run(file_tool.edit_file(fp, "aaa", "111"))
        assert "Successfully" in result1
        result2 = asyncio.run(file_tool.edit_file(fp, "ccc", "333"))
        assert "Successfully" in result2
        assert Path(fp).read_text() == "111 bbb 333"

    def test_python_error_hint_null_literal(self):
        """NameError on JSON `null`/`true`/`false` gets a structured hint
        pointing the LLM at the source rather than retrying blindly."""
        from agentica.tools.buildin_tools import _detect_python_error_hint
        sample = "Traceback ...\n  File ...\nNameError: name 'null' is not defined"
        hint = _detect_python_error_hint(sample)
        assert hint is not None
        assert "JSON literal" in hint
        assert "None" in hint

    def test_python_error_hint_syntax_error(self):
        from agentica.tools.buildin_tools import _detect_python_error_hint
        sample = "  File 'x.py', line 5\n    if x = 1\nSyntaxError: invalid syntax"
        hint = _detect_python_error_hint(sample)
        assert hint is not None
        assert "SyntaxError" in hint

    def test_python_error_hint_module_not_found(self):
        from agentica.tools.buildin_tools import _detect_python_error_hint
        sample = "ModuleNotFoundError: No module named 'foobar_widget'"
        hint = _detect_python_error_hint(sample)
        assert hint is not None
        assert "dependency" in hint or "pip install" in hint

    def test_python_error_hint_returns_none_for_normal_output(self):
        from agentica.tools.buildin_tools import _detect_python_error_hint
        assert _detect_python_error_hint("") is None
        assert _detect_python_error_hint("Hello world") is None
        assert _detect_python_error_hint("AssertionError: 1 != 2") is None  # genuine logic bug

    def test_edit_replace_all(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "replall.txt")
        Path(fp).write_text("x x x")
        self._read(file_tool, fp)
        result = asyncio.run(file_tool.edit_file(fp, "x", "y", replace_all=True))
        assert "Successfully" in result
        assert Path(fp).read_text() == "y y y"

    def test_quote_fallback_preserves_unrelated_typographic_quotes(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "quotes.txt")
        Path(fp).write_text(
            'title = \u201ckeep\u201d\nvalue = \u2018old\u2019\n',
            encoding="utf-8",
        )
        result = asyncio.run(file_tool.edit_file(fp, "value = 'old'", "value = 'new'"))
        assert "Successfully" in result
        assert Path(fp).read_text(encoding="utf-8") == (
            'title = \u201ckeep\u201d\nvalue = \'new\'\n'
        )

    def test_quote_fallback_replace_all_preserves_unrelated_content(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "quotes_all.txt")
        Path(fp).write_text(
            'title = \u201ckeep\u201d\nvalue = \'old\'\nvalue = \'old\'\n',
            encoding="utf-8",
        )
        result = asyncio.run(file_tool.edit_file(
            fp,
            "value = \u2018old\u2019",
            "value = 'new'",
            replace_all=True,
        ))
        assert "Successfully" in result
        assert Path(fp).read_text(encoding="utf-8") == (
            'title = \u201ckeep\u201d\nvalue = \'new\'\nvalue = \'new\'\n'
        )

    def test_edit_string_not_found(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "nf.txt")
        Path(fp).write_text("hello")
        self._read(file_tool, fp)
        with pytest.raises(ValueError):
            asyncio.run(file_tool.edit_file(fp, "zzz", "yyy"))
        # File should be unchanged
        assert Path(fp).read_text() == "hello"

    def test_edit_nonexistent_file(self, file_tool):
        with pytest.raises(FileNotFoundError):
            asyncio.run(file_tool.edit_file("/no/such/file.txt", "a", "b"))

    def test_edit_multiple_matches_no_replace_all(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "dup.txt")
        Path(fp).write_text("foo bar foo")
        self._read(file_tool, fp)
        with pytest.raises(ValueError, match="Found 2 occurrences"):
            asyncio.run(file_tool.edit_file(fp, "foo", "baz"))
        # File unchanged
        assert Path(fp).read_text() == "foo bar foo"

    def test_edit_no_side_effect_on_failure(self, file_tool, tmp_dir):
        """A failed edit should not modify the file."""
        fp = os.path.join(tmp_dir, "atomic.txt")
        Path(fp).write_text("aaa bbb")
        self._read(file_tool, fp)
        with pytest.raises(ValueError):
            asyncio.run(file_tool.edit_file(fp, "zzz", "999"))
        # File should be unchanged
        assert Path(fp).read_text() == "aaa bbb"


class TestRemovedMultiEditFileTool:
    """multi_edit_file is no longer exposed; use apply_patch for multi-hunk edits."""

    def test_not_registered(self):
        tk = BuiltinFileTool()
        assert "multi_edit_file" not in tk.functions

    def test_other_tools_still_use_auto_schema(self):
        """Sanity: parameters_override is opt-in. Tools that don't set it
        must still get their schema auto-derived from the signature."""
        tk = BuiltinFileTool()
        fn = tk.functions["read_file"]
        assert fn.parameters_override is None
        fn.process_entrypoint(strict=False)
        # auto-derived schema should have file_path as a string property
        assert fn.parameters["properties"]["file_path"]["type"] == "string"


class TestBuiltinFileToolGlob:
    def test_glob_py_files(self, file_tool, tmp_dir):
        Path(tmp_dir, "a.py").write_text("")
        Path(tmp_dir, "b.py").write_text("")
        Path(tmp_dir, "c.txt").write_text("")
        result = asyncio.run(file_tool.glob("*.py", tmp_dir))
        files = json.loads(result)
        assert len(files) == 2
        assert all(f.endswith(".py") for f in files)

    def test_glob_recursive(self, file_tool, tmp_dir):
        sub = Path(tmp_dir, "sub")
        sub.mkdir()
        Path(sub, "deep.py").write_text("")
        Path(tmp_dir, "top.py").write_text("")
        result = asyncio.run(file_tool.glob("**/*.py", tmp_dir))
        files = json.loads(result)
        assert len(files) == 2

    def test_glob_no_matches(self, file_tool, tmp_dir):
        result = asyncio.run(file_tool.glob("*.xyz", tmp_dir))
        files = json.loads(result)
        assert files == []

    def test_glob_nonexistent_dir(self, file_tool):
        with pytest.raises(FileNotFoundError):
            asyncio.run(file_tool.glob("*", "/nonexistent_dir_xyz"))


class TestBuiltinFileToolGrep:
    def test_grep_default_returns_content_with_line_numbers(self, file_tool, tmp_dir):
        # Default output_mode is "content" — must return matching lines with
        # line numbers, not just a path list. A bare path-only response was the
        # root cause of dumb-model retry loops where the model couldn't tell
        # whether it had actually seen the code yet.
        Path(tmp_dir, "a.txt").write_text("hello world\n")
        Path(tmp_dir, "b.txt").write_text("goodbye world\n")
        Path(tmp_dir, "c.txt").write_text("nothing here\n")
        result = asyncio.run(file_tool.grep("hello", tmp_dir))
        assert "a.txt" in result
        assert "hello world" in result, "default mode must include matched line content"
        assert "c.txt" not in result

    def test_grep_files_with_matches_mode_returns_paths_only(self, file_tool, tmp_dir):
        Path(tmp_dir, "a.txt").write_text("hello world\n")
        Path(tmp_dir, "c.txt").write_text("nothing here\n")
        result = asyncio.run(
            file_tool.grep("hello", tmp_dir, output_mode="files_with_matches")
        )
        assert "a.txt" in result
        assert "hello world" not in result
        assert "c.txt" not in result

    def test_grep_content_mode(self, file_tool, tmp_dir):
        Path(tmp_dir, "code.py").write_text("def foo():\n    pass\ndef bar():\n    pass\n")
        result = asyncio.run(file_tool.grep("def", tmp_dir, output_mode="content"))
        assert "def foo" in result
        assert "def bar" in result

    def test_grep_no_matches(self, file_tool, tmp_dir):
        Path(tmp_dir, "empty.txt").write_text("nothing\n")
        result = asyncio.run(file_tool.grep("zzzzz", tmp_dir))
        assert "No matches" in result

    def test_grep_nonexistent_path(self, file_tool):
        with pytest.raises(FileNotFoundError, match="Path not found"):
            asyncio.run(file_tool.grep("test", "/nonexistent_xyz"))

    def test_grep_accepts_file_path(self, file_tool, tmp_dir):
        fp = Path(tmp_dir, "single.py")
        fp.write_text("gate_passed = True\n")
        result = asyncio.run(file_tool.grep("gate_passed", str(fp), include="*.py"))
        assert "gate_passed = True" in result

    def test_grep_fallback_accepts_file_path(self, file_tool, tmp_dir):
        fp = Path(tmp_dir, "single.py")
        fp.write_text("commit_pass = True\n")
        with patch("agentica.tools.buildin_tools.shutil.which", return_value=None):
            result = asyncio.run(file_tool.grep("commit_pass", str(fp), include="*.py"))
        assert "commit_pass = True" in result

    def test_grep_case_insensitive(self, file_tool, tmp_dir):
        Path(tmp_dir, "mixed.txt").write_text("Hello WORLD\n")
        result = asyncio.run(file_tool.grep("hello", tmp_dir, case_insensitive=True, output_mode="content"))
        assert "Hello" in result

    def test_grep_fixed_strings(self, file_tool, tmp_dir):
        Path(tmp_dir, "regex.txt").write_text("price is $10.00\n")
        # $ and . are special in regex; fixed_strings should match literally
        result = asyncio.run(file_tool.grep("$10.00", tmp_dir, fixed_strings=True, output_mode="content"))
        assert "$10.00" in result

    def test_grep_manages_own_timeout(self, file_tool):
        """grep must self-limit so the outer 120s executor wrapper is skipped."""
        fn = file_tool.functions["grep"]
        assert fn.manages_own_timeout is True

    def test_grep_fallback_times_out(self, tmp_dir):
        """When rg is unavailable, the pure-Python fallback still hard-times-out
        instead of running up to the outer 120s executor limit."""
        import time as _time
        from agentica.tools import buildin_tools
        tool = BuiltinFileTool(work_dir=tmp_dir)

        # Slow sync fallback worker; the real _run_grep_fallback wraps it with
        # asyncio.wait_for, so the timeout fires well before this returns.
        def slow_worker(*args, **kwargs):
            _time.sleep(0.5)
            return "should not reach"
        tool._grep_fallback = slow_worker

        with patch("agentica.tools.buildin_tools.shutil.which", return_value=None), \
             patch("agentica.tools.buildin_tools._GREP_TIMEOUT", 0.1):
            with pytest.raises(TimeoutError, match=r"grep timed out"):
                asyncio.run(tool.grep("x", str(tmp_dir)))

    def test_grep_timeout_arg_respected(self, tmp_dir):
        """The LLM-passed `timeout` arg is used as-is, overriding the default
        (no clamping, no upper cap)."""
        import time as _time
        tool = BuiltinFileTool(work_dir=tmp_dir)

        def slow_worker(*args, **kwargs):
            _time.sleep(2)
            return "should not reach"
        tool._grep_fallback = slow_worker

        # Patch the default _GREP_TIMEOUT up to 100s; passing timeout=1 must
        # still fire at 1s, proving the caller's value wins and is not clamped
        # back toward the default.
        with patch("agentica.tools.buildin_tools._GREP_TIMEOUT", 100), \
             patch("agentica.tools.buildin_tools.shutil.which", return_value=None):
            with pytest.raises(TimeoutError, match=r"grep timed out after 1 seconds"):
                asyncio.run(tool.grep("x", str(tmp_dir), timeout=1))

    def test_grep_default_timeout_still_bounds(self, tmp_dir):
        """When no timeout arg is passed, the module default still bounds the
        search (a timeout must always be set — bad disk / regex hang)."""
        import time as _time
        tool = BuiltinFileTool(work_dir=tmp_dir)

        def slow_worker(*args, **kwargs):
            _time.sleep(2)
            return "should not reach"
        tool._grep_fallback = slow_worker

        with patch("agentica.tools.buildin_tools._GREP_TIMEOUT", 1), \
             patch("agentica.tools.buildin_tools.shutil.which", return_value=None):
            with pytest.raises(TimeoutError, match=r"grep timed out after 1 seconds"):
                asyncio.run(tool.grep("x", str(tmp_dir)))

    def test_grep_cancellation_cleans_up_subprocess(self, tmp_dir):
        async def cancel_running_grep():
            process = BlockingSubprocess()
            cleanup = AsyncMock()
            with patch(
                "agentica.tools.buildin_tools.asyncio.create_subprocess_exec",
                new=AsyncMock(return_value=process),
            ), patch(
                "agentica.tools.buildin_tools.terminate_subprocess",
                cleanup,
            ), patch(
                "agentica.tools.buildin_tools.shutil.which",
                return_value="/usr/bin/rg",
            ):
                tool = BuiltinFileTool(work_dir=tmp_dir)
                task = asyncio.create_task(tool.grep("needle", str(tmp_dir)))
                await process.started.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

            cleanup.assert_awaited_once_with(process)

        asyncio.run(cancel_running_grep())


# ===========================================================================
# BuiltinExecuteTool tests
# ===========================================================================

class TestBuiltinExecuteTool:
    def test_execute_registered_as_raw_string_tool(self, execute_tool):
        function = execute_tool.functions["execute"]
        function.process_entrypoint(strict=False)

        assert function.sanitize_arguments is False
        assert "passed unchanged" in function.description

    def test_background_command_management_tools_not_registered(self, execute_tool):
        assert "list_background_commands" not in execute_tool.functions
        assert "stop_background_command" not in execute_tool.functions

    def test_execute_simple_command(self, execute_tool):
        result = asyncio.run(execute_tool.execute("echo hello"))
        assert "hello" in result

    def test_execute_background_registers_process(self, tmp_dir, monkeypatch):
        agentica_home = Path(tmp_dir) / "agentica-home"
        monkeypatch.setenv("AGENTICA_HOME", str(agentica_home))
        registry = BackgroundProcessRegistry(user_id="alice@example.com")
        tool = BuiltinExecuteTool(
            work_dir=tmp_dir,
            background_process_registry=registry,
        )
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote('import time; time.sleep(30)')}"

        result = asyncio.run(tool.execute(command, background=True))

        try:
            running = registry.list()
            assert len(running) == 1
            item = running[0]
            assert f"PID {item.pid}" in result
            assert f"id: {item.id}" in result
            assert f"/stop {item.id}" in result
            assert f"kill -- -{item.pid}" in result
            assert "/stop" in result
            assert Path(item.log_path).exists()
            assert str(Path(tmp_dir) / ".agentica") not in item.log_path
            assert str(agentica_home / "projects" / "alice@example.com") in item.log_path
            assert str(agentica_home / "projects" / "default") not in item.log_path
        finally:
            registry.stop()
        assert registry.running_count() == 0

    def test_execute_background_result_tells_the_model_not_to_wait(self, tmp_dir):
        """The model followed the old "inspect progress by reading the log" text
        with `execute("sleep 70; tail ...")`, which re-blocked the very turn
        backgrounding had just freed. The result must state that the exit goes
        to the user and that waiting is wrong."""
        registry = BackgroundProcessRegistry()
        tool = BuiltinExecuteTool(work_dir=tmp_dir, background_process_registry=registry)
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote('import time; time.sleep(30)')}"

        try:
            result = asyncio.run(tool.execute(command, background=True))
        finally:
            registry.stop()

        assert "reported to the user, not to you" in result
        assert "no sleep, no polling, no blocking tail" in result
        assert "Inspect progress" not in result
        # The only sanctioned way back into this conversation.
        assert 'wait(id="term_1")' in result

    def test_execute_runs_self_detaching_command_but_flags_it(self, execute_tool):
        """`nohup ... &` stays allowed — it is a standard idiom and a caller may
        want a raw orphan — but the result has to say what it costs, since /ps,
        /stop and the completion notice all miss an untracked child."""
        result = asyncio.run(execute_tool.execute("nohup echo started > /dev/null 2>&1 &"))

        assert "It is untracked" in result
        assert "background=True" in result

    def test_execute_refuses_self_detaching_command_in_background_mode(self, execute_tool):
        """Here the '&' is not merely untracked but wrong: the registry would
        watch a shell that exits at once and announce a completion while the
        command is still running."""
        with pytest.raises(ValueError, match="Remove the trailing '&'"):
            asyncio.run(execute_tool.execute("python3 run.py &", background=True))

    def test_execute_leaves_plain_commands_unflagged(self, execute_tool):
        """`2>&1` contains an ampersand without detaching anything."""
        result = asyncio.run(execute_tool.execute("echo ok 2>&1"))

        assert "ok" in result
        assert "untracked" not in result

    def test_execute_refuses_long_foreground_sleep(self, execute_tool):
        """The observed poll: background the job, then `sleep 330 && tail log`,
        which re-blocks the turn backgrounding had just freed."""
        with pytest.raises(ValueError, match="Refusing to hold this turn") as excinfo:
            asyncio.run(execute_tool.execute("sleep 330 && tail -2 /tmp/run.log"))

        # A caller waiting on something Agentica does not track needs the correct
        # form, not just a refusal.
        assert "until curl -sf" in str(excinfo.value)
        # The refusal must name the primitive that replaces the blind sleep.
        assert "wait(id=...)" in str(excinfo.value)

    def test_execute_allows_retry_loop_waiting_on_external_condition(self, execute_tool):
        """The recommended form exits on success, so it must not be refused."""
        result = asyncio.run(
            execute_tool.execute("until echo ready; do sleep 5; done")
        )
        assert "ready" in result

    def test_execute_allows_short_sleep_for_service_startup(self, execute_tool):
        result = asyncio.run(execute_tool.execute("sleep 1 && echo up"))
        assert "up" in result

    def test_wait_returns_as_soon_as_the_command_exits(self, tmp_dir):
        """The point of `wait` over `sleep N`: a generous timeout costs only as
        much wall time as the command itself."""
        registry = BackgroundProcessRegistry()
        tool = BuiltinExecuteTool(work_dir=tmp_dir, background_process_registry=registry)
        script = 'import time; time.sleep(1); print("summary ready")'
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

        async def scenario():
            started = await tool.execute(command, background=True)
            item_id = registry.list(include_finished=True)[0].id
            assert f'wait(id="{item_id}")' in started
            return await tool.wait(item_id, timeout=120)

        began = time.monotonic()
        try:
            result = asyncio.run(scenario())
        finally:
            registry.stop()
        elapsed = time.monotonic() - began

        assert "exited with code 0" in result
        assert "summary ready" in result
        assert elapsed < 30

    def test_wait_reports_progress_without_stopping_the_command(self, tmp_dir):
        registry = BackgroundProcessRegistry()
        tool = BuiltinExecuteTool(work_dir=tmp_dir, background_process_registry=registry)
        script = 'import time; print("phase 1", flush=True); time.sleep(30)'
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

        async def scenario():
            await tool.execute(command, background=True)
            item_id = registry.list()[0].id
            return await tool.wait(item_id, timeout=1)

        try:
            result = asyncio.run(scenario())
            assert "still running" in result
            assert "phase 1" in result
            assert registry.running_count() == 1
            # A job on the scale of hours must not be waited on in a loop; the
            # user's completion notice is what should drive the next step.
            assert "stop waiting: end your turn" in result
        finally:
            registry.stop()

    def test_wait_on_finished_command_returns_immediately(self, tmp_dir):
        """A command that finished while the agent did something else must still
        be reportable — otherwise the result is lost to the conversation."""
        registry = BackgroundProcessRegistry()
        tool = BuiltinExecuteTool(work_dir=tmp_dir, background_process_registry=registry)
        script = 'print("early")'
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

        async def scenario():
            await tool.execute(command, background=True)
            item = registry.list(include_finished=True)[0]
            assert item.finished.wait(timeout=30)
            return await tool.wait(item.id, timeout=300)

        try:
            result = asyncio.run(scenario())
        finally:
            registry.stop()

        assert "exited with code 0" in result
        assert "early" in result

    def test_wait_reports_a_failing_command_exit_code(self, tmp_dir):
        registry = BackgroundProcessRegistry()
        tool = BuiltinExecuteTool(work_dir=tmp_dir, background_process_registry=registry)
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote('raise SystemExit(3)')}"

        async def scenario():
            await tool.execute(command, background=True)
            return await tool.wait(registry.list(include_finished=True)[0].id, timeout=60)

        try:
            result = asyncio.run(scenario())
        finally:
            registry.stop()

        assert "exited with code 3" in result

    def test_wait_on_unknown_id_lists_known_ids(self, tmp_dir):
        registry = BackgroundProcessRegistry()
        tool = BuiltinExecuteTool(work_dir=tmp_dir, background_process_registry=registry)
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote('print(1)')}"

        async def scenario():
            await tool.execute(command, background=True)
            await tool.wait("term_99", timeout=1)

        try:
            with pytest.raises(ValueError, match="No background command 'term_99'") as excinfo:
                asyncio.run(scenario())
        finally:
            registry.stop()

        assert "term_1" in str(excinfo.value)

    def test_wait_caps_a_single_call(self, execute_tool):
        """One call must not hold the turn indefinitely: the caller returns
        through the model loop, which is where the user can interrupt."""
        assert execute_tool.functions["wait"].manages_own_timeout is True

        registry = execute_tool._background_process_registry
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote('import time; time.sleep(30)')}"

        async def scenario():
            await execute_tool.execute(command, background=True)
            item_id = registry.list()[0].id
            with patch("agentica.tools.buildin_tools._MAX_WAIT_SECONDS", 1):
                return await execute_tool.wait(item_id, timeout=300)

        try:
            result = asyncio.run(scenario())
        finally:
            registry.stop()

        assert "still running" in result

    def test_execute_background_emits_completion_event(self, tmp_dir, monkeypatch):
        agentica_home = Path(tmp_dir) / "agentica-home"
        monkeypatch.setenv("AGENTICA_HOME", str(agentica_home))
        registry = BackgroundProcessRegistry(user_id="alice@example.com")
        tool = BuiltinExecuteTool(
            work_dir=tmp_dir,
            background_process_registry=registry,
        )
        script = 'print("done")'
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

        result = asyncio.run(tool.execute(command, background=True))
        event = registry.wait_completed(timeout=5)

        assert "Started background command #1" in result
        assert event.id == "term_1"
        assert event.num == 1
        assert event.returncode == 0
        assert event.stop_requested is False
        assert "done" in Path(event.log_path).read_text(encoding="utf-8")
        assert registry.running_count() == 0

    def test_background_stop_marks_completion_event_as_stop_requested(self, tmp_dir):
        registry = BackgroundProcessRegistry()
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote('import time; time.sleep(30)')}"
        item = registry.start(command, cwd=tmp_dir)

        stopped = registry.stop(item.id)
        event = registry.wait_completed(timeout=5)

        assert stopped == [item]
        assert event.id == item.id
        assert event.stop_requested is True
        with pytest.raises(queue.Empty):
            registry.wait_completed(timeout=0.01)

    def test_background_stop_tolerates_wait_after_sigkill_timeout(self):
        registry = BackgroundProcessRegistry()
        process = MagicMock(pid=12345)
        process.poll.return_value = None
        process.wait.side_effect = [
            subprocess.TimeoutExpired("command", 2),
            subprocess.TimeoutExpired("command", 2),
        ]
        item = BackgroundProcess(
            id="term_1",
            num=1,
            process=process,
            command="command",
            cwd=None,
            log_path="/tmp/term_1.log",
            started_at=0,
        )
        registry._items[item.id] = item

        with patch("agentica.tools.background_processes.os.killpg") as killpg:
            stopped = registry.stop(item.id)

        assert stopped == [item]
        assert killpg.call_count == 2

    def test_background_start_removes_log_when_popen_fails(self, tmp_dir, monkeypatch):
        agentica_home = Path(tmp_dir) / "agentica-home"
        monkeypatch.setenv("AGENTICA_HOME", str(agentica_home))
        registry = BackgroundProcessRegistry()

        with patch(
            "agentica.tools.background_processes.subprocess.Popen",
            side_effect=OSError("cannot start"),
        ), pytest.raises(OSError, match="cannot start"):
            registry.start("command", cwd=str(tmp_dir))

        assert list(agentica_home.rglob("*.log")) == []

    def test_execute_returns_exit_code_on_failure(self, execute_tool):
        with pytest.raises(RuntimeError, match="exit(ed)? (with )?code 42"):
            asyncio.run(execute_tool.execute("exit 42"))

    def test_execute_treats_python_module_linter_exit_one_as_diagnostics(self, execute_tool, tmp_dir):
        Path(tmp_dir, "ruff.py").write_text(
            "import sys\n"
            "print('UP009 UTF-8 encoding declaration is unnecessary')\n"
            "sys.exit(1)\n"
        )

        result = asyncio.run(
            execute_tool.execute(
                f"PYTHONPATH={shlex.quote(tmp_dir)} python3 -m ruff check sample.py"
            )
        )

        assert "UP009" in result
        assert "[Exit code: 1]" in result
        assert "Diagnostics found" in result

    def test_execute_still_raises_for_plain_python3_exit_one(self, execute_tool):
        with pytest.raises(RuntimeError, match="Command exited with code 1"):
            asyncio.run(execute_tool.execute("python3 -c 'import sys; sys.exit(1)'"))

    def test_execute_captures_stderr(self, execute_tool):
        result = asyncio.run(execute_tool.execute("echo error_msg >&2"))
        assert "error_msg" in result

    def test_execute_timeout(self):
        tool = BuiltinExecuteTool(timeout=1)
        with pytest.raises(TimeoutError, match="timed out"):
            asyncio.run(tool.execute("sleep 30"))

    @pytest.mark.skipif(os.name == "nt", reason="POSIX process-group cleanup")
    def test_execute_cancellation_reaps_subprocess_group(self, tmp_dir):
        pid_file = Path(tmp_dir, "child.pid")
        script = (
            "import os, time; "
            f"open({str(pid_file)!r}, 'w').write(str(os.getpid())); "
            "time.sleep(60)"
        )
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"
        tool = BuiltinExecuteTool(work_dir=tmp_dir)
        child_pid = None

        async def cancel_running_command():
            task = asyncio.create_task(tool.execute(command))
            for _ in range(200):
                if pid_file.exists():
                    break
                await asyncio.sleep(0.01)
            assert pid_file.exists(), "child process did not start"

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            pid = int(pid_file.read_text())
            for _ in range(200):
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    return pid
                await asyncio.sleep(0.01)
            return pid

        try:
            child_pid = asyncio.run(cancel_running_command())
            with pytest.raises(ProcessLookupError):
                os.kill(child_pid, 0)
        finally:
            if child_pid is not None:
                try:
                    os.kill(child_pid, 9)
                except ProcessLookupError:
                    pass

    def test_execute_python_code(self, execute_tool):
        result = asyncio.run(execute_tool.execute("python3 -c 'print(2+3)'"))
        assert "5" in result

    def test_execute_preserves_python_single_quotes(self, execute_tool):
        python = shlex.quote(sys.executable)
        command = f'{python} -c "from pathlib import Path; print(Path(\'.\').resolve().name)"'

        result = asyncio.run(execute_tool.execute(command))

        assert result == Path(execute_tool._work_dir).name

    def test_execute_preserves_literal_backslash_n(self, execute_tool):
        python = shlex.quote(sys.executable)
        command = f'''{python} -c 'print("\\n".join(["a", "b"]))' '''.strip()

        result = asyncio.run(execute_tool.execute(command))

        assert result == "a\nb"

    def test_execute_call_arguments_remain_exact(self, execute_tool):
        from agentica.tools.base import get_function_call

        execute_tool.functions["execute"].process_entrypoint(strict=False)
        commands = [
            "true",
            "  printf preserved  ",
            'python3 -c "print(True, False, None)"',
        ]
        for command in commands:
            function_call = get_function_call(
                "execute",
                json.dumps({"command": command, "timeout": "5"}),
                functions=execute_tool.functions,
            )
            assert function_call.error is None
            assert function_call.arguments["command"] == command
            assert function_call.arguments["timeout"] == 5

    def test_execute_multiline_python(self, execute_tool):
        cmd = '''python3 -c "def f(n):
    return n * 2
print(f(21))"'''
        result = asyncio.run(execute_tool.execute(cmd))
        assert "42" in result

    def test_execute_cwd(self, tmp_dir):
        tool = BuiltinExecuteTool(work_dir=tmp_dir)
        result = asyncio.run(tool.execute("pwd"))
        assert tmp_dir in result


class TestShellTool:
    def test_execute_preserves_command(self, tmp_dir):
        tool = ShellTool(work_dir=tmp_dir)
        python = shlex.quote(sys.executable)
        command = f'''{python} -c 'print("\\n".join(["a", "b"]))' '''.strip()

        result = asyncio.run(tool.execute(command))

        assert tool.functions["execute"].sanitize_arguments is False
        assert result == "a\nb"

    def test_execute_cancellation_cleans_up_subprocess(self, tmp_dir):
        async def cancel_running_command():
            process = BlockingSubprocess()
            cleanup = AsyncMock()
            with patch(
                "agentica.tools.shell_tool.asyncio.create_subprocess_shell",
                new=AsyncMock(return_value=process),
            ), patch(
                "agentica.tools.shell_tool.terminate_subprocess",
                cleanup,
            ):
                tool = ShellTool(work_dir=tmp_dir)
                task = asyncio.create_task(tool.execute("sleep 60"))
                await process.started.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

            cleanup.assert_awaited_once_with(process, process_group=True)

        asyncio.run(cancel_running_command())


# ===========================================================================
# BuiltinWebSearchTool tests
# ===========================================================================


def test_web_tool_legacy_exports_point_to_canonical_classes():
    assert BuiltinWebSearchTool is CanonicalBuiltinWebSearchTool
    assert BuiltinFetchUrlTool is CanonicalBuiltinFetchUrlTool


def test_task_state_tool_legacy_exports_point_to_canonical_classes():
    assert BuiltinTodoTool is CanonicalBuiltinTodoTool


def test_get_builtin_tools_still_returns_expected_tool_types():
    tools = get_builtin_tools(work_dir="/tmp")
    tool_names = {tool.__class__.__name__ for tool in tools}
    assert "BuiltinFileTool" in tool_names
    assert "BuiltinExecuteTool" in tool_names
    assert "BuiltinWebSearchTool" in tool_names
    assert "BuiltinFetchUrlTool" in tool_names
    assert "BuiltinTodoTool" in tool_names
    assert "BuiltinTaskTool" in tool_names

class TestBuiltinWebSearchTool:
    def test_web_search_delegates_to_baidu(self):
        """Verify web_search calls BaiduSearchTool.baidu_search under the hood."""
        tool = BuiltinWebSearchTool()

        mock_result = json.dumps([{"title": "test", "url": "http://example.com", "content": "result"}])
        tool._search.baidu_search = AsyncMock(return_value=mock_result)

        result = asyncio.run(tool.web_search("test query"))
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert parsed[0]["title"] == "test"
        tool._search.baidu_search.assert_awaited_once_with("test query", max_results=5)

    def test_web_search_multiple_queries(self):
        tool = BuiltinWebSearchTool()
        mock_result = json.dumps({"q1": [], "q2": []})
        tool._search.baidu_search = AsyncMock(return_value=mock_result)

        result = asyncio.run(tool.web_search(["q1", "q2"], max_results=3))
        tool._search.baidu_search.assert_awaited_once_with(["q1", "q2"], max_results=3)

    def test_web_search_error_handling(self):
        tool = BuiltinWebSearchTool()
        tool._search.baidu_search = AsyncMock(side_effect=Exception("network error"))

        # After方案A: search failures propagate as exceptions instead of Error strings.
        # Runner/FunctionCall.invoke captures them into function_call.error.
        with pytest.raises(Exception, match="network error"):
            asyncio.run(tool.web_search("fail"))


# ===========================================================================
# BuiltinFetchUrlTool tests
# ===========================================================================

class TestBuiltinFetchUrlTool:
    def test_fetch_url_delegates_to_crawler(self):
        """Verify fetch_url calls UrlCrawlerTool.url_crawl under the hood."""
        tool = BuiltinFetchUrlTool()

        mock_result = json.dumps({"url": "http://example.com", "content": "page content", "save_path": "/tmp/x"})
        tool._crawler.url_crawl = AsyncMock(return_value=mock_result)

        result = asyncio.run(tool.fetch_url("http://example.com"))
        parsed = json.loads(result)
        assert parsed["url"] == "http://example.com"
        assert parsed["content"] == "page content"
        tool._crawler.url_crawl.assert_awaited_once_with("http://example.com")


# ===========================================================================
# BuiltinTodoTool tests
# ===========================================================================

class TestBuiltinTodoTool:
    def test_write_todos_basic(self, todo_tool):
        result = todo_tool.write_todos([
            {"content": "Task A", "status": "pending"},
            {"content": "Task B", "status": "in_progress"},
        ])
        assert result == "Todos updated (2 items: 1 in progress, 1 pending)."
        assert len(todo_tool.todos) == 2

    def test_write_todos_result_does_not_echo_the_list(self, todo_tool):
        """The model just sent this list; echoing it back is pure context cost,
        and at one update per finished step that cost repeats all session."""
        result = todo_tool.write_todos([
            {"content": "Review the model layer", "status": "completed"},
            {"content": "Review the runner", "status": "in_progress"},
            {"content": "Summarise findings", "status": "pending"},
        ])
        assert "Review the model layer" not in result
        assert len(result) < 120

    def test_write_todos_description_disambiguates_steps_from_tool_calls(self, todo_tool):
        """"3+ steps" alone reads as "3+ tool calls", which opens a todo list for
        almost every request. The description must draw that distinction and must
        not carry a bias toward calling the tool when unsure."""
        description = todo_tool.functions["write_todos"].description or ""

        assert "not 3 tool calls" in description
        assert "in_progress" in description
        assert "when in doubt" not in description.lower()

    def test_write_todos_invalid_status(self, todo_tool):
        with pytest.raises(ValueError):
            todo_tool.write_todos([{"content": "Bad", "status": "unknown"}])

    def test_write_todos_missing_content(self, todo_tool):
        with pytest.raises(ValueError):
            todo_tool.write_todos([{"status": "pending"}])

    def test_write_todos_none(self, todo_tool):
        with pytest.raises(ValueError):
            todo_tool.write_todos(None)

    def test_write_todos_empty_list(self, todo_tool):
        with pytest.raises(ValueError):
            todo_tool.write_todos([])

    def test_write_todos_overwrites(self, todo_tool):
        """Writing new todos replaces old ones entirely."""
        todo_tool.write_todos([{"content": "Old", "status": "pending"}])
        todo_tool.write_todos([{"content": "New1", "status": "pending"}, {"content": "New2", "status": "pending"}])
        assert len(todo_tool.todos) == 2
        contents = [t["content"] for t in todo_tool.todos]
        assert "Old" not in contents
        assert "New1" in contents

    def test_set_agent_stores_on_agent(self):
        """When set_agent is called, todos are stored on agent.todos."""
        tool = BuiltinTodoTool()
        mock_agent = MagicMock()
        mock_agent.todos = []
        tool.set_agent(mock_agent)

        tool.write_todos([
            {"content": "Task X", "status": "pending"},
            {"content": "Task Y", "status": "in_progress"},
        ])
        # Todos should be stored on mock_agent.todos
        assert len(mock_agent.todos) == 2
        assert mock_agent.todos[0]["content"] == "Task X"
        assert mock_agent.todos[1]["content"] == "Task Y"

    def test_standalone_mode_uses_local_todos(self):
        """Without set_agent, todos are stored locally on the tool."""
        tool = BuiltinTodoTool()
        tool.write_todos([{"content": "Local task", "status": "pending"}])
        assert len(tool.todos) == 1
        assert tool.todos[0]["content"] == "Local task"
        # _agent should be None
        assert tool._agent is None

    def test_todos_property_reads_from_agent(self):
        """The todos property should read from agent when agent is set."""
        tool = BuiltinTodoTool()
        mock_agent = MagicMock()
        mock_agent.todos = [{"id": "1", "content": "Agent task", "status": "completed"}]
        tool.set_agent(mock_agent)
        assert tool.todos == mock_agent.todos

    # ---- Auto-clear tests (mirrors CC allDone logic) ----

    def test_auto_clear_when_all_completed(self, todo_tool):
        """All-completed todos should auto-clear the list."""
        result = todo_tool.write_todos([
            {"content": "Task A", "status": "completed"},
            {"content": "Task B", "status": "completed"},
        ])
        assert result == "All 2 todos completed; list cleared."
        assert len(todo_tool.todos) == 0

    def test_no_auto_clear_when_not_all_completed(self, todo_tool):
        """Partial completion should NOT clear the list."""
        result = todo_tool.write_todos([
            {"content": "Task A", "status": "completed"},
            {"content": "Task B", "status": "in_progress"},
        ])
        assert result == "Todos updated (2 items: 1 done, 1 in progress)."
        assert len(todo_tool.todos) == 2

    # ---- Verification nudge tests (mirrors CC structural nudge) ----

    def test_verification_nudge_3plus_all_completed_no_verify(self, todo_tool):
        """3+ all-completed tasks with no verification keyword -> nudge fires."""
        result = todo_tool.write_todos([
            {"content": "Implement feature A", "status": "completed"},
            {"content": "Implement feature B", "status": "completed"},
            {"content": "Implement feature C", "status": "completed"},
        ])
        assert "NOTE:" in result

    def test_no_nudge_when_less_than_3_tasks(self, todo_tool):
        """< 3 tasks all completed -> no nudge."""
        result = todo_tool.write_todos([
            {"content": "Task A", "status": "completed"},
            {"content": "Task B", "status": "completed"},
        ])
        assert "NOTE:" not in result

    def test_no_nudge_when_not_all_completed(self, todo_tool):
        """3+ tasks but not all completed -> no nudge."""
        result = todo_tool.write_todos([
            {"content": "Task A", "status": "completed"},
            {"content": "Task B", "status": "completed"},
            {"content": "Task C", "status": "in_progress"},
        ])
        assert "NOTE:" not in result

    def test_no_nudge_when_verification_keyword_present(self, todo_tool):
        """3+ all completed but one mentions 'verify' -> no nudge."""
        result = todo_tool.write_todos([
            {"content": "Implement feature", "status": "completed"},
            {"content": "Verify implementation", "status": "completed"},
            {"content": "Deploy to staging", "status": "completed"},
        ])
        assert "NOTE:" not in result

    def test_no_nudge_when_test_keyword_present(self, todo_tool):
        """3+ all completed but one mentions 'test' -> no nudge."""
        result = todo_tool.write_todos([
            {"content": "Implement feature", "status": "completed"},
            {"content": "Write unit tests", "status": "completed"},
            {"content": "Update docs", "status": "completed"},
        ])
        assert "NOTE:" not in result

    def test_no_nudge_when_lint_keyword_present(self, todo_tool):
        """3+ all completed but one mentions 'lint' -> no nudge."""
        result = todo_tool.write_todos([
            {"content": "Refactor module", "status": "completed"},
            {"content": "Run linting", "status": "completed"},
            {"content": "Deploy", "status": "completed"},
        ])
        assert "NOTE:" not in result

    # ---- _needs_verification_nudge static method tests ----

    def test_needs_verification_nudge_static(self):
        """Direct test of the static nudge detection method."""
        assert BuiltinTodoTool._needs_verification_nudge([
            {"content": "A", "status": "completed"},
            {"content": "B", "status": "completed"},
            {"content": "C", "status": "completed"},
        ]) is True

        # Has 'check' keyword
        assert BuiltinTodoTool._needs_verification_nudge([
            {"content": "A", "status": "completed"},
            {"content": "Check results", "status": "completed"},
            {"content": "C", "status": "completed"},
        ]) is False

        # Has 'review' keyword
        assert BuiltinTodoTool._needs_verification_nudge([
            {"content": "A", "status": "completed"},
            {"content": "Code review", "status": "completed"},
            {"content": "C", "status": "completed"},
        ]) is False

        # Has 'validate' keyword
        assert BuiltinTodoTool._needs_verification_nudge([
            {"content": "A", "status": "completed"},
            {"content": "Validate output", "status": "completed"},
            {"content": "C", "status": "completed"},
        ]) is False

    # ---- Tool result message format tests ----

    def test_tool_result_message_is_neutral(self, todo_tool):
        """Tool result message confirms the update without nudging re-calls."""
        result = todo_tool.write_todos([
            {"content": "Task A", "status": "pending"},
        ])
        assert result == "Todos updated (1 items: 1 pending)."


# ===========================================================================
# BuiltinTaskTool tests (requires mocking LLM / Agent)
# ===========================================================================

class TestBuiltinTaskTool:
    """``BuiltinTaskTool`` is a thin LLM-facing adapter around
    ``SubagentRegistry.spawn``. Tests focus on the adapter contract; the
    runtime behavior of ``spawn`` itself is covered by ``test_subagent.py``.
    """

    def test_task_without_parent_returns_error(self):
        """Unbound tool (no parent agent) cannot spawn anything."""
        tool = BuiltinTaskTool()
        result = asyncio.run(tool.task("do something"))
        parsed = json.loads(result)
        assert parsed["success"] is False
        assert "not bound" in parsed["error"]

    def test_task_forwards_to_spawn_and_serializes_completed(self):
        """Adapter calls ``SubagentRegistry().spawn`` and JSON-serializes the result."""
        tool = BuiltinTaskTool()
        tool.set_parent_agent(MagicMock())

        spawn_result = {
            "status": "completed",
            "agent_type": "code",
            "subagent_name": "Code Agent",
            "content": "answer is 42",
            "tool_calls_summary": [{"name": "read_file", "info": "x.py"}],
            "tool_count": 1,
            "execution_time": 0.123,
            "run_id": "abc",
        }

        async def fake_spawn(self, **kwargs):
            assert kwargs["task"] == "compute 6 * 7"
            assert kwargs["agent_type"] == "code"
            return spawn_result

        with patch("agentica.subagent.SubagentRegistry.spawn", new=fake_spawn):
            result = asyncio.run(tool.task("compute 6 * 7", subagent_type="code"))

        parsed = json.loads(result)
        assert parsed["success"] is True
        assert parsed["subagent_type"] == "code"
        assert parsed["subagent_name"] == "Code Agent"
        assert parsed["result"] == "answer is 42"
        assert parsed["tool_count"] == 1
        assert parsed["execution_time"] == 0.123

    def test_task_serializes_error_result(self):
        """Adapter surfaces spawn errors through the LLM-facing JSON envelope."""
        tool = BuiltinTaskTool()
        tool.set_parent_agent(MagicMock())

        async def fake_spawn(self, **kwargs):
            return {
                "status": "error",
                "error": "Subagent timed out after 5 seconds",
                "agent_type": "code",
                "content": "",
            }

        with patch("agentica.subagent.SubagentRegistry.spawn", new=fake_spawn):
            result = asyncio.run(tool.task("slow"))

        parsed = json.loads(result)
        assert parsed["success"] is False
        assert "timed out" in parsed["error"]
        assert parsed["subagent_type"] == "code"

    def test_format_tool_brief_read_file(self):
        brief = BuiltinTaskTool._format_tool_brief("read_file", {"file_path": "/a/b/c.py"})
        assert "c.py" in brief

    def test_format_tool_brief_grep(self):
        brief = BuiltinTaskTool._format_tool_brief("grep", {"pattern": "hello"}, "found 3 matches")
        assert "hello" in brief

    def test_format_tool_brief_execute(self):
        brief = BuiltinTaskTool._format_tool_brief("execute", {"command": "ls -la /tmp"})
        assert "ls -la" in brief

    def test_format_tool_brief_default(self):
        brief = BuiltinTaskTool._format_tool_brief("unknown_tool", {"key": "value"})
        assert "key=" in brief

    def test_set_parent_agent(self):
        tool = BuiltinTaskTool()
        mock_agent = MagicMock()
        tool.set_parent_agent(mock_agent)
        assert tool._parent_agent is mock_agent

    def test_task_declares_own_timeout_management(self):
        tool = BuiltinTaskTool()
        assert tool.functions["task"].manages_own_timeout is True
        assert tool.functions["task"].interrupt_behavior == "block"

    def test_task_passes_auxiliary_model_to_spawn(self):
        """When ``auxiliary_model`` is set, the adapter forwards it to spawn as
        the cheap-tier model (main-tier types ignore it)."""
        custom_model = MagicMock()
        tool = BuiltinTaskTool(auxiliary_model=custom_model)
        tool.set_parent_agent(MagicMock())

        captured: Dict[str, Any] = {}

        async def fake_spawn(self, **kwargs):
            captured.update(kwargs)
            return {"status": "completed", "agent_type": "code", "content": "ok",
                    "tool_calls_summary": [], "tool_count": 0, "execution_time": 0}

        with patch("agentica.subagent.SubagentRegistry.spawn", new=fake_spawn):
            asyncio.run(tool.task("test", subagent_type="code"))

        assert captured["auxiliary_model_override"] is custom_model


# ===========================================================================
# Agent auto-wire tests (Agent.__init__ wires TodoTool / TaskTool)
# ===========================================================================

class TestAgentAutoWire:
    """Agent.__init__ clones stateful tools per-agent (so the user's original
    instance is never overwritten when the same logical tool is reused across
    multiple agents) and wires the per-agent clone to ``self``."""

    def test_agent_wires_todo_tool(self):
        """Agent.__init__ stores a per-agent clone of BuiltinTodoTool wired to self."""
        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat

        todo_tool = BuiltinTodoTool()
        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            tools=[todo_tool],
        )
        # User's original tool is left untouched (isolation contract)
        assert todo_tool._agent is None
        # Agent owns its own clone, wired to itself
        wired = next(t for t in agent.tools if isinstance(t, BuiltinTodoTool))
        assert wired is not todo_tool
        assert wired._agent is agent

    def test_agent_wires_task_tool(self):
        """Agent.__init__ stores a per-agent clone of BuiltinTaskTool wired to self."""
        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat

        task_tool = BuiltinTaskTool()
        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            tools=[task_tool],
        )
        assert task_tool._parent_agent is None
        wired = next(t for t in agent.tools if isinstance(t, BuiltinTaskTool))
        assert wired is not task_tool
        assert wired._parent_agent is agent

    def test_todo_tool_stores_on_agent(self):
        """After wiring, write_todos on the agent's clone stores todos on agent.todos."""
        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat

        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            tools=[BuiltinTodoTool()],
        )
        wired = next(t for t in agent.tools if isinstance(t, BuiltinTodoTool))
        wired.write_todos([
            {"content": "Test task", "status": "pending"},
        ])
        assert len(agent.todos) == 1
        assert agent.todos[0]["content"] == "Test task"


# ===========================================================================
# OpenAI stream_finish_reason capture tests
# ===========================================================================

class TestOpenAIStreamFinishReason:
    """Test that OpenAIChat.response_stream correctly captures finish_reason."""

    def _make_openai_chat(self):
        from agentica.model.openai import OpenAIChat
        return OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")

    def test_finish_reason_captured_from_last_chunk(self):
        """stream_finish_reason should be captured from the chunk where finish_reason is not None."""
        model = self._make_openai_chat()

        # Build mock stream chunks
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].finish_reason = None
        chunk1.choices[0].delta = MagicMock()
        chunk1.choices[0].delta.content = "Hello"
        chunk1.choices[0].delta.reasoning_content = None
        chunk1.choices[0].delta.audio = None
        chunk1.choices[0].delta.tool_calls = None
        chunk1.usage = None

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].finish_reason = "stop"
        chunk2.choices[0].delta = MagicMock()
        chunk2.choices[0].delta.content = " World"
        chunk2.choices[0].delta.reasoning_content = None
        chunk2.choices[0].delta.audio = None
        chunk2.choices[0].delta.tool_calls = None
        chunk2.usage = None

        async def mock_invoke_stream(messages):
            yield chunk1
            yield chunk2

        model.invoke_stream = mock_invoke_stream

        messages = [Message(role="user", content="Hi")]
        collected = []

        async def run():
            async for resp in model.response_stream(messages=messages):
                collected.append(resp)

        asyncio.run(run())
        assert model.last_finish_reason == "stop"

    def test_finish_reason_length_captured(self):
        """When output is truncated, finish_reason should be 'length'."""
        model = self._make_openai_chat()

        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].finish_reason = "length"
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = "partial output..."
        chunk.choices[0].delta.reasoning_content = None
        chunk.choices[0].delta.audio = None
        chunk.choices[0].delta.tool_calls = None
        chunk.usage = None

        async def mock_invoke_stream(messages):
            yield chunk

        model.invoke_stream = mock_invoke_stream

        messages = [Message(role="user", content="Hi")]

        async def run():
            async for _ in model.response_stream(messages=messages):
                pass

        asyncio.run(run())
        assert model.last_finish_reason == "length"

    def test_finish_reason_none_when_no_choices(self):
        """When stream has no choices, finish_reason should remain None."""
        model = self._make_openai_chat()

        chunk = MagicMock()
        chunk.choices = []
        chunk.usage = None

        async def mock_invoke_stream(messages):
            yield chunk

        model.invoke_stream = mock_invoke_stream

        messages = [Message(role="user", content="Hi")]

        async def run():
            async for _ in model.response_stream(messages=messages):
                pass

        asyncio.run(run())
        assert model.last_finish_reason is None


# ===========================================================================
# Guard: BuiltinFileTool functions MUST reach Model.get_tools_for_api()
# ===========================================================================

class TestFileToolRegistrationGuard:
    """End-to-end guard: every BuiltinFileTool function must be visible in
    the final tool schema sent to the LLM.

    This test exists because a past bug placed self.register() calls outside
    __init__(), causing read_file / ls / glob to silently disappear from the
    model's tool list while execute remained available — the model then fell
    back to shell commands.
    """

    EXPECTED_FUNCTIONS = {"ls", "read_file", "write_file", "edit_file",
                          "apply_patch", "glob", "grep"}

    def test_file_tool_functions_in_tool_dict(self):
        """Tool.functions dict must contain all file operations after init."""
        tool = BuiltinFileTool(work_dir="/tmp")
        registered = set(tool.functions.keys())
        missing = self.EXPECTED_FUNCTIONS - registered
        assert not missing, f"Functions missing from Tool.functions: {missing}"

    def test_file_tool_functions_reach_model_api_schema(self):
        """After Agent.update_model(), every file function must appear in
        Model.get_tools_for_api() — the payload actually sent to the LLM."""
        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat

        file_tool = BuiltinFileTool(work_dir="/tmp")
        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            tools=[file_tool],
        )
        agent.update_model()

        api_tools = agent.model.get_tools_for_api()
        api_names = {t["function"]["name"] for t in api_tools}
        missing = self.EXPECTED_FUNCTIONS - api_names
        assert not missing, (
            f"Functions missing from Model.get_tools_for_api(): {missing}. "
            f"Likely cause: self.register() not called in __init__."
        )


# ===========================================================================
# AskUserQuestionTool tests
# ===========================================================================

class TestAskUserQuestionTool:
    def test_ask_user_question_manages_own_timeout(self):
        """ask_user_question/confirm must wait indefinitely for the user (CC/Cursor
        semantics), not be auto-passed by the outer ~120s tool-executor timeout."""
        from agentica.tools.ask_user_question_tool import AskUserQuestionTool

        tool = AskUserQuestionTool(input_callback=lambda p, o=None: "ok")
        assert tool.functions["ask_user_question"].manages_own_timeout is True
        assert tool.functions["confirm"].manages_own_timeout is True

    def test_ask_user_question_uses_callback(self):
        from agentica.tools.ask_user_question_tool import AskUserQuestionTool

        captured = {}

        def cb(prompt, options=None):
            captured["prompt"] = prompt
            captured["options"] = options
            return "my answer"

        tool = AskUserQuestionTool(input_callback=cb)
        result = json.loads(tool.ask_user_question(prompt="What now?", mode="text"))
        assert result["response"] == "my answer"
        assert "What now?" in captured["prompt"]

    def test_callback_less_instance_uses_registered_default(self):
        """A callback-less AskUserQuestionTool (subagent / cron / regression)
        must route through the process-wide default callback registered by the
        TUI, instead of deadlocking on bare input() while prompt_toolkit owns
        stdin."""
        from agentica.tools.ask_user_question_tool import (
            AskUserQuestionTool,
            set_default_ask_user_question_callback,
        )

        captured = {}

        def default_cb(prompt, options=None):
            captured["prompt"] = prompt
            captured["options"] = options
            return "default answer"

        set_default_ask_user_question_callback(default_cb)
        try:
            tool = AskUserQuestionTool()  # no explicit callback
            assert tool.input_callback is None
            result = json.loads(tool.ask_user_question(prompt="Pick", mode="text"))
            assert result["response"] == "default answer"
            assert "Pick" in captured["prompt"]
        finally:
            set_default_ask_user_question_callback(None)

    def test_no_callback_no_default_falls_back_to_input(self):
        """Without the TUI registered, a callback-less tool keeps the legacy
        bare-input() behavior (non-interactive scripts). We can't drive input()
        in a unit test, so we verify the routing decision indirectly: the
        default holder is None and the instance callback is None."""
        from agentica.tools.ask_user_question_tool import (
            AskUserQuestionTool,
            set_default_ask_user_question_callback,
        )

        set_default_ask_user_question_callback(None)
        tool = AskUserQuestionTool()
        assert tool.input_callback is None
