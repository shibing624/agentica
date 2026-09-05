# -*- coding: utf-8 -*-
"""Tests for BuiltinFileTool edit / apply_patch / grant_path_access."""
import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from agentica.tools.builtin import BuiltinFileTool


class TestBuiltinFileToolGrantPathAccess:
    """Sandbox / sensitive-path escalation via the public grant API."""

    def _sandboxed_tool(self, tmp_dir):
        from agentica.agent.config import SandboxConfig

        sandbox = SandboxConfig(enabled=True, writable_dirs=[tmp_dir])
        return BuiltinFileTool(work_dir=tmp_dir, sandbox_config=sandbox)

    def test_write_outside_writable_dirs_is_blocked(self, tmp_dir):
        with tempfile.TemporaryDirectory() as other_dir:
            tool = self._sandboxed_tool(tmp_dir)
            fp = os.path.join(other_dir, "outside.txt")
            with pytest.raises(PermissionError, match="blocked"):
                asyncio.run(tool.write_file(fp, "content"))

    def test_grant_exact_unblocks_that_file_only(self, tmp_dir):
        with tempfile.TemporaryDirectory() as other_dir:
            tool = self._sandboxed_tool(tmp_dir)
            allowed = os.path.join(other_dir, "now_allowed.txt")
            sibling = os.path.join(other_dir, "still_blocked.txt")
            tool.grant_path_access(allowed, prefix=False)
            write_result = asyncio.run(tool.write_file(allowed, "content"))
            assert "Created" in write_result
            assert Path(allowed).read_text() == "content"
            with pytest.raises(PermissionError):
                asyncio.run(tool.write_file(sibling, "nope"))

    def test_grant_prefix_unblocks_parent_directory(self, tmp_dir):
        with tempfile.TemporaryDirectory() as other_dir:
            tool = self._sandboxed_tool(tmp_dir)
            allowed = os.path.join(other_dir, "a.txt")
            sibling = os.path.join(other_dir, "b.txt")
            tool.grant_path_access(allowed, prefix=True)
            asyncio.run(tool.write_file(allowed, "a"))
            asyncio.run(tool.write_file(sibling, "b"))
            assert Path(allowed).read_text() == "a"
            assert Path(sibling).read_text() == "b"

    def test_sensitive_write_blocked_when_sandbox_enabled(self, tmp_dir):
        from agentica.agent.config import SandboxConfig

        tool = BuiltinFileTool(
            work_dir=tmp_dir,
            sandbox_config=SandboxConfig(enabled=True, writable_dirs=[]),
        )
        with pytest.raises(PermissionError, match="sensitive"):
            asyncio.run(tool.write_file("/etc/hosts", "content"))
        assert "request_path_access" not in tool.functions

    def test_sensitive_write_not_guarded_when_sandbox_disabled(self, tmp_dir):
        tool = BuiltinFileTool(work_dir=tmp_dir, sandbox_config=None)
        assert tool._sensitive_write_guard("/etc/hosts") is not None
        assert "request_path_access" not in tool.functions

    def test_sensitive_grant_is_exact_file_even_when_prefix_requested(self, tmp_dir):
        tool = BuiltinFileTool(work_dir=tmp_dir, sandbox_config=None)
        tool.grant_path_access("/etc/hosts", prefix=True)
        assert tool._sensitive_write_guard("/etc/hosts") is None
        assert tool._sensitive_write_guard("/etc/passwd") is not None

    def test_read_blocked_path_component_is_grantable(self, tmp_dir):
        blocked_dir = os.path.join(tmp_dir, ".ssh")
        os.makedirs(blocked_dir, exist_ok=True)
        target = os.path.join(blocked_dir, "id_rsa")
        Path(target).write_text("fake-key")

        tool = self._sandboxed_tool(tmp_dir)
        with pytest.raises(PermissionError, match="blocked"):
            asyncio.run(tool.read_file(target))

        tool.grant_path_access(target, prefix=False)
        content = asyncio.run(tool.read_file(target))
        assert "fake-key" in content

    def test_request_path_access_not_in_schema(self, tmp_dir):
        allow = BuiltinFileTool(work_dir=tmp_dir, permission_mode="allow-all")
        ask = BuiltinFileTool(work_dir=tmp_dir, permission_mode="ask")
        auto = BuiltinFileTool(work_dir=tmp_dir, permission_mode="auto")
        assert "request_path_access" not in allow.functions
        assert "request_path_access" not in ask.functions
        assert "request_path_access" not in auto.functions
        allow.set_permission_mode("auto")
        assert "request_path_access" not in allow.functions


class TestBuiltinFileToolApplyPatch:
    def test_result_carries_actual_changes_for_all_actions(self, file_tool, tmp_dir):
        Path(tmp_dir, "update.py").write_text("VALUE = 1\n")
        Path(tmp_dir, "delete.py").write_text("obsolete\n")
        patch_text = """*** Begin Patch
*** Update File: update.py
@@
-VALUE = 1
+VALUE = 2
*** Add File: add.py
+created = True
*** Delete File: delete.py
*** End Patch"""

        result = asyncio.run(file_tool.apply_patch(patch_text))

        assert result.display_meta == {
            "files": [
                {
                    "path": "update.py", "action": "update",
                    "before": "VALUE = 1\n", "after": "VALUE = 2\n",
                },
                {
                    "path": "add.py", "action": "add",
                    "before": None, "after": "created = True",
                },
                {
                    "path": "delete.py", "action": "delete",
                    "before": "obsolete\n", "after": None,
                },
            ]
        }

    def test_registered_as_serial_destructive_raw_string_tool(self, file_tool):
        function = file_tool.functions["apply_patch"]
        function.process_entrypoint(strict=False)

        assert function.is_destructive is True
        assert function.concurrency_safe is False
        assert function.parameters["required"] == ["patch"]
        assert function.parameters["properties"]["patch"]["type"] == "string"

    def test_description_states_envelope_and_hunk_prefixes(self, file_tool):
        function = file_tool.functions["apply_patch"]
        function.process_entrypoint(strict=False)
        description = function.description
        assert "*** Begin Patch" in description
        assert "*** Update File: app.py" in description
        assert "*** Update File: tests/test_app.py" in description
        assert (
            "+# matching via BFS\n"
            " def run():\n"
            "-    timeout = 10\n"
            "+    timeout = 30"
        ) in description
        assert "-    retries = 1\n+    retries = 3" in description
        assert "one ``@@`` under the same" in description
        assert "One Update File per path" in description
        assert "spaced copy of the file is a no-op" in description
        assert "then one patch" in description
        assert "Read the current file" not in description
        assert "MUST call read_file" not in description

    def test_applies_update_when_envelope_markers_are_omitted(self, file_tool, tmp_dir):
        Path(tmp_dir, "app.py").write_text("VALUE = 1\nKEEP = True\n")
        result = asyncio.run(file_tool.apply_patch(
            "*** Update File: app.py\n@@\n-VALUE = 1\n+VALUE = 2\n KEEP = True\n"
        ))
        assert "Successfully applied patch" in result
        assert Path(tmp_dir, "app.py").read_text() == "VALUE = 2\nKEEP = True\n"

    def test_two_hunks_in_one_update_file_apply_together(self, file_tool, tmp_dir):
        Path(tmp_dir, "app.py").write_text("timeout = 10\nname = \"demo\"\nretries = 1\n")
        result = asyncio.run(file_tool.apply_patch(
            "*** Begin Patch\n"
            "*** Update File: app.py\n"
            "@@\n"
            "-timeout = 10\n"
            "+timeout = 30\n"
            "@@\n"
            "-retries = 1\n"
            "+retries = 3\n"
            "*** End Patch\n"
        ))
        assert "Successfully applied patch" in result
        assert Path(tmp_dir, "app.py").read_text() == "timeout = 30\nname = \"demo\"\nretries = 3\n"

    def test_applies_update_when_wrapped_in_markdown_fence(self, file_tool, tmp_dir):
        Path(tmp_dir, "app.py").write_text("VALUE = 1\nKEEP = True\n")
        result = asyncio.run(file_tool.apply_patch(
            "```patch\n"
            "here is the edit\n"
            "*** Update File: app.py\n"
            "@@\n"
            "-VALUE = 1\n"
            "+VALUE = 2\n"
            " KEEP = True\n"
            "```\n"
        ))
        assert "Successfully applied patch" in result
        assert Path(tmp_dir, "app.py").read_text() == "VALUE = 2\nKEEP = True\n"

    def test_noop_keep_only_hunk_says_to_use_minus_not_malformed(self, file_tool, tmp_dir):
        Path(tmp_dir, "app.py").write_text("VALUE = 1\n# keep me\n")
        patch_text = """*** Begin Patch
*** Update File: app.py
@@
 VALUE = 1
 # keep me
*** End Patch"""
        with pytest.raises(ValueError) as exc:
            asyncio.run(file_tool.apply_patch(patch_text))
        message = str(exc.value)
        assert "does not change" in message
        assert "Malformed patch" not in message
        assert "start it with '-'" in message
        assert Path(tmp_dir, "app.py").read_text() == "VALUE = 1\n# keep me\n"

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
        assert "Patch not applied for 3 files" in message
        assert "- first.py:" in message
        assert "Hunk 1: context not found" in message
        assert "Hunk 2: context not found" in message
        assert "- second.py:" in message
        assert "- existing.py:" in message
        assert "Expected context:" not in message
        assert "Read or re-read" not in message
        assert "short unique context" not in message
        assert first.read_text() == "FIRST = 1\n"
        assert second.read_text() == "SECOND = 2\n"
        assert existing.read_text() == "keep\n"

    def test_context_mismatch_does_not_dump_expected_actual(self, file_tool, tmp_dir):
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
        assert "Patch context not found for 1 file" in message
        assert "Hunk 1: context not found" in message
        assert "beta" in message
        assert "Expected context:" not in message
        assert "Actual from line" not in message
        assert "First difference" not in message
        assert "beta-current" not in message
        assert target.read_text() == "alpha\nbeta-current\ngamma\n"

    def test_unprefixed_file_line_is_recovered_as_keep(self, file_tool, tmp_dir):
        target = Path(tmp_dir, "bipartite.py")
        target.write_text("def max_matching(n_left, n_right, adj):\n    return 0\n")
        result = asyncio.run(file_tool.apply_patch(
            "*** Begin Patch\n"
            "*** Update File: bipartite.py\n"
            "@@\n"
            "def max_matching(n_left, n_right, adj):\n"
            "-    return 0\n"
            "+    return 1\n"
            "*** End Patch"
        ))
        assert "Successfully applied patch" in result
        assert target.read_text() == "def max_matching(n_left, n_right, adj):\n    return 1\n"

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

