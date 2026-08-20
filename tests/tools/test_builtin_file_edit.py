# -*- coding: utf-8 -*-
"""Tests for BuiltinFileTool edit / apply_patch / request_path_access."""
import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agentica.tools.builtin import BuiltinFileTool


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
        assert "Read or re-read" not in message
        assert "short unique context" not in message
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
        # The mismatching line is marked '>' in both the expected and actual block.
        assert "> beta-current" in message
        assert "> beta" in message
        assert "First difference at context line 2 (file line 2)" in message
        assert "Read or re-read" not in message

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

    def test_edit_result_carries_the_exact_change_executed(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "edit.txt")
        Path(fp).write_text("first = 1\nsecond = 1\n")

        result = asyncio.run(file_tool.edit_file(fp, "second = 1", "second = 2"))

        assert result.display_meta == {
            "files": [{
                "path": str(Path(fp).resolve()),
                "action": "update",
                "before": "first = 1\nsecond = 1\n",
                "after": "first = 1\nsecond = 2\n",
            }]
        }

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
        from agentica.tools.builtin.execute_tool import _detect_python_error_hint
        sample = "Traceback ...\n  File ...\nNameError: name 'null' is not defined"
        hint = _detect_python_error_hint(sample)
        assert hint is not None
        assert "JSON literal" in hint
        assert "None" in hint

    def test_python_error_hint_syntax_error(self):
        from agentica.tools.builtin.execute_tool import _detect_python_error_hint
        sample = "  File 'x.py', line 5\n    if x = 1\nSyntaxError: invalid syntax"
        hint = _detect_python_error_hint(sample)
        assert hint is not None
        assert "SyntaxError" in hint

    def test_python_error_hint_module_not_found(self):
        from agentica.tools.builtin.execute_tool import _detect_python_error_hint
        sample = "ModuleNotFoundError: No module named 'foobar_widget'"
        hint = _detect_python_error_hint(sample)
        assert hint is not None
        assert "dependency" in hint or "pip install" in hint

    def test_python_error_hint_returns_none_for_normal_output(self):
        from agentica.tools.builtin.execute_tool import _detect_python_error_hint
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

