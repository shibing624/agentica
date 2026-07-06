# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for edit-time LSP diagnostics (no real LSP server needed).
"""
import asyncio
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.lsp_diagnostics import Diagnostic, format_diagnostics, _parse, LspDiagnosticsChecker
from agentica.tools.buildin_tools import BuiltinFileTool


class TestDiagnosticParsing(unittest.TestCase):
    def test_parse_raw_diagnostic(self):
        raw = {
            "range": {"start": {"line": 4, "character": 2}},
            "severity": 1,
            "message": "Undefined name 'foo'",
            "source": "Pyright",
            "code": "reportUndefinedVariable",
        }
        d = _parse("/x/app.py", raw)
        self.assertEqual(d.line, 5)        # 1-based
        self.assertEqual(d.character, 3)
        self.assertEqual(d.severity, "error")
        self.assertIn("Undefined", d.message)

    def test_format_diagnostics_empty(self):
        self.assertEqual(format_diagnostics([]), "")

    def test_format_diagnostics_text(self):
        d = Diagnostic(file="/x/app.py", line=3, character=1, severity="error",
                       message="bad", source="Pyright", code="X")
        out = format_diagnostics([d], header="New:")
        self.assertIn("New:", out)
        self.assertIn("app.py:3:1", out)
        self.assertIn("bad", out)


class _FakeChecker:
    """Duck-typed stand-in for LspDiagnosticsChecker used by BuiltinFileTool."""
    def __init__(self, new_text="", supported=True):
        self._new_text = new_text
        self._supported = supported
        self.snapshot_calls = 0
        self.after_calls = 0

    def snapshot_before(self, file_path):
        self.snapshot_calls += 1

    def report_after(self, file_path):
        self.after_calls += 1
        return self._new_text


class TestFileToolDiagnosticsIntegration(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.work = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_write_file_appends_new_diagnostics(self):
        checker = _FakeChecker(new_text="New diagnostics introduced by this edit:\n  error app.py:1:1 boom")
        tool = BuiltinFileTool(work_dir=self.work, diagnostics_checker=checker)
        result = asyncio.run(tool.write_file("app.py", "x = 1\n"))
        self.assertIn("boom", result)
        self.assertEqual(checker.snapshot_calls, 1)
        self.assertEqual(checker.after_calls, 1)

    def test_write_file_no_diagnostics_no_suffix(self):
        checker = _FakeChecker(new_text="")
        tool = BuiltinFileTool(work_dir=self.work, diagnostics_checker=checker)
        result = asyncio.run(tool.write_file("app.py", "x = 1\n"))
        self.assertNotIn("diagnostics", result.lower())

    def test_no_checker_is_noop(self):
        tool = BuiltinFileTool(work_dir=self.work)
        result = asyncio.run(tool.write_file("app.py", "x = 1\n"))
        self.assertIn("absolute path", result)

    def test_edit_file_appends_diagnostics(self):
        path = os.path.join(self.work, "app.py")
        with open(path, "w") as f:
            f.write("a = 1\n")
        checker = _FakeChecker(new_text="New diagnostics introduced by this edit:\n  error app.py:1:1 oops")
        tool = BuiltinFileTool(work_dir=self.work, diagnostics_checker=checker)
        result = asyncio.run(tool.edit_file("app.py", "a = 1", "a = undefined_name"))
        self.assertIn("oops", result)


class TestCheckerReportAfterDiff(unittest.TestCase):
    """report_after diffs cached baseline vs current; only NEW diagnostics surface."""

    def _make_checker(self, current_diags):
        checker = LspDiagnosticsChecker.__new__(LspDiagnosticsChecker)
        checker.errors_only = False
        checker._baselines = {}
        checker.has_client = lambda fp: True
        checker.diagnostics = lambda fp: current_diags
        return checker

    def test_only_new_diagnostics_returned(self):
        pre = [Diagnostic("/x/app.py", 1, 1, "error", "old problem")]
        checker = self._make_checker(pre)
        # Seed the baseline with the pre-edit state.
        checker.snapshot_before("/x/app.py")
        # Now the file gains a new problem.
        checker.diagnostics = lambda fp: pre + [
            Diagnostic("/x/app.py", 9, 5, "error", "new problem"),
        ]
        text = checker.report_after("/x/app.py")
        self.assertIn("new problem", text)
        self.assertNotIn("old problem", text)

    def test_no_new_diagnostics_returns_empty(self):
        same = [Diagnostic("/x/app.py", 1, 1, "error", "same")]
        checker = self._make_checker(same)
        checker.snapshot_before("/x/app.py")
        self.assertEqual(checker.report_after("/x/app.py"), "")

    def test_same_problem_not_reported_twice_across_edits(self):
        """A problem introduced by edit 1 must not re-surface on edit 2."""
        base = []
        checker = self._make_checker(base)
        checker.snapshot_before("/x/app.py")  # baseline empty
        introduced = [Diagnostic("/x/app.py", 3, 1, "error", "boom")]
        checker.diagnostics = lambda fp: introduced
        first = checker.report_after("/x/app.py")
        self.assertIn("boom", first)
        # Edit 2: same problem still present -> should NOT be reported again.
        second = checker.report_after("/x/app.py")
        self.assertEqual(second, "")


if __name__ == "__main__":
    unittest.main()
