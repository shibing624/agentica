# -*- coding: utf-8 -*-
"""Regression tests for loguru-style module path resolution in
``agentica.utils.log``.

The console/file formatters previously hard-coded the literal string
``"agentica"`` for every log line, which made it impossible to tell which
file actually emitted a record. These tests lock in the new behaviour:
records carry the full dotted module path of their source file.
"""
import io
import logging
import os
import unittest

from agentica.utils.log import (
    LoguruStyleFormatter,
    _PlainLoguruStyleFormatter,
    _dotted_module_from_path,
)


class TestDottedModuleFromPath(unittest.TestCase):
    """``_dotted_module_from_path`` must produce a real importable path
    when the file lives inside a package, and degrade cleanly otherwise."""

    def setUp(self):
        # All resolution is path-based and must not depend on cwd.
        self.repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )

    def test_resolves_top_level_package_module(self):
        path = os.path.join(self.repo_root, "agentica", "runner.py")
        self.assertEqual(_dotted_module_from_path(path), "agentica.runner")

    def test_resolves_nested_package_module(self):
        path = os.path.join(self.repo_root, "agentica", "utils", "log.py")
        self.assertEqual(_dotted_module_from_path(path), "agentica.utils.log")

    def test_resolves_deep_subpackage(self):
        path = os.path.join(
            self.repo_root, "agentica", "tools", "buildin_tools.py"
        )
        self.assertEqual(
            _dotted_module_from_path(path), "agentica.tools.buildin_tools"
        )

    def test_falls_back_to_stem_for_non_package_files(self):
        # /tmp has no __init__.py — should degrade to the bare stem,
        # not crash and not silently strip useful info.
        self.assertEqual(
            _dotted_module_from_path("/tmp/standalone_script.py"),
            "standalone_script",
        )

    def test_handles_empty_or_missing_pathname(self):
        self.assertEqual(_dotted_module_from_path(""), "?")

    def test_results_are_cached(self):
        # Cheap correctness signal: the lru_cache wrapper exposes cache_info.
        before = _dotted_module_from_path.cache_info().hits
        path = os.path.join(self.repo_root, "agentica", "runner.py")
        _dotted_module_from_path(path)
        _dotted_module_from_path(path)
        self.assertGreater(
            _dotted_module_from_path.cache_info().hits, before,
            "second call must hit the cache — log formatting is hot-path",
        )


class _RecordFactory:
    """Minimal helper to build a LogRecord pointing at an arbitrary file."""

    @staticmethod
    def make(pathname: str, funcName: str = "do_thing", lineno: int = 42,
             message: str = "hi", level: int = logging.DEBUG) -> logging.LogRecord:
        record = logging.LogRecord(
            name="agentica",
            level=level,
            pathname=pathname,
            lineno=lineno,
            msg=message,
            args=(),
            exc_info=None,
            func=funcName,
        )
        return record


class TestLoguruStyleFormatter(unittest.TestCase):
    """Console formatter must surface the dotted module path, not 'agentica'."""

    def setUp(self):
        self.repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        self.formatter = LoguruStyleFormatter()

    def _strip_ansi(self, s: str) -> str:
        import re
        return re.sub(r"\x1b\[[0-9;]*m", "", s)

    def test_real_file_yields_full_module_path(self):
        path = os.path.join(self.repo_root, "agentica", "runner.py")
        rec = _RecordFactory.make(path, funcName="_run_core", lineno=855)
        out = self._strip_ansi(self.formatter.format(rec))
        self.assertIn("agentica.runner:_run_core:855", out)
        self.assertNotIn(" agentica:_run_core:", out,
                         "old hard-coded 'agentica:func:lineno' must not return")

    def test_does_not_crash_on_synthetic_pathname(self):
        rec = _RecordFactory.make("<string>", funcName="<module>", lineno=1)
        out = self._strip_ansi(self.formatter.format(rec))
        # It should still emit *something* identifying the location.
        self.assertIn(":<module>:1", out)


class TestPlainLoguruStyleFormatter(unittest.TestCase):
    """File formatter must emit the same locator without ANSI escape codes."""

    def setUp(self):
        self.repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        self.formatter = _PlainLoguruStyleFormatter()

    def test_no_ansi_in_file_output(self):
        path = os.path.join(self.repo_root, "agentica", "tools", "buildin_tools.py")
        rec = _RecordFactory.make(path, funcName="ls", lineno=380,
                                  message="Listed 37 items")
        out = self.formatter.format(rec)
        self.assertNotIn("\x1b[", out, "file formatter must not write ANSI codes")
        self.assertIn("agentica.tools.buildin_tools:ls:380", out)
        self.assertIn("Listed 37 items", out)

    def test_levelname_padded(self):
        path = os.path.join(self.repo_root, "agentica", "runner.py")
        rec = _RecordFactory.make(path, level=logging.INFO, message="x")
        rec.levelname = "INFO"
        out = self.formatter.format(rec)
        # left-justified width 8 — preserves grep-friendly column alignment
        self.assertIn(" INFO     | ", out)


class TestLoggerEndToEnd(unittest.TestCase):
    """Wire the formatter into a logger and assert real records carry the
    dotted module path captured at emission time."""

    def test_logger_emits_dotted_module_for_real_caller(self):
        log = logging.getLogger("agentica.test_log_module_path")
        log.handlers.clear()
        log.setLevel(logging.DEBUG)
        log.propagate = False

        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(_PlainLoguruStyleFormatter())
        log.addHandler(handler)

        try:
            log.debug("end-to-end probe")
        finally:
            log.removeHandler(handler)

        out = buf.getvalue()
        # The caller of log.debug() lives in this very test file. Its dotted
        # module name must appear in the formatted line — not the literal
        # string "agentica".
        self.assertIn("test_log_module_path:test_logger_emits_dotted_module_for_real_caller", out)
        self.assertIn("end-to-end probe", out)


if __name__ == "__main__":
    unittest.main()
