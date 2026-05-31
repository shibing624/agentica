# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for the environment doctor (agentica.diagnostics).
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

from agentica.diagnostics import run_doctor, DoctorReport, OK, WARN, FAIL


class TestDoctor(unittest.TestCase):
    def test_returns_report_with_core_checks(self):
        report = run_doctor()
        self.assertIsInstance(report, DoctorReport)
        names = [c.name for c in report.checks]
        for expected in ["Python version", "Agentica version", "Home dir writable",
                         "Configured provider", "API key", "LSP (pyright)", "MCP config"]:
            self.assertIn(expected, names)

    def test_all_statuses_valid(self):
        report = run_doctor()
        for c in report.checks:
            self.assertIn(c.status, (OK, WARN, FAIL))

    def test_python_version_ok_on_supported_runtime(self):
        report = run_doctor()
        py = next(c for c in report.checks if c.name == "Python version")
        # The test runner itself is >= 3.10.
        self.assertEqual(py.status, OK)

    def test_summary_and_counts_consistent(self):
        report = run_doctor()
        counts = report.counts()
        self.assertEqual(sum(counts.values()), len(report.checks))
        self.assertIn("ok", report.summary())

    def test_missing_api_key_is_failure(self):
        with patch("agentica.cli.setup.has_api_key", return_value=False):
            report = run_doctor()
        api = next(c for c in report.checks if c.name == "API key")
        self.assertEqual(api.status, FAIL)
        self.assertFalse(report.ok)

    def test_ok_property_true_when_no_failures(self):
        with patch("agentica.cli.setup.has_api_key", return_value=True):
            report = run_doctor()
        # ok ignores warnings (e.g. pyright/MCP may warn in CI).
        api = next(c for c in report.checks if c.name == "API key")
        self.assertEqual(api.status, OK)


if __name__ == "__main__":
    unittest.main()
