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
        # The provider check splits the key source into two rows
        # (config.yaml + env) so users can tell which one is in play.
        for expected in ["Python version", "Agentica version", "Home dir writable",
                         "Active profile", "API key (config.yaml)", "API key (env)",
                         "LSP diagnostics", "LSP workspace", "LSP server (pyright)",
                         "MCP config"]:
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
        """Final 'API key' FAIL row appears only when BOTH sources are empty."""
        with (
            patch("agentica.cli.setup.get_profile_api_key", return_value=None),
            patch.dict(os.environ, {}, clear=True),
        ):
            report = run_doctor()
        api = next(c for c in report.checks if c.name == "API key")
        self.assertEqual(api.status, FAIL)
        # Remediation copy should steer users toward config.yaml, not .env.
        self.assertIn("config.yaml", api.detail)
        self.assertFalse(report.ok)

    def test_ok_property_true_when_config_yaml_has_key(self):
        """A key in config.yaml alone is enough — env var is just a fallback."""
        with (
            patch("agentica.cli.setup.get_profile_api_key", return_value="sk-from-config"),
            patch.dict(os.environ, {}, clear=True),
        ):
            report = run_doctor()
        # No final "API key" FAIL row when config.yaml provides the key.
        self.assertFalse(any(c.name == "API key" and c.status == FAIL for c in report.checks))
        profile_row = next(c for c in report.checks if c.name == "API key (config.yaml)")
        self.assertEqual(profile_row.status, OK)

    def test_diagnostics_enabled_is_reported(self):
        report = run_doctor(enable_diagnostics=True, diagnostics_servers=["pyright"], work_dir=os.getcwd())
        diag = next(c for c in report.checks if c.name == "LSP diagnostics")
        self.assertEqual(diag.status, OK)
        self.assertIn("enabled", diag.detail)


if __name__ == "__main__":
    unittest.main()
