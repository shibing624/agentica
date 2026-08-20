# -*- coding: utf-8 -*-
"""Keep eval-harness unit tests out of the real ~/.agentica tree."""
import os
import tempfile

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("AGENTICA_LOG_FILE", "")


@pytest.fixture(autouse=True)
def _isolate_agentica_dirs(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("AGENTICA_PROJECTS_DIR", tmpdir)
        monkeypatch.setenv("AGENTICA_CACHE_DIR", tmpdir)
        yield
