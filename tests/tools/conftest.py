# -*- coding: utf-8 -*-
"""Shared fixtures for builtin-tool tests."""
import shutil
import tempfile

import pytest

from agentica.tools.background_processes import BackgroundProcessRegistry
from agentica.tools.builtin import BuiltinExecuteTool, BuiltinFileTool, BuiltinTodoTool


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for file operation tests."""
    d = tempfile.mkdtemp(prefix="test_builtin_tools_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def file_tool(tmp_dir):
    """BuiltinFileTool scoped to a temp directory."""
    return BuiltinFileTool(work_dir=tmp_dir)


@pytest.fixture
def execute_tool(tmp_dir):
    """BuiltinExecuteTool as an SDK caller gets it — no background registry."""
    return BuiltinExecuteTool(work_dir=tmp_dir)


@pytest.fixture
def bg_execute_tool(tmp_dir):
    """BuiltinExecuteTool as the CLI builds it — a registry owns detached commands."""
    registry = BackgroundProcessRegistry()
    try:
        yield BuiltinExecuteTool(work_dir=tmp_dir, background_process_registry=registry)
    finally:
        registry.stop()


@pytest.fixture
def todo_tool():
    return BuiltinTodoTool()
