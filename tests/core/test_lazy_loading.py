# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for __init__.py lazy loading mechanism.
"""
import sys
import os
import threading
import importlib
import subprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===========================================================================
# TestCoreImports
# ===========================================================================


class TestCoreImports:
    """Test that core modules are imported eagerly."""

    def test_agent_importable(self):
        from agentica import Agent
        assert Agent is not None

    def test_model_importable(self):
        from agentica import Model
        assert Model is not None

    def test_openai_chat_importable(self):
        from agentica import OpenAIChat
        assert OpenAIChat is not None

    def test_message_importable(self):
        from agentica import Message
        assert Message is not None

    def test_run_response_importable(self):
        from agentica import RunResponse
        assert RunResponse is not None

    def test_tool_importable(self):
        from agentica import Tool
        assert Tool is not None

    def test_function_importable(self):
        from agentica import Function
        assert Function is not None

    def test_workspace_importable(self):
        from agentica import Workspace
        assert Workspace is not None

    def test_workflow_importable(self):
        from agentica import Workflow
        assert Workflow is not None

    def test_working_memory_importable(self):
        from agentica import WorkingMemory
        assert WorkingMemory is not None


# ===========================================================================
# TestLazyLoadOptional
# ===========================================================================


class TestLazyLoadOptional:
    """Test that optional modules use lazy loading."""

    def test_agentica_import_does_not_load_provider_sdks(self):
        """Bare SDK import should not eagerly import provider/OCR dependencies."""
        script = """
import json
import sys
import agentica
loaded = [name for name in ('openai', 'anthropic', 'imgocr', 'cv2') if name in sys.modules]
print(json.dumps(loaded))
raise SystemExit(1 if loaded else 0)
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_cli_main_import_does_not_load_provider_sdks_or_ocr(self):
        """Importing the CLI entrypoint should stay lightweight before model creation."""
        script = """
import json
import sys
import agentica.cli.main
loaded = [name for name in ('openai', 'anthropic', 'imgocr', 'cv2') if name in sys.modules]
print(json.dumps(loaded))
raise SystemExit(1 if loaded else 0)
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_guardrails_importable(self):
        """Guardrails should be accessible."""
        try:
            from agentica.guardrails import InputGuardrail, OutputGuardrail
            assert InputGuardrail is not None
        except ImportError:
            pytest.skip("Guardrails not available")


# ===========================================================================
# TestThreadSafety
# ===========================================================================


class TestThreadSafety:
    """Test that concurrent imports don't cause issues."""

    def test_concurrent_imports_no_error(self):
        """Multiple threads importing agentica simultaneously should not crash."""
        errors = []

        def _import_agentica():
            try:
                import agentica
                _ = agentica.Agent
                _ = agentica.OpenAIChat
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_import_agentica) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0, f"Import errors: {errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
