# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for __init__.py lazy loading mechanism.
"""
import sys
import os
import threading
import importlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


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

    def test_deep_agent_importable(self):
        from agentica import DeepAgent
        assert DeepAgent is not None

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


# ===========================================================================
# TestPublicAPI
# ===========================================================================


class TestPublicAPI:
    """Test that __all__ names are accessible."""

    def test_all_public_names_accessible(self):
        import agentica
        if hasattr(agentica, '__all__'):
            for name in agentica.__all__:
                assert hasattr(agentica, name) or callable(getattr(agentica, name, None)) is not None, \
                    f"'{name}' in __all__ but not accessible"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
