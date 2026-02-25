# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests verifying all Model providers implement the async-only interface.
"""
import asyncio
import inspect
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from agentica.model.base import Model


# ---------------------------------------------------------------------------
# Collect provider classes
# ---------------------------------------------------------------------------

def _get_provider_classes():
    """Dynamically import all provider model classes."""
    providers = []
    try:
        from agentica.model.openai.chat import OpenAIChat
        providers.append(("OpenAIChat", OpenAIChat))
    except ImportError:
        pass
    try:
        from agentica.model.openai.like import OpenAILike
        providers.append(("OpenAILike", OpenAILike))
    except ImportError:
        pass
    try:
        from agentica.model.anthropic.claude import Claude
        providers.append(("Claude", Claude))
    except ImportError:
        pass
    try:
        from agentica.model.ollama.chat import OllamaChat
        providers.append(("OllamaChat", OllamaChat))
    except ImportError:
        pass
    return providers


PROVIDER_CLASSES = _get_provider_classes()


# ===========================================================================
# TestProviderInterface
# ===========================================================================


class TestProviderInterface:
    """Verify all providers implement async-only interface."""

    @pytest.mark.parametrize("name,cls", PROVIDER_CLASSES, ids=[p[0] for p in PROVIDER_CLASSES])
    def test_response_is_async(self, name, cls):
        assert asyncio.iscoroutinefunction(cls.response), f"{name}.response should be async"

    @pytest.mark.parametrize("name,cls", PROVIDER_CLASSES, ids=[p[0] for p in PROVIDER_CLASSES])
    def test_response_stream_is_async(self, name, cls):
        # response_stream may be an async generator, which is also async
        is_async = (
            asyncio.iscoroutinefunction(cls.response_stream)
            or inspect.isasyncgenfunction(cls.response_stream)
        )
        assert is_async, f"{name}.response_stream should be async"

    @pytest.mark.parametrize("name,cls", PROVIDER_CLASSES, ids=[p[0] for p in PROVIDER_CLASSES])
    def test_invoke_is_async(self, name, cls):
        assert asyncio.iscoroutinefunction(cls.invoke), f"{name}.invoke should be async"

    @pytest.mark.parametrize("name,cls", PROVIDER_CLASSES, ids=[p[0] for p in PROVIDER_CLASSES])
    def test_invoke_stream_is_async(self, name, cls):
        is_async = (
            asyncio.iscoroutinefunction(cls.invoke_stream)
            or inspect.isasyncgenfunction(cls.invoke_stream)
        )
        assert is_async, f"{name}.invoke_stream should be async"

    @pytest.mark.parametrize("name,cls", PROVIDER_CLASSES, ids=[p[0] for p in PROVIDER_CLASSES])
    def test_no_sync_response_method(self, name, cls):
        """No provider should have legacy aresponse/ainvoke methods."""
        assert not hasattr(cls, "aresponse"), f"{name} should not have 'aresponse'"
        assert not hasattr(cls, "ainvoke"), f"{name} should not have 'ainvoke'"
        assert not hasattr(cls, "aresponse_stream"), f"{name} should not have 'aresponse_stream'"

    @pytest.mark.parametrize("name,cls", PROVIDER_CLASSES, ids=[p[0] for p in PROVIDER_CLASSES])
    def test_inherits_from_model(self, name, cls):
        assert issubclass(cls, Model), f"{name} should inherit from Model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
