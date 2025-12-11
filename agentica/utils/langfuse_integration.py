# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Langfuse integration for observability and tracing

Langfuse is an open-source LLM observability platform that helps you trace,
monitor, and debug your LLM applications.

Langfuse uses OpenTelemetry auto-instrumentation to automatically trace all
OpenAI API calls when configured via environment variables.

Usage:
    1. Install langfuse:
        pip install langfuse

    2. Set environment variables:
        LANGFUSE_SECRET_KEY = "sk-lf-xxx"
        LANGFUSE_PUBLIC_KEY = "pk-lf-xxx"
        LANGFUSE_HOST = "https://cloud.langfuse.com"  # or your self-hosted URL

    3. Use agentica as normal - Langfuse will automatically trace all LLM calls:
        from agentica import Agent
        from agentica.model.openai import OpenAIChat

        agent = Agent(
            name="My Agent",
            user_id="user-123",  # Automatically passed to Langfuse metadata
            session_id="session-abc",  # Automatically passed to Langfuse metadata
            model=OpenAIChat(
                id="gpt-4o",
                langfuse_tags=["demo", "test"],  # Optional tags
            ),
        )

        response = agent.run("Hello!")
        # All LLM calls are automatically traced in Langfuse dashboard
"""
import os
from typing import Optional, List, Dict, Any
from agentica.config import LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY
from agentica.utils.log import logger


# Global flag to track if Langfuse is available
_langfuse_available: Optional[bool] = None


def is_langfuse_configured() -> bool:
    """Check if Langfuse environment variables are configured."""
    return bool(LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY)


def is_langfuse_available() -> bool:
    """
    Check if Langfuse package is installed and configured.

    Langfuse uses OpenTelemetry auto-instrumentation to trace OpenAI calls.
    """
    global _langfuse_available
    if _langfuse_available is not None:
        return _langfuse_available

    try:
        import langfuse  # noqa: F401
        _langfuse_available = is_langfuse_configured()
        if _langfuse_available:
            logger.debug("Langfuse is available and configured (using OpenTelemetry auto-instrumentation)")
        else:
            logger.warning(f"Langfuse package installed but not configured (missing env vars), LANGFUSE_PUBLIC_KEY: {LANGFUSE_PUBLIC_KEY}")
    except ImportError:
        _langfuse_available = False
        logger.debug("Langfuse package not installed. Install with: pip install langfuse")

    return _langfuse_available


def build_langfuse_metadata(
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build metadata dict for Langfuse tracing.

    According to Langfuse docs, the metadata should include:
    - langfuse_user_id: User identifier
    - langfuse_session_id: Session identifier
    - langfuse_tags: List of tags

    Args:
        user_id: User identifier
        session_id: Session identifier
        tags: List of tags
        extra_metadata: Additional metadata to include

    Returns:
        Dict with Langfuse metadata
    """
    metadata: Dict[str, Any] = {}

    if user_id:
        metadata["langfuse_user_id"] = user_id
    if session_id:
        metadata["langfuse_session_id"] = session_id
    if tags:
        metadata["langfuse_tags"] = tags
    if extra_metadata:
        metadata.update(extra_metadata)

    return metadata


def get_langfuse_openai_client():
    """
    Get OpenAI client classes wrapped with Langfuse observability.

    Returns:
        Tuple of (LangfuseOpenAI, LangfuseAsyncOpenAI) classes,
        or (None, None) if Langfuse is not available.
    """
    if not is_langfuse_available():
        return None, None

    try:
        from langfuse.openai import OpenAI as LangfuseOpenAI
        from langfuse.openai import AsyncOpenAI as LangfuseAsyncOpenAI

        return LangfuseOpenAI, LangfuseAsyncOpenAI
    except ImportError:
        logger.warning("langfuse.openai module not available")
        return None, None


def flush_langfuse():
    """Flush any pending Langfuse events."""
    try:
        from langfuse import Langfuse
        if is_langfuse_configured():
            client = Langfuse()
            client.flush()
            logger.debug("Langfuse events flushed")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to flush Langfuse: {e}")


def shutdown_langfuse():
    """Shutdown Langfuse and flush pending events."""
    try:
        from langfuse import Langfuse
        if is_langfuse_configured():
            client = Langfuse()
            client.shutdown()
            logger.debug("Langfuse shutdown completed")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to shutdown Langfuse: {e}")
