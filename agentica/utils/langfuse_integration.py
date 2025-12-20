# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Langfuse integration for observability and tracing

Langfuse is an open-source LLM observability platform that helps you trace,
monitor, and debug your LLM applications.

This module provides:
1. OpenAI client wrapper with Langfuse instrumentation
2. Trace context manager for grouping multi-turn conversations
3. Session and user tracking across traces

Usage:
    1. Install langfuse:
        pip install langfuse

    2. Set environment variables:
        LANGFUSE_SECRET_KEY = "sk-lf-xxx"
        LANGFUSE_PUBLIC_KEY = "pk-lf-xxx"
        LANGFUSE_BASE_URL = "https://cloud.langfuse.com"  # or your self-hosted URL

    3. Use agentica as normal - Langfuse will automatically trace all LLM calls:
        from agentica import Agent
        from agentica.model.openai import OpenAIChat

        agent = Agent(
            name="My Agent",
            user_id="user-123",  # Automatically passed to Langfuse
            session_id="session-abc",  # Groups multi-turn conversations
            model=OpenAIChat(id="gpt-4o"),
        )

        response = agent.run("Hello!")
        # All LLM calls within this run are grouped in a single trace
"""

import logging
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Generator
from agentica.config import LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_BASE_URL
from agentica.utils.log import logger


# Global flag to track if Langfuse is available
_langfuse_available: Optional[bool] = None


def _suppress_opentelemetry_errors():
    """
    Suppress OpenTelemetry exporter errors (e.g., timeout errors when Langfuse host is unreachable).
    
    This is useful in internal network environments where Langfuse host may be unreachable,
    preventing noisy error messages from being printed to the console.
    """
    # Suppress OpenTelemetry SDK export errors
    logging.getLogger("opentelemetry.sdk._shared_internal").setLevel(logging.CRITICAL)
    logging.getLogger("opentelemetry.exporter.otlp.proto.http.trace_exporter").setLevel(logging.CRITICAL)
    # Also suppress requests library connection errors related to OTLP
    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)


# Global flag to track if timeout warning has been shown
_langfuse_timeout_warned: bool = False


def _check_langfuse_connection() -> bool:
    """
    Check if Langfuse host is reachable.
    
    Returns:
        True if connection is successful, False otherwise.
    """
    global _langfuse_timeout_warned
    
    langfuse_host = LANGFUSE_BASE_URL
    
    try:
        import urllib.request
        import urllib.error
        
        # Simple HEAD request with short timeout
        req = urllib.request.Request(langfuse_host, method="HEAD")
        urllib.request.urlopen(req, timeout=5)
        return True
    except urllib.error.URLError as e:
        if not _langfuse_timeout_warned:
            _langfuse_timeout_warned = True
            logger.warning(
                f"Langfuse host unreachable: {langfuse_host}. "
                f"Tracing data will not be sent. Error: {e.reason}"
            )
        return False
    except Exception as e:
        if not _langfuse_timeout_warned:
            _langfuse_timeout_warned = True
            logger.warning(
                f"Langfuse connection check failed: {langfuse_host}. "
                f"Tracing may not work. Error: {e}"
            )
        return False


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
            # Suppress OpenTelemetry exporter errors (e.g., timeout when host is unreachable)
            _suppress_opentelemetry_errors()
            # Check connection and warn user if host is unreachable
            if _check_langfuse_connection():
                logger.debug("Langfuse is available and configured (using OpenTelemetry auto-instrumentation)")
            else:
                # Still return True to allow graceful degradation
                # The warning has already been logged in _check_langfuse_connection
                pass
        else:
            logger.debug(f"Langfuse package installed but not configured (missing env vars), LANGFUSE_PUBLIC_KEY: {LANGFUSE_PUBLIC_KEY}")
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


@contextmanager
def langfuse_trace_context(
        name: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        input_data: Optional[Any] = None,
) -> Generator[Any, None, None]:
    """
    Context manager for grouping multiple LLM calls into a single Langfuse trace.

    This is essential for multi-turn agent interactions where a single Agent.run()
    may invoke the LLM multiple times (e.g., with tool calls). All calls within
    this context will be grouped under the same trace with the specified session_id.

    Args:
        name: Name for the trace (e.g., "agent-run", "chat-completion")
        session_id: Session identifier for grouping multi-turn conversations
        user_id: User identifier
        tags: Optional list of tags for filtering
        metadata: Optional additional metadata
        input_data: Optional input data to log with the trace

    Yields:
        A LangfuseTraceWrapper object that allows setting output after execution

    Example:
        with langfuse_trace_context(
            name="my-agent-run",
            session_id="session-123",
            user_id="user-456"
        ) as trace:
            response1 = openai.chat.completions.create(...)
            response2 = openai.chat.completions.create(...)
            trace.set_output(response2.choices[0].message.content)
    """
    if not is_langfuse_available():
        yield _DummySpanWrapper()
        return

    try:
        from langfuse import get_client

        langfuse = get_client()

        # Start a span as the root observation for this agent run
        # This automatically creates a trace
        # OpenAI calls within this context will automatically become children
        with langfuse.start_as_current_observation(
                as_type="span",
                name=name,
                input=input_data,
                metadata=metadata,
        ) as root_span:
            # Set trace-level attributes at the beginning
            root_span.update_trace(
                session_id=session_id,
                user_id=user_id,
                tags=tags,
            )

            # Create a wrapper to collect output
            wrapper = _LangfuseTraceWrapper(root_span, langfuse)

            yield wrapper

            # IMPORTANT: After all child observations are complete,
            # update trace output as the LAST step using update_current_trace()
            # This prevents child spans from overwriting the trace output
            if wrapper._output is not None:
                # Update root span output first
                root_span.update(output=wrapper._output)
                # Then use update_current_trace() as the final step
                # This is the recommended way to set trace output when there are nested observations
                langfuse.update_current_trace(
                    input=input_data,
                    output=wrapper._output,
                )

            if wrapper._metadata:
                root_span.update(metadata=wrapper._metadata)

    except ImportError:
        logger.debug("Langfuse context manager not available")
        yield _DummySpanWrapper()
    except Exception as e:
        logger.warning(f"Failed to create Langfuse trace context: {e}")
        yield _DummySpanWrapper()


class _LangfuseTraceWrapper:
    """Wrapper to collect output and update trace."""

    def __init__(self, span: Any, langfuse_client: Any):
        self._span = span
        self._langfuse = langfuse_client
        self._output: Optional[Any] = None
        self._metadata: Dict[str, Any] = {}

    def set_output(self, output: Any) -> None:
        """Set the output to be recorded when the context exits."""
        self._output = output

    def set_metadata(self, key: str, value: Any) -> None:
        """Add metadata to be recorded when the context exits."""
        self._metadata[key] = value

    def update(self, **kwargs) -> None:
        """Directly update the underlying span."""
        if self._span:
            try:
                self._span.update(**kwargs)
            except Exception as e:
                logger.debug(f"Failed to update span: {e}")


class _LangfuseSpanWrapper:
    """Wrapper to collect output and update span before context exits."""

    def __init__(self, span: Any, langfuse_client: Any = None):
        self._span = span
        self._langfuse = langfuse_client
        self._output: Optional[Any] = None
        self._metadata: Dict[str, Any] = {}

    def set_output(self, output: Any) -> None:
        """Set the output to be recorded when the context exits."""
        self._output = output

    def set_metadata(self, key: str, value: Any) -> None:
        """Add metadata to be recorded when the context exits."""
        self._metadata[key] = value

    def update(self, **kwargs) -> None:
        """Directly update the underlying span."""
        if self._span:
            try:
                self._span.update(**kwargs)
            except Exception as e:
                logger.debug(f"Failed to update span: {e}")


class _DummySpanWrapper:
    """Dummy wrapper when Langfuse is not available."""

    def set_output(self, _output: Any) -> None:
        pass

    def set_metadata(self, _key: str, _value: Any) -> None:
        pass

    def update(self, **_kwargs) -> None:
        pass


@contextmanager
def langfuse_span_context(
        name: str,
        input_data: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
) -> Generator[Any, None, None]:
    """
    Context manager for creating a nested span within an existing trace.

    Use this for grouping related operations within an agent run,
    such as tool execution or retrieval steps.

    Args:
        name: Name for the span (e.g., "tool-execution", "retrieval")
        input_data: Optional input data to log
        metadata: Optional additional metadata

    Yields:
        The Langfuse span object if available, None otherwise
    """
    if not is_langfuse_available():
        yield None
        return

    try:
        from langfuse import get_client

        langfuse = get_client()

        with langfuse.start_as_current_observation(
                as_type="span",
                name=name,
                input=input_data,
                metadata=metadata,
        ) as span:
            yield span

    except ImportError:
        yield None
    except Exception as e:
        logger.warning(f"Failed to create Langfuse span: {e}")
        yield None


def update_langfuse_span(
        span: Any,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        level: Optional[str] = None,
        status_message: Optional[str] = None,
) -> None:
    """
    Update a Langfuse span with output and additional information.

    Args:
        span: The Langfuse span object
        output: Output data to log
        metadata: Additional metadata to add
        level: Log level (DEBUG, DEFAULT, WARNING, ERROR)
        status_message: Status message
    """
    if span is None:
        return

    try:
        update_kwargs = {}
        if output is not None:
            update_kwargs["output"] = output
        if metadata is not None:
            update_kwargs["metadata"] = metadata
        if level is not None:
            update_kwargs["level"] = level
        if status_message is not None:
            update_kwargs["status_message"] = status_message

        if update_kwargs:
            span.update(**update_kwargs)
    except Exception as e:
        logger.debug(f"Failed to update Langfuse span: {e}")
