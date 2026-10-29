# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Langfuse integration for observability and tracing

Langfuse is an open-source LLM observability platform that helps you trace,
monitor, and debug your LLM applications.

This module provides:
    1. One root trace per Agent.run user request
    2. Request input and final response output
    3. Child observations for real model calls and tool executions
    4. Session, user, and tag attribution

Usage:
    1. Install langfuse:
        pip install langfuse

    2. Set environment variables:
        LANGFUSE_SECRET_KEY = "sk-lf-xxx"
        LANGFUSE_PUBLIC_KEY = "pk-lf-xxx"
        LANGFUSE_BASE_URL = "https://cloud.langfuse.com"  # or your self-hosted URL

    3. Use agentica as normal - Langfuse records one root trace per request:
        from agentica import Agent
        from agentica.model.openai import OpenAIChat

        agent = Agent(
            name="My Agent",
            user_id="user-123",  # Automatically passed to Langfuse
            session_id="session-abc",  # Groups multi-turn conversations
            model=OpenAIChat(id="gpt-4o"),
        )

        response = agent.run("Hello!")
        # Model calls and tools are children of this trace; hooks are not

    4. Bounded shutdown at process exit (langfuse's own atexit shutdown
       blocks for ~2s: span force_flush + consumer-thread poll joins).
       The CLI does this automatically; SDK apps should call it from their
       shutdown hook:
        from agentica.utils.langfuse_integration import shutdown_langfuse_bounded
        shutdown_langfuse_bounded()  # overlaps nothing, caps the wait
"""

import base64
import logging
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional, List, Dict, Any, Generator, Tuple
from agentica.config import LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_BASE_URL
from agentica.security.redact import redact_sensitive_text
from agentica.utils.log import logger


# Global flag to track if Langfuse is available
_langfuse_available: Optional[bool] = None


_langfuse_root_active: ContextVar[bool] = ContextVar(
    "langfuse_root_active",
    default=False,
)


def bound_langfuse_value(value: Any, depth: int = 0) -> Any:
    """Bound and redact a value before sending it to Langfuse.

    A ``data:...;base64,...`` URI is left whole when it decodes. Chopping it
    at 4000 chars (or redacting a ``sk-`` run inside the payload) leaves a
    string Langfuse still treats as media, and ``b64decode`` then logs
    ``Error parsing base64 data URI`` / ``Incorrect padding``. Langfuse
    replaces a valid URI with a short media reference before the span
    attribute is stored, so the cap is not what keeps the field small.
    """
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        if _decode_base64_data_uri(value) is not None:
            return value
        if _looks_like_base64_data_uri(value):
            return _omit_broken_data_uri(value)
        text = redact_sensitive_text(value, level="strict") or ""
        return text if len(text) <= 4000 else text[:4000] + "...[truncated]"
    if depth >= 6:
        return f"<depth-truncated: {type(value).__name__}>"
    if isinstance(value, dict):
        return {
            str(key): bound_langfuse_value(item, depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [bound_langfuse_value(item, depth + 1) for item in value]
    return bound_langfuse_value(str(value), depth + 1)


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


def _looks_like_base64_data_uri(data: Any) -> bool:
    """True only for strings shaped like ``data:...;base64,...``.

    Langfuse's MediaManager treats every string starting with ``data:`` as a
    base64 media URI. SSE frames (``data: {"choices":...}``), Swift type
    annotations, and other ``data:`` prefixes then explode as
    ``Error parsing base64 data URI`` / ``Data is not base64 encoded``.
    Shape is not enough: a URI chopped mid-payload still matches and then
    fails with ``Incorrect padding``. Decode with ``_decode_base64_data_uri``.
    """
    if not isinstance(data, str) or not data.startswith("data:"):
        return False
    header, sep, _rest = data[5:].partition(",")
    if not sep or not header:
        return False
    return "base64" in header.split(";")


def _decode_base64_data_uri(data: str) -> Optional[Tuple[bytes, str]]:
    """Decode a media data URI, or None when it is not valid base64.

    Does not log. Langfuse's own parser logs ``Error parsing base64 data URI``
    on the same failure, which is how a truncated screenshot reached the CLI.
    """
    if not _looks_like_base64_data_uri(data):
        return None
    header, _, payload = data[5:].partition(",")
    mime = header.split(";", 1)[0]
    if not mime or not payload:
        return None
    translated = "".join(payload.split()).translate(str.maketrans("-_", "+/"))
    try:
        return base64.b64decode(translated, validate=True), mime
    except Exception:
        return None


def _omit_broken_data_uri(data: str) -> str:
    """Replace an undecodable data URI with text Langfuse will not parse."""
    header, _, payload = data[5:].partition(",")
    mime = header.split(";", 1)[0] or "application/octet-stream"
    return f"[media omitted mime={mime} chars={len(payload)}]"


def _install_langfuse_media_guard() -> None:
    """Parse media data URIs ourselves so a bad payload does not get logged.

    Langfuse catches the decode error and logs it with a traceback. Returning
    ``None, None`` is the same outcome, without the log line.
    """
    try:
        from langfuse.media import LangfuseMedia
    except ImportError:
        return
    original = LangfuseMedia._parse_base64_data_uri
    if getattr(original, "_agentica_guarded", False):
        return

    def _parse_base64_data_uri(self, data: str):
        parsed = _decode_base64_data_uri(data)
        if parsed is None:
            return None, None
        return parsed

    _parse_base64_data_uri._agentica_guarded = True  # type: ignore[attr-defined]
    LangfuseMedia._parse_base64_data_uri = _parse_base64_data_uri  # type: ignore[method-assign]


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
            _install_langfuse_media_guard()
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
    """Shutdown Langfuse and flush pending events (unbounded — see below)."""
    try:
        if is_langfuse_configured():
            from langfuse import get_client
            get_client().shutdown()
            logger.debug("Langfuse shutdown completed")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to shutdown Langfuse: {e}")


def start_shutdown_thread() -> Optional["threading.Thread"]:
    """Kick Langfuse shutdown on a daemon thread; return it (None when off).

    ``resource_manager.shutdown`` blocks on two serial sleeps — the OTEL
    span ``force_flush`` plus pausing/joining the score/media consumer
    threads, whose poll loop costs ~1s each even when idle. Started early, the
    work overlaps the caller's own teardown; a bounded ``join`` at the very end
    caps it. The thread is daemon: if the join times out the process exits
    anyway, dropping at most the tail of telemetry (spans inside one flush
    interval), which the langfuse atexit handler would otherwise hold the
    process hostage for — with a slow/unreachable host that is tens of
    seconds, which we measured as ~2s even on a healthy one.
    """
    if not is_langfuse_configured():
        return None
    try:
        # Peek the singleton registry first: get_client() lazily *creates* a
        # client (spawning its consumer threads), so calling it in a session
        # that never traced would pay the very shutdown cost we are avoiding.
        from langfuse._client.resource_manager import LangfuseResourceManager
        with LangfuseResourceManager._lock:
            if not LangfuseResourceManager._instances:
                return None
        from langfuse import get_client
        client = get_client()
    except ImportError:
        return None
    except Exception as e:
        # Private-API drift or a quirky environment must never take down the
        # caller's exit path — degrade to "no bounded shutdown".
        logger.warning(f"Langfuse shutdown thread not started: {e}")
        return None

    def _shutdown():
        try:
            # rm.shutdown is ordered right for us: it unregisters its own
            # atexit FIRST (so a process exit during the flush never blocks
            # again), then force_flushes the spans (the actual network send),
            # then pauses/joins the score & media poll threads (~1s poll cycle
            # each even when idle — the bulk of the wait, and the part a
            # bounded join may safely cut: those queues sit empty in the CLI).
            client.shutdown()
            logger.debug("Langfuse shutdown completed")
        except Exception as e:
            logger.warning(f"Failed to shutdown Langfuse: {e}")

    thread = threading.Thread(target=_shutdown, daemon=True, name="langfuse-shutdown")
    thread.start()
    return thread


def shutdown_langfuse_bounded(timeout: float = 0.8) -> None:
    """Bounded convenience wrapper: start + join with a timeout."""
    thread = start_shutdown_thread()
    if thread is not None:
        thread.join(timeout=timeout)


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
    Context manager for one user-request Langfuse trace.

    Real model calls and tool executions are child observations of this
    trace. Hooks stay internal. A nested agent reuses the active trace.

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
            result = run_agent_workflow()
            trace.set_output(result)
    """
    if _langfuse_root_active.get() or not is_langfuse_available():
        yield _DummySpanWrapper()
        return

    try:
        from langfuse import get_client, propagate_attributes

        langfuse = get_client()
        observation = langfuse.start_as_current_observation(
            as_type="agent",
            name=name,
            input=input_data,
            metadata=metadata,
        )
    except ImportError:
        logger.debug("Langfuse context manager not available")
        yield _DummySpanWrapper()
        return
    except Exception as e:
        logger.warning(f"Failed to create Langfuse trace context: {e}")
        yield _DummySpanWrapper()
        return

    with observation as root_span:
        with propagate_attributes(
            trace_name=name,
            session_id=session_id,
            user_id=user_id,
            tags=tags,
        ):
            wrapper = _LangfuseTraceWrapper(root_span)
            root_token = _langfuse_root_active.set(True)
            try:
                yield wrapper
            finally:
                _langfuse_root_active.reset(root_token)
                try:
                    if wrapper._output is not None:
                        root_span.update(output=wrapper._output)

                    if wrapper._metadata:
                        root_span.update(metadata=wrapper._metadata)
                except Exception as e:
                    logger.debug(f"Failed to finalize Langfuse trace context: {e}")


class _LangfuseTraceWrapper:
    """Wrapper to collect output and update trace."""

    def __init__(self, span: Any):
        self._span = span
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
def langfuse_observation_context(
        name: str,
        *,
        as_type: str,
        input_data: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
) -> Generator[Any, None, None]:
    """Create a child observation inside the active user-request trace."""
    if not _langfuse_root_active.get() or not is_langfuse_available():
        yield _DummySpanWrapper()
        return

    try:
        from langfuse import get_client

        observation = get_client().start_as_current_observation(
            as_type=as_type,
            name=name,
            input=input_data,
            metadata=metadata,
            model=model,
        )
    except ImportError:
        yield _DummySpanWrapper()
        return
    except Exception as error:
        logger.warning(f"Failed to create Langfuse observation: {error}")
        yield _DummySpanWrapper()
        return

    with observation as span:
        wrapper = _LangfuseTraceWrapper(span)
        try:
            yield wrapper
        except BaseException as error:
            wrapper.update(level="ERROR", status_message=str(error)[:1000])
            raise
        finally:
            try:
                if wrapper._output is not None:
                    span.update(output=wrapper._output)
                if wrapper._metadata:
                    span.update(metadata=wrapper._metadata)
            except Exception as error:
                logger.debug(f"Failed to finalize Langfuse observation: {error}")
