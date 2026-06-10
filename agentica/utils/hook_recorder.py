# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: HookRecorder - lifecycle-agnostic hook execution recorder.

A single recording point for hook invocations across Runner / Model / Composite.
Replaces the previous duplicate helpers (_run_hook_with_langfuse,
_run_tool_hook_with_record, _serialize_hook_result, _has_hook_payload, ...)
spread across runner.py and model/base.py.

Design goals:
- Backend-neutral name (no "langfuse" in identifiers); the trace exporter is
  the only consumer of records().
- Safe serialization: depth-bounded, cycle-detecting, length-capped, so a
  hook returning a large/cyclic object cannot blow up trace upload.
- Override-aware: skips recording for unmodified base hook methods, so a
  default RunHooks() / AgentHooks() never pollutes the trace metadata.
- Per-run lifecycle: Agent owns one HookRecorder; Runner resets it at start.

The recorder does NOT decide whether to call the hook — that is the
composite/runner's job. It only wraps the awaitable, times it, and stores
a JSON-friendly audit record.
"""
import time
from typing import Any, Awaitable, Dict, List, Optional


# Reserved record fields. Caller-supplied metadata is nested under "meta" to
# guarantee it can never overwrite system fields (hook_type, ok, error, ...).
_RESERVED_FIELDS = frozenset({
    "hook_type", "hook_class", "hook_module", "method", "ok",
    "duration_ms", "output", "meta", "error", "error_type",
})


class HookRecorder:
    """Per-run audit log for hook invocations.

    Use ``await recorder.run(hook, hook_type, method, awaitable, base_class=...)``
    to execute a hook awaitable while capturing duration, output, and errors.
    Call ``recorder.export()`` at run end to dump the records into trace
    metadata.

    Not thread-safe: one recorder per run is the assumed model.
    """

    # Serializer caps. Tuned for trace upload friendliness:
    # - 8 levels covers typical Message/dict/list nesting without tail risk.
    # - 4000 chars per string keeps a single field under Langfuse's 4-5KB
    #   per-key practical limit while preserving useful context.
    MAX_DEPTH = 8
    MAX_STR = 4000

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self.records = []

    def export(self) -> List[Dict[str, Any]]:
        """Return a shallow copy of the records collected so far."""
        return list(self.records)

    def __bool__(self) -> bool:
        return bool(self.records)

    def __len__(self) -> int:
        return len(self.records)

    async def run(
        self,
        hook: Any,
        hook_type: str,
        method: str,
        awaitable: Awaitable[Any],
        *,
        base_class: Optional[type] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute the hook awaitable and append its audit record.

        - On success, records the call when the method is overridden vs
          ``base_class`` OR metadata was supplied OR the result has payload.
          Unmodified base methods returning None are not recorded.
        - On failure, records the error unconditionally and re-raises.

        The ``metadata`` dict (e.g. ``{"tool_name": ..., "tool_call_id": ...}``)
        is nested under ``record["meta"]`` so it can never collide with system
        fields.
        """
        start = time.perf_counter()
        try:
            result = await awaitable
        except Exception as error:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.records.append(self._build_record(
                hook=hook,
                hook_type=hook_type,
                method=method,
                elapsed_ms=elapsed_ms,
                ok=False,
                error=error,
                metadata=metadata,
            ))
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        if not self._should_record_success(hook, method, base_class, result, metadata):
            return result
        self.records.append(self._build_record(
            hook=hook,
            hook_type=hook_type,
            method=method,
            elapsed_ms=elapsed_ms,
            ok=True,
            result=result,
            metadata=metadata,
        ))
        return result

    @classmethod
    def _should_record_success(
        cls,
        hook: Any,
        method: str,
        base_class: Optional[type],
        result: Any,
        metadata: Optional[Dict[str, Any]],
    ) -> bool:
        # Always record when the caller passed metadata (e.g. tool name) — that
        # is itself the audit value, even if the hook returns None.
        if metadata:
            return True
        # Record when there is a non-empty payload.
        if cls._has_payload(cls._serialize(result)):
            return True
        # Record when the hook subclass overrides the method, even with `pass`.
        # The override itself is the user's signal that this lifecycle slot
        # matters and should appear on the trace.
        if base_class is not None and cls._method_overridden(hook, base_class, method):
            return True
        return False

    @staticmethod
    def _method_overridden(hook: Any, base_class: type, method: str) -> bool:
        """True when the hook's class redefines ``method`` vs ``base_class``.

        Pure class-level comparison: AsyncMock instances and runtime
        monkey-patching are correctly handled by the composite calling the
        hook directly; the recorder only filters which records to keep.
        """
        cls_method = getattr(type(hook), method, None)
        base_method = getattr(base_class, method, None)
        if cls_method is None or base_method is None:
            return False
        return cls_method is not base_method

    @classmethod
    def _build_record(
        cls,
        hook: Any,
        hook_type: str,
        method: str,
        elapsed_ms: float,
        ok: bool,
        result: Any = None,
        error: Optional[BaseException] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "hook_type": hook_type,
            "hook_class": type(hook).__name__,
            "hook_module": type(hook).__module__,
            "method": method,
            "ok": ok,
            "duration_ms": round(elapsed_ms, 3),
        }
        if ok:
            output = cls._serialize(result)
            if cls._has_payload(output):
                record["output"] = output
        if metadata:
            # Nest metadata under "meta" so user-supplied keys can never
            # overwrite system fields.
            record["meta"] = {
                str(key): cls._serialize(value)
                for key, value in metadata.items()
            }
        if error is not None:
            record["error_type"] = type(error).__name__
            record["error"] = str(error)
        return record

    @classmethod
    def _serialize(cls, value: Any, _depth: int = 0, _seen: Optional[set] = None) -> Any:
        """Best-effort JSON-friendly conversion with depth/cycle/length caps.

        Handles: scalars, str (capped), dict, list/tuple, objects exposing
        ``to_model_dict()`` (Message) or ``model_dump()`` (Pydantic). Anything
        else falls back to ``str(value)`` (also capped) so the recorder can
        never raise out of the serialization path.
        """
        if value is None or isinstance(value, (int, float, bool)):
            return value
        if isinstance(value, str):
            if len(value) > cls.MAX_STR:
                return value[:cls.MAX_STR] + "...[truncated]"
            return value

        if _depth >= cls.MAX_DEPTH:
            return f"<depth-truncated: {type(value).__name__}>"
        if _seen is None:
            _seen = set()
        if id(value) in _seen:
            return "<cycle>"
        _seen = _seen | {id(value)}

        if isinstance(value, dict):
            return {
                str(key): cls._serialize(item, _depth + 1, _seen)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [cls._serialize(item, _depth + 1, _seen) for item in value]

        # Duck-typed conversion for richer types. Both Message and Pydantic
        # BaseModel are common hook outputs; converting them to dict/list
        # preserves structure for the trace consumer. The conversion call is
        # the audit boundary — if the producer's serializer is broken, we
        # fall back to a string representation rather than killing the run.
        to_model_dict = getattr(value, "to_model_dict", None)
        if callable(to_model_dict):
            try:
                converted = to_model_dict()
            except Exception as conv_err:
                return f"<unserializable {type(value).__name__}: {conv_err}>"
            return cls._serialize(converted, _depth + 1, _seen)

        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                converted = model_dump()
            except Exception as conv_err:
                return f"<unserializable {type(value).__name__}: {conv_err}>"
            return cls._serialize(converted, _depth + 1, _seen)

        text = str(value)
        if len(text) > cls.MAX_STR:
            return text[:cls.MAX_STR] + "...[truncated]"
        return text

    @classmethod
    def _has_payload(cls, value: Any) -> bool:
        """True when ``value`` carries non-empty audit content."""
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (int, float, bool)):
            return True
        if isinstance(value, dict):
            return any(cls._has_payload(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(cls._has_payload(item) for item in value)
        return True
