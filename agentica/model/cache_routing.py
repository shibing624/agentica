# -*- coding: utf-8 -*-
"""
@description: Sticky-routing session ids for prompt caches on caching proxies.

A prompt cache only pays off when consecutive requests land on the same backend
machine. Aggregating proxies that fan out across several upstreams let the
caller pin that choice with a request header. Both the Anthropic and the
OpenAI provider need the same id resolution, so it lives here rather than under
one provider's module — importing ``model/openai/chat.py`` from the Anthropic
side would drag the whole OpenAI SDK in.
"""
import json
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

CACHE_ROUTING_FILENAME = "cache_routing.json"


def persistent_cache_session_id(base_url: Any, home: Optional[Path] = None) -> str:
    """Load-or-create the sticky cache-routing id for this endpoint.

    Persisted in ``~/.agentica/cache/cache_routing.json`` keyed by base_url so
    a new CLI process keeps the same routing id and lands on the same proxy
    backend (a fresh random id would guarantee a cold cache). Best-effort: if
    the file is unreadable/unwritable, an in-memory id is used for this process.

    ``home`` overrides ``~`` for tests; production callers leave it None.
    """
    path = (home or Path.home()) / ".agentica" / "cache" / CACHE_ROUTING_FILENAME
    key = str(base_url or "default").rstrip("/")
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    sid = data.get(key)
    if not sid:
        sid = f"agentica-cache-{uuid4()}"
        data[key] = sid
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass  # persistence is best-effort; the in-memory id still works
    return sid


def resolve_cache_session_id(
    session_id: Optional[str] = None,
    base_url: Any = None,
    home: Optional[Path] = None,
) -> str:
    """Pick the sticky-routing value: the live session when there is one.

    Conversation scope wins because it is what the routing decision is really
    about — one session shares one prompt prefix, and it also gives the user an
    escape hatch: on a proxy with a misbehaving backend, starting a new session
    re-roles the dice instead of being stuck on the same machine forever. The
    endpoint-persistent id is the fallback for bare SDK use, where no session
    exists but a stable key still beats none.

    Call this per request; do NOT memoize the result into an instance field.
    ``Model.session_id`` is assigned by ``Agent.update_model()`` on every turn
    and is legitimately None on an earlier call (bare SDK use, or a model
    invoked before the agent wires identity). Caching that first answer would
    freeze the fallback forever and the real session would never be picked up.
    ``persistent_cache_session_id`` is one small JSON read and is constant per
    endpoint, so resolving fresh costs nothing worth caching.
    """
    return session_id or persistent_cache_session_id(base_url, home=home)
