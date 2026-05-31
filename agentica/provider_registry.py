# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Pluggable LLM provider registry.

Lets users and third-party packages add a model provider WITHOUT editing
agentica core. A provider is just a factory ``(**kwargs) -> Model``.

Two ways to register:

1. In-process:
       from agentica import register_provider, OpenAIChat
       register_provider("myllm", lambda **kw: OpenAIChat(
           base_url="https://my-llm/v1", **kw))
       model = create_provider("myllm", id="my-model", api_key="...")

2. Via packaging entry points (auto-discovered, no import needed):
       # pyproject.toml of a third-party package
       [project.entry-points."agentica.providers"]
       myllm = "my_pkg.providers:make_model"

Built-in providers (openai/kimi/anthropic/azure + every OpenAI-compatible
factory in ``PROVIDER_FACTORIES``) are seeded automatically.
"""
import threading
from typing import Callable, Dict, List, Optional

from agentica.utils.log import logger

# slug -> factory callable(**kwargs) -> Model
_REGISTRY: Dict[str, Callable] = {}
_LOCK = threading.RLock()
_seeded = False


def register_provider(slug: str, factory: Callable, *, overwrite: bool = False) -> Callable:
    """Register a provider factory under ``slug``.

    Args:
        slug: Provider identifier (case-insensitive), e.g. "myllm".
        factory: Callable returning a Model instance, accepting kwargs like
            ``id`` / ``api_key`` / ``base_url``.
        overwrite: Allow replacing an existing registration.

    Raises:
        ValueError: If slug is already registered and overwrite is False.
    """
    slug = slug.strip().lower()
    if not slug:
        raise ValueError("Provider slug must be non-empty")
    if not callable(factory):
        raise ValueError("Provider factory must be callable")
    with _LOCK:
        if slug in _REGISTRY and not overwrite:
            raise ValueError(
                f"Provider '{slug}' already registered (pass overwrite=True to replace)"
            )
        _REGISTRY[slug] = factory
    logger.debug(f"Registered LLM provider: {slug}")
    return factory


def unregister_provider(slug: str) -> bool:
    """Remove a provider registration. Returns True if it existed."""
    with _LOCK:
        return _REGISTRY.pop(slug.strip().lower(), None) is not None


def get_provider_factory(slug: str) -> Optional[Callable]:
    """Return the factory for a provider slug, or None if unknown."""
    _ensure_seeded()
    return _REGISTRY.get(slug.strip().lower())


def list_providers() -> List[str]:
    """Return all known provider slugs (built-in + plugin), sorted."""
    _ensure_seeded()
    with _LOCK:
        return sorted(_REGISTRY)


def create_provider(slug: str, **kwargs):
    """Instantiate a model from a registered provider.

    Raises:
        ValueError: If the provider slug is not registered.
    """
    factory = get_provider_factory(slug)
    if factory is None:
        raise ValueError(
            f"Unknown provider '{slug}'. Available: {list_providers()}"
        )
    return factory(**kwargs)


# ── seeding ──────────────────────────────────────────────────────────────
def _ensure_seeded() -> None:
    global _seeded
    if _seeded:
        return
    with _LOCK:
        if _seeded:
            return

        def _openai(**kw):
            from agentica.model.openai import OpenAIChat
            return OpenAIChat(**kw)

        def _kimi(**kw):
            from agentica.model.kimi.chat import KimiChat
            return KimiChat(**kw)

        def _claude(**kw):
            from agentica.model.anthropic.claude import Claude
            return Claude(**kw)

        def _azure(**kw):
            from agentica.model.azure import AzureOpenAIChat
            return AzureOpenAIChat(**kw)

        builtins: Dict[str, Callable] = {
            "openai": _openai,
            "kimi": _kimi,
            "anthropic": _claude,
            "claude": _claude,
            "azure": _azure,
        }
        try:
            from agentica import PROVIDER_FACTORIES
            builtins.update(PROVIDER_FACTORIES)
        except Exception as e:  # pragma: no cover - import-time safety only
            logger.debug(f"PROVIDER_FACTORIES seeding skipped: {e}")

        # setdefault: never clobber a user registration done before first use.
        for slug, factory in builtins.items():
            _REGISTRY.setdefault(slug, factory)

        # Mark seeded BEFORE loading plugins: a plugin's import side effects may
        # call create_provider() reentrantly (RLock allows it); with _seeded
        # already True that nested call won't recursively re-run seeding.
        _seeded = True
        _load_entry_point_providers()


def _load_entry_point_providers() -> None:
    """Discover providers declared under the ``agentica.providers`` entry point."""
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - py<3.8 only
        return
    try:
        eps = entry_points()
        if hasattr(eps, "select"):  # py3.10+
            group = eps.select(group="agentica.providers")
        else:  # pragma: no cover - legacy mapping API
            group = eps.get("agentica.providers", [])
        for ep in group:
            try:
                _REGISTRY.setdefault(ep.name.strip().lower(), ep.load())
                logger.debug(f"Loaded provider plugin from entry point: {ep.name}")
            except Exception as e:
                logger.warning(f"Failed to load provider plugin '{ep.name}': {e}")
    except Exception as e:  # pragma: no cover - discovery is best-effort
        logger.debug(f"Provider entry-point discovery failed: {e}")
