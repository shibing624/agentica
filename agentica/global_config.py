# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Unified, hand-editable configuration shared by the SDK and the CLI.

Historically agentica had three disconnected configuration surfaces:

* the SDK core (``agentica/config.py`` + model classes) read **only** process
  environment variables (populated at import time by ``python-dotenv`` from the
  project ``.env`` and ``~/.agentica/.env``);
* the CLI (``agentica/cli/setup.py``) read ``~/.agentica/cli_config.json``;
* the gateway (``agentica/gateway/config.py``) read its own ``AGENTICA_*`` env
  vars.

This module introduces a single, structured, human-editable file —
``~/.agentica/agentica.json`` — that both the SDK and the CLI can share. It
supports **named profiles** (e.g. ``default``, ``gpt``, ``local``) so a user can
keep several provider/model setups and switch the active one at runtime.

Design principles
-----------------
* **SDK contract is preserved.** The SDK keeps reading plain environment
  variables. On import, :func:`apply_global_config` projects the *active*
  profile's ``api_key`` (and any free-form ``env`` block) into ``os.environ``.
  No model class or tool needs to change.
* **Shell env always wins.** Values are injected with ``setdefault`` semantics
  (we never override a variable that is already present in the process
  environment). This matches 12-factor / dotenv conventions and avoids the
  "stale key in a file silently shadows my shell export" footgun.
* **``.env`` stays supported.** ``~/.agentica/.env`` is still loaded (for users
  who maintain it by hand, MCP tools, CI, etc.). Precedence, highest first:
  shell env  >  .env  >  agentica.json.
* **Hand-editable.** The file is plain JSON with a stable schema; ``agentica
  setup`` writes it, but users may edit it directly.

Schema (``~/.agentica/agentica.json``)::

    {
      "active_profile": "default",
      "profiles": {
        "default": {
          "model_provider": "deepseek",
          "model_name": "deepseek-v4-flash",
          "base_url": "https://api.deepseek.com",
          "api_key": "sk-...",

          // --- optional model tuning (omit to use model/factory defaults) ---
          "reasoning_effort": "max",   // thinking depth: low|medium|high|max
          "max_tokens": 8192,          // output limit (max output tokens)
          "context_window": 1000000,   // context limit; overrides catalog value
          "temperature": 0.7,
          "top_p": 0.95
        },
        "gpt": {
          "model_provider": "openai",
          "model_name": "gpt-4o",
          "base_url": "https://api.openai.com/v1",
          "api_key": "sk-...",
          "reasoning_effort": "high"
        }
      },
      "env": {
        "SERPER_API_KEY": "..."
      }
    }

``reasoning_effort`` maps a model's "thinking depth" (OpenAI o-series / gpt-5.x
and DeepSeek use this; Claude uses a separate thinking budget). ``max_tokens``
is the output token cap. ``context_window`` is a capability value used for
context-budget display and compression — it is NOT sent to the API; setting it
overrides the value auto-detected from the model catalog.

The file is written with ``chmod 0o600`` because profiles hold secrets.
"""

import json
import os
from typing import Dict, Optional, Any

# Map a provider slug to the environment variable its model factory reads for
# the API key. Kept in sync with the provider factories in agentica/__init__.py
# and model/defaults.py. A profile's ``api_key`` is injected into this env var
# so the existing env-based SDK key resolution just works.
PROVIDER_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "zhipuai": "ZAI_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "kimi": "KIMI_API_KEY",
    "ark": "ARK_API_KEY",
    "yi": "YI_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "together": "TOGETHER_API_KEY",
    "xai": "XAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "sambanova": "SAMBANOVA_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "internlm": "INTERNLM_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
}

# The base_url env var used by the OpenAI-compatible base client. Injecting this
# lets a custom endpoint defined purely in agentica.json work without flags.
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"

DEFAULT_PROFILE_NAME = "default"


def global_config_path() -> str:
    """Return the path to ``agentica.json`` (honours ``AGENTICA_HOME``).

    Resolved lazily (not import-time-frozen) so tests can point
    ``AGENTICA_HOME`` elsewhere.
    """
    home = os.path.expanduser(os.getenv("AGENTICA_HOME", "~/.agentica"))
    return os.path.join(home, "agentica.json")


def provider_api_key_env(provider: str) -> str:
    """Return the env var a provider reads for its API key (OPENAI fallback)."""
    return PROVIDER_API_KEY_ENV.get(provider, "OPENAI_API_KEY")


def load_global_config() -> Dict[str, Any]:
    """Load ``agentica.json``, or an empty dict if missing/invalid."""
    path = global_config_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_global_config(config: Dict[str, Any]) -> None:
    """Persist ``agentica.json`` to disk with ``0o600`` perms (holds secrets)."""
    path = global_config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    try:
        os.chmod(path, 0o600)
    except OSError:
        # Non-fatal on platforms where chmod has no effect; contents are saved.
        pass


def get_active_profile_name(config: Optional[Dict[str, Any]] = None) -> str:
    """Return the active profile name, defaulting to ``default``."""
    data = load_global_config() if config is None else config
    return data.get("active_profile") or DEFAULT_PROFILE_NAME


def get_profiles(config: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """Return the ``profiles`` map, or an empty dict."""
    data = load_global_config() if config is None else config
    profiles = data.get("profiles")
    return profiles if isinstance(profiles, dict) else {}


def get_profile(name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a single profile dict (the active one if ``name`` is None)."""
    data = load_global_config() if config is None else config
    profiles = get_profiles(data)
    if name is None:
        name = get_active_profile_name(data)
    profile = profiles.get(name)
    return profile if isinstance(profile, dict) else {}


def set_active_profile(name: str) -> bool:
    """Mark ``name`` as the active profile. Returns False if it does not exist."""
    config = load_global_config()
    if name not in get_profiles(config):
        return False
    config["active_profile"] = name
    save_global_config(config)
    return True


def upsert_profile(name: str, profile: Dict[str, Any], make_active: bool = True) -> None:
    """Create or update a named profile and optionally make it active."""
    config = load_global_config()
    profiles = config.get("profiles")
    if not isinstance(profiles, dict):
        profiles = {}
    # Drop None values so the stored profile stays clean / hand-editable.
    profiles[name] = {k: v for k, v in profile.items() if v is not None}
    config["profiles"] = profiles
    if make_active or "active_profile" not in config:
        config["active_profile"] = name
    save_global_config(config)


def _inject_env(key: str, value: Optional[str]) -> None:
    """Set ``os.environ[key] = value`` only if not already set (shell wins)."""
    if value and not os.environ.get(key):
        os.environ[key] = str(value)


def apply_global_config() -> Dict[str, Any]:
    """Project the active profile + free-form ``env`` block into ``os.environ``.

    Called once at import time from :mod:`agentica.config`, *after* ``.env`` has
    been loaded, so the precedence (highest first) is::

        shell env  >  .env  >  agentica.json

    Injection uses ``setdefault`` semantics — an already-present env var is never
    overwritten. Returns the active profile dict (possibly empty) so callers can
    reuse the resolved model config without re-reading the file.
    """
    config = load_global_config()
    if not config:
        return {}

    # 1. Free-form env block (arbitrary keys: tool API keys, tracing, etc.).
    env_block = config.get("env")
    if isinstance(env_block, dict):
        for k, v in env_block.items():
            _inject_env(str(k), v if v is None else str(v))

    # 2. Active profile -> provider-specific env vars.
    profile = get_profile(config=config)
    if profile:
        provider = profile.get("model_provider")
        api_key = profile.get("api_key")
        base_url = profile.get("base_url")
        if provider and api_key:
            _inject_env(provider_api_key_env(provider), api_key)
        # For OpenAI-compatible custom endpoints, also seed OPENAI_BASE_URL so a
        # base_url defined purely in agentica.json takes effect without a flag.
        if provider in ("openai",) and base_url:
            _inject_env(OPENAI_BASE_URL_ENV, base_url)
    return profile
