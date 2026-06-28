# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Unified, hand-editable configuration shared by the SDK and the CLI.

A single hand-editable file — ``~/.agentica/config.yaml`` — is the source of
truth for both the SDK and the CLI. It is **YAML** (not JSON) so users can add
comments and read it easily. Named profiles (e.g. ``default``, ``gpt``,
``local``) let a user keep several provider/model setups and switch the active
one at runtime (``/model profile <name>``).

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
  shell env  >  .env  >  config.yaml.
* **Hand-editable + comments preserved.** The file is plain YAML with a stable
  schema; ``agentica setup`` writes it, but users may edit it directly and add
  comments. Programmatic writes (onboarding, ``/model profile``) use
  ``ruamel.yaml`` round-tripping so user comments are NOT lost.

Two model concepts
------------------
The SDK has two model roles: the **main model** (user-facing turns) and the
**auxiliary model** (background LLM calls: memory extraction, context
compression, user-correction classification, goal judging, skill upgrade — and
the ``task`` subagent tool). The CLI config follows this: each profile has the
main model fields at the top, plus an optional ``aux_model`` sub-block for a
cheaper/faster model. When ``aux_model`` is omitted, the aux role reuses the
main model.

Schema (``~/.agentica/config.yaml``)::

    # Active profile name; switch at runtime with: /model profile <name>
    active_profile: default

    profiles:
      default:
        # --- main model (user-facing turns) ---
        model_provider: deepseek
        model_name: deepseek-v4-flash
        base_url: https://api.deepseek.com
        api_key: sk-...

        # optional model tuning (omit to use model/factory defaults)
        # reasoning_effort: max      # low|medium|high|max (OpenAI/DeepSeek)
        # max_tokens: 8192           # output token limit
        # context_window: 1000000    # context limit; overrides catalog value
        # temperature: 0.7
        # top_p: 0.95

        # --- optional aux model (background calls + `task` subagent tool) ---
        # Omit the whole block to reuse the main model for aux work. When aux
        # shares the main provider, any field may be omitted and inherits from
        # the main model. When it uses a different provider, base_url defaults
        # to that provider's preset and api_key must be set here or available
        # via the provider env var.
        # aux_model:
        #   model_provider: zhipuai
        #   model_name: glm-4.7-flash
        #   base_url: https://open.bigmodel.cn/api/paas/v4
        #   api_key: sk-...

    # Free-form env block: arbitrary keys injected into os.environ (tool API
    # keys, tracing, etc.). Shell/env-file values still win over these.
    env:
      SERPER_API_KEY: "..."

The file is written with ``chmod 0o600`` because profiles hold secrets.
"""

import os
from typing import Dict, Optional, Any, List

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

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
# lets a custom endpoint defined purely in config.yaml work without flags.
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"

DEFAULT_PROFILE_NAME = "default"

_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.indent(mapping=2, sequence=4, offset=2)


def global_config_path() -> str:
    """Return the path to ``config.yaml`` (honours ``AGENTICA_HOME``).

    Resolved lazily (not import-time-frozen) so tests can point
    ``AGENTICA_HOME`` elsewhere.
    """
    home = os.path.expanduser(os.getenv("AGENTICA_HOME", "~/.agentica"))
    return os.path.join(home, "config.yaml")


def provider_api_key_env(provider: str) -> str:
    """Return the env var a provider reads for its API key (OPENAI fallback)."""
    return PROVIDER_API_KEY_ENV.get(provider, "OPENAI_API_KEY")


def _load_commented() -> CommentedMap:
    """Load config.yaml as a ruamel CommentedMap (preserves comments).

    Returns an empty CommentedMap when the file is missing or invalid.
    """
    path = global_config_path()
    if not os.path.exists(path):
        return CommentedMap()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = _yaml.load(f)
    except Exception:
        return CommentedMap()
    return data if isinstance(data, CommentedMap) else CommentedMap()


def _to_plain(obj: Any) -> Any:
    """Recursively convert ruamel containers to plain dict/list for read API."""
    if isinstance(obj, CommentedMap):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    return obj


def _to_commented(obj: Any) -> Any:
    """Recursively convert plain dict/list to ruamel containers."""
    if isinstance(obj, dict):
        cm = CommentedMap()
        for k, v in obj.items():
            cm[k] = _to_commented(v)
        return cm
    if isinstance(obj, list):
        return [_to_commented(v) for v in obj]
    return obj


def load_global_config() -> Dict[str, Any]:
    """Load ``config.yaml`` as a plain dict, or an empty dict if missing/invalid."""
    return _to_plain(_load_commented())


def _save_commented(data: CommentedMap) -> None:
    """Persist a CommentedMap to config.yaml with ``0o600`` perms (holds secrets)."""
    path = global_config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        _yaml.dump(data, f)
    try:
        os.chmod(path, 0o600)
    except OSError:
        # Non-fatal on platforms where chmod has no effect; contents are saved.
        pass


def save_global_config(config: Dict[str, Any]) -> None:
    """Persist a plain dict to config.yaml with ``0o600`` perms.

    Note: this performs a fresh dump from a plain dict, so any comments that
    existed in the file are lost. For comment-preserving writes use
    :func:`upsert_profile` / :func:`set_active_profile`, which round-trip the
    existing file. This function is kept for tests and ad-hoc callers that
    build a config dict from scratch.
    """
    _save_commented(_to_commented(config))


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
    """Mark ``name`` as the active profile. Returns False if it does not exist.

    Round-trips the YAML file so user comments are preserved.
    """
    data = _load_commented()
    profiles = data.get("profiles")
    if not isinstance(profiles, CommentedMap) or name not in profiles:
        return False
    data["active_profile"] = name
    _save_commented(data)
    return True


def upsert_profile(name: str, profile: Dict[str, Any], make_active: bool = True) -> None:
    """Create or update a named profile and optionally make it active.

    Round-trips the YAML file so user comments are preserved. ``None`` values
    are dropped so the stored profile stays clean / hand-editable.
    """
    data = _load_commented()
    profiles = data.get("profiles")
    if not isinstance(profiles, CommentedMap):
        profiles = CommentedMap()
        data["profiles"] = profiles
    profiles[name] = CommentedMap(
        (k, _to_commented(v)) for k, v in profile.items() if v is not None
    )
    if make_active or "active_profile" not in data:
        data["active_profile"] = name
    _save_commented(data)


def find_profile_for_provider(provider: str, base_url: Optional[str] = None) -> Dict[str, Any]:
    """Return the first profile matching ``provider`` (and ``base_url`` if given).

    Profiles are the key store now that cli_config.json is gone, so this is how
    a previously-saved key for a different provider is looked up (e.g. when the
    user switches providers in ``/model``, or when the aux model uses a
    different provider whose key lives in another profile).
    """
    for p in get_profiles().values():
        if p.get("model_provider") == provider:
            if base_url is None:
                return p
            p_base = (p.get("base_url") or "").rstrip("/")
            if p_base == base_url.rstrip("/"):
                return p
    return {}


def _inject_env(key: str, value: Optional[str]) -> None:
    """Set ``os.environ[key] = value`` only if not already set (shell wins)."""
    if value and not os.environ.get(key):
        os.environ[key] = str(value)


def apply_global_config() -> Dict[str, Any]:
    """Project the active profile + free-form ``env`` block into ``os.environ``.

    Called once at import time from :mod:`agentica.config`, *after* ``.env`` has
    been loaded, so the precedence (highest first) is::

        shell env  >  .env  >  config.yaml

    Injection uses ``setdefault`` semantics — an already-present env var is
    never overwritten. Returns the active profile dict (possibly empty) so
    callers can reuse the resolved model config without re-reading the file.
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
        # base_url defined purely in config.yaml takes effect without a flag.
        if provider in ("openai",) and base_url:
            _inject_env(OPENAI_BASE_URL_ENV, base_url)
    return profile


def write_commented_template(path: Optional[str] = None) -> None:
    """Write a fresh, fully-commented config.yaml template (first-run onboarding).

    Overwrites ``path`` (defaults to :func:`global_config_path`) with the
    documented schema and ``0o600`` perms. Intended only when no config.yaml
    exists yet, so we never clobber a user's hand-edited file.
    """
    target = path or global_config_path()
    os.makedirs(os.path.dirname(target), exist_ok=True)
    template = _CONFIG_TEMPLATE
    with open(target, "w", encoding="utf-8") as f:
        f.write(template)
    try:
        os.chmod(target, 0o600)
    except OSError:
        pass


_CONFIG_TEMPLATE = """\
# Agentica configuration — shared by the SDK and the CLI.
# Hand-edit freely; comments are preserved on programmatic writes.
# Switch the active profile at runtime with: /model profile <name>

active_profile: default

profiles:
  default:
    # --- main model (user-facing turns) ---
    model_provider: deepseek
    model_name: deepseek-v4-flash
    base_url: https://api.deepseek.com
    api_key: REPLACE_ME

    # optional model tuning (omit to use model/factory defaults)
    # reasoning_effort: max      # low|medium|high|max (OpenAI/DeepSeek)
    # max_tokens: 8192           # output token limit
    # context_window: 1000000    # context limit; overrides catalog value
    # temperature: 0.7
    # top_p: 0.95

    # --- optional aux model (background calls + `task` subagent tool) ---
    # A cheaper/faster model here saves cost on memory extraction, context
    # compression, and delegated subtasks. Omit to reuse the main model.
    # aux_model:
    #   model_provider: zhipuai
    #   model_name: glm-4.7-flash
    #   base_url: https://open.bigmodel.cn/api/paas/v4
    #   api_key: sk-...

# Free-form env block: arbitrary keys injected into os.environ.
# Shell / .env values still win over these (setdefault semantics).
env: {}
"""
