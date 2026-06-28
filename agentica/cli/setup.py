# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: First-run model provider onboarding wizard + model config resolution.

The single source of truth for SDK + CLI model configuration is
``~/.agentica/config.yaml`` (see :mod:`agentica.global_config`). It holds named
profiles, each with a **main model** and an optional **aux model**. The aux
model is the cheap/fast model used for all non-user-facing LLM work: memory
extraction, context compression, user-correction classification, goal judging,
skill upgrade, AND the ``task`` subagent tool. Omit ``aux_model`` to reuse the
main model for aux work.

``~/.agentica/.env`` is still loaded at startup (for users who maintain it by
hand for MCP tools and similar); the wizard never writes to it. Precedence
(highest first): shell env > .env > config.yaml.

There is no longer a separate ``cli_config.json`` — API keys live in the
profile's ``api_key`` field (or provider env vars). A custom OpenAI-compatible
endpoint is just a host-suffixed profile (e.g. ``openai@my-llm.local``) with its
own ``api_key``, so it can never silently reuse the real OpenAI key.
"""

import os
import sys
from typing import Dict, Optional

from prompt_toolkit import prompt as pt_prompt

from agentica.global_config import (
    DEFAULT_PROFILE_NAME,
    get_profile,
    get_profiles,
    get_active_profile_name,
    upsert_profile,
    set_active_profile,
    find_profile_for_provider,
    global_config_path,
    write_commented_template,
)

# Provider presets: provider slug -> metadata used by the wizard.
# ``env`` is the environment variable each provider factory reads for its key
# (see agentica/__init__.py). It is consulted as a *fallback* only when no key
# is stored in a profile. ``base_url`` is the default endpoint shown to the
# user. Every provider here is backed by MODEL_REGISTRY in cli/config.py.
PROVIDER_PRESETS = {
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-v4-flash",
    },
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "env": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
    },
    "zhipuai": {
        "label": "ZhipuAI (GLM)",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "env": "ZAI_API_KEY",
        "default_model": "glm-4.7-flash",
    },
    "moonshot": {
        "label": "Moonshot (Kimi)",
        "base_url": "https://api.moonshot.cn/v1",
        "env": "MOONSHOT_API_KEY",
        "default_model": "kimi-k2.5",
    },
    "ark": {
        "label": "ByteDance Volcengine Ark (Doubao)",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "env": "ARK_API_KEY",
        "default_model": "doubao-1.5-pro-32k",
    },
    "yi": {
        "label": "01.AI (Yi)",
        "base_url": "https://api.lingyiwanwu.com/v1",
        "env": "YI_API_KEY",
        "default_model": "yi-lightning",
    },
    "anthropic": {
        "label": "Anthropic (Claude)",
        "base_url": "https://api.anthropic.com",
        "env": "ANTHROPIC_API_KEY",
        "default_model": "claude-opus-4.8",
    },
}

# Order shown in the picker. "custom" is appended as the final option and maps
# to an OpenAI-compatible endpoint (provider=openai + user base_url).
_PROVIDER_ORDER = ["deepseek", "openai", "anthropic", "zhipuai", "moonshot", "ark", "yi"]

# Hardcoded fallback when nothing is configured.
DEFAULT_PROVIDER = "deepseek"
DEFAULT_MODEL = "deepseek-v4-flash"


def default_base_url(provider: str) -> Optional[str]:
    """Return the default base_url for a known provider."""
    preset = PROVIDER_PRESETS.get(provider)
    if preset is None:
        return None
    return preset["base_url"]


def default_model_name(provider: str) -> str:
    """Return the default model name for a known provider."""
    preset = PROVIDER_PRESETS.get(provider)
    if preset is None:
        return DEFAULT_MODEL
    return preset["default_model"]


def provider_env_var(provider: str) -> str:
    """Return the API-key env var name for a provider (OPENAI_API_KEY fallback).

    Used as a *fallback* lookup only when no key is stored in a profile — the
    wizard itself no longer writes to env vars.
    """
    preset = PROVIDER_PRESETS.get(provider)
    return preset["env"] if preset else "OPENAI_API_KEY"


def _is_custom_openai(provider: str, base_url: Optional[str] = None) -> bool:
    """True for a Custom OpenAI-compatible endpoint (openai + non-default base_url).

    Such an endpoint must NOT fall back to ``OPENAI_API_KEY``: that variable
    belongs to OpenAI proper, and silently reusing it for an unrelated endpoint
    is a footgun.
    """
    if provider != "openai":
        return False
    canonical = PROVIDER_PRESETS["openai"]["base_url"].rstrip("/")
    return bool(base_url) and base_url.rstrip("/") != canonical


def _profile_name_for(provider: str, base_url: Optional[str] = None) -> str:
    """Derive a stable config.yaml profile name from provider + base_url.

    Known providers map to their slug (``deepseek``, ``openai``, ...). A custom
    OpenAI-compatible endpoint gets a host-suffixed name (``openai@my-llm.local``)
    so it does not clobber the canonical ``openai`` profile.
    """
    if provider == "openai" and base_url:
        canonical = PROVIDER_PRESETS["openai"]["base_url"].rstrip("/")
        if base_url.rstrip("/") != canonical:
            from urllib.parse import urlparse

            host = urlparse(base_url).netloc or base_url.rstrip("/")
            return f"openai@{host}"
    return provider


def get_profile_api_key(provider: str, base_url: Optional[str] = None) -> Optional[str]:
    """Look up an API key stored in a config.yaml profile for this provider/base_url.

    Profiles are the key store now that cli_config.json is gone. Returns the
    ``api_key`` of the first profile whose main model matches the given
    provider (and base_url if given), else ``None``.
    """
    p = find_profile_for_provider(provider, base_url)
    return p.get("api_key") if p else None


def has_api_key(provider: str, base_url: Optional[str] = None) -> bool:
    """True if an API key is available in a profile or the process env.

    Resolution order: a config.yaml profile matching provider/base_url with an
    ``api_key`` wins; otherwise the provider env var is consulted (kept as a
    backwards-compatible fallback for users who still export keys in their
    shell or maintain ``~/.agentica/.env`` by hand).

    For a custom OpenAI-compatible endpoint we do NOT consult
    ``OPENAI_API_KEY`` as a fallback (see :func:`_is_custom_openai`).
    """
    if get_profile_api_key(provider, base_url):
        return True
    if _is_custom_openai(provider, base_url):
        return False
    env_var = provider_env_var(provider)
    if os.getenv(env_var):
        return True
    # ZhipuAI keeps a legacy fallback name.
    if provider == "zhipuai" and os.getenv("ZHIPUAI_API_KEY"):
        return True
    return False


def is_profile_complete(profile: Optional[Dict] = None) -> bool:
    """True when a profile has all four main-model fields set."""
    p = profile if profile is not None else get_profile()
    return bool(
        p.get("model_provider")
        and p.get("model_name")
        and p.get("base_url")
        and p.get("api_key")
    )


def should_onboard(provider: str, base_url: Optional[str] = None) -> bool:
    """Decide whether to run the first-launch wizard.

    Skip when the active profile is complete, or when a profile matching the
    resolved provider already has a key. Otherwise require a TTY.
    """
    if is_profile_complete():
        return False
    if has_api_key(provider, base_url):
        return False
    return sys.stdin.isatty() and sys.stdout.isatty()


def _select_provider(console) -> str:
    """Show the numbered provider picker and return the chosen slug."""
    console.print()
    console.print("  Select a model provider:", style="bold cyan")
    for idx, slug in enumerate(_PROVIDER_ORDER, start=1):
        preset = PROVIDER_PRESETS[slug]
        marker = " [dim](default)[/dim]" if slug == DEFAULT_PROVIDER else ""
        console.print(f"    {idx}. {preset['label']}{marker}")
    custom_idx = len(_PROVIDER_ORDER) + 1
    console.print(f"    {custom_idx}. Custom (any OpenAI-compatible endpoint)")
    console.print()

    while True:
        raw = pt_prompt(f"  Provider [1-{custom_idx}, default 1]: ").strip()
        if not raw:
            return _PROVIDER_ORDER[0]
        if raw.isdigit():
            n = int(raw)
            if 1 <= n <= len(_PROVIDER_ORDER):
                return _PROVIDER_ORDER[n - 1]
            if n == custom_idx:
                return "custom"
        # Allow typing the slug directly too.
        slug = raw.lower()
        if slug in PROVIDER_PRESETS or slug == "custom":
            return slug
        console.print(f"  [red]Invalid choice: {raw}[/red]")


_REASONING_EFFORT_CHOICES = ("low", "medium", "high", "max")


def _prompt_int(label: str) -> Optional[int]:
    """Prompt for an optional positive integer (blank = skip)."""
    while True:
        raw = pt_prompt(label).strip()
        if not raw:
            return None
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
        # Re-prompt on invalid input; the label already says "blank to skip".


def _prompt_float(label: str) -> Optional[float]:
    """Prompt for an optional float (blank = skip)."""
    while True:
        raw = pt_prompt(label).strip()
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            pass


def _prompt_advanced_params(console, provider: str) -> Dict:
    """Optionally collect advanced model tuning params during onboarding.

    Returns a dict that may contain ``reasoning_effort``, ``max_tokens``,
    ``context_window``, ``temperature`` and ``top_p`` (only the keys the user
    set). Keys left blank are omitted so the model factory keeps its defaults.
    """
    console.print()
    answer = pt_prompt("  Configure advanced model params (thinking depth, limits)? [y/N]: ").strip().lower()
    if answer not in ("y", "yes"):
        return {}

    params: Dict = {}
    console.print("  [dim]Press Enter to skip any field.[/dim]")

    # Thinking depth / reasoning effort.
    effort = pt_prompt(f"  Reasoning effort {list(_REASONING_EFFORT_CHOICES)} (blank to skip): ").strip().lower()
    if effort in _REASONING_EFFORT_CHOICES:
        params["reasoning_effort"] = effort
    elif effort:
        console.print(f"  [yellow]Ignored invalid reasoning effort: {effort}[/yellow]")

    # Output limit.
    max_tokens = _prompt_int("  Max output tokens (output limit, blank to skip): ")
    if max_tokens is not None:
        params["max_tokens"] = max_tokens

    # Context limit (overrides the catalog auto-detected value).
    context_window = _prompt_int("  Context window (context limit, blank to skip): ")
    if context_window is not None:
        params["context_window"] = context_window

    # Sampling.
    temperature = _prompt_float("  Temperature (blank to skip): ")
    if temperature is not None:
        params["temperature"] = temperature
    top_p = _prompt_float("  Top-p (blank to skip): ")
    if top_p is not None:
        params["top_p"] = top_p

    return params


def _prompt_aux_model(console, main_provider: str) -> Dict:
    """Optionally configure the aux model (background calls + `task` subagent).

    Fully skippable: the whole section is gated by a ``[y/N]`` question. When
    the user declines, returns ``{}``. Otherwise prompts for provider, base_url,
    api_key and model_name (with sensible defaults / blank-to-skip) and returns
    an ``aux_model`` block dict suitable for storing in a config.yaml profile.
    The key is persisted into the profile (no separate key-slot file).
    """
    console.print()
    console.print("  Aux model (background tasks + `task` subagent) - optional", style="bold cyan")
    console.print("  [dim]A cheaper/faster model here saves cost on memory extraction,[/dim]")
    console.print("  [dim]context compression, and delegated subtasks.[/dim]")
    answer = pt_prompt("  Configure a separate aux model? [y/N]: ").strip().lower()
    if answer not in ("y", "yes"):
        return {}

    provider_choice = _select_provider(console)
    is_custom = provider_choice == "custom"
    if is_custom:
        provider = "openai"
        console.print()
        console.print("  Custom OpenAI-compatible endpoint", style="bold cyan")
        base_url = pt_prompt("  Base URL: ").strip()
        while not base_url:
            console.print("  [red]Base URL is required for a custom endpoint.[/red]")
            base_url = pt_prompt("  Base URL: ").strip()
        default_model = ""
    else:
        provider = provider_choice
        preset = PROVIDER_PRESETS[provider]
        default_base = preset["base_url"]
        console.print()
        console.print(f"  {preset['label']} selected", style="bold cyan")
        base_url = pt_prompt(f"  Base URL [{default_base}]: ").strip() or default_base
        default_model = preset["default_model"]

    # API key: prefer a previously stored profile key, else the provider env
    # var (so users with shell-exported keys don't get re-prompted). Custom
    # endpoints never look at OPENAI_API_KEY.
    existing_key = get_profile_api_key(provider, base_url)
    if not existing_key and not is_custom:
        existing_key = os.getenv(provider_env_var(provider))
    key_hint = " (press Enter to keep existing)" if existing_key else " (blank to skip)"
    entered_key = pt_prompt(f"  API key{key_hint}: ", is_password=True).strip()
    key = entered_key or existing_key

    model_prompt = f"  Model name [{default_model}]: " if default_model else "  Model name: "
    model_name = pt_prompt(model_prompt).strip() or default_model
    while not model_name:
        console.print("  [red]Model name is required (or answer N to skip the aux model).[/red]")
        model_name = pt_prompt("  Model name: ").strip()

    block: Dict = {
        "model_provider": provider,
        "model_name": model_name,
        "base_url": base_url,
    }
    if key:
        block["api_key"] = key
    if provider == main_provider:
        console.print("  [dim]Same provider as main model — you could also just set model_name.[/dim]")
    return block


def run_onboarding(console) -> Dict:
    """Interactive first-run wizard.

    Persists the main model (provider/model/base_url/api_key + optional tuning)
    and the optional aux model to ``~/.agentica/config.yaml`` as a named profile,
    made active so the SDK picks up the api_key via env injection on next run.
    Returns ``{model_provider, model_name, base_url, api_key, ...aux fields}``;
    the api_key field may be the freshly entered value, a previously stored one,
    or ``None`` if the user declined to enter one.
    """
    console.print()
    console.print("=" * min(getattr(console, "width", 80), 80), style="bright_cyan")
    console.print("  Welcome to Agentica CLI - let's set up your model provider", style="bold bright_green")
    console.print("  (You can change this later with /model or by running: agentica setup)", style="dim")
    console.print("=" * min(getattr(console, "width", 80), 80), style="bright_cyan")

    provider_choice = _select_provider(console)

    is_custom = provider_choice == "custom"
    if is_custom:
        provider = "openai"
        console.print()
        console.print("  Custom OpenAI-compatible endpoint", style="bold cyan")
        base_url = pt_prompt("  Base URL: ").strip()
        while not base_url:
            console.print("  [red]Base URL is required for a custom endpoint.[/red]")
            base_url = pt_prompt("  Base URL: ").strip()
        default_model = ""
    else:
        provider = provider_choice
        preset = PROVIDER_PRESETS[provider]
        default_base = preset["base_url"]
        console.print()
        console.print(f"  {preset['label']} selected", style="bold cyan")
        base_url = pt_prompt(f"  Base URL [{default_base}]: ").strip() or default_base
        default_model = preset["default_model"]

    # API key - masked input. Prefer a previously stored profile key, then fall
    # back to the provider env var. Custom endpoints never look at
    # OPENAI_API_KEY (see _is_custom_openai).
    existing_key = get_profile_api_key(provider, base_url)
    if not existing_key and not is_custom:
        existing_key = os.getenv(provider_env_var(provider))
    key_hint = " (press Enter to keep existing)" if existing_key else ""
    entered_key = pt_prompt(f"  API key{key_hint}: ", is_password=True).strip()
    if entered_key:
        console.print("  [green]API key will be saved to ~/.agentica/config.yaml[/green]")
    elif existing_key:
        console.print("  [dim]Keeping existing API key.[/dim]")
    else:
        console.print("  [yellow]No API key entered - re-run `agentica setup` or pass --api_key later.[/yellow]")

    # Model name.
    model_prompt = f"  Model name [{default_model}]: " if default_model else "  Model name: "
    model_name = pt_prompt(model_prompt).strip() or default_model
    while not model_name:
        console.print("  [red]Model name is required.[/red]")
        model_name = pt_prompt("  Model name: ").strip()

    # Optional advanced tuning (thinking depth, output/context limits, sampling).
    # Skipped by default to keep first-run quick; users can also hand-edit
    # ~/.agentica/config.yaml afterwards.
    advanced = _prompt_advanced_params(console, provider)

    # Optional aux model (background calls + `task` subagent) — fully skippable.
    aux_block = _prompt_aux_model(console, provider)

    # Persist to config.yaml as a named profile. On first run (no file yet)
    # write a fully-commented template first so the user gets a readable base;
    # upsert then round-trips the file, preserving those comments.
    if not os.path.exists(global_config_path()):
        write_commented_template()
    resolved_key = entered_key or existing_key
    profile_name = _profile_name_for(provider, base_url)
    profile_data = {
        "model_provider": provider,
        "model_name": model_name,
        "base_url": base_url,
        "api_key": resolved_key,
    }
    profile_data.update(advanced)  # None values are dropped by upsert_profile
    if aux_block:
        profile_data["aux_model"] = aux_block
    upsert_profile(profile_name, profile_data, make_active=True)

    console.print()
    console.print(f"  [bright_green]Configured: {provider}/{model_name}[/bright_green]")
    console.print(f"  [dim]Endpoint: {base_url}[/dim]")
    console.print(f"  [dim]Saved as profile '{profile_name}' in ~/.agentica/config.yaml[/dim]")
    if aux_block:
        console.print(
            f"  [dim]Aux model: {aux_block['model_provider']}/{aux_block['model_name']}[/dim]"
        )
    console.print()

    result = {
        "model_provider": provider,
        "model_name": model_name,
        "base_url": base_url,
        "api_key": resolved_key,
        **advanced,
    }
    result.update(_aux_resolution(aux_block, provider, base_url, resolved_key))
    return result


def _aux_resolution(
    aux_block: Dict, main_provider: str, main_base_url: Optional[str], main_api_key: Optional[str]
) -> Dict:
    """Resolve the aux_model block into flat aux_model_* fields (no CLI flags).

    Same provider as main -> inherit main base_url/api_key for omitted fields.
    Different provider -> base_url from that provider's preset, api_key from the
    block (or a matching profile), never the main key.
    """
    aux_name = aux_block.get("model_name")
    if not aux_name:
        return {"aux_model_provider": None, "aux_model_name": None,
                "aux_base_url": None, "aux_api_key": None}
    aux_provider = aux_block.get("model_provider") or main_provider
    aux_base = aux_block.get("base_url")
    aux_key = aux_block.get("api_key")
    if not aux_base:
        aux_base = main_base_url if aux_provider == main_provider else default_base_url(aux_provider)
    if not aux_key:
        if aux_provider == main_provider:
            aux_key = main_api_key
        else:
            aux_key = get_profile_api_key(aux_provider, aux_base)
    return {
        "aux_model_provider": aux_provider,
        "aux_model_name": aux_name,
        "aux_base_url": aux_base,
        "aux_api_key": aux_key,
    }


def resolve_model_config(args, console=None) -> Dict:
    """Resolve provider/model/base_url/api_key with CLI args > profile > env.

    Triggers the first-run wizard when appropriate. Returns a dict with the
    main ``model_provider``/``model_name``/``base_url``/``api_key``, optional
    model-tuning keys, and the optional aux model keys ``aux_model_provider`` /
    ``aux_model_name`` / ``aux_base_url`` / ``aux_api_key`` (all None when no
    aux model is configured, in which case aux work reuses the main model).

    Resolution precedence (highest first):
        1. CLI flags (``--model_provider``/``--model_name``/... and
           ``--aux_model_*``)
        2. ``~/.agentica/config.yaml`` active profile (new single source;
           includes the optional ``aux_model`` sub-block)
        3. provider preset defaults

    The ``api_key`` is the key stored in a profile for the resolved
    provider/base_url (or freshly entered during onboarding). ``None`` means the
    model factory should fall back to its env-var lookup. The CLI ``--api_key``
    flag, when provided, still wins at the call site (see ``cli/main.py``). The
    aux model follows the same precedence via ``--aux_model_*`` flags over the
    profile ``aux_model`` block; a different provider never reuses the main
    model's api_key or base_url.
    """
    # Active profile in config.yaml is the single source of truth.
    active_profile = get_profile()
    profile_provider = active_profile.get("model_provider")

    provider = args.model_provider or profile_provider or DEFAULT_PROVIDER
    use_profile = provider == profile_provider

    model_name = (
        args.model_name
        or (active_profile.get("model_name") if use_profile else None)
        or default_model_name(provider)
    )
    base_url = (
        args.base_url
        or (active_profile.get("base_url") if use_profile else None)
        or default_base_url(provider)
    )

    # Only consider onboarding when the user didn't pin a provider via flags.
    resolved_key: Optional[str] = None
    if args.model_provider is None and console is not None and should_onboard(provider, base_url):
        result = run_onboarding(console)
        provider = args.model_provider or result["model_provider"]
        model_name = args.model_name or result["model_name"]
        base_url = args.base_url or result["base_url"]
        resolved_key = result.get("api_key")
        # Onboarding may have written/activated a new profile.
        active_profile = get_profile()
        use_profile = provider == active_profile.get("model_provider")

    # API key resolution: active profile -> any profile matching provider/base_url.
    if resolved_key is None and use_profile:
        resolved_key = active_profile.get("api_key")
    if resolved_key is None:
        resolved_key = get_profile_api_key(provider, base_url)

    # Model tuning params come from the active profile only (no preset
    # defaults): reasoning_effort / max_tokens / context_window / temperature /
    # top_p. They are None unless the user set them, so the model factory keeps
    # its own defaults. CLI flags still override these at the call site.
    profile_params = active_profile if use_profile else {}

    # Optional aux model. Resolution precedence (highest first):
    #   1. CLI flags (--aux_model_provider / --aux_model_name / ...)
    #   2. profile ``aux_model`` block in config.yaml
    #   3. main model (when same provider) or provider preset / matching profile
    # When no aux_model_name is resolved, all aux fields are None and aux work
    # reuses the main model (see cli/config.py::_build_sibling_model).
    aux_block = (active_profile.get("aux_model") or {}) if use_profile else {}
    aux_provider = args.aux_model_provider or aux_block.get("model_provider") or provider
    aux_name = args.aux_model_name or aux_block.get("model_name")
    aux_base = args.aux_base_url or aux_block.get("base_url")
    aux_key = args.aux_api_key or aux_block.get("api_key")
    if aux_name:
        if not aux_base:
            # Same provider as main -> reuse main endpoint; otherwise use the
            # aux provider's preset default (a different provider's endpoint is
            # never the main model's base_url).
            aux_base = base_url if aux_provider == provider else default_base_url(aux_provider)
        if not aux_key:
            if aux_provider == provider:
                aux_key = resolved_key
            else:
                # Different provider: never reuse the main key. Fall back to a
                # key stored in a matching profile, else None so the model
                # factory tries the provider env var.
                aux_key = get_profile_api_key(aux_provider, aux_base)
    else:
        aux_provider = aux_base = aux_key = None

    return {
        "model_provider": provider,
        "model_name": model_name,
        "base_url": base_url,
        "api_key": resolved_key,
        "max_tokens": profile_params.get("max_tokens"),
        "temperature": profile_params.get("temperature"),
        "reasoning_effort": profile_params.get("reasoning_effort"),
        "top_p": profile_params.get("top_p"),
        "context_window": profile_params.get("context_window"),
        "aux_model_provider": aux_provider,
        "aux_model_name": aux_name,
        "aux_base_url": aux_base,
        "aux_api_key": aux_key,
    }
