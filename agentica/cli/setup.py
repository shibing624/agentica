# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: First-run model provider onboarding wizard for the CLI.

On first launch (missing saved CLI config or missing API key) the user is
walked through picking a model provider and filling in base_url, api_key and
model_name. The non-secret choices are persisted to
``~/.agentica/cli_config.json``; the API key is written to the existing
``~/.agentica/.env`` (AGENTICA_DOTENV_PATH), so secrets never live in JSON.
"""
import json
import os
import sys
from typing import Dict, Optional

from prompt_toolkit import prompt as pt_prompt

from agentica.config import AGENTICA_HOME, AGENTICA_DOTENV_PATH

CLI_CONFIG_PATH = os.path.join(AGENTICA_HOME, "cli_config.json")

# Provider presets: provider slug -> metadata used by the wizard.
# ``env`` is the environment variable each provider factory reads for its key
# (see agentica/__init__.py). ``base_url`` is the default endpoint shown to the
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


def load_cli_config() -> Dict:
    """Load the saved CLI config, or an empty dict if none/invalid."""
    if not os.path.exists(CLI_CONFIG_PATH):
        return {}
    try:
        with open(CLI_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_cli_config(config: Dict) -> None:
    """Persist the CLI config (non-secret fields only) to disk."""
    os.makedirs(os.path.dirname(CLI_CONFIG_PATH), exist_ok=True)
    with open(CLI_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def is_onboarded() -> bool:
    """True once the user has completed (or explicitly skipped) onboarding."""
    return bool(load_cli_config().get("onboarded"))


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


def is_cli_config_complete(config: Optional[Dict] = None) -> bool:
    """Return True when non-secret CLI model config is complete."""
    data = load_cli_config() if config is None else config
    return bool(
        data.get("onboarded")
        and data.get("model_provider")
        and data.get("model_name")
        and data.get("base_url")
    )


def provider_env_var(provider: str) -> str:
    """Return the API-key env var name for a provider (OPENAI_API_KEY fallback)."""
    preset = PROVIDER_PRESETS.get(provider)
    return preset["env"] if preset else "OPENAI_API_KEY"


def has_api_key(provider: str) -> bool:
    """True if an API key is already available in the environment."""
    env_var = provider_env_var(provider)
    if os.getenv(env_var):
        return True
    # ZhipuAI keeps a legacy fallback name.
    if provider == "zhipuai" and os.getenv("ZHIPUAI_API_KEY"):
        return True
    return False


def save_api_key_to_env(env_var: str, value: str) -> None:
    """Write/update a key in ~/.agentica/.env and the current process env."""
    os.makedirs(os.path.dirname(AGENTICA_DOTENV_PATH), exist_ok=True)

    lines = []
    if os.path.exists(AGENTICA_DOTENV_PATH):
        with open(AGENTICA_DOTENV_PATH, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

    prefix = f"{env_var}="
    replaced = False
    for i, line in enumerate(lines):
        if line.strip().startswith(prefix):
            lines[i] = f"{env_var}={value}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{env_var}={value}")

    with open(AGENTICA_DOTENV_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Make the key visible to the running process immediately.
    os.environ[env_var] = value


def should_onboard(provider: str) -> bool:
    """Decide whether to run the first-launch wizard.

    Run the wizard whenever either side of the CLI setup is incomplete:
    non-secret config (provider/model/base_url) or the provider API key.
    A complete config plus an API key skips onboarding.
    """
    if is_cli_config_complete() and has_api_key(provider):
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


def run_onboarding(console) -> Dict:
    """Interactive first-run wizard. Returns resolved {provider, model, base_url}.

    Persists the non-secret choice to cli_config.json and the API key to .env.
    """
    console.print()
    console.print("=" * min(getattr(console, "width", 80), 80), style="bright_cyan")
    console.print("  Welcome to Agentica CLI - let's set up your model provider", style="bold bright_green")
    console.print("  (You can change this later with /model or by running: agentica setup)", style="dim")
    console.print("=" * min(getattr(console, "width", 80), 80), style="bright_cyan")

    provider_choice = _select_provider(console)

    if provider_choice == "custom":
        provider = "openai"
        console.print()
        console.print("  Custom OpenAI-compatible endpoint", style="bold cyan")
        base_url = pt_prompt("  Base URL: ").strip()
        while not base_url:
            console.print("  [red]Base URL is required for a custom endpoint.[/red]")
            base_url = pt_prompt("  Base URL: ").strip()
        env_var = "OPENAI_API_KEY"
        default_model = ""
    else:
        provider = provider_choice
        preset = PROVIDER_PRESETS[provider]
        default_base = preset["base_url"]
        console.print()
        console.print(f"  {preset['label']} selected", style="bold cyan")
        base_url = pt_prompt(f"  Base URL [{default_base}]: ").strip() or default_base
        env_var = preset["env"]
        default_model = preset["default_model"]

    # API key — masked input.
    existing_key = os.getenv(env_var)
    key_hint = " (press Enter to keep existing)" if existing_key else ""
    api_key = pt_prompt(f"  API key{key_hint}: ", is_password=True).strip()
    if api_key:
        save_api_key_to_env(env_var, api_key)
        console.print("  [green]API key saved to ~/.agentica/.env[/green]")
    elif existing_key:
        console.print("  [dim]Keeping existing API key.[/dim]")
    else:
        console.print("  [yellow]No API key entered - set it later in ~/.agentica/.env[/yellow]")

    # Model name.
    model_prompt = f"  Model name [{default_model}]: " if default_model else "  Model name: "
    model_name = pt_prompt(model_prompt).strip() or default_model
    while not model_name:
        console.print("  [red]Model name is required.[/red]")
        model_name = pt_prompt("  Model name: ").strip()

    config = load_cli_config()
    config.update({
        "onboarded": True,
        "model_provider": provider,
        "model_name": model_name,
        "base_url": base_url,
    })
    save_cli_config(config)

    console.print()
    console.print(f"  [bright_green]Configured: {provider}/{model_name}[/bright_green]")
    console.print(f"  [dim]Endpoint: {base_url}[/dim]")
    console.print()

    return {"model_provider": provider, "model_name": model_name, "base_url": base_url}


def resolve_model_config(args, console=None) -> Dict:
    """Resolve provider/model/base_url with CLI args > saved config > defaults.

    Triggers the first-run wizard when appropriate. Returns a dict with
    ``model_provider``, ``model_name`` and ``base_url`` keys.
    """
    saved = load_cli_config()
    saved_provider = saved.get("model_provider")

    provider = args.model_provider or saved_provider or DEFAULT_PROVIDER
    use_saved_provider_config = provider == saved_provider
    model_name = (
        args.model_name
        or (saved.get("model_name") if use_saved_provider_config else None)
        or default_model_name(provider)
    )
    base_url = (
        args.base_url
        or (saved.get("base_url") if use_saved_provider_config else None)
        or default_base_url(provider)
    )

    # Only consider onboarding when the user didn't pin a provider via flags.
    if args.model_provider is None and console is not None and should_onboard(provider):
        result = run_onboarding(console)
        provider = args.model_provider or result["model_provider"]
        model_name = args.model_name or result["model_name"]
        base_url = args.base_url or result["base_url"]

    return {"model_provider": provider, "model_name": model_name, "base_url": base_url}
