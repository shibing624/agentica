# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: CLI self-management primitives.

Lets the agentica CLI (and the agent itself, via tools) inspect and modify its
own runtime: self-upgrade the pip package, read/edit the unified config files
(``~/.agentica/config.yaml`` and ``~/.agentica/.env``). These are the building
blocks shared by the slash commands (human-facing) and the agent tools
(conversation-facing) so both paths run identical logic.
"""

import json
import os
import re
import subprocess
import sys
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from agentica.version import __version__
from agentica.config import AGENTICA_DOTENV_PATH
from agentica.global_config import (
    load_global_config,
    get_profile,
    get_active_profile_name,
    upsert_profile,
    global_config_path,
)

# Keys whose values must never be echoed back in clear text.
_SECRET_KEY_PATTERN = re.compile(r"(api_key|token|secret|password|passwd)", re.IGNORECASE)


def mask_secret(key: str, value: Any) -> Any:
    """Mask a value if its key looks secret. Returns value unchanged otherwise."""
    if value is None:
        return value
    if _SECRET_KEY_PATTERN.search(str(key)):
        s = str(value)
        if len(s) <= 8:
            return "****"
        return f"{s[:4]}...{s[-4:]}"
    return value


# ==================== Self-upgrade ====================

def get_current_version() -> str:
    """Return the installed agentica version string."""
    return __version__


def get_latest_version(timeout: float = 5.0) -> Optional[str]:
    """Query PyPI for the latest published agentica version.

    Returns the version string, or None if the lookup fails (offline, etc.).
    The failure is intentionally soft — the caller decides how to surface it.
    """
    url = "https://pypi.org/pypi/agentica/json"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("info", {}).get("version")
    except Exception:
        return None


def _version_tuple(v: str) -> Tuple[int, ...]:
    """Best-effort numeric version parse for comparison (e.g. '1.4.6' -> (1,4,6))."""
    parts = []
    for chunk in re.split(r"[._-]", v):
        m = re.match(r"\d+", chunk)
        parts.append(int(m.group()) if m else 0)
    return tuple(parts)


def is_upgrade_available(current: str, latest: Optional[str]) -> bool:
    """True if ``latest`` is strictly newer than ``current``."""
    if not latest:
        return False
    return _version_tuple(latest) > _version_tuple(current)


def run_pip_upgrade(target: str = "agentica", pre: bool = False) -> Tuple[int, str]:
    """Run ``pip install -U`` for the target package in the current interpreter.

    Returns ``(exit_code, combined_output)``. Output (stdout+stderr) is captured
    and returned verbatim so callers can surface the real pip signal rather than
    swallowing failures.
    """
    cmd = [sys.executable, "-m", "pip", "install", "-U", target]
    if pre:
        cmd.append("--pre")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, output


# ==================== config.yaml editing ====================

def config_file_path() -> str:
    """Absolute path of the unified config.yaml."""
    return global_config_path()


def read_config_summary(reveal_secrets: bool = False) -> Dict[str, Any]:
    """Return a masked snapshot of the global config for display.

    Secrets in profiles and the env block are masked unless ``reveal_secrets``.
    """
    config = load_global_config()
    if not config:
        return {}
    snapshot: Dict[str, Any] = {}
    snapshot["active_profile"] = get_active_profile_name(config)
    profiles_out: Dict[str, Any] = {}
    for name, profile in (config.get("profiles") or {}).items():
        if not isinstance(profile, dict):
            continue
        profiles_out[name] = {
            k: (v if reveal_secrets else mask_secret(k, v))
            for k, v in profile.items()
        }
    snapshot["profiles"] = profiles_out
    env_block = config.get("env")
    if isinstance(env_block, dict):
        snapshot["env"] = {
            k: (v if reveal_secrets else mask_secret(k, v))
            for k, v in env_block.items()
        }
    return snapshot


# Profile fields the agent is allowed to tune. api_key is editable too but is
# masked on read; deleting required identity fields is blocked.
_EDITABLE_PROFILE_FIELDS = {
    "model_provider", "model_name", "base_url", "api_key",
    "max_tokens", "temperature", "reasoning_effort", "top_p", "context_window",
}

# Values that are numeric on the model side; coerced from string input.
_NUMERIC_FIELDS = {"max_tokens", "context_window"}
_FLOAT_FIELDS = {"temperature", "top_p"}


def _coerce_profile_value(field: str, value: str) -> Any:
    """Coerce a string CLI/tool input into the right type for a profile field."""
    if value == "" or value.lower() in ("none", "null"):
        return None
    if field in _NUMERIC_FIELDS:
        return int(value)
    if field in _FLOAT_FIELDS:
        return float(value)
    return value


def set_profile_field(
    field: str,
    value: str,
    profile_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Set a single field on a config.yaml profile (comment-preserving write).

    Returns the updated (masked) profile dict. Raises ValueError on an
    unknown/forbidden field so the caller surfaces a clear error.
    """
    if field not in _EDITABLE_PROFILE_FIELDS:
        raise ValueError(
            f"Field '{field}' is not editable. Allowed: "
            f"{', '.join(sorted(_EDITABLE_PROFILE_FIELDS))}"
        )
    name = profile_name or get_active_profile_name()
    profile = dict(get_profile(name))
    coerced = _coerce_profile_value(field, value)
    if coerced is None:
        profile.pop(field, None)
    else:
        profile[field] = coerced
    upsert_profile(name, profile, make_active=False)
    return {k: mask_secret(k, v) for k, v in get_profile(name).items()}


# ==================== .env editing ====================

def dotenv_path() -> str:
    """Absolute path of the global .env file."""
    return AGENTICA_DOTENV_PATH


def read_dotenv(reveal_secrets: bool = False) -> Dict[str, str]:
    """Parse the global .env into a dict, masking secret-looking values."""
    path = dotenv_path()
    result: Dict[str, str] = {}
    if not os.path.exists(path):
        return result
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            result[key] = val if reveal_secrets else mask_secret(key, val)
    return result


def set_dotenv_var(key: str, value: Optional[str]) -> None:
    """Upsert (or delete, if ``value`` is None) a key in the global .env file.

    Preserves all other lines/comments. Creates the file (0o600) if missing.
    Also updates ``os.environ`` so the change is live in-process.
    """
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
        raise ValueError(f"Invalid env var name: {key!r}")
    path = dotenv_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines: List[str] = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    new_line = None if value is None else f'{key}="{value}"\n'
    found = False
    out: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            existing_key = stripped.partition("=")[0].strip()
            if existing_key == key:
                found = True
                if new_line is not None:
                    out.append(new_line)
                continue  # delete -> drop the line
        out.append(line)
    if not found and new_line is not None:
        if out and not out[-1].endswith("\n"):
            out[-1] = out[-1] + "\n"
        out.append(new_line)

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(out)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass

    # Make it live immediately for the running process.
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value