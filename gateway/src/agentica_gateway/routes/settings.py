# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Config routes: /api/config/*, /api/models, /api/model, /api/open, /api/status.
"""
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, fields
from pathlib import Path
from io import StringIO
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from .. import deps
from ..config import settings
from agentica.version import __version__
from agentica.model.providers import list_providers, get_provider_factory
from agentica.config.profiles import (
    get_profiles,
    get_active_profile_name,
    get_profile,
    get_setting,
    upsert_profile,
    delete_profile,
    load_global_config,
)
from agentica.cli import self_manage
from agentica.config import AGENTICA_HOME
from agentica.utils.log import logger
from agentica.model.cost import get_model_supports_images
from agentica.compression.manager import parse_compact_token_limit
from ..models import (
    ModelSwitchRequest,
    ProfileSwitchRequest,
    ProfileUpsertRequest,
    BaseDirRequest,
    OpenRequest,
)
from ..services.agent_service import AgentService
from ..services.media_understanding import is_gemini, media_model_label

router = APIRouter()

_DIR_HISTORY_MAX = 20
_UPDATE_CACHE_TTL_S = 600.0
_update_cache: dict[str, Any] = {"checked_at": 0.0, "latest": None}
_update_lock = asyncio.Lock()


@dataclass
class ProfileFields:
    """Single definition of the config.yaml profile schema (see the profile
    shape documented in agentica/config/profiles.py). Profiles themselves stay
    plain dicts end-to-end (loaded/saved via ruamel.yaml), but the field list
    is declared once here and reused by _profile_summary/get_profile_detail/
    _profile_body_to_dict instead of being hand-written 3 times."""
    model_provider: str = ""
    model_name: str = ""
    base_url: str = ""
    api_key: str = ""
    wire_api: str = ""
    reasoning: str = ""
    reasoning_effort: str = ""
    max_tokens: int = 0
    context_window: int = 0
    compact_token_limit: int = 0
    temperature: float = 0.0
    top_p: float = 0.0
    auxiliary_model: Optional[dict] = None
    env: Optional[dict] = None


PROFILE_FIELD_NAMES = tuple(f.name for f in fields(ProfileFields))
TUNING_FIELD_NAMES = (
    "wire_api", "reasoning", "reasoning_effort", "max_tokens",
    "temperature", "top_p", "context_window", "compact_token_limit",
)


# ============== Root + Status ==============

@router.get("/")
async def root():
    return {"name": "Agentica Gateway", "version": __version__, "status": "running"}


@router.get("/health")
@router.get("/api/health")
async def health():
    from agentica.cron.jobs import list_jobs
    active_jobs = len(list_jobs(include_disabled=False))
    return {
        "status": "ok",
        "version": __version__,
        "channels": deps.channel_manager.get_status() if deps.channel_manager else {},
        "scheduler": {"active_jobs": active_jobs},
    }


@router.get("/api/update/check")
async def check_update(force: bool = False):
    """Return installed version and latest PyPI version (cached briefly unless forced)."""
    current = self_manage.get_current_version()
    now = time.monotonic()
    if (
        force
        or _update_cache["latest"] is None
        or (now - float(_update_cache["checked_at"])) > _UPDATE_CACHE_TTL_S
    ):
        latest = await asyncio.to_thread(self_manage.get_latest_version)
        if latest:
            _update_cache["latest"] = latest
            _update_cache["checked_at"] = now
    else:
        latest = _update_cache["latest"]
    return {
        "current_version": current,
        "latest_version": latest,
        "update_available": self_manage.is_upgrade_available(current, latest),
    }


@router.post("/api/update/apply")
async def apply_update(request: Request):
    """Upgrade agentica + agentica-gateway via pip/uv and restart the gateway."""
    principal = request.state.principal
    if not principal.is_admin:
        raise HTTPException(status_code=403, detail="Administrator access required")
    if _update_lock.locked():
        raise HTTPException(status_code=409, detail="Update already in progress")
    async with _update_lock:
        current = self_manage.get_current_version()
        code, output = await asyncio.to_thread(
            self_manage.run_pip_upgrade, ("agentica", "agentica-gateway")
        )
        if code != 0:
            tail = "\n".join((output or "").strip().splitlines()[-12:])
            raise HTTPException(
                status_code=500,
                detail=tail or f"pip exited with code {code}",
            )
        latest = await asyncio.to_thread(self_manage.get_latest_version)
        _update_cache["latest"] = latest or current
        _update_cache["checked_at"] = time.monotonic()
        from .. import main as gw_main
        restarting = gw_main._server is not None
        if restarting:
            asyncio.get_running_loop().call_later(0.35, gw_main.request_restart)
        return {
            "status": "ok",
            "previous_version": current,
            "version": latest or current,
            "restarting": restarting,
            "output": (output or "").strip(),
        }


@router.get("/api/status")
async def status():
    from agentica.cron.jobs import list_jobs
    active_jobs = len(list_jobs(include_disabled=False))
    scheduler_status = {"active_jobs": active_jobs}

    svc = deps.agent_service
    context_window = svc.get_context_window() if svc else 128000

    active_profile = get_active_profile_name()
    config_path = self_manage.config_file_path()
    model_name = svc.model_name if svc else settings.model_name
    profile = get_profile(active_profile) or {}
    compact_token_limit = parse_compact_token_limit(profile.get("compact_token_limit"))
    if compact_token_limit is None:
        compact_token_limit = parse_compact_token_limit(get_setting("compact_token_limit", None))
    return {
        "workspace": str(settings.workspace_path),
        "base_dir": str(settings.base_dir),
        "model": f"{svc.model_provider}/{svc.model_name}" if svc else f"{settings.model_provider}/{settings.model_name}",
        "model_provider": svc.model_provider if svc else settings.model_provider,
        "model_name": model_name,
        "supports_images": get_model_supports_images(model_name) or is_gemini(model_name),
        "media_model": media_model_label(),
        "model_thinking": settings.model_thinking or "",
        "context_window": context_window,
        "compact_token_limit": compact_token_limit or 0,
        "version": __version__,
        "channels": deps.channel_manager.get_status() if deps.channel_manager else {},
        "scheduler": scheduler_status,
        "active_profile": active_profile,
        "config_path": str(config_path),
        "tuning": {
            "wire_api": svc.model_wire_api if svc else settings.model_wire_api,
            "reasoning": svc.model_reasoning if svc else settings.model_reasoning,
            "max_tokens": svc.max_tokens if svc else settings.max_tokens,
            "temperature": svc.temperature if svc else settings.temperature,
            "top_p": svc.top_p if svc else settings.top_p,
            "reasoning_effort": svc.model_reasoning_effort if svc else settings.model_reasoning_effort,
        },
    }


_PREFS_THEME = {"auto", "light", "dark"}
_PREFS_LANG = {"en", "zh"}
_PREFS_APPROVAL = {"ask", "auto", "allow-all"}


def _prefs_path(user_id: str) -> Path:
    return Path(AGENTICA_HOME).expanduser() / "gateway" / "prefs" / f"{user_id}.json"


def _read_prefs(user_id: str) -> dict:
    """Prefs for one account; an unreadable file reads as "nothing chosen yet".

    This is on the IM turn path (``approval_mode_pref``), so a half-written or
    hand-edited file must not fail every message. The write is not atomic, and
    the reader is not allowed to turn that into an outage.
    """
    path = _prefs_path(user_id)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8") or "{}")
    except (OSError, ValueError) as exc:
        logger.warning(f"gateway prefs unreadable, using defaults: {path}: {exc}")
        return {}
    return data if isinstance(data, dict) else {}


def _write_prefs(user_id: str, data: dict) -> None:
    path = _prefs_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _clean_prefs(raw: dict) -> dict:
    out: dict = {}
    theme = raw.get("theme")
    if theme in _PREFS_THEME:
        out["theme"] = theme
    lang = raw.get("lang")
    if lang in _PREFS_LANG:
        out["lang"] = lang
    mode = raw.get("approval_mode")
    if mode in _PREFS_APPROVAL:
        out["approval_mode"] = mode
    last = raw.get("last_session_id")
    if isinstance(last, str) and last.strip():
        out["last_session_id"] = last.strip()
    if isinstance(raw.get("auto_extract_memory"), bool):
        out["auto_extract_memory"] = raw["auto_extract_memory"]
    if isinstance(raw.get("show_cli_sessions"), bool):
        out["show_cli_sessions"] = raw["show_cli_sessions"]
    if "workspaces" in raw:
        out["workspaces"] = _clean_workspaces(raw["workspaces"])
    return out


_MAX_WORKSPACES = 200
_MAX_WORKSPACE_NAME = 80


def _clean_workspaces(raw: Any) -> list:
    """Sidebar workspaces the user added or renamed: ``[{"dir", "name"?}]``.

    A workspace is a directory; sessions group under it by their work dir. An
    entry exists here only when it carries something the session list cannot
    derive — a folder added before its first chat, or a display name. Names are
    labels, never a rename of the folder on disk.
    """
    if not isinstance(raw, list):
        return []
    seen: set = set()
    out: list = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        d = item.get("dir")
        if not isinstance(d, str) or not d.strip():
            continue
        d = d.strip().rstrip("/\\") or d.strip()
        if d in seen:
            continue
        seen.add(d)
        entry = {"dir": d}
        name = item.get("name")
        if isinstance(name, str) and name.strip():
            entry["name"] = name.strip()[:_MAX_WORKSPACE_NAME]
        out.append(entry)
        if len(out) >= _MAX_WORKSPACES:
            break
    return out


def show_cli_sessions_pref(user_id: str) -> bool:
    """Whether this account's web session list includes CLI sessions (default: no)."""
    return bool(_clean_prefs(_read_prefs(user_id)).get("show_cli_sessions", False))


def approval_mode_pref(user_id: str) -> Optional[str]:
    """The approval tier this account last picked in the web selector, if any."""
    return _clean_prefs(_read_prefs(user_id)).get("approval_mode")


@router.get("/api/prefs")
async def get_prefs(request: Request):
    """Per-account UI prefs. localStorage is a first-paint cache of this file."""
    return _clean_prefs(_read_prefs(request.state.principal.user_id))


@router.put("/api/prefs")
async def put_prefs(request: Request):
    uid = request.state.principal.user_id
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Expected a JSON object")
    previous = _read_prefs(uid)
    if "workspaces" in body:
        known = {w["dir"] for w in _clean_workspaces(previous.get("workspaces"))}
        for w in _clean_workspaces(body["workspaces"]):
            if w["dir"] in known:
                continue
            # Same rule as the working directory: we never create a folder on
            # the user's behalf, so a new workspace must already exist.
            p = Path(w["dir"]).expanduser()
            if not p.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"Directory does not exist: {p}. Create it first, then try again.",
                )
            if not p.is_dir():
                raise HTTPException(status_code=400, detail=f"Path is not a directory: {p}")
    merged = _clean_prefs({**previous, **body})
    _write_prefs(uid, merged)
    old_extract = previous.get("auto_extract_memory", True)
    new_extract = merged.get("auto_extract_memory", True)
    if bool(old_extract) != bool(new_extract) and deps.agent_service is not None:
        await deps.agent_service._invalidate_cache()
    return merged


# ============== Models ==============

@router.get("/api/models")
async def list_models():
    """Return current model only. The model catalog is no longer hardcoded —
    users type `provider/model_name` in the UI's custom input."""
    svc = deps.agent_service
    current_provider = svc.model_provider if svc else settings.model_provider
    current_name = svc.model_name if svc else settings.model_name
    return {
        "current_provider": current_provider,
        "current_name": current_name,
        "current": f"{current_provider}/{current_name}",
    }


@router.post("/api/model")
async def switch_model(
    request: ModelSwitchRequest,
    svc: AgentService = Depends(deps.get_agent_service),
):
    # Validate provider against the SDK registry so an unknown slug (e.g.
    # "doubao" instead of "ark") fails here with a helpful list, not lazily
    # on the next agent build. For full profile-based switches use
    # POST /api/profile/switch.
    if get_provider_factory(request.model_provider) is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown model_provider '{request.model_provider}'. "
                f"Supported: {list_providers()}"
            ),
        )
    # AgentService.model_provider/model_name proxy directly to settings, so a
    # single write here is enough — no separate svc.xxx write needed.
    settings.model_provider = request.model_provider
    settings.model_name = request.model_name
    await svc._invalidate_cache()
    return {"status": "ok", "model": f"{request.model_provider}/{request.model_name}"}


# ============== Providers + Profiles ==============

@router.get("/api/providers")
async def list_providers_api():
    """Return all SDK-supported provider slugs (single source of truth)."""
    return {"providers": list_providers()}


def _mask_key(key: str) -> str:
    """Mask an API key for display: show first 4 + last 4 chars."""
    if not key:
        return ""
    if len(key) <= 8:
        return "****"
    return key[:4] + "****" + key[-4:]


def _profile_summary(name: str, profile: dict) -> dict:
    """Build a UI-friendly profile summary (api_key masked)."""
    aux = profile.get("auxiliary_model") or {}
    if not isinstance(aux, dict):
        aux = {}
    tuning = []
    for field_name in TUNING_FIELD_NAMES:
        value = profile.get(field_name)
        if value:
            label = "effort" if field_name == "reasoning_effort" else field_name
            tuning.append(f"{label}={value}")
    return {
        "name": name,
        "model_provider": profile.get("model_provider", ""),
        "model_name": profile.get("model_name", ""),
        "base_url": profile.get("base_url", ""),
        "api_key_masked": _mask_key(profile.get("api_key", "")),
        "has_api_key": bool(profile.get("api_key")),
        "tuning": tuning,
        "auxiliary": (
            {
                "model_provider": aux.get("model_provider", ""),
                "model_name": aux.get("model_name", ""),
                "base_url": aux.get("base_url", ""),
                "wire_api": aux.get("wire_api", ""),
                "reasoning": aux.get("reasoning", ""),
                "reasoning_effort": aux.get("reasoning_effort", ""),
                "has_api_key": bool(aux.get("api_key")),
                "api_key_masked": _mask_key(aux.get("api_key", "")),
            }
            if aux
            else None
        ),
    }


@router.get("/api/profiles")
async def list_profiles():
    """List all config.yaml profiles with the active one marked."""
    profiles = get_profiles()
    active = get_active_profile_name()
    return {
        "active": active,
        "profiles": [_profile_summary(name, p) for name, p in profiles.items()],
    }


@router.post("/api/profile/switch")
async def switch_profile(
    request: ProfileSwitchRequest,
    svc: AgentService = Depends(deps.get_agent_service),
):
    name = request.name.strip()
    profiles = get_profiles()
    if name not in profiles:
        raise HTTPException(
            status_code=404,
            detail=f"Profile '{name}' not found. Available: {list(profiles.keys())}",
        )
    # reload_profile() writes the full tuning set (model_provider/model_name/
    # reasoning_effort/max_tokens/temperature/top_p/context_window/...)
    # directly to `settings`, so no manual patch-up is needed here. A session
    # that is mid-run finishes on its current model and rebuilds afterwards.
    await svc.reload_profile(name)
    return {
        "status": "ok",
        "active_profile": name,
        "model": f"{svc.model_provider}/{svc.model_name}",
    }


# ============== Profile CRUD ==============

@router.get("/api/profile/{name}")
async def get_profile_detail(name: str):
    p = get_profile(name)
    if not p:
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")
    aux = p.get("auxiliary_model") or {}
    if not isinstance(aux, dict):
        aux = {}
    return {
        "name": name,
        "model_provider": p.get("model_provider", ""),
        "model_name": p.get("model_name", ""),
        "base_url": p.get("base_url", ""),
        "api_key_masked": _mask_key(p.get("api_key", "")),
        "has_api_key": bool(p.get("api_key")),
        **{field_name: p.get(field_name) for field_name in TUNING_FIELD_NAMES},
        "auxiliary_model": (
            {
                "model_provider": aux.get("model_provider", ""),
                "model_name": aux.get("model_name", ""),
                "base_url": aux.get("base_url", ""),
                "wire_api": aux.get("wire_api", ""),
                "reasoning": aux.get("reasoning", ""),
                "reasoning_effort": aux.get("reasoning_effort", ""),
                "has_api_key": bool(aux.get("api_key")),
                "api_key_masked": _mask_key(aux.get("api_key", "")),
            }
            if aux
            else None
        ),
        "env": p.get("env") or {},
    }


def _profile_body_to_dict(body: ProfileUpsertRequest) -> dict:
    """Convert a ProfileUpsertRequest to a profile dict, dropping empty/None
    fields. api_key is only included when non-empty (empty means "keep existing"
    on update)."""
    d: dict = {}
    for k in PROFILE_FIELD_NAMES:
        if k == "api_key":
            continue
        v = getattr(body, k)
        if v is not None and v != "":
            d[k] = v
    if body.api_key:
        d["api_key"] = body.api_key
    return d


@router.post("/api/profile")
async def create_profile(
    body: ProfileUpsertRequest,
):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Profile name must not be empty")
    upsert_profile(name, _profile_body_to_dict(body), make_active=False)
    return {"status": "ok", "name": name}


@router.put("/api/profile/{name}")
async def update_profile(
    name: str,
    body: ProfileUpsertRequest,
):
    existing = get_profile(name)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")
    # Merge: keep existing fields the user left empty (esp. api_key).
    merged = dict(existing)
    for k, v in _profile_body_to_dict(body).items():
        merged[k] = v
    upsert_profile(name, merged, make_active=False)
    return {"status": "ok", "name": name}


@router.delete("/api/profile/{name}")
async def remove_profile(
    name: str,
):
    if not get_profile(name):
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")
    if not delete_profile(name):
        raise HTTPException(status_code=400, detail="Delete failed")
    return {"status": "deleted", "name": name}


# ============== Working directory ==============

@router.post("/api/config/base_dir")
async def set_base_dir(request: BaseDirRequest):
    """Set the working directory for the current/new session.

    The directory must already exist — we never create it on the user's
    behalf. Setting a directory that isn't the current project's dir is also
    how a new project gets created (see ensureProjectForSession on the
    frontend): each distinct dir maps 1:1 to a project.
    """
    raw = request.base_dir.strip()
    if not raw:
        raise HTTPException(status_code=400, detail="Path must not be empty")
    p = Path(raw).expanduser().resolve()
    if not p.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Directory does not exist: {p}. Create it first, then try again.",
        )
    if not p.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {p}")

    settings.base_dir = p
    svc = deps.agent_service
    if svc:
        svc.update_work_dir(str(p))
    await _add_dir_history(str(p))
    return {"status": "ok", "base_dir": str(p)}


@router.get("/api/config/dir_history")
async def get_dir_history():
    """Recent working dirs. Missing / inaccessible paths are dropped so
    leftover pytest tmpdirs do not stay in the settings chips forever."""
    history = await _pruned_dir_history()
    return {"history": history}


@router.delete("/api/config/dir_history")
async def delete_dir_history(path: Optional[str] = None):
    """Remove one history entry (``?path=``) or clear the list."""
    history = await _pruned_dir_history()
    if path:
        history = [p for p in history if p != path]
    else:
        history = []
    await _save_dir_history(history)
    return {"status": "ok", "history": history}


@router.get("/api/config/file")
async def get_config_file():
    """``config.yaml`` text for the settings preview. Secrets are masked."""
    from ruamel.yaml import YAML

    path = self_manage.config_file_path()
    masked = _mask_config_tree(load_global_config())
    buf = StringIO()
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.allow_unicode = True
    yaml.dump(masked or {}, buf)
    return {"path": path, "content": buf.getvalue()}


def _mask_config_tree(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: _mask_config_tree(v) if isinstance(v, (dict, list)) else self_manage.mask_secret(k, v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_mask_config_tree(v) for v in obj]
    return obj


def _dir_history_file() -> Path:
    from agentica.config import AGENTICA_CACHE_DIR
    return Path(AGENTICA_CACHE_DIR).expanduser() / "dir_history.json"


def _existing_dirs(paths: list) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in paths:
        if not isinstance(raw, str) or not raw.strip():
            continue
        p = Path(raw).expanduser()
        try:
            if not p.is_dir():
                continue
        except OSError:
            continue
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


async def _load_dir_history() -> list[str]:
    f = _dir_history_file()
    if f.exists():
        try:
            text = await asyncio.to_thread(f.read_text)
            data = json.loads(text)
            return data if isinstance(data, list) else []
        except Exception:
            pass
    return []


async def _save_dir_history(history: list[str]) -> None:
    f = _dir_history_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(history, ensure_ascii=False)
    await asyncio.to_thread(f.write_text, data)


async def _pruned_dir_history() -> list[str]:
    history = _existing_dirs(await _load_dir_history())
    await _save_dir_history(history)
    return history


async def _add_dir_history(path: str) -> None:
    history = await _pruned_dir_history()
    if path in history:
        history.remove(path)
    history.insert(0, path)
    await _save_dir_history(history[:_DIR_HISTORY_MAX])


# ============== Filesystem browsing (folder picker) ==============

@router.get("/api/fs/browse")
async def browse_fs(path: Optional[str] = None):
    """List subdirectories of a path for the web UI's folder picker
    (read-only). Falls back to settings.base_dir when path is omitted."""
    base = Path(path).expanduser().resolve() if path else settings.base_dir
    if not base.exists() or not base.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {base}")
    try:
        entries = sorted(base.iterdir(), key=lambda p: p.name.lower())
    except PermissionError:
        entries = []
    dirs = [
        {"name": entry.name, "path": str(entry)}
        for entry in entries
        if entry.is_dir() and not entry.name.startswith(".")
    ]
    parent = str(base.parent) if base.parent != base else None
    return {"path": str(base), "parent": parent, "dirs": dirs}


@router.post("/api/fs/temp")
async def make_temp_workspace():
    """Create a throwaway working directory for a new web chat.

    Same idea as leaving the CLI cwd blank: the directory must exist, so the
    server makes one under ``$AGENTICA_HOME/tmp/web-chats``.
    """
    root = Path(AGENTICA_HOME) / "tmp" / "web-chats"
    root.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="chat-", dir=str(root)))
    return {"path": str(path)}


# ============== Open in Finder / Terminal / an app ==============

# dsh's open-in-app catalog, macOS part: (id, kind, bundle names). An app is
# offered only when one of its bundles is installed; `kind` groups the menu
# (file manager, editors, terminals). Finder and Terminal ship with macOS.
_MAC_APPS = (
    ("finder", "files", ("/System/Library/CoreServices/Finder.app",)),
    ("cursor", "editor", ("Cursor.app",)),
    ("vscode", "editor", ("Visual Studio Code.app",)),
    ("vscodeinsiders", "editor", ("Visual Studio Code - Insiders.app",)),
    ("windsurf", "editor", ("Windsurf.app",)),
    ("zed", "editor", ("Zed.app", "Zed Preview.app")),
    ("sublimetext", "editor", ("Sublime Text.app",)),
    ("pycharm", "editor", ("PyCharm.app", "PyCharm CE.app", "PyCharm Professional Edition.app")),
    ("idea", "editor", ("IntelliJ IDEA.app", "IntelliJ IDEA CE.app", "IntelliJ IDEA Ultimate.app")),
    ("xcode", "editor", ("Xcode.app",)),
    ("terminal", "terminal", ("/System/Applications/Utilities/Terminal.app",)),
    ("iterm", "terminal", ("iTerm.app",)),
    ("warp", "terminal", ("Warp.app",)),
    ("ghostty", "terminal", ("Ghostty.app",)),
    ("cmux", "terminal", ("cmux.app",)),
    ("kitty", "terminal", ("kitty.app",)),
)
_APP_NAMES = {
    "finder": "Finder", "cursor": "Cursor", "vscode": "VS Code", "vscodeinsiders": "VS Code Insiders",
    "windsurf": "Windsurf", "zed": "Zed", "sublimetext": "Sublime Text", "pycharm": "PyCharm",
    "idea": "IntelliJ IDEA", "xcode": "Xcode", "terminal": "Terminal", "iterm": "iTerm2",
    "warp": "Warp", "ghostty": "Ghostty", "cmux": "cmux", "kitty": "kitty",
}


def _mac_bundle(names, home: Optional[str] = None) -> Optional[str]:
    """The first installed bundle: an absolute path as is, a bare name under
    /Applications or ~/Applications."""
    home = home or os.path.expanduser("~")
    for name in names:
        candidates = [name] if name.startswith("/") else [f"/Applications/{name}", f"{home}/Applications/{name}"]
        for path in candidates:
            if os.path.isdir(path):
                return path
    return None


def installed_apps(platform: Optional[str] = None) -> list:
    """Apps a folder can be opened in on this machine, in menu order."""
    platform = platform or sys.platform
    if platform != "darwin":
        # Elsewhere the gateway knows the file manager and a terminal only.
        return [{"id": "finder", "name": "Files", "kind": "files"},
                {"id": "terminal", "name": "Terminal", "kind": "terminal"}]
    out = []
    for app_id, kind, names in _MAC_APPS:
        if _mac_bundle(names):
            out.append({"id": app_id, "name": _APP_NAMES[app_id], "kind": kind})
    return out


@router.get("/api/open/apps")
async def open_apps():
    return {"apps": installed_apps()}


def _bundle_icon_png(bundle: str, out: Path) -> bool:
    """The bundle's icon as a 64px PNG at ``out`` (``plutil`` + ``sips``, as dsh)."""
    resources = Path(bundle) / "Contents" / "Resources"
    icon = None
    try:
        info = subprocess.run(
            ["plutil", "-extract", "CFBundleIconFile", "raw", "-o", "-", str(Path(bundle) / "Contents" / "Info.plist")],
            capture_output=True, text=True, timeout=5,
        )
        name = info.stdout.strip()
        if info.returncode == 0 and name:
            icon = resources / (name if name.endswith(".icns") else f"{name}.icns")
    except (OSError, subprocess.TimeoutExpired):
        pass
    if icon is None or not icon.is_file():
        found = sorted(resources.glob("*.icns")) if resources.is_dir() else []
        icon = found[0] if found else None
    if icon is None:
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        done = subprocess.run(
            ["sips", "-s", "format", "png", "-Z", "64", str(icon), "--out", str(out)],
            capture_output=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0 and out.is_file()


@router.get("/api/open/apps/{app_id}/icon")
async def open_app_icon(app_id: str):
    """The app's own icon, so the menu shows Finder / Cursor / iTerm2 as they look.
    404 off macOS or when the bundle has none; the page then draws a glyph."""
    if sys.platform != "darwin":
        raise HTTPException(status_code=404, detail="No icon")
    names = next((n for i, _k, n in _MAC_APPS if i == app_id), None)
    bundle = _mac_bundle(names) if names else None
    if bundle is None:
        raise HTTPException(status_code=404, detail="No icon")
    out = Path(AGENTICA_HOME) / "cache" / "app-icons" / f"{app_id}.png"
    if not out.is_file() and not await asyncio.to_thread(_bundle_icon_png, bundle, out):
        raise HTTPException(status_code=404, detail="No icon")
    return FileResponse(out, media_type="image/png", headers={"Cache-Control": "max-age=86400"})

@router.post("/api/open")
async def open_path(request: OpenRequest):
    """Open a local path or an http(s) URL with the OS default handler, or a
    folder in one of the apps ``/api/open/apps`` lists."""
    # Reject unknown values instead of silently opening with the wrong handler.
    if request.app not in ("finder", "editor", "reveal", "terminal") and request.app not in _APP_NAMES:
        raise HTTPException(status_code=422, detail=f"Unknown app: {request.app}")
    url = (request.url or "").strip()
    if url:
        if not url.startswith(("http://", "https://")):
            raise HTTPException(status_code=400, detail="Only http(s) URLs can be opened")
        try:
            _open_os(url, app="finder")
            return {"status": "ok"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    if not request.path:
        raise HTTPException(status_code=400, detail="path or url is required")
    p = Path(request.path).expanduser()
    if not p.exists():
        raise HTTPException(status_code=404, detail="Path not found")

    try:
        _open_os(str(p), app=request.app)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _open_command(target: str, app: str, platform: Optional[str] = None) -> Optional[list]:
    """The argv that opens ``target`` the way ``app`` asks, on this OS.

    ``finder``: open with the default app (a folder opens in the file manager).
    ``editor``: open in the default text editor, whatever the file type.
    ``reveal``: show the item selected in its folder. ``terminal``: a shell there.
    None when nothing can do it (Linux with no known terminal). Pure, so tests
    check it directly and never have to spawn anything.
    """
    platform = platform or sys.platform
    if platform == "darwin":
        for app_id, _kind, names in _MAC_APPS:
            if app == app_id and app_id not in ("finder", "terminal"):
                bundle = _mac_bundle(names)
                return ["open", "-a", bundle, target] if bundle else None
        if app == "terminal":
            return ["open", "-a", "Terminal", target]
        if app == "reveal":
            return ["open", "-R", target]
        if app == "editor":
            return ["open", "-t", target]
        return ["open", target]
    if platform.startswith("linux"):
        if app == "reveal":
            # No portable "select this file" on Linux file managers. Resolve so
            # a relative path never turns into `xdg-open ""`.
            folder = target if os.path.isdir(target) else os.path.dirname(os.path.abspath(target))
            return ["xdg-open", folder]
        if app == "terminal":
            for term in ["gnome-terminal", "xterm", "konsole"]:
                if shutil.which(term):
                    return [term, f"--working-directory={target}"]
            return None
        return ["xdg-open", target]
    if app == "reveal":
        return ["explorer", f"/select,{target}"]
    if app == "editor":
        return ["notepad", target]
    return ["explorer", target]


def _open_os(target: str, app: str) -> None:
    """Hand ``target`` to the desktop. The only place the gateway pops a window,
    which is why tests/conftest.py replaces exactly this function."""
    cmd = _open_command(target, app)
    if cmd:
        subprocess.Popen(cmd)
