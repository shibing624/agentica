# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Config routes: /api/config/*, /api/models, /api/model, /api/open, /api/status.
"""
import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from io import StringIO
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from .. import deps
from ..config import settings
from agentica.version import __version__
from agentica.provider_registry import list_providers, get_provider_factory
from agentica.global_config import (
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
from agentica.cost_tracker import get_model_supports_images
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


@dataclass
class ProfileFields:
    """Single definition of the config.yaml profile schema (see the profile
    shape documented in agentica/global_config.py). Profiles themselves stay
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
    from agentica.compression.manager import parse_compact_token_limit
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
    path = _prefs_path(user_id)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8") or "{}")
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
    return out


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
    if svc.has_active_runs():
        raise HTTPException(
            status_code=409,
            detail="Cannot switch model while a run is active. Wait for it to finish or cancel it first.",
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
    if svc.has_active_runs():
        raise HTTPException(
            status_code=409,
            detail="Cannot switch profile while a run is active. Wait for it to finish or cancel it first.",
        )
    # reload_profile() writes the full tuning set (model_provider/model_name/
    # reasoning_effort/max_tokens/temperature/top_p/context_window/...)
    # directly to `settings`, so no manual patch-up is needed here.
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
    svc: AgentService = Depends(deps.get_agent_service),
):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Profile name must not be empty")
    if svc.has_active_runs():
        raise HTTPException(status_code=409, detail="Cannot change config while a run is active.")
    upsert_profile(name, _profile_body_to_dict(body), make_active=False)
    return {"status": "ok", "name": name}


@router.put("/api/profile/{name}")
async def update_profile(
    name: str,
    body: ProfileUpsertRequest,
    svc: AgentService = Depends(deps.get_agent_service),
):
    existing = get_profile(name)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")
    if svc.has_active_runs():
        raise HTTPException(status_code=409, detail="Cannot change config while a run is active.")
    # Merge: keep existing fields the user left empty (esp. api_key).
    merged = dict(existing)
    for k, v in _profile_body_to_dict(body).items():
        merged[k] = v
    upsert_profile(name, merged, make_active=False)
    return {"status": "ok", "name": name}


@router.delete("/api/profile/{name}")
async def remove_profile(
    name: str,
    svc: AgentService = Depends(deps.get_agent_service),
):
    if not get_profile(name):
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")
    if svc.has_active_runs():
        raise HTTPException(status_code=409, detail="Cannot change config while a run is active.")
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


# ============== Open in Finder / Terminal ==============

@router.post("/api/open")
async def open_path(request: OpenRequest):
    """Open a local path or an http(s) URL with the OS default handler."""
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


def _open_os(target: str, app: str) -> None:
    if sys.platform == "darwin":
        if app == "terminal":
            subprocess.Popen(["open", "-a", "Terminal", target])
        else:
            subprocess.Popen(["open", target])
    elif sys.platform == "linux":
        if app == "terminal":
            for term in ["gnome-terminal", "xterm", "konsole"]:
                if shutil.which(term):
                    subprocess.Popen([term, f"--working-directory={target}"])
                    break
        else:
            subprocess.Popen(["xdg-open", target])
    else:
        subprocess.Popen(["explorer", target])
