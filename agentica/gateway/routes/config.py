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
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse

from .. import deps
from ..config import settings
from agentica.version import __version__
from agentica.provider_registry import list_providers, get_provider_factory
from agentica.global_config import (
    get_profiles,
    get_active_profile_name,
    get_profile,
    upsert_profile,
    delete_profile,
)
from agentica.cli import self_manage
from ..models import (
    ModelSwitchRequest,
    ProfileSwitchRequest,
    ProfileUpsertRequest,
    ThinkingToggleRequest,
    BaseDirRequest,
    OpenRequest,
)
from ..services.agent_service import AgentService

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
    reasoning_effort: str = ""
    max_tokens: int = 0
    context_window: int = 0
    temperature: float = 0.0
    top_p: float = 0.0
    auxiliary_model: Optional[dict] = None
    env: Optional[dict] = None


PROFILE_FIELD_NAMES = tuple(f.name for f in fields(ProfileFields))
TUNING_FIELD_NAMES = ("reasoning_effort", "max_tokens", "temperature", "top_p", "context_window")


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
    return {
        "workspace": str(settings.workspace_path),
        "base_dir": str(settings.base_dir),
        "model": f"{svc.model_provider}/{svc.model_name}" if svc else f"{settings.model_provider}/{settings.model_name}",
        "model_provider": svc.model_provider if svc else settings.model_provider,
        "model_name": svc.model_name if svc else settings.model_name,
        "model_thinking": settings.model_thinking or "",
        "context_window": context_window,
        "version": __version__,
        "channels": deps.channel_manager.get_status() if deps.channel_manager else {},
        "scheduler": scheduler_status,
        "active_profile": active_profile,
        "config_path": str(config_path),
        "tuning": {
            "max_tokens": svc.max_tokens if svc else settings.max_tokens,
            "temperature": svc.temperature if svc else settings.temperature,
            "top_p": svc.top_p if svc else settings.top_p,
            "reasoning_effort": svc.model_reasoning_effort if svc else settings.model_reasoning_effort,
        },
    }


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


# ============== Thinking ==============

@router.post("/api/config/thinking")
async def toggle_thinking(request: ThinkingToggleRequest):
    new_val = "enabled" if request.enabled else ""
    settings.model_thinking = new_val
    svc = deps.agent_service
    if svc:
        await svc._invalidate_cache()
    return {"status": "ok", "thinking": new_val}


@router.get("/api/config/thinking")
async def get_thinking():
    return {"thinking": settings.model_thinking or ""}


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
    history = await _load_dir_history()
    current = str(settings.base_dir)
    if current not in history:
        history.insert(0, current)
        await _save_dir_history(history)
    return {"history": history}


@router.delete("/api/config/dir_history")
async def clear_dir_history():
    await _save_dir_history([str(settings.base_dir)])
    return {"status": "ok"}


def _dir_history_file() -> Path:
    from agentica.config import AGENTICA_CACHE_DIR
    return Path(AGENTICA_CACHE_DIR).expanduser() / "dir_history.json"


async def _load_dir_history() -> list[str]:
    f = _dir_history_file()
    if f.exists():
        try:
            text = await asyncio.to_thread(f.read_text)
            return json.loads(text)
        except Exception:
            pass
    return []


async def _save_dir_history(history: list[str]) -> None:
    f = _dir_history_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(history, ensure_ascii=False)
    await asyncio.to_thread(f.write_text, data)


async def _add_dir_history(path: str) -> None:
    history = await _load_dir_history()
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


# ============== Open in Finder / Terminal ==============

@router.post("/api/open")
async def open_path(request: OpenRequest):
    """Open a path in Finder or Terminal (local deployments only)."""
    p = Path(request.path).expanduser()
    if not p.exists():
        raise HTTPException(status_code=404, detail="Path not found")

    try:
        if sys.platform == "darwin":
            if request.app == "terminal":
                subprocess.Popen(["open", "-a", "Terminal", str(p)])
            else:
                subprocess.Popen(["open", str(p)])
        elif sys.platform == "linux":
            if request.app == "terminal":
                for term in ["gnome-terminal", "xterm", "konsole"]:
                    if shutil.which(term):
                        subprocess.Popen([term, f"--working-directory={str(p)}"])
                        break
            else:
                subprocess.Popen(["xdg-open", str(p)])
        else:
            subprocess.Popen(["explorer", str(p)])
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
