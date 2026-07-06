# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
Plugins routes: /api/tools (read-only introspection of built-in code tools),
/api/mcp/servers (CRUD over MCP tool servers), and /api/skills (full CRUD
over markdown SKILL.md files).

Built-in tools are Python-code tools wired up in agentica.tools.buildin_tools —
they cannot be created or edited from the web UI, only listed for visibility.
MCP servers are stored in ~/.agentica/mcp_config.json (the same file/schema
DeepAgent auto-loads on construction — see agentica.mcp.config.MCPConfig), so
adding/removing one here just rewrites that file and invalidates the agent
cache; the next request rebuilds the agent and it picks up the new tools.
Skills are plain SKILL.md files under AGENTICA_SKILL_DIR and are fully
user-editable, so the web UI supports create/update/delete for them.
"""
import json
import re
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from agentica.config import AGENTICA_HOME, AGENTICA_SKILL_DIR
from agentica.skills.skill_loader import SkillLoader
from agentica.tools.buildin_tools import get_builtin_tools

from .. import deps
from ..models import McpServerRequest, SkillCreateRequest, SkillUpdateRequest
from ..services.agent_service import AgentService

router = APIRouter(prefix="/api")


# ============== Tools (read-only) ==============

@router.get("/tools")
async def list_builtin_tools():
    """Flatten the built-in Tool groups into individual callable functions."""
    tools = get_builtin_tools(include_skills=False, include_ask_user_question=False)
    out = []
    for t in tools:
        for fname, func in t.functions.items():
            desc = (func.description or "").strip().split("\n")[0][:200]
            out.append({
                "name": fname,
                "tool_group": t.name,
                "description": desc,
                "is_read_only": func.is_read_only,
            })
    out.sort(key=lambda x: x["name"])
    return {"tools": out, "total": len(out)}


# ============== MCP servers (user-editable CRUD) ==============

def _mcp_config_path() -> Path:
    return Path(AGENTICA_HOME).expanduser() / "mcp_config.json"


def _load_mcp_servers() -> dict:
    p = _mcp_config_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8") or "{}")
    except Exception:
        return {}
    return data.get("mcpServers") or {}


def _save_mcp_servers(servers: dict) -> None:
    p = _mcp_config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"mcpServers": servers}, indent=2, ensure_ascii=False), encoding="utf-8")


@router.get("/mcp/servers")
async def list_mcp_servers():
    servers = _load_mcp_servers()
    out = []
    for name, cfg in servers.items():
        out.append({
            "name": name,
            "type": "sse" if cfg.get("url") else "stdio",
            "command": cfg.get("command", ""),
            "args": cfg.get("args") or [],
            "url": cfg.get("url", ""),
            "env_keys": list((cfg.get("env") or {}).keys()),
        })
    out.sort(key=lambda x: x["name"])
    return {"servers": out, "total": len(out)}


@router.post("/mcp/servers")
async def create_mcp_server(
    body: McpServerRequest,
    svc: AgentService = Depends(deps.get_agent_service),
):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Server name must not be empty")
    if not body.command and not body.url:
        raise HTTPException(status_code=400, detail="Either command or url is required")
    servers = _load_mcp_servers()
    if name in servers:
        raise HTTPException(status_code=400, detail=f"MCP server '{name}' already exists")
    entry: dict = {}
    if body.command:
        entry["command"] = body.command
        entry["args"] = body.args or []
    if body.url:
        entry["url"] = body.url
    if body.env:
        entry["env"] = body.env
    if body.headers:
        entry["headers"] = body.headers
    if body.timeout:
        entry["timeout"] = body.timeout
    servers[name] = entry
    _save_mcp_servers(servers)
    # New tool must be visible to the agent(s) already backing this web session,
    # so force every cached DeepAgent to rebuild (re-running auto MCP load) on
    # its next turn instead of only affecting brand-new sessions.
    await svc._invalidate_cache()
    return {"status": "created", "name": name}


@router.delete("/mcp/servers/{name}")
async def delete_mcp_server(
    name: str,
    svc: AgentService = Depends(deps.get_agent_service),
):
    servers = _load_mcp_servers()
    if name not in servers:
        raise HTTPException(status_code=404, detail=f"MCP server '{name}' not found")
    del servers[name]
    _save_mcp_servers(servers)
    await svc._invalidate_cache()
    return {"status": "deleted", "name": name}


# ============== Skills (user-editable CRUD) ==============

def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip().lower()).strip("-")
    return slug or "skill"


def _skill_dir() -> Path:
    d = Path(AGENTICA_SKILL_DIR).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_registry():
    return SkillLoader().load_all()


def _write_skill_md(path: Path, name: str, description: str, content: str, trigger: str = "") -> None:
    frontmatter = [f"name: {name}", f"description: {description}"]
    if trigger:
        frontmatter.append(f"trigger: {trigger}")
    md = "---\n" + "\n".join(frontmatter) + "\n---\n\n" + content
    (path / "SKILL.md").write_text(md, encoding="utf-8")


@router.get("/skills")
async def list_skills():
    registry = _load_registry()
    skills = []
    for s in registry.list_all():
        d = s.to_dict(include_content=False)
        d["editable"] = s.location == "user"
        skills.append(d)
    skills.sort(key=lambda x: x["name"])
    return {"skills": skills, "total": len(skills)}


@router.get("/skills/{name}")
async def get_skill(name: str):
    registry = _load_registry()
    skill = registry.get(name)
    if not skill:
        raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")
    d = skill.to_dict(include_content=True)
    d["editable"] = skill.location == "user"
    return d


@router.post("/skills")
async def create_skill(body: SkillCreateRequest):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Skill name must not be empty")
    slug = _slugify(name)
    skill_path = _skill_dir() / slug
    if skill_path.exists():
        raise HTTPException(status_code=400, detail=f"Skill '{slug}' already exists")
    skill_path.mkdir(parents=True)
    _write_skill_md(skill_path, name, body.description, body.content, body.trigger or "")
    return {"status": "created", "name": name, "slug": slug}


@router.put("/skills/{name}")
async def update_skill(name: str, body: SkillUpdateRequest):
    registry = _load_registry()
    skill = registry.get(name)
    if not skill or skill.location != "user":
        raise HTTPException(status_code=404, detail="Editable skill not found")
    description = body.description if body.description is not None else skill.description
    trigger = body.trigger if body.trigger is not None else (skill.trigger or "")
    content = body.content if body.content is not None else skill.content
    _write_skill_md(skill.path, skill.name, description, content, trigger)
    return {"status": "updated", "name": name}


@router.delete("/skills/{name}")
async def delete_skill(name: str):
    registry = _load_registry()
    skill = registry.get(name)
    if not skill or skill.location != "user":
        raise HTTPException(status_code=404, detail="Editable skill not found")
    shutil.rmtree(skill.path, ignore_errors=True)
    return {"status": "deleted", "name": name}
