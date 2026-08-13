# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Skills Hub — Source adapters and hub state management.

Provides:
  - SkillMeta / SkillBundle: data models for search results and downloadable skill packages
  - SkillSource ABC: interface for all skill registry adapters
  - GitHubSource: fetch skills from GitHub repos via Contents/Trees API
  - SkillsShSource: discover skills via skills.sh, delegate fetch to GitHub
  - LobeHubSource: fetch system-prompt skills from LobeHub marketplace
  - HubLockFile: track provenance of installed hub skills
  - TapsManager: manage custom GitHub repo sources
  - Security scanning (static pattern check) + quarantine install pipeline
  - unified_search(): search all sources and merge results

Used by agentica/cli/commands/tools_skills.py for /skills slash commands.
"""
import hashlib
import json
import os
import re
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

from agentica.config import AGENTICA_SKILL_DIR
from agentica.utils.log import logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SKILLS_DIR = Path(AGENTICA_SKILL_DIR).expanduser()
HUB_DIR = SKILLS_DIR / ".hub"
LOCK_FILE = HUB_DIR / "lock.json"
QUARANTINE_DIR = HUB_DIR / "quarantine"
AUDIT_LOG = HUB_DIR / "audit.log"
TAPS_FILE = HUB_DIR / "taps.json"
INDEX_CACHE_DIR = HUB_DIR / "index-cache"

INDEX_CACHE_TTL = 3600  # 1 hour


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SkillMeta:
    """Minimal metadata returned by search results."""
    name: str
    description: str
    source: str           # "github", "skills-sh", "lobehub"
    identifier: str       # source-specific unique ID
    trust_level: str      # "trusted" | "community"
    repo: Optional[str] = None
    path: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillBundle:
    """A downloaded skill ready for quarantine/scanning/installation."""
    name: str
    files: Dict[str, Union[str, bytes]]  # relative_path -> file content
    source: str
    identifier: str
    trust_level: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Path validation (security)
# ---------------------------------------------------------------------------

def _normalize_bundle_path(path_value: str, *, field_name: str, allow_nested: bool) -> str:
    """Normalize and validate bundle-controlled paths before touching disk."""
    raw = str(path_value).strip()
    if not raw:
        raise ValueError(f"Unsafe {field_name}: empty path")
    normalized = raw.replace("\\", "/")
    path = PurePosixPath(normalized)
    parts = [part for part in path.parts if part not in ("", ".")]
    if normalized.startswith("/") or path.is_absolute():
        raise ValueError(f"Unsafe {field_name}: {path_value}")
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"Unsafe {field_name}: {path_value}")
    if re.fullmatch(r"[A-Za-z]:", parts[0]):
        raise ValueError(f"Unsafe {field_name}: {path_value}")
    if not allow_nested and len(parts) != 1:
        raise ValueError(f"Unsafe {field_name}: {path_value}")
    return "/".join(parts)


def _validate_skill_name(name: str) -> str:
    return _normalize_bundle_path(name, field_name="skill name", allow_nested=False)


def _validate_category_name(category: str) -> str:
    return _normalize_bundle_path(category, field_name="category", allow_nested=False)


def _validate_bundle_rel_path(rel_path: str) -> str:
    return _normalize_bundle_path(rel_path, field_name="bundle file path", allow_nested=True)


# ---------------------------------------------------------------------------
# GitHub Authentication
# ---------------------------------------------------------------------------

class GitHubAuth:
    """GitHub API authentication. Tries GITHUB_TOKEN env, then anonymous."""

    def __init__(self):
        self._token: Optional[str] = None

    def get_headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/vnd.github.v3+json"}
        token = self._resolve_token()
        if token:
            headers["Authorization"] = f"token {token}"
        return headers

    def is_authenticated(self) -> bool:
        return self._resolve_token() is not None

    def _resolve_token(self) -> Optional[str]:
        if self._token is not None:
            return self._token
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if token:
            self._token = token
            return token
        # Try gh CLI
        import subprocess
        try:
            result = subprocess.run(
                ["gh", "auth", "token"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                self._token = result.stdout.strip()
                return self._token
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return None


# ---------------------------------------------------------------------------
# Source adapter interface
# ---------------------------------------------------------------------------

class SkillSource(ABC):
    """Abstract base for all skill registry adapters."""

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        ...

    @abstractmethod
    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        ...

    @abstractmethod
    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        ...

    @abstractmethod
    def source_id(self) -> str:
        ...

    def trust_level_for(self, identifier: str) -> str:
        return "community"


# ---------------------------------------------------------------------------
# Trusted repos (skills from these repos get "trusted" trust level)
# ---------------------------------------------------------------------------

TRUSTED_REPOS = {
    "shibing624/skills",
    "openai/skills",
    "anthropics/skills",
}


# ---------------------------------------------------------------------------
# GitHub source adapter
# ---------------------------------------------------------------------------

class GitHubSource(SkillSource):
    """Fetch skills from GitHub repos via the Contents API."""

    DEFAULT_TAPS = [
        {"repo": "shibing624/skills", "path": "skills/"},
        {"repo": "openai/skills", "path": "skills/"},
        {"repo": "anthropics/skills", "path": "skills/"},
    ]

    def __init__(self, auth: GitHubAuth, extra_taps: Optional[List[Dict]] = None):
        self.auth = auth
        self.taps = list(self.DEFAULT_TAPS)
        if extra_taps:
            self.taps.extend(extra_taps)

    def source_id(self) -> str:
        return "github"

    def trust_level_for(self, identifier: str) -> str:
        parts = identifier.split("/", 2)
        if len(parts) >= 2:
            repo = f"{parts[0]}/{parts[1]}"
            if repo in TRUSTED_REPOS:
                return "trusted"
        return "community"

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        results: List[SkillMeta] = []
        query_lower = query.lower()
        for tap in self.taps:
            skills = self._list_skills_in_repo(tap["repo"], tap.get("path", ""))
            for skill in skills:
                searchable = f"{skill.name} {skill.description} {' '.join(skill.tags)}".lower()
                if query_lower in searchable:
                    results.append(skill)
        # Deduplicate by name
        seen: Dict[str, SkillMeta] = {}
        for r in results:
            if r.name not in seen:
                seen[r.name] = r
        return list(seen.values())[:limit]

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        parts = identifier.split("/", 2)
        if len(parts) < 3:
            return None
        repo = f"{parts[0]}/{parts[1]}"
        skill_path = parts[2]
        files = self._download_directory(repo, skill_path)
        if not files or "SKILL.md" not in files:
            return None
        skill_name = skill_path.rstrip("/").split("/")[-1]
        return SkillBundle(
            name=skill_name, files=files, source="github",
            identifier=identifier, trust_level=self.trust_level_for(identifier),
        )

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        parts = identifier.split("/", 2)
        if len(parts) < 3:
            return None
        repo = f"{parts[0]}/{parts[1]}"
        skill_path = parts[2].rstrip("/")
        content = self._fetch_file_content(repo, f"{skill_path}/SKILL.md")
        if not content:
            return None
        fm = _parse_frontmatter_quick(content)
        return SkillMeta(
            name=fm.get("name", skill_path.split("/")[-1]),
            description=str(fm.get("description", "")),
            source="github", identifier=identifier,
            trust_level=self.trust_level_for(identifier),
            repo=repo, path=skill_path,
            tags=fm.get("tags", []) if isinstance(fm.get("tags"), list) else [],
        )

    def _list_skills_in_repo(self, repo: str, path: str) -> List[SkillMeta]:
        cache_key = f"gh_{repo}_{path}".replace("/", "_")
        cached = _read_index_cache(cache_key)
        if cached is not None:
            return [SkillMeta(**s) for s in cached]

        url = f"https://api.github.com/repos/{repo}/contents/{path.rstrip('/')}"
        try:
            resp = httpx.get(url, headers=self.auth.get_headers(), timeout=15, follow_redirects=True)
            if resp.status_code != 200:
                return []
        except httpx.HTTPError:
            return []

        entries = resp.json()
        if not isinstance(entries, list):
            return []

        skills: List[SkillMeta] = []
        for entry in entries:
            if entry.get("type") != "dir" or entry["name"].startswith((".", "_")):
                continue
            prefix = path.rstrip("/")
            skill_id = f"{repo}/{prefix}/{entry['name']}" if prefix else f"{repo}/{entry['name']}"
            meta = self.inspect(skill_id)
            if meta:
                skills.append(meta)

        _write_index_cache(cache_key, [_skill_meta_to_dict(s) for s in skills])
        return skills

    def _download_directory(self, repo: str, path: str) -> Dict[str, str]:
        """Download all files from a GitHub directory via Trees API, fallback to Contents API."""
        files = self._download_via_tree(repo, path)
        if files is not None:
            return files
        return self._download_recursive(repo, path)

    def _download_via_tree(self, repo: str, path: str) -> Optional[Dict[str, str]]:
        headers = self.auth.get_headers()
        path = path.rstrip("/")
        try:
            resp = httpx.get(f"https://api.github.com/repos/{repo}", headers=headers, timeout=15)
            if resp.status_code != 200:
                return None
            branch = resp.json().get("default_branch", "main")
            resp = httpx.get(
                f"https://api.github.com/repos/{repo}/git/trees/{branch}",
                params={"recursive": "1"}, headers=headers, timeout=30,
            )
            if resp.status_code != 200 or resp.json().get("truncated"):
                return None
        except httpx.HTTPError:
            return None

        prefix = f"{path}/"
        files: Dict[str, str] = {}
        for item in resp.json().get("tree", []):
            if item.get("type") != "blob" or not item.get("path", "").startswith(prefix):
                continue
            rel_path = item["path"][len(prefix):]
            content = self._fetch_file_content(repo, item["path"])
            if content is not None:
                files[rel_path] = content
        return files if files else None

    def _download_recursive(self, repo: str, path: str) -> Dict[str, str]:
        url = f"https://api.github.com/repos/{repo}/contents/{path.rstrip('/')}"
        try:
            resp = httpx.get(url, headers=self.auth.get_headers(), timeout=15, follow_redirects=True)
            if resp.status_code != 200:
                return {}
        except httpx.HTTPError:
            return {}

        entries = resp.json()
        if not isinstance(entries, list):
            return {}

        files: Dict[str, str] = {}
        for entry in entries:
            name = entry.get("name", "")
            if entry.get("type") == "file":
                content = self._fetch_file_content(repo, entry.get("path", ""))
                if content is not None:
                    files[name] = content
            elif entry.get("type") == "dir":
                for sub_name, sub_content in self._download_recursive(repo, entry.get("path", "")).items():
                    files[f"{name}/{sub_name}"] = sub_content
        return files

    def _fetch_file_content(self, repo: str, path: str) -> Optional[str]:
        url = f"https://api.github.com/repos/{repo}/contents/{path}"
        try:
            resp = httpx.get(
                url, headers={**self.auth.get_headers(), "Accept": "application/vnd.github.v3.raw"},
                timeout=15, follow_redirects=True,
            )
            if resp.status_code == 200:
                return resp.text
        except httpx.HTTPError:
            pass
        return None


# ---------------------------------------------------------------------------
# skills.sh source adapter
# ---------------------------------------------------------------------------

class SkillsShSource(SkillSource):
    """Discover skills via skills.sh, delegate content fetch to GitHub."""

    BASE_URL = "https://skills.sh"
    SEARCH_URL = f"{BASE_URL}/api/search"

    def __init__(self, auth: GitHubAuth):
        self.auth = auth
        self.github = GitHubSource(auth=auth)

    def source_id(self) -> str:
        return "skills-sh"

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        if not query.strip():
            # Empty query: fetch featured skills from homepage
            return self._featured_skills(limit)
        cache_key = f"skills_sh_{hashlib.md5(query.encode()).hexdigest()}"
        cached = _read_index_cache(cache_key)
        if cached is not None:
            return [SkillMeta(**s) for s in cached][:limit]

        try:
            resp = httpx.get(self.SEARCH_URL, params={"q": query, "limit": limit}, timeout=20)
            if resp.status_code != 200:
                return []
            data = resp.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            return []

        items = data.get("skills", []) if isinstance(data, dict) else []
        results: List[SkillMeta] = []
        for item in items[:limit]:
            canonical = item.get("id", "")
            if "/" not in canonical:
                continue
            parts = canonical.split("/", 2)
            if len(parts) < 3:
                continue
            repo = f"{parts[0]}/{parts[1]}"
            skill_path = parts[2]
            results.append(SkillMeta(
                name=str(item.get("name") or skill_path.split("/")[-1]),
                description=f"Indexed by skills.sh from {repo}",
                source="skills-sh", identifier=f"skills-sh/{canonical}",
                trust_level=self.github.trust_level_for(canonical),
                repo=repo, path=skill_path,
                extra={"detail_url": f"{self.BASE_URL}/{canonical}"},
            ))
        _write_index_cache(cache_key, [_skill_meta_to_dict(s) for s in results])
        return results

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        canonical = identifier.removeprefix("skills-sh/")
        # Try direct and standard paths
        for candidate in self._candidate_identifiers(canonical):
            bundle = self.github.fetch(candidate)
            if bundle:
                bundle.source = "skills-sh"
                bundle.identifier = f"skills-sh/{canonical}"
                return bundle
        return None

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        canonical = identifier.removeprefix("skills-sh/")
        for candidate in self._candidate_identifiers(canonical):
            meta = self.github.inspect(candidate)
            if meta:
                meta.source = "skills-sh"
                meta.identifier = f"skills-sh/{canonical}"
                return meta
        return None

    def _featured_skills(self, limit: int) -> List[SkillMeta]:
        """Scrape skills.sh homepage for featured skill links."""
        cached = _read_index_cache("skills_sh_featured")
        if cached is not None:
            return [SkillMeta(**s) for s in cached][:limit]
        try:
            resp = httpx.get(self.BASE_URL, timeout=20)
            if resp.status_code != 200:
                return []
        except httpx.HTTPError:
            return []
        link_re = re.compile(r'href=["\']/((?!agents/|_next/|api/)[^"\'/]+/[^"\'/]+/[^"\'/]+)["\']')
        seen: set = set()
        results: List[SkillMeta] = []
        for match in link_re.finditer(resp.text):
            canonical = match.group(1)
            if canonical in seen:
                continue
            seen.add(canonical)
            parts = canonical.split("/", 2)
            if len(parts) < 3:
                continue
            repo = f"{parts[0]}/{parts[1]}"
            skill_path = parts[2]
            results.append(SkillMeta(
                name=skill_path.split("/")[-1],
                description=f"Featured on skills.sh from {repo}",
                source="skills-sh", identifier=f"skills-sh/{canonical}",
                trust_level=self.github.trust_level_for(canonical),
                repo=repo, path=skill_path,
            ))
            if len(results) >= limit:
                break
        _write_index_cache("skills_sh_featured", [_skill_meta_to_dict(s) for s in results])
        return results

    @staticmethod
    def _candidate_identifiers(identifier: str) -> List[str]:
        parts = identifier.split("/", 2)
        if len(parts) < 3:
            return [identifier]
        repo = f"{parts[0]}/{parts[1]}"
        skill_path = parts[2].lstrip("/")
        return [
            f"{repo}/{skill_path}",
            f"{repo}/skills/{skill_path}",
            f"{repo}/.agentica/skills/{skill_path}",
            f"{repo}/.claude/skills/{skill_path}",
        ]


# ---------------------------------------------------------------------------
# LobeHub source adapter
# ---------------------------------------------------------------------------

class LobeHubSource(SkillSource):
    """Fetch skills from LobeHub's agent marketplace (system prompt templates)."""

    INDEX_URL = "https://chat-agents.lobehub.com/index.json"

    def source_id(self) -> str:
        return "lobehub"

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        index = self._fetch_index()
        if not index:
            return []
        query_lower = query.lower()
        agents = index.get("agents", index) if isinstance(index, dict) else index
        if not isinstance(agents, list):
            return []

        results: List[SkillMeta] = []
        for agent in agents:
            meta = agent.get("meta", agent)
            title = meta.get("title", agent.get("identifier", ""))
            desc = meta.get("description", "")
            tags = meta.get("tags", [])
            searchable = f"{title} {desc} {' '.join(tags) if isinstance(tags, list) else ''}".lower()
            if query_lower in searchable:
                aid = agent.get("identifier", title.lower().replace(" ", "-"))
                results.append(SkillMeta(
                    name=aid, description=desc[:200], source="lobehub",
                    identifier=f"lobehub/{aid}", trust_level="community",
                    tags=tags if isinstance(tags, list) else [],
                ))
            if len(results) >= limit:
                break
        return results

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        agent_id = identifier.removeprefix("lobehub/")
        data = self._fetch_agent(agent_id)
        if not data:
            return None
        skill_md = self._convert_to_skill_md(data)
        return SkillBundle(
            name=agent_id, files={"SKILL.md": skill_md},
            source="lobehub", identifier=f"lobehub/{agent_id}",
            trust_level="community",
        )

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        agent_id = identifier.removeprefix("lobehub/")
        index = self._fetch_index()
        if not index:
            return None
        agents = index.get("agents", index) if isinstance(index, dict) else index
        if not isinstance(agents, list):
            return None
        for agent in agents:
            if agent.get("identifier") == agent_id:
                meta = agent.get("meta", agent)
                return SkillMeta(
                    name=agent_id, description=meta.get("description", ""),
                    source="lobehub", identifier=f"lobehub/{agent_id}",
                    trust_level="community",
                    tags=meta.get("tags", []) if isinstance(meta.get("tags"), list) else [],
                )
        return None

    def _fetch_index(self) -> Optional[Any]:
        cached = _read_index_cache("lobehub_index")
        if cached is not None:
            return cached
        try:
            resp = httpx.get(self.INDEX_URL, timeout=30)
            if resp.status_code != 200:
                return None
            data = resp.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            return None
        _write_index_cache("lobehub_index", data)
        return data

    def _fetch_agent(self, agent_id: str) -> Optional[dict]:
        try:
            resp = httpx.get(f"https://chat-agents.lobehub.com/{agent_id}.json", timeout=15)
            if resp.status_code == 200:
                return resp.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            pass
        return None

    @staticmethod
    def _convert_to_skill_md(agent_data: dict) -> str:
        meta = agent_data.get("meta", agent_data)
        identifier = agent_data.get("identifier", "lobehub-agent")
        title = meta.get("title", identifier)
        description = meta.get("description", "")
        tags = meta.get("tags", [])
        system_role = agent_data.get("config", {}).get("systemRole", "")

        tag_list = tags if isinstance(tags, list) else []
        fm = "\n".join([
            "---",
            f"name: {identifier}",
            f"description: {description[:500]}",
            f"tags: [{', '.join(str(t) for t in tag_list)}]",
            "---",
        ])
        body = "\n".join([
            f"# {title}", "", description, "",
            "## Instructions", "",
            system_role if system_role else "(No system role defined)",
        ])
        return fm + "\n\n" + body + "\n"


# ---------------------------------------------------------------------------
# Security scanning (static pattern check)
# ---------------------------------------------------------------------------

# Patterns that indicate potentially dangerous content in SKILL.md
_DANGEROUS_PATTERNS = [
    re.compile(r"rm\s+-rf\s+/", re.IGNORECASE),
    re.compile(r"curl\s+.*\|\s*(?:bash|sh|zsh)", re.IGNORECASE),
    re.compile(r"wget\s+.*\|\s*(?:bash|sh|zsh)", re.IGNORECASE),
    re.compile(r"eval\s*\(", re.IGNORECASE),
    re.compile(r"exec\s*\(", re.IGNORECASE),
    re.compile(r"__import__\s*\(", re.IGNORECASE),
    re.compile(r"subprocess\.(?:call|run|Popen)", re.IGNORECASE),
    re.compile(r"os\.system\s*\(", re.IGNORECASE),
]

_SUSPICIOUS_PATTERNS = [
    re.compile(r"(?:api[_-]?key|password|secret|token)\s*[=:]\s*['\"]", re.IGNORECASE),
    re.compile(r"chmod\s+[0-7]*777", re.IGNORECASE),
    re.compile(r"/etc/(?:passwd|shadow)", re.IGNORECASE),
    re.compile(r"ssh-keygen|authorized_keys", re.IGNORECASE),
]


@dataclass
class ScanResult:
    """Result of a security scan on a skill."""
    verdict: str  # "clean" | "suspicious" | "dangerous"
    findings: List[Dict[str, str]] = field(default_factory=list)  # [{pattern, file, line}]


def scan_skill(skill_path: Path) -> ScanResult:
    """Static security scan on a skill directory. Checks all text files for dangerous patterns."""
    findings: List[Dict[str, str]] = []
    verdict = "clean"

    for f in skill_path.rglob("*"):
        if not f.is_file() or f.suffix in (".pyc", ".pyo", ".so", ".dll"):
            continue
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        rel = str(f.relative_to(skill_path))
        for pattern in _DANGEROUS_PATTERNS:
            for match in pattern.finditer(content):
                line_num = content[:match.start()].count("\n") + 1
                findings.append({"severity": "dangerous", "pattern": pattern.pattern,
                                 "file": rel, "line": str(line_num)})
                verdict = "dangerous"

        if verdict != "dangerous":
            for pattern in _SUSPICIOUS_PATTERNS:
                for match in pattern.finditer(content):
                    line_num = content[:match.start()].count("\n") + 1
                    findings.append({"severity": "suspicious", "pattern": pattern.pattern,
                                     "file": rel, "line": str(line_num)})
                    if verdict == "clean":
                        verdict = "suspicious"

    return ScanResult(verdict=verdict, findings=findings)


def should_allow_install(result: ScanResult, force: bool = False) -> Tuple[bool, str]:
    """Check if a scan result allows installation."""
    if result.verdict == "dangerous" and not force:
        return False, f"Dangerous patterns found ({len(result.findings)} findings). Use --force to override."
    return True, ""


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _read_index_cache(key: str) -> Optional[Any]:
    cache_file = INDEX_CACHE_DIR / f"{key}.json"
    if not cache_file.exists():
        return None
    try:
        if time.time() - cache_file.stat().st_mtime > INDEX_CACHE_TTL:
            return None
        return json.loads(cache_file.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_index_cache(key: str, data: Any) -> None:
    INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        (INDEX_CACHE_DIR / f"{key}.json").write_text(
            json.dumps(data, ensure_ascii=False, default=str))
    except OSError:
        pass


def _skill_meta_to_dict(meta: SkillMeta) -> dict:
    return {
        "name": meta.name, "description": meta.description,
        "source": meta.source, "identifier": meta.identifier,
        "trust_level": meta.trust_level, "repo": meta.repo,
        "path": meta.path, "tags": meta.tags, "extra": meta.extra,
    }


def _parse_frontmatter_quick(content: str) -> dict:
    """Parse YAML frontmatter from SKILL.md content."""
    if not content.startswith("---"):
        return {}
    match = re.search(r'\n---\s*\n', content[3:])
    if not match:
        return {}
    try:
        import yaml
        parsed = yaml.safe_load(content[3:match.start() + 3])
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def content_hash(path: Path) -> str:
    """Compute deterministic hash of a skill directory."""
    h = hashlib.sha256()
    for f in sorted(path.rglob("*")):
        if f.is_file():
            h.update(f.read_bytes())
    return f"sha256:{h.hexdigest()[:16]}"


# ---------------------------------------------------------------------------
# Lock file management
# ---------------------------------------------------------------------------

class HubLockFile:
    """Manages skills/.hub/lock.json — tracks provenance of installed hub skills."""

    def __init__(self, path: Path = LOCK_FILE):
        self.path = path

    def load(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "installed": {}}
        try:
            return json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return {"version": 1, "installed": {}}

    def save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    def record_install(self, name: str, source: str, identifier: str,
                       trust_level: str, scan_verdict: str, skill_hash: str,
                       install_path: str, files: List[str]) -> None:
        data = self.load()
        data["installed"][name] = {
            "source": source, "identifier": identifier,
            "trust_level": trust_level, "scan_verdict": scan_verdict,
            "content_hash": skill_hash, "install_path": install_path,
            "files": files,
            "installed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.save(data)

    def record_uninstall(self, name: str) -> None:
        data = self.load()
        data["installed"].pop(name, None)
        self.save(data)

    def get_installed(self, name: str) -> Optional[dict]:
        return self.load()["installed"].get(name)

    def list_installed(self) -> List[dict]:
        return [{"name": k, **v} for k, v in self.load()["installed"].items()]


# ---------------------------------------------------------------------------
# Taps management
# ---------------------------------------------------------------------------

class TapsManager:
    """Manages taps.json — custom GitHub repo sources."""

    def __init__(self, path: Path = TAPS_FILE):
        self.path = path

    def load(self) -> List[dict]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text()).get("taps", [])
        except (json.JSONDecodeError, OSError):
            return []

    def save(self, taps: List[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"taps": taps}, indent=2) + "\n")

    def add(self, repo: str, path: str = "skills/") -> bool:
        taps = self.load()
        if any(t["repo"] == repo for t in taps):
            return False
        taps.append({"repo": repo, "path": path})
        self.save(taps)
        return True

    def remove(self, repo: str) -> bool:
        taps = self.load()
        new_taps = [t for t in taps if t["repo"] != repo]
        if len(new_taps) == len(taps):
            return False
        self.save(new_taps)
        return True

    def list_taps(self) -> List[dict]:
        return self.load()


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def append_audit_log(action: str, skill_name: str, source: str,
                     trust_level: str, verdict: str, extra: str = "") -> None:
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    parts = [ts, action, skill_name, f"{source}:{trust_level}", verdict]
    if extra:
        parts.append(extra)
    try:
        with open(AUDIT_LOG, "a") as f:
            f.write(" ".join(parts) + "\n")
    except OSError:
        pass
    except Exception as e:
        pass


# ---------------------------------------------------------------------------
# Hub operations (high-level)
# ---------------------------------------------------------------------------

def ensure_hub_dirs() -> None:
    """Create the .hub directory structure."""
    HUB_DIR.mkdir(parents=True, exist_ok=True)
    QUARANTINE_DIR.mkdir(exist_ok=True)
    INDEX_CACHE_DIR.mkdir(exist_ok=True)


def quarantine_bundle(bundle: SkillBundle) -> Path:
    """Write a skill bundle to the quarantine directory for scanning."""
    ensure_hub_dirs()
    skill_name = _validate_skill_name(bundle.name)
    validated_files: List[Tuple[str, Union[str, bytes]]] = []
    for rel_path, file_content in bundle.files.items():
        safe_path = _validate_bundle_rel_path(rel_path)
        validated_files.append((safe_path, file_content))

    dest = QUARANTINE_DIR / skill_name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    for rel_path, file_content in validated_files:
        file_dest = dest.joinpath(*rel_path.split("/"))
        file_dest.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(file_content, bytes):
            file_dest.write_bytes(file_content)
        else:
            file_dest.write_text(file_content, encoding="utf-8")
    return dest


def install_from_quarantine(quarantine_path: Path, skill_name: str,
                            category: str, bundle: SkillBundle,
                            scan_result: ScanResult) -> Path:
    """Move a scanned skill from quarantine into the skills directory."""
    safe_name = _validate_skill_name(skill_name)
    safe_category = _validate_category_name(category) if category else ""

    # Verify quarantine path is under QUARANTINE_DIR
    if not quarantine_path.resolve().is_relative_to(QUARANTINE_DIR.resolve()):
        raise ValueError(f"Unsafe quarantine path: {quarantine_path}")

    install_dir = SKILLS_DIR / safe_category / safe_name if safe_category else SKILLS_DIR / safe_name
    if install_dir.exists():
        shutil.rmtree(install_dir)

    install_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(quarantine_path), str(install_dir))

    # Record in lock file
    lock = HubLockFile()
    lock.record_install(
        name=safe_name, source=bundle.source, identifier=bundle.identifier,
        trust_level=bundle.trust_level, scan_verdict=scan_result.verdict,
        skill_hash=content_hash(install_dir),
        install_path=str(install_dir.relative_to(SKILLS_DIR)),
        files=list(bundle.files.keys()),
    )
    append_audit_log("INSTALL", safe_name, bundle.source,
                     bundle.trust_level, scan_result.verdict,
                     content_hash(install_dir))
    return install_dir


def uninstall_skill(skill_name: str) -> Tuple[bool, str]:
    """Remove a hub-installed skill."""
    lock = HubLockFile()
    entry = lock.get_installed(skill_name)
    if not entry:
        return False, f"'{skill_name}' is not a hub-installed skill"

    install_path = SKILLS_DIR / entry["install_path"]
    if install_path.exists():
        shutil.rmtree(install_path)

    lock.record_uninstall(skill_name)
    append_audit_log("UNINSTALL", skill_name, entry["source"],
                     entry["trust_level"], "n/a", "user_request")
    return True, f"Uninstalled '{skill_name}' from {entry['install_path']}"


# ---------------------------------------------------------------------------
# Hub install (full pipeline: fetch -> quarantine -> scan -> install)
# ---------------------------------------------------------------------------

def hub_install(identifier: str, category: str = "", force: bool = False,
                sources: Optional[List[SkillSource]] = None) -> Tuple[bool, str]:
    """Full install pipeline: fetch, quarantine, scan, install.

    Returns (success, message).
    """
    if sources is None:
        sources = create_source_router()

    # Short name resolution
    if "/" not in identifier:
        resolved = resolve_short_name(identifier, sources)
        if not resolved:
            # Show all candidates with this name for debugging
            all_results = unified_search(identifier, sources, limit=50, deduplicate=False)
            exact = [r for r in all_results if r.name.lower() == identifier.lower()]
            if exact:
                lines = [f"Multiple sources have '{identifier}', specify full identifier:"]
                for r in exact:
                    lines.append(f"  /skills install {r.identifier}")
                return False, "\n".join(lines)
            return False, f"No skill named '{identifier}' found in any source."
        identifier = resolved

    # Fetch bundle — try all sources
    bundle = None
    for src in sources:
        bundle = src.fetch(identifier)
        if bundle:
            break
    if not bundle:
        return False, (f"Could not fetch '{identifier}' from any source.\n"
                       f"  Check that the repo exists and contains SKILL.md.")

    # Check if already installed
    lock = HubLockFile()
    existing = lock.get_installed(bundle.name)
    if existing and not force:
        return False, f"'{bundle.name}' is already installed. Use --force to reinstall."

    # Quarantine
    q_path = quarantine_bundle(bundle)

    # Security scan
    result = scan_skill(q_path)
    allowed, reason = should_allow_install(result, force=force)
    if not allowed:
        shutil.rmtree(q_path, ignore_errors=True)
        append_audit_log("BLOCKED", bundle.name, bundle.source,
                         bundle.trust_level, result.verdict, reason)
        return False, f"Installation blocked: {reason}"

    # Install
    install_dir = install_from_quarantine(q_path, bundle.name, category, bundle, result)
    return True, (f"Installed '{bundle.name}' (user-level)\n"
                  f"  Path: {install_dir}\n"
                  f"  Source: {bundle.source} ({bundle.identifier})\n"
                  f"  Files: {', '.join(bundle.files.keys())}")


def resolve_short_name(name: str, sources: List[SkillSource]) -> str:
    """Resolve a short skill name to a full identifier by searching all sources.

    Priority: trusted > community. If multiple trusted, pick first.
    """
    results = unified_search(name, sources, limit=50, deduplicate=False)
    exact = [r for r in results if r.name.lower() == name.lower()]
    if not exact:
        return ""
    if len(exact) == 1:
        return exact[0].identifier
    # Multiple exact matches: prefer trusted
    trusted = [r for r in exact if r.trust_level == "trusted"]
    if trusted:
        return trusted[0].identifier
    # All community: pick first
    return exact[0].identifier


# ---------------------------------------------------------------------------
# Source router + unified search
# ---------------------------------------------------------------------------

def create_source_router(auth: Optional[GitHubAuth] = None) -> List[SkillSource]:
    """Create all configured source adapters."""
    if auth is None:
        auth = GitHubAuth()
    taps_mgr = TapsManager()
    extra_taps = taps_mgr.list_taps()

    return [
        SkillsShSource(auth=auth),
        GitHubSource(auth=auth, extra_taps=extra_taps),
        LobeHubSource(),
    ]


def unified_search(query: str, sources: Optional[List[SkillSource]] = None,
                   source_filter: str = "all", limit: int = 10,
                   deduplicate: bool = True) -> List[SkillMeta]:
    """Search all sources and merge results."""
    if sources is None:
        sources = create_source_router()

    all_results: List[SkillMeta] = []
    for src in sources:
        if source_filter != "all" and src.source_id() != source_filter:
            continue
        try:
            all_results.extend(src.search(query, limit=limit))
        except Exception as e:
            logger.debug(f"Search failed for {src.source_id()}: {e}")

    if not deduplicate:
        return all_results[:limit]

    # Deduplicate by name, preferring higher trust levels
    _TRUST_RANK = {"trusted": 2, "community": 1}
    seen: Dict[str, SkillMeta] = {}
    for r in all_results:
        if r.name not in seen:
            seen[r.name] = r
        elif _TRUST_RANK.get(r.trust_level, 0) > _TRUST_RANK.get(seen[r.name].trust_level, 0):
            seen[r.name] = r

    return list(seen.values())[:limit]
