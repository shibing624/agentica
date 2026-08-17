# -*- coding: utf-8 -*-
"""Shared paths, model/agent factory, result IO, and code extraction."""
from __future__ import annotations

import json
import os
import re
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from urllib.request import urlopen

from .metrics import aggregate_metrics, empty_metrics, format_summary_metrics, usage_from_request_entries

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
CACHE_DIR = HERE / ".cache"
OUTPUT_DIR = HERE / "outputs"

INSTRUCTIONS_ADDENDUM = """
####

Use the above instructions to modify the supplied files: {file_list}
Don't change the names of existing functions or classes, as they may be referenced from other code like unit tests, etc.
Only use standard libraries, don't suggest installing any packages.

Run tests from this directory with:
python -m pytest --rootdir=. --noconftest -q
Do not run pytest from a parent directory; this tree sits inside another Python repo.
"""

TEST_FAILURES = """
####

See the testing errors above.
The tests are correct, don't try and change them.
Fix the code in {file_list} to resolve the errors.
"""


@dataclass
class SampleResult:
    bench: str
    task_id: str
    passed: bool
    duration_s: float
    error: str = ""
    prediction: str = ""
    tries: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=empty_metrics)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_path(*parts: str) -> Path:
    return ensure_dir(CACHE_DIR).joinpath(*parts)


def iter_jsonl_url(url: str, *, limit: int = 0, timeout: int = 60):
    """Yield JSON objects from a remote jsonl, stopping after ``limit`` rows (0 = all)."""
    import io

    with urlopen(url, timeout=timeout) as resp:
        for line in io.TextIOWrapper(resp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            if limit and limit > 0:
                limit -= 1
                if limit <= 0:
                    return


def download_file(url: str, dest: Path, timeout: int = 60) -> Path:
    """Download url to dest unless dest already exists and is non-empty."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    ensure_dir(dest.parent)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urlopen(url, timeout=timeout) as resp, tmp.open("wb") as fh:
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            fh.write(chunk)
    tmp.replace(dest)
    return dest


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_code(text: str) -> str:
    """Pull the last fenced Python block, else the whole response."""
    if not text:
        return ""
    fences = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fences:
        return fences[-1].strip()
    return text.strip()


def summarize(bench: str, model: str, results: List[SampleResult], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    agg = aggregate_metrics(results)
    payload = {
        "bench": bench,
        "model": model or agg.get("model") or "",
        "total": total,
        "passed": passed,
        "accuracy": round(100.0 * passed / total, 2) if total else 0.0,
        "avg_duration_s": agg.get("avg_wall_clock_s", 0.0),
        "metrics": agg,
        "tasks": agg.get("tasks") or [],
        "results": [r.to_dict() for r in results],
    }
    if extra:
        payload.update(extra)
    return payload


def print_summary(payload: Dict[str, Any]) -> None:
    print("=" * 60)
    print(f"bench   : {payload['bench']}")
    print(f"model   : {payload['model']}")
    print(f"score   : {payload['passed']}/{payload['total']}  ({payload['accuracy']}%)")
    print(f"avg time: {payload['avg_duration_s']}s")
    metrics = payload.get("metrics") or {}
    if metrics:
        print(format_summary_metrics(metrics))
    print("=" * 60)


@contextmanager
def isolated_home(root: Path) -> Iterator[Path]:
    """Point AGENTICA_HOME at a throwaway dir so eval never writes ~/.agentica."""
    home = ensure_dir(root)
    previous = os.environ.get("AGENTICA_HOME")
    os.environ["AGENTICA_HOME"] = str(home)
    try:
        yield home
    finally:
        if previous is None:
            os.environ.pop("AGENTICA_HOME", None)
        else:
            os.environ["AGENTICA_HOME"] = previous


def build_model(
    model_id: str,
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
):
    from agentica import OpenAIChat

    kwargs: Dict[str, Any] = {"id": model_id}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    if extra_body:
        kwargs["extra_body"] = extra_body
    if timeout is not None:
        kwargs["timeout"] = timeout
    return OpenAIChat(**kwargs)


def build_coding_agent(model, work_dir: Path, isolated_home_dir: Path, *, tool_call_limit: int = 40):
    """DeepAgent with file + execute only — the Polyglot / agent-loop surface."""
    from agentica import DeepAgent, Workspace
    from agentica.agent.config import PromptConfig, ToolConfig, WorkspaceMemoryConfig

    workspace = Workspace(str(isolated_home_dir / "workspace"), user_id="eval")
    return DeepAgent(
        model=model,
        work_dir=str(work_dir),
        workspace=workspace,
        user_id="eval",
        include_file_tools=True,
        include_execute=True,
        include_web_search=False,
        include_fetch_url=False,
        include_todos=True,
        include_task=False,
        include_skills=False,
        include_ask_user_question=False,
        enable_long_term_memory=False,
        enable_experience_capture=False,
        enable_session_log=False,
        permission_mode="allow-all",
        long_term_memory_config=WorkspaceMemoryConfig(
            auto_archive=False,
            auto_extract_memory=False,
            load_workspace_context=False,
            load_workspace_memory=False,
        ),
        tool_config=ToolConfig(
            auto_load_mcp=False,
            permission_mode="allow-all",
            tool_call_limit=tool_call_limit,
        ),
        prompt_config=PromptConfig(
            markdown=True,
            enable_agentic_prompt=True,
            add_datetime_to_instructions=False,
            add_name_to_instructions=False,
        ),
    )


async def generate_text(model, prompt: str):
    from agentica import Message

    usage_obj = model.usage
    before = len(usage_obj.request_usage_entries)
    response = await model.response([Message(role="user", content=prompt)])
    entries = usage_obj.request_usage_entries[before:]
    metrics = usage_from_request_entries(entries, model_id=model.id or "")
    return response.content or "", metrics


def slice_items(items: List[Any], max_samples: int, offset: int = 0) -> List[Any]:
    if offset:
        items = items[offset:]
    if max_samples and max_samples > 0:
        return items[:max_samples]
    return items


def now_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S")
