# -*- coding: utf-8 -*-
"""Wrap Claude Code / Codex CLI so Polyglot can PK them with the same judge.

We own wall-clock, pytest, crash/timeout, honesty, and collateral.
API/token/cost come from the CLI's own JSON (headless mode). Tool-call
counts are best-effort: Codex jsonl has item events; Claude's result JSON
only exposes num_turns (mapped to api_calls).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agentica.model.usage import split_prompt_usage

from .execute import run_command
from .metrics import empty_metrics

_CODEX_TOOL_ITEMS = {
    "command_execution",
    "mcp_tool_call",
    "file_change",
    "web_search",
    "web_fetch",
}


def metrics_from_claude_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Map `claude --print --output-format json` result object onto eval metrics."""
    metrics = empty_metrics()
    usage = payload.get("usage") or {}
    model_usage = payload.get("modelUsage") or {}
    fresh = int(usage.get("input_tokens") or 0)
    cached = int(usage.get("cache_read_input_tokens") or 0)
    write = int(usage.get("cache_creation_input_tokens") or 0)
    output = int(usage.get("output_tokens") or 0)
    cost = float(payload.get("total_cost_usd") or 0)
    if fresh + cached + write + output == 0:
        for stat in model_usage.values():
            fresh += int(stat.get("inputTokens") or 0)
            cached += int(stat.get("cacheReadInputTokens") or 0)
            write += int(stat.get("cacheCreationInputTokens") or 0)
            output += int(stat.get("outputTokens") or 0)
            cost += float(stat.get("costUSD") or 0)
        if payload.get("total_cost_usd") is not None:
            cost = float(payload["total_cost_usd"])
    prompt = fresh + cached + write
    turns = payload.get("num_turns")
    models = list(model_usage)
    metrics.update(
        {
            "model": models[0] if models else "",
            "api_calls": int(turns) if turns is not None else None,
            "tool_calls": None,
            "input_tokens": prompt,
            "fresh_input_tokens": fresh,
            "cached_input_tokens": cached,
            "cache_write_tokens": write,
            "output_tokens": output,
            "cache_hit_rate": round(cached / prompt, 4) if prompt else None,
            "cost_usd": round(cost, 6),
            "aborted": bool(payload.get("is_error")),
            "abort_reason": ",".join(payload.get("errors") or []) or str(payload.get("subtype") or ""),
        }
    )
    return metrics


def metrics_from_codex_jsonl(text: str) -> Tuple[str, Dict[str, Any]]:
    """Map `codex exec --json` JSONL onto (last agent message, metrics)."""
    metrics = empty_metrics()
    api_calls = 0
    tool_calls = 0
    last_text = ""
    input_tokens = 0
    cached = 0
    output = 0
    failed = False
    abort_reason = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        event = json.loads(line)
        kind = event.get("type") or ""
        if kind == "turn.completed":
            api_calls += 1
            usage = event.get("usage") or {}
            input_tokens += int(usage.get("input_tokens") or 0)
            cached += int(usage.get("cached_input_tokens") or 0)
            output += int(usage.get("output_tokens") or 0)
        elif kind == "turn.failed":
            failed = True
            abort_reason = str(event.get("error") or "turn.failed")
        elif kind == "item.completed":
            item = event.get("item") or {}
            item_type = item.get("type") or ""
            if item_type in _CODEX_TOOL_ITEMS:
                tool_calls += 1
            if item_type == "agent_message":
                last_text = item.get("text") or last_text
        elif kind == "error":
            failed = True
            abort_reason = str(event.get("message") or event.get("error") or "error")
    fresh, read, write = split_prompt_usage(
        input_tokens,
        {"cached_tokens": cached},
    )
    prompt = fresh + read + write
    metrics.update(
        {
            "model": "codex",
            "api_calls": api_calls,
            "tool_calls": tool_calls,
            "input_tokens": prompt,
            "fresh_input_tokens": fresh,
            "cached_input_tokens": read,
            "cache_write_tokens": write,
            "output_tokens": output,
            "cache_hit_rate": round(read / prompt, 4) if prompt else None,
            "aborted": failed,
            "abort_reason": abort_reason,
        }
    )
    return last_text, metrics


def resolve_codex_reasoning_effort(extra_body: Optional[Dict[str, Any]]) -> str:
    """Map --extra-body onto Codex ``model_reasoning_effort``.

    Some OpenAI-compatible Responses endpoints use
    ``{"reasoning": {"effort": "none"}}`` to turn thinking
    off. Chat Completions leftover keys (``reasoning_effort``,
    ``thinking_enabled: false``) are accepted so the same flag works both ways.
    Missing extra_body still defaults to ``high`` (Codex CLI's own default).
    """
    if not extra_body:
        return "high"
    reasoning = extra_body.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort") not in (None, ""):
        return str(reasoning["effort"])
    if extra_body.get("reasoning_effort") not in (None, ""):
        return str(extra_body["reasoning_effort"])
    if extra_body.get("thinking_enabled") is False:
        return "none"
    return "high"


def write_isolated_codex_home(
    root: Path,
    *,
    model_id: str,
    base_url: str,
    reasoning_effort: str = "high",
) -> Path:
    """CODEX_HOME for one eval run. Codex only speaks Responses; the proxy must too."""
    home = Path(root)
    home.mkdir(parents=True, exist_ok=True)
    config = (
        f'model = "{model_id}"\n'
        'model_provider = "gateway"\n'
        f'model_reasoning_effort = "{reasoning_effort}"\n'
        "\n"
        "[model_providers.gateway]\n"
        'name = "Gateway"\n'
        f'base_url = "{base_url.rstrip("/")}"\n'
        'env_key = "OPENAI_API_KEY"\n'
        'wire_api = "responses"\n'
        "requires_openai_auth = false\n"
        "stream_idle_timeout_ms = 600000\n"
    )
    (home / "config.toml").write_text(config, encoding="utf-8")
    return home


def anthropic_base_url(openai_compatible_base: str) -> str:
    """Map an OpenAI-compatible ``.../v1`` root to an Anthropic SDK base.

    Claude Code appends ``/v1/messages``. Leaving the client on ``.../v1``
    would request ``.../v1/v1/messages``. Some proxies mount Anthropic under
    ``.../anthropic`` when the OpenAI root is ``.../v1``.
    """
    base = openai_compatible_base.rstrip("/")
    if base.endswith("/anthropic"):
        return base
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    if base.endswith("/llmproxy"):
        return base + "/anthropic"
    return base


def claude_env(
    *,
    config_dir: Path,
    base_url: str,
    api_key: str,
    model_id: str,
) -> Dict[str, str]:
    """Isolated Claude Code env pointed at an Anthropic /v1/messages proxy."""
    config_dir.mkdir(parents=True, exist_ok=True)
    return {
        "CLAUDE_CONFIG_DIR": str(config_dir),
        "ANTHROPIC_BASE_URL": anthropic_base_url(base_url),
        "ANTHROPIC_API_KEY": api_key,
        # Some proxies reject x-api-key and accept Authorization: Bearer.
        # Claude --bare still requires ANTHROPIC_API_KEY to exist; AUTH_TOKEN
        # is what actually goes on the wire as Bearer.
        "ANTHROPIC_AUTH_TOKEN": api_key,
        "ANTHROPIC_MODEL": model_id,
    }


def claude_command(prompt: str, *, model: str = "", effort: str = "") -> List[str]:
    command = [
        "claude",
        "--print",
        "--output-format",
        "json",
        "--dangerously-skip-permissions",
        "--bare",
    ]
    if model:
        command.extend(["--model", model])
    if effort:
        command.extend(["--effort", effort])
    command.append(prompt)
    return command


def codex_command(prompt: str, *, model: str = "") -> List[str]:
    command = [
        "codex",
        "exec",
        "--json",
        "--sandbox",
        "workspace-write",
        "-c",
        "approval_policy=never",
        "--ephemeral",
        "--skip-git-repo-check",
    ]
    if model:
        command.extend(["-m", model])
    command.append(prompt)
    return command


def run_cli_agent(
    kind: str,
    work: Path,
    prompt: str,
    timeout: int,
    *,
    model: str = "",
    env: Optional[Dict[str, str]] = None,
) -> Tuple[str, Dict[str, Any], bool, str]:
    """Run one headless CLI turn in ``work``. Returns prediction, metrics, crashed, abort_reason."""
    if kind == "claude":
        binary, command = "claude", claude_command(
            prompt,
            model=model,
            effort=str((env or {}).get("CODE_BENCH_CLAUDE_EFFORT") or ""),
        )
    elif kind == "codex":
        binary, command = "codex", codex_command(prompt, model=model)
    else:
        raise ValueError(f"unknown CLI agent {kind!r}")
    if shutil.which(binary) is None:
        metrics = empty_metrics()
        metrics["model"] = kind
        metrics["tool_calls"] = None
        metrics["api_calls"] = None
        metrics["crashed"] = True
        metrics["aborted"] = True
        metrics["abort_reason"] = f"{binary}_not_found"
        return "", metrics, True, f"{binary}_not_found"

    merged = os.environ.copy()
    if env:
        merged.update(env)
    result = run_command(
        command,
        cwd=work,
        timeout=timeout,
        stdin="",
        env=merged,
        combine_output=False,
        max_output=0,
    )
    if result.timed_out:
        metrics = empty_metrics()
        metrics["model"] = kind
        metrics["tool_calls"] = None
        metrics["api_calls"] = None
        metrics["timed_out"] = True
        metrics["aborted"] = True
        metrics["abort_reason"] = "timeout"
        return result.output[-2000:], metrics, False, "timeout"

    try:
        if kind == "claude":
            payload = json.loads(result.output)
            metrics = metrics_from_claude_result(payload)
            text = payload.get("result") or ""
            crashed = bool(payload.get("is_error"))
            abort = metrics.get("abort_reason") or ""
            return text, metrics, crashed, abort
        if not result.ok and not (result.output or "").strip():
            metrics = empty_metrics()
            metrics["model"] = model or kind
            metrics["tool_calls"] = None
            metrics["api_calls"] = None
            metrics["crashed"] = True
            metrics["aborted"] = True
            tail = (result.stderr or "").strip().splitlines()
            reason = tail[-1][:200] if tail else f"exit_{result.returncode}"
            metrics["abort_reason"] = reason
            return (result.stderr or "")[-2000:], metrics, True, reason
        text, metrics = metrics_from_codex_jsonl(result.output)
        if model:
            metrics["model"] = model
        crashed = bool(metrics.get("aborted")) or (not result.ok)
        return text, metrics, crashed, str(metrics.get("abort_reason") or "")
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        metrics = empty_metrics()
        metrics["model"] = kind
        metrics["tool_calls"] = None
        metrics["api_calls"] = None
        metrics["crashed"] = True
        metrics["aborted"] = True
        metrics["abort_reason"] = type(exc).__name__
        detail = (result.stderr or result.output or str(exc))[-2000:]
        return detail, metrics, True, type(exc).__name__
