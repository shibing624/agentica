# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: CostTracker — per-run LLM cost accounting.

Tracks token usage and estimates USD cost for every API call within
a single agent run.  The instance is attached to RunResponse.cost_tracker
and can be printed via RunResponse.cost_summary.

Pricing is loaded at first use from a local cache of models.dev/api.json,
refreshed every 24 hours.  Falls back to a hardcoded table when offline.

Usage::

    response = agent.run("...")
    print(response.cost_summary)
    print(f"total: ${response.total_cost_usd:.4f}")
"""
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import httpx

from agentica.config import AGENTICA_CACHE_DIR
from agentica.utils.log import logger

# ---------------------------------------------------------------------------
# Hardcoded fallback — used when models.dev is unreachable.
# USD per 1 M tokens. Synchronized with models.dev on 2026-07-23;
# entries removed from the live catalog retain their last maintained values.
# ---------------------------------------------------------------------------
_FALLBACK_PRICING: Dict[str, Dict[str, Any]] = {
    # ---- OpenAI (official) ----
    "gpt-5": {"input": 1.25, "output": 10.0, "cache_read": 0.125, "cache_write": 0.0, "context_window": 400000, "input_modalities": ('text', 'image')},
    "gpt-5-mini": {"input": 0.25, "output": 2.0, "cache_read": 0.025, "cache_write": 0.0, "context_window": 400000, "input_modalities": ('text', 'image')},
    "gpt-5.1": {"input": 1.25, "output": 10.0, "cache_read": 0.125, "cache_write": 0.0, "context_window": 400000, "input_modalities": ('text', 'image')},
    "gpt-5.2": {"input": 1.75, "output": 14.0, "cache_read": 0.175, "cache_write": 0.0, "context_window": 400000, "input_modalities": ('text', 'image')},
    "gpt-4.1": {"input": 2.0, "output": 8.0, "cache_read": 0.5, "cache_write": 0.0, "context_window": 1047576, "input_modalities": ('text', 'image', 'pdf')},
    "gpt-4.1-mini": {"input": 0.4, "output": 1.6, "cache_read": 0.1, "cache_write": 0.0, "context_window": 1047576, "input_modalities": ('text', 'image', 'pdf')},
    "gpt-4.1-nano": {"input": 0.1, "output": 0.4, "cache_read": 0.025, "cache_write": 0.0, "context_window": 1047576, "input_modalities": ('text', 'image')},
    "gpt-4o": {"input": 2.5, "output": 10.0, "cache_read": 1.25, "cache_write": 0.0, "context_window": 128000, "input_modalities": ('text', 'image', 'pdf')},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6, "cache_read": 0.075, "cache_write": 0.0, "context_window": 128000, "input_modalities": ('text', 'image', 'pdf')},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0, "cache_read": 0.0, "cache_write": 0.0, "context_window": 128000, "input_modalities": ('text', 'image')},
    "gpt-4": {"input": 30.0, "output": 60.0, "cache_read": 0.0, "cache_write": 0.0, "context_window": 8192, "input_modalities": ('text',)},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5, "cache_read": 0.0, "cache_write": 0.0, "context_window": 16385, "input_modalities": ('text',)},
    "o4-mini": {"input": 1.1, "output": 4.4, "cache_read": 0.275, "cache_write": 0.0, "context_window": 200000, "input_modalities": ('text', 'image')},
    "o3": {"input": 2.0, "output": 8.0, "cache_read": 0.5, "cache_write": 0.0, "context_window": 200000, "input_modalities": ('text', 'image', 'pdf')},
    "o3-mini": {"input": 1.1, "output": 4.4, "cache_read": 0.55, "cache_write": 0.0, "context_window": 200000, "input_modalities": ('text',)},
    "o1": {"input": 15.0, "output": 60.0, "cache_read": 7.5, "cache_write": 0.0, "context_window": 200000, "input_modalities": ('text', 'image', 'pdf')},
    "o1-mini": {"input": 1.1, "output": 4.4, "cache_read": 0.55, "cache_write": 0.0, "context_window": 128000, "input_modalities": ('text',)},
    # ---- Anthropic (official) ----
    "claude-opus-4-6": {"input": 5.0, "output": 25.0, "cache_read": 0.5, "cache_write": 6.25, "context_window": 1000000, "input_modalities": ('text', 'image', 'pdf')},
    "claude-opus-4-5": {"input": 5.0, "output": 25.0, "cache_read": 0.5, "cache_write": 6.25, "context_window": 200000, "input_modalities": ('text', 'image', 'pdf')},
    "claude-opus-4": {"input": 15.0, "output": 75.0, "cache_read": 1.5, "cache_write": 18.75, "context_window": 200000, "input_modalities": ('text', 'image')},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75, "context_window": 1000000, "input_modalities": ('text', 'image', 'pdf')},
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75, "context_window": 1000000, "input_modalities": ('text', 'image', 'pdf')},
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0, "cache_read": 0.0, "cache_write": 0.0, "context_window": 200000, "input_modalities": ('text', 'image', 'pdf')},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0, "cache_read": 0.1, "cache_write": 1.25, "context_window": 200000, "input_modalities": ('text', 'image', 'pdf')},
    "claude-haiku-3-5":         {"input": 0.80,  "output": 4.00,  "cache_read": 0.08, "cache_write": 1.00,  "context_window": 200000, "input_modalities": ("text", "image", "pdf")},
    "claude-3-5-sonnet":        {"input": 3.00,  "output": 15.00, "cache_read": 0.30, "cache_write": 3.75,  "context_window": 200000, "input_modalities": ("text", "image")},
    "claude-3-haiku":           {"input": 0.25,  "output": 1.25,  "cache_read": 0.03, "cache_write": 0.30,  "context_window": 200000, "input_modalities": ("text", "image")},
    # ---- Google Gemini (official) ----
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0, "cache_read": 0.125, "cache_write": 0.0, "context_window": 1048576, "input_modalities": ('text', 'image', 'audio', 'video', 'pdf')},
    "gemini-2.5-flash": {"input": 0.3, "output": 2.5, "cache_read": 0.03, "cache_write": 0.0, "context_window": 1048576, "input_modalities": ('text', 'image', 'audio', 'video', 'pdf')},
    "gemini-2.5-flash-lite": {"input": 0.1, "output": 0.4, "cache_read": 0.01, "cache_write": 0.0, "context_window": 1048576, "input_modalities": ('text', 'image', 'audio', 'video', 'pdf')},
    "gemini-2.0-flash": {"input": 0.1, "output": 0.4, "cache_read": 0.025, "cache_write": 0.0, "context_window": 1048576, "input_modalities": ('text', 'image', 'audio', 'video', 'pdf')},
    "gemini-1.5-pro":       {"input": 1.25,  "output": 5.00,  "cache_read": 0.312, "cache_write": 0.00, "context_window": 1000000, "input_modalities": ("text", "image", "audio", "video", "pdf")},
    "gemini-1.5-flash":     {"input": 0.075, "output": 0.30,  "cache_read": 0.019, "cache_write": 0.00, "context_window": 1000000, "input_modalities": ("text", "image", "audio", "video", "pdf")},
    # ---- DeepSeek (official, CNY list price converted to USD at ~7.2 CNY/USD) ----
    "deepseek-v4-flash": {"input": 0.14, "output": 0.28, "cache_read": 0.0028, "cache_write": 0.0, "context_window": 1000000, "input_modalities": ('text',)},
    "deepseek-v4-pro": {"input": 0.435, "output": 0.87, "cache_read": 0.003625, "cache_write": 0.0, "context_window": 1000000, "input_modalities": ('text',)},
    "deepseek-chat": {"input": 0.14, "output": 0.28, "cache_read": 0.0028, "cache_write": 0.0, "context_window": 1000000, "input_modalities": ('text',)},
    "deepseek-reasoner": {"input": 0.14, "output": 0.28, "cache_read": 0.0028, "cache_write": 0.0, "context_window": 1000000, "input_modalities": ('text',)},
    # ---- ZhipuAI (official) ----
    "glm-5": {"input": 1.0, "output": 3.2, "cache_read": 0.2, "cache_write": 0.0, "context_window": 204800, "input_modalities": ('text',)},
    "glm-5.1": {"input": 1.4, "output": 4.4, "cache_read": 0.26, "cache_write": 0.0, "context_window": 200000, "input_modalities": ('text',)},
    "glm-4.7-flash": {"input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0, "context_window": 200000, "input_modalities": ('text',)},
    "glm-4-plus": {"input": 7.497, "output": 7.497, "cache_read": 0.0, "cache_write": 0.0, "context_window": 128000, "input_modalities": ('text',)},
    "glm-4-flash": {"input": 0.1003, "output": 0.1003, "cache_read": 0.0, "cache_write": 0.0, "context_window": 128000, "input_modalities": ('text',)},
    "glm-4-air": {"input": 0.2006, "output": 0.2006, "cache_read": 0.0, "cache_write": 0.0, "context_window": 128000, "input_modalities": ('text',)},
    "glm-4": {"input": 14.994, "output": 14.994, "cache_read": 0.0, "cache_write": 0.0, "context_window": 128000, "input_modalities": ('text',)},
    # ---- Kimi / Moonshot ----
    "kimi-k2.5": {"input": 0.6, "output": 3.0, "cache_read": 0.1, "cache_write": 0.0, "context_window": 256000, "input_modalities": ('text', 'image')},
    "moonshot-v1-8k":       {"input": 0.18,  "output": 0.18,  "cache_read": 0.00,  "cache_write": 0.00, "context_window": 8192, "input_modalities": ("text",)},
    "moonshot-v1-32k":      {"input": 0.35,  "output": 0.35,  "cache_read": 0.00,  "cache_write": 0.00, "context_window": 32768, "input_modalities": ("text",)},
    # ---- MiniMax ----
    "minimax-m2.7": {"input": 0.3, "output": 1.2, "cache_read": 0.06, "cache_write": 0.375, "context_window": 204800, "input_modalities": ('text',)},
    "minimax-m2.5": {"input": 0.11, "output": 0.95, "cache_read": 0.02, "cache_write": 0.375, "context_window": 204800, "input_modalities": ('text',)},
    "minimax-m2": {"input": 0.17, "output": 1.53, "cache_read": 0.0, "cache_write": 0.0, "context_window": 200000, "input_modalities": ('text',)},
    # ---- Qwen (Alibaba) ----
    "qwen-turbo": {"input": 0.05, "output": 0.2, "cache_read": 0.0, "cache_write": 0.0, "context_window": 1000000, "input_modalities": ('text',)},
    "qwen-plus": {"input": 0.4, "output": 1.2, "cache_read": 0.0, "cache_write": 0.0, "context_window": 1000000, "input_modalities": ('text',)},
    "qwen-max": {"input": 1.6, "output": 6.4, "cache_read": 0.0, "cache_write": 0.0, "context_window": 32768, "input_modalities": ('text',)},
    # ---- Ark (Volcengine) — Doubao / Seed family from ByteDance ----
    "doubao-seed-2.0-pro": {"input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0, "context_window": 256000, "input_modalities": ('text', 'image', 'video')},
    "doubao-seed-2.0-lite": {"input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0, "context_window": 256000, "input_modalities": ('text', 'image', 'video')},
    "doubao-seed-1.8":      {"input": 0.11,  "output": 0.29,  "cache_read": 0.00,  "cache_write": 0.00, "context_window": 224000, "input_modalities": ("text", "image", "video")},
    "doubao-pro-4k":        {"input": 0.11,  "output": 0.32,  "cache_read": 0.00,  "cache_write": 0.00, "context_window": 4096, "input_modalities": ("text",)},
    "doubao-lite-4k":       {"input": 0.04,  "output": 0.08,  "cache_read": 0.00,  "cache_write": 0.00, "context_window": 4096, "input_modalities": ("text",)},
    # ---- Yi ----
    "yi-lightning": {"input": 0.2006, "output": 0.2006, "cache_read": 0.0, "cache_write": 0.0, "context_window": 12000, "input_modalities": ('text',)},
    # ---- Groq (fast inference) ----
    "llama3-70b-8192":      {"input": 0.59,  "output": 0.79,  "cache_read": 0.00,  "cache_write": 0.00, "context_window": 8192, "input_modalities": ("text",)},
    "mixtral-8x7b-32768":   {"input": 0.27,  "output": 0.27,  "cache_read": 0.00,  "cache_write": 0.00, "context_window": 32768, "input_modalities": ("text",)},
    # ---- Mistral ----
    "mistral-large":        {"input": 2.00,  "output": 6.00,  "cache_read": 0.00,  "cache_write": 0.00, "context_window": 128000, "input_modalities": ("text",)},
    "mistral-small": {"input": 0.075, "output": 0.2, "cache_read": 0.0, "cache_write": 0.0, "context_window": 128000, "input_modalities": ('text', 'image')},
}

# ---------------------------------------------------------------------------
# models.dev catalog: fetch, cache, parse
# ---------------------------------------------------------------------------
_CATALOG_URL = "https://models.dev/api.json"
_CACHE_TTL = 86400  # 24 hours
_CACHE_SCHEMA_VERSION = 2

# Official providers whose values take priority over third-party resellers.
# model_prefix -> official provider_id
_OFFICIAL_PROVIDERS: Dict[str, str] = {
    "gpt-": "openai", "o1": "openai", "o3": "openai", "o4": "openai",
    "claude-": "anthropic",
    "glm-": "zhipuai",
    "deepseek-": "deepseek",
    "qwen-": "alibaba",
    "gemini-": "google",
    "mistral-": "mistral",
    "llama-": "meta",
}

# Per-model entry: pricing, context window, and static capabilities.
ModelEntry = Dict[str, Any]

# Singleton — loaded once per process
_MODEL_CATALOG: Optional[Dict[str, ModelEntry]] = None


def _get_cache_path() -> str:
    return os.path.join(AGENTICA_CACHE_DIR, "model_pricing_cache.json")


def _is_official(provider_id: str, model_id: str) -> bool:
    for prefix, official in _OFFICIAL_PROVIDERS.items():
        if model_id.startswith(prefix) and provider_id == official:
            return True
    return False


def _parse_catalog(catalog: dict) -> Dict[str, ModelEntry]:
    """Convert models.dev data into flat pricing and capability entries.

    Official providers take priority over third-party entries for the same
    model identifier.
    """
    result: Dict[str, ModelEntry] = {}
    seen_official: set = set()

    for provider_id, provider_data in catalog.items():
        if not isinstance(provider_data, dict):
            continue
        models = provider_data.get("models")
        if not isinstance(models, dict):
            continue
        for model_id, model_info in models.items():
            if not isinstance(model_info, dict):
                continue
            cost = model_info.get("cost") or {}
            if not isinstance(cost, dict):
                cost = {}

            key = model_id.lower().strip()
            official = _is_official(provider_id, key)

            if key in seen_official and not official:
                continue

            limit = model_info.get("limit") or {}
            modalities = model_info.get("modalities") or {}
            input_modalities = modalities.get("input") or []
            if not isinstance(input_modalities, list):
                input_modalities = []
            entry: ModelEntry = {
                "input": float(cost.get("input", 0)),
                "output": float(cost.get("output", 0)),
                "cache_read": float(cost.get("cache_read", 0)),
                "cache_write": float(cost.get("cache_write", 0)),
                "context_window": int(limit.get("context", 0)),
                "input_modalities": tuple(str(item) for item in input_modalities),
            }

            if key in result and not official:
                existing = result[key]
                if existing["input"] > 0 and entry["input"] == 0:
                    continue
                if existing["input"] == 0 and entry["input"] > 0:
                    pass  # upgrade from free to paid
                else:
                    continue

            result[key] = entry
            if official:
                seen_official.add(key)

    return result


def _fetch_and_cache() -> Optional[Dict[str, ModelEntry]]:
    """Fetch models.dev catalog and write to local cache. Returns None on failure."""
    try:
        resp = httpx.get(
            _CATALOG_URL,
            timeout=10,
            headers={"User-Agent": "agentica"},
        )
        resp.raise_for_status()
        catalog = resp.json()
    except Exception as e:
        logger.debug(f"Failed to fetch model catalog from {_CATALOG_URL}: {e}")
        return None

    entries = _parse_catalog(catalog)
    if not entries:
        return None

    cache_path = _get_cache_path()
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(
                {"schema_version": _CACHE_SCHEMA_VERSION, "models": entries},
                f,
            )
    except OSError:
        pass
    return entries


def _load_cached() -> Optional[Dict[str, ModelEntry]]:
    """Load catalog from local cache if file mtime is within TTL."""
    cache_path = _get_cache_path()
    try:
        mtime = os.path.getmtime(cache_path)
        if time.time() - mtime > _CACHE_TTL:
            return None
        with open(cache_path) as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return None
        if payload.get("schema_version") != _CACHE_SCHEMA_VERSION:
            return None
        data = payload.get("models")
        if not isinstance(data, dict):
            return None
        if data and not isinstance(next(iter(data.values())), dict):
            return None
        for entry in data.values():
            modalities = entry.get("input_modalities", ())
            if isinstance(modalities, list):
                entry["input_modalities"] = tuple(modalities)
        return data
    except (OSError, json.JSONDecodeError, TypeError, StopIteration):
        return None


def _get_catalog() -> Dict[str, ModelEntry]:
    """Return the merged catalog (cached remote + hardcoded fallback).

    Remote entries fill in the ~2000 models not in fallback.
    Fallback is authoritative for models we've explicitly listed
    (manually calibrated pricing / context_window).
    """
    global _MODEL_CATALOG
    if _MODEL_CATALOG is not None:
        return _MODEL_CATALOG

    remote = _load_cached()
    if remote is None:
        remote = _fetch_and_cache()

    merged: Dict[str, ModelEntry] = {
        key: dict(entry) for key, entry in _FALLBACK_PRICING.items()
    }
    if remote:
        merged.update(remote)
    _MODEL_CATALOG = merged
    return _MODEL_CATALOG


def get_model_pricing() -> Dict[str, ModelEntry]:
    """Return the pricing table (backward-compatible wrapper)."""
    return _get_catalog()


def _get_model_entry(model_id: str) -> Optional[ModelEntry]:
    """Look up a catalog entry by exact id, then longest matching prefix."""
    normalised = CostTracker._normalise(model_id)
    catalog = _get_catalog()
    entry = catalog.get(normalised)
    if entry is not None:
        return entry

    matching_keys = [
        key
        for key in catalog
        if normalised.startswith(key)
        and normalised[len(key):].startswith(("-", ".", "_", ":"))
    ]
    if not matching_keys:
        return None
    return catalog[max(matching_keys, key=len)]


def get_model_context_window(model_id: str, default: int = 128000) -> int:
    """Look up context window for a model from the catalog."""
    entry = _get_model_entry(model_id)
    if entry and entry.get("context_window", 0) > 0:
        return int(entry["context_window"])
    return default


def get_model_supports_images(model_id: str) -> bool:
    """Return whether the catalog declares image as an input modality."""
    entry = _get_model_entry(model_id)
    if entry is None:
        return False
    return "image" in entry.get("input_modalities", ())


# Backward-compatible alias used by tests
MODEL_PRICING = _FALLBACK_PRICING


@dataclass
class ModelUsageStat:
    """Token usage statistics for a single model."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    requests: int = 0


@dataclass
class CostTracker:
    """Full-session cost tracker.

    One instance is created per Agent.run() call and attached to RunResponse.

    Attributes:
        total_cost_usd:      Accumulated USD cost across all API calls.
        total_input_tokens:  Accumulated input tokens.
        total_output_tokens: Accumulated output tokens.
        turns:               Number of API calls recorded.
        has_unknown_model:   True if any model was not in MODEL_PRICING.
        model_usage:         Per-model breakdown (keyed by normalised model id).
    """

    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    last_input_tokens: int = 0
    turns: int = 0
    has_unknown_model: bool = False
    model_usage: Dict[str, ModelUsageStat] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        """Record one API call and return its cost in USD."""
        normalised = self._normalise(model_id)
        pricing = self._lookup_pricing(normalised)

        cost = (
            input_tokens       * pricing["input"]       / 1_000_000
            + output_tokens    * pricing["output"]      / 1_000_000
            + cache_read_tokens  * pricing["cache_read"]  / 1_000_000
            + cache_write_tokens * pricing["cache_write"] / 1_000_000
        )

        stat = self.model_usage.setdefault(normalised, ModelUsageStat())
        stat.input_tokens        += input_tokens
        stat.output_tokens       += output_tokens
        stat.cache_read_tokens   += cache_read_tokens
        stat.cache_write_tokens  += cache_write_tokens
        stat.cost_usd            += cost
        stat.requests            += 1

        self.total_cost_usd      += cost
        self.total_input_tokens  += input_tokens
        self.total_output_tokens += output_tokens
        self.last_input_tokens    = input_tokens
        self.turns               += 1

        return cost

    def summary(self) -> str:
        """Return a human-readable cost summary."""
        lines = [f"Total cost:   ${self.total_cost_usd:.4f}"]
        if self.has_unknown_model:
            lines.append("              ⚠ unknown model(s) — costs may be underestimated")
        lines.append(
            f"Total tokens: {self.total_input_tokens:,} input"
            f" + {self.total_output_tokens:,} output"
        )
        lines.append(f"API calls:    {self.turns}")
        if self.model_usage:
            lines.append("Usage by model:")
            for model, stat in self.model_usage.items():
                parts = [f"{stat.input_tokens:,} in", f"{stat.output_tokens:,} out"]
                if stat.cache_read_tokens:
                    parts.append(f"{stat.cache_read_tokens:,} cache_read")
                if stat.cache_write_tokens:
                    parts.append(f"{stat.cache_write_tokens:,} cache_write")
                lines.append(f"  {model}: {', '.join(parts)} (${stat.cost_usd:.4f})")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lookup_pricing(self, normalised_id: str) -> Dict[str, float]:
        """Look up pricing from the merged catalog+fallback table."""
        table = get_model_pricing()

        if normalised_id in table:
            return table[normalised_id]

        # Prefix match — e.g. "gpt-4o-2024-11-20" → "gpt-4o"
        for key in table:
            if normalised_id.startswith(key):
                return table[key]

        # Best-effort family match (first dash-separated segment).
        # Prefer a non-zero pricing entry.
        family = normalised_id.split("-")[0]
        best_family: Optional[Dict[str, float]] = None
        for key in table:
            if key.split("-")[0] == family:
                pricing = table[key]
                if pricing["input"] > 0 or pricing["output"] > 0:
                    return pricing
                if best_family is None:
                    best_family = pricing
        if best_family is not None:
            self.has_unknown_model = True
            return best_family

        self.has_unknown_model = True
        return {"input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0}

    @staticmethod
    def _normalise(model_id: str) -> str:
        """Strip provider prefixes and lowercase the model identifier."""
        _PREFIXES = (
            "openai/",
            "anthropic/",
            "accounts/fireworks/models/",
            "together_ai/",
            "groq/",
            "cohere/",
        )
        lower = model_id.lower().strip()
        for prefix in _PREFIXES:
            if lower.startswith(prefix):
                lower = lower[len(prefix):]
                break
        return lower
