# agentica/utils/json_parse.py
# -*- coding: utf-8 -*-
"""Robust JSON extraction from LLM text responses.

LLMs that don't go through the OpenAI tool-call protocol often return JSON
wrapped in prose, markdown code fences, or with a stray trailing token.
These two helpers tolerate that, returning ``None`` when no valid JSON of
the expected top-level shape can be recovered.

Used by ``Swarm`` for parsing coordinator output and available to user code
that asks an LLM to return structured JSON without a tool-call schema.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


_FENCE_PATTERN = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    """Return content inside the first ```...``` fence, or text unchanged."""
    m = _FENCE_PATTERN.search(text)
    return m.group(1).strip() if m else text


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object (dict) from text.

    Tolerates surrounding prose and markdown code fences. Returns ``None`` if
    no top-level JSON object can be recovered.

    Examples::

        extract_json_object('{"a": 1}')                  # {"a": 1}
        extract_json_object('Result: {"a": 1}\\n')       # {"a": 1}
        extract_json_object('```json\\n{"a": 1}\\n```') # {"a": 1}
        extract_json_object('[1, 2, 3]')                 # None
    """
    if not text:
        return None
    candidate = _strip_code_fence(text.strip())

    try:
        result = json.loads(candidate)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(candidate[start:end + 1])
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def extract_json_array(text: str) -> Optional[List[Any]]:
    """Extract a JSON array (list) from text.

    Tolerates surrounding prose and markdown code fences. Returns ``None`` if
    no top-level JSON array can be recovered.

    Examples::

        extract_json_array('[1, 2, 3]')                  # [1, 2, 3]
        extract_json_array('Here: [1, 2] done.')         # [1, 2]
        extract_json_array('```\\n[1]\\n```')            # [1]
        extract_json_array('{"a": 1}')                   # None
    """
    if not text:
        return None
    candidate = _strip_code_fence(text.strip())

    try:
        result = json.loads(candidate)
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    start = candidate.find("[")
    end = candidate.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(candidate[start:end + 1])
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
    return None
