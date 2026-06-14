# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Sensitive text redaction shared by logs, tools, archives, and compression.

Two redaction levels:

- ``high_confidence`` (default): masks unambiguous secrets only — vendor-prefixed
  keys (sk-*, ghp_*, AKIA*, ...), JWTs, ``-----BEGIN PRIVATE KEY-----`` blocks,
  ``Authorization: Bearer …`` headers, DB connection strings with embedded
  passwords, URL query tokens, and JSON ``"api_key": "…"``-style fields.
  These shapes only appear when something genuinely sensitive is present, so
  they're safe to apply by default.

- ``strict``: additionally masks env-style ``API_KEY=value`` and key:value
  forms. These are aggressive: ordinary source code (variable names containing
  ``key`` / ``token`` / ``password``, e.g. ``api_key=existing_key``) gets
  rewritten too, which breaks tools like ``edit_file`` whose ``old_string``
  must match the file byte-for-byte. Use only for log lines / archives where
  no downstream tool ever has to round-trip the text back into source.

Two top-level toggles control whether redaction runs at all in the model
loop (so end users can opt out):

- ``AGENTICA_REDACT_TOOL_OUTPUTS`` (env, default ``0``): redact tool result
  text before it reaches the LLM. OFF by default because it corrupts
  ``edit_file`` round-trips.
- ``AGENTICA_REDACT_STREAMED_TEXT`` (env, default ``0``): redact streamed
  ``show_result`` chunks. OFF by default for the same reason.

Unterminated ``-----BEGIN PRIVATE KEY-----`` blocks are *always* redacted
regardless of these toggles — that's a hard safety floor.
"""

import os
import re
from typing import Optional

# ---------------------------------------------------------------------------
# High-confidence patterns — safe to apply by default. Each only matches
# shapes that are essentially never legitimate source code.
# ---------------------------------------------------------------------------

_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----"
)

_DB_CONNSTR_RE = re.compile(
    r"((?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^:\s/@]+:)([^@\s]+)(@)",
    re.IGNORECASE,
)

_AUTH_HEADER_RE = re.compile(r"(Authorization:\s*Bearer\s+)(\S+)", re.IGNORECASE)

_BARE_BEARER_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(Bearer\s+)([A-Za-z0-9._~+/=-]{20,})(?![A-Za-z0-9_-])",
    re.IGNORECASE,
)

_URL_QUERY_SECRET_RE = re.compile(
    r"([?&](?:access_token|refresh_token|id_token|api[_-]?key|apikey|auth[_-]?token|token|secret|password|signature|sig|client_secret|jwt|key|code)=)"
    r"([^&#\s]+)",
    re.IGNORECASE,
)

_JSON_SECRET_FIELD_RE = re.compile(
    r'("(?:api_?key|token|secret|password|access_token|refresh_token|auth_token|bearer|private_key)")\s*:\s*"([^"]+)"',
    re.IGNORECASE,
)

_PREFIX_RE = re.compile(
    r"(?<![A-Za-z0-9_-])("
    r"sk-proj-[A-Za-z0-9_-]{20,}|"
    r"sk-[A-Za-z0-9_-]{20,}|"
    r"ghp_[A-Za-z0-9]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|"
    r"gho_[A-Za-z0-9]{20,}|"
    r"ghu_[A-Za-z0-9]{20,}|"
    r"ghs_[A-Za-z0-9]{20,}|"
    r"ghr_[A-Za-z0-9]{20,}|"
    r"AKIA[A-Z0-9]{16}|"
    r"AIza[A-Za-z0-9_-]{30,}|"
    r"hf_[A-Za-z0-9]{20,}|"
    r"gsk_[A-Za-z0-9]{20,}|"
    r"pypi-[A-Za-z0-9_-]{20,}"
    r")(?![A-Za-z0-9_-])"
)

_JWT_RE = re.compile(
    r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{10,}(?:\.[A-Za-z0-9_=-]{4,}){0,2}(?![A-Za-z0-9_-])"
)

# ---------------------------------------------------------------------------
# Strict patterns — opt-in. Match ordinary source code, so they break
# byte-exact round-trips (edit_file old_string, multi_edit_file).
# ---------------------------------------------------------------------------

_ENV_ASSIGN_RE = re.compile(
    r"(?<![?&A-Za-z0-9_])([A-Z0-9_]*(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)[A-Z0-9_]*)\s*=\s*(['\"]?)(\S+)\2",
    re.IGNORECASE,
)

_KEY_VALUE_SECRET_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"([A-Z0-9_-]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)[A-Z0-9_-]*\s*:\s*)"
    r"(['\"]?)([^\s,'\"}]+)\2",
    re.IGNORECASE,
)


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def redact_tool_outputs_enabled() -> bool:
    """Whether to redact non-streamed tool result text before sending to the LLM.

    OFF by default: redaction rewrites strings the LLM later has to match
    byte-exactly with ``edit_file`` / ``multi_edit_file``, causing spurious
    "String not found" errors. Operators handling truly sensitive tool
    outputs (production secrets in logs, etc.) can opt in via
    ``AGENTICA_REDACT_TOOL_OUTPUTS=1``. Private-key blocks are always
    redacted regardless of this flag — see ``redact_private_key_blocks``.
    """
    return _env_truthy("AGENTICA_REDACT_TOOL_OUTPUTS", default=False)


def redact_streamed_text_enabled() -> bool:
    """Whether to redact streamed ``show_result=True`` tool chunks.

    OFF by default for the same reason as ``redact_tool_outputs_enabled``.
    Toggle via ``AGENTICA_REDACT_STREAMED_TEXT=1``.
    """
    return _env_truthy("AGENTICA_REDACT_STREAMED_TEXT", default=False)


def redact_sensitive_text(
    text: Optional[str],
    *,
    level: str = "high_confidence",
) -> Optional[str]:
    """Mask common secret shapes in text.

    Args:
        text: text to scan.
        level: ``"high_confidence"`` (default) — vendor-prefixed keys, JWTs,
            private key blocks, auth headers, DB connstr passwords, URL query
            tokens, JSON secret fields. Safe to apply to source code: leaves
            ordinary identifiers like ``api_key=existing_key`` alone.
            ``"strict"`` — additionally masks env-style and key:value forms.
            Aggressive; will rewrite source code containing ``key``/``token``/
            ``password`` identifiers.

    This is a logging and persistence safety net, not a permission boundary.
    Callers should still avoid sending secrets into prompts or archives.
    """
    if not text:
        return text

    redacted = _PRIVATE_KEY_RE.sub("***REDACTED_PRIVATE_KEY***", text)
    redacted = _DB_CONNSTR_RE.sub(r"\1***REDACTED***\3", redacted)
    redacted = _AUTH_HEADER_RE.sub(r"\1***REDACTED***", redacted)
    redacted = _BARE_BEARER_RE.sub(r"\1***REDACTED***", redacted)
    redacted = _URL_QUERY_SECRET_RE.sub(r"\1***", redacted)
    redacted = _JSON_SECRET_FIELD_RE.sub(r'\1: "***REDACTED***"', redacted)
    redacted = _PREFIX_RE.sub("***REDACTED_SECRET***", redacted)
    redacted = _JWT_RE.sub("***REDACTED_JWT***", redacted)

    if level == "strict":
        redacted = _ENV_ASSIGN_RE.sub(r"\1=***REDACTED***", redacted)
        redacted = _KEY_VALUE_SECRET_RE.sub(r"\1\2***REDACTED***\2", redacted)

    return redacted
