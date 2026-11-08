# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Zero-JS capability-URL card page (/c/<token>) for IM approvals and questions.

Adapted from agent-autoloop's ``kernel/card-page.ts`` and ``kernel/credentials.ts``:
- Capability URL: the link token is ``HMAC-SHA256(secret, ["card", id, created_at])``
  encoded as base64url, so opening ``/c/<token>`` in a mobile webview needs no
  login cookie and grants authority over that single card only.
- Zero JavaScript + strict CSP: the 9-line adaptive light/dark stylesheet is
  pinned by its SHA-256 hash in ``Content-Security-Policy: default-src 'none'``.
- Two separate ``<form method="post">`` blocks when free-text input is allowed
  so pressing Enter in the text input never triggers the first option button.
- Post-Redirect-Get (``303 See Other``): submitting a form redirects back to
  ``GET /c/<token>``, rendering the settled decision read-only.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

STYLE = (
    ":root{color-scheme:light dark;font:16px/1.6 system-ui,sans-serif}\n"
    "body{max-width:34rem;margin:0 auto;padding:32px 16px;overflow-wrap:anywhere}\n"
    "h1{font-size:1.25rem;line-height:1.4;margin:0 0 12px}\n"
    "p{white-space:pre-wrap;margin:0 0 20px}\n"
    "form{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 12px}\n"
    "button,input{font:inherit;min-height:44px;border-radius:8px;border:1px solid GrayText;"
    "padding:0 16px;background:Canvas;color:CanvasText}\n"
    "button.best{background:CanvasText;color:Canvas;border-color:CanvasText}\n"
    "input{flex:1 1 12rem;min-width:0}"
)

STYLE_SHA = base64.b64encode(hashlib.sha256(STYLE.encode("utf-8")).digest()).decode("ascii")
POLICY = (
    f"default-src 'none'; style-src 'sha256-{STYLE_SHA}'; "
    "form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
)

PAGE_HEADERS: Dict[str, str] = {
    "content-type": "text/html; charset=utf-8",
    "cache-control": "no-store",
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "content-security-policy": POLICY,
}

LINK_LIFETIME_SECONDS = 7 * 86400.0

_ENTITIES = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
}


def escaped(words: str) -> str:
    """Escape HTML special characters including single/double quotes."""
    return "".join(_ENTITIES.get(ch, ch) for ch in (words or ""))


def credential(secret: str, purpose: str, *parts: Any) -> str:
    """Derive a capability token via HMAC-SHA256 (unpadded base64url)."""
    payload = json.dumps([purpose, *parts], separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    digest = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def shell(title: str, content: str) -> str:
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{escaped(title)}</title>"
        f"<style>{STYLE}</style></head><body>{content}</body></html>"
    )


def words_page(words: str) -> str:
    return shell(words, f"<p>{escaped(words)}</p>")


@dataclass
class CardItem:
    """One interactive card (approval, ask_user_question, or notice)."""

    id: str
    token: str
    kind: str  # "approval" | "ask" | "notice"
    title: str
    text: str = ""
    options: List[str] = field(default_factory=list)
    values: List[str] = field(default_factory=list)
    recommend: int = 0
    allow_custom_text: bool = True
    state: str = "open"  # "open" | "closed" | "expired"
    choice: Optional[int] = None
    answer: Optional[str] = None
    result: Optional[str] = None
    by: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    decided_at: Optional[float] = None
    session_id: str = ""
    request_id: str = ""
    on_resolve: Optional[Callable[["CardItem"], Any]] = None
    _waiters: List[asyncio.Future] = field(default_factory=list, repr=False)

    @property
    def selected_value(self) -> str:
        """The resolved value to hand back to the caller (decision key or answer text)."""
        if self.choice is not None:
            if 0 <= self.choice < len(self.values):
                return self.values[self.choice]
            if 0 <= self.choice < len(self.options):
                return self.options[self.choice]
        return self.answer or ""

    @property
    def selected_label(self) -> str:
        if self.choice is not None and 0 <= self.choice < len(self.options):
            return self.options[self.choice]
        return self.answer or ""


def card_page(card: CardItem) -> str:
    """Render ``card`` as a standalone zero-JS HTML document."""
    words = f"<h1>{escaped(card.title)}</h1>"
    if card.text:
        words += f"<p>{escaped(card.text)}</p>"

    if card.state == "open":
        focused = card.recommend if 0 <= card.recommend < len(card.options) else 0
        buttons: List[str] = []
        for idx, option in enumerate(card.options):
            best = ' class="best"' if idx == card.recommend else ""
            focus = " autofocus" if idx == focused else ""
            buttons.append(
                f'<button name="choice" value="{idx}"{best}{focus}>{escaped(option)}</button>'
            )
        options_form = f'<form method="post">{"".join(buttons)}</form>' if buttons else ""
        # Its own form: pressing Enter in the text input must not press the first option button.
        other_form = ""
        if card.kind == "notice":
            box = '<input name="text" placeholder="我实际怎么处理的（可不填）" aria-label="处理说明">'
            other_form = f"<form method=\"post\">{box}<button>记录并结案</button></form>"
        elif card.allow_custom_text:
            box = '<input name="text" placeholder="别的想法" aria-label="别的想法" required>'
            other_form = f"<form method=\"post\">{box}<button>发送</button></form>"
        return shell(card.title, f"{words}{options_form}{other_form}")

    chosen = None
    if card.kind in ("ask", "approval"):
        if card.choice is not None and 0 <= card.choice < len(card.options):
            chosen = card.options[card.choice]
        elif card.answer:
            chosen = card.answer
    outcome_parts = [f"你选了「{chosen}」" if chosen else "", card.result or ""]
    outcome = "。".join(p for p in outcome_parts if p)
    settled = (
        f"<p>处理说明：{escaped(card.answer)}</p>"
        if card.kind == "notice" and card.answer
        else ""
    )
    body = f"{words}{f'<p>{escaped(outcome)}</p>' if outcome else ''}{settled}"
    return shell(card.title, body)


class CardRegistry:
    """In-memory store of capability-URL cards."""

    def __init__(self, secret_provider: Optional[Callable[[], str]] = None) -> None:
        self._secret_provider = secret_provider
        self._FallbackSecret = uuid.uuid4().hex
        self._by_token: Dict[str, CardItem] = {}
        self._by_id: Dict[str, CardItem] = {}

    def _secret(self) -> str:
        if self._secret_provider is not None:
            s = self._secret_provider()
            if s:
                return s
        return self._FallbackSecret

    def create(
        self,
        *,
        kind: str,
        title: str,
        text: str = "",
        options: Optional[List[str]] = None,
        values: Optional[List[str]] = None,
        recommend: int = 0,
        allow_custom_text: Optional[bool] = None,
        session_id: str = "",
        request_id: str = "",
        on_resolve: Optional[Callable[[CardItem], Any]] = None,
    ) -> CardItem:
        self.gc()
        card_id = uuid.uuid4().hex[:12]
        created_at = time.time()
        token = credential(self._secret(), "card", card_id, f"{created_at:.6f}")
        opts = list(options) if options else (["已处理"] if kind == "notice" else [])
        vals = list(values) if values else list(opts)
        if allow_custom_text is None:
            allow_custom_text = kind != "approval"
        card = CardItem(
            id=card_id,
            token=token,
            kind=kind,
            title=title,
            text=text,
            options=opts,
            values=vals,
            recommend=recommend,
            allow_custom_text=allow_custom_text,
            created_at=created_at,
            session_id=session_id,
            request_id=request_id,
            on_resolve=on_resolve,
        )
        self._by_token[token] = card
        self._by_id[card_id] = card
        return card

    def get_by_token(self, token: str) -> Optional[CardItem]:
        self.gc()
        return self._by_token.get(token)

    def get_by_id(self, card_id: str) -> Optional[CardItem]:
        self.gc()
        return self._by_id.get(card_id)

    def find_open(self, *, session_id: str = "", request_id: str = "") -> Optional[CardItem]:
        for card in self._by_id.values():
            if card.state != "open":
                continue
            if request_id and card.request_id == request_id:
                return card
            if session_id and not request_id and card.session_id == session_id:
                return card
        return None

    def answer(
        self,
        token_or_id: str,
        *,
        choice: Optional[int] = None,
        text: Optional[str] = None,
        by: str = "链接",
        result: Optional[str] = None,
        fire_callback: bool = True,
    ) -> bool:
        """Settle an open card. Returns True only for the first writer."""
        card = self._by_token.get(token_or_id) or self._by_id.get(token_or_id)
        if card is None or card.state != "open":
            return False

        cleaned_text = (text or "").strip()
        if choice is not None:
            if not (0 <= choice < len(card.options)):
                return False
            card.choice = choice
            card.answer = card.options[choice]
        elif cleaned_text:
            if not card.allow_custom_text and card.kind == "approval":
                return False
            card.answer = cleaned_text
        elif card.kind == "notice":
            card.choice = 0
            card.answer = card.options[0] if card.options else "已处理"
        else:
            return False

        card.state = "closed"
        card.by = by
        card.decided_at = time.time()
        if result is not None:
            card.result = result

        for fut in list(card._waiters):
            if not fut.done():
                fut.set_result(card)
        card._waiters.clear()

        if fire_callback and card.on_resolve is not None:
            cb = card.on_resolve
            card.on_resolve = None
            res = cb(card)
            if asyncio.iscoroutine(res):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(res)
                except RuntimeError:
                    pass
        return True

    def expire(self, token_or_id: str, reason: str = "该请求已超时关闭") -> bool:
        card = self._by_token.get(token_or_id) or self._by_id.get(token_or_id)
        if card is None or card.state != "open":
            return False
        card.state = "expired"
        card.result = reason
        card.decided_at = time.time()
        card.on_resolve = None
        for fut in list(card._waiters):
            if not fut.done():
                fut.set_result(card)
        card._waiters.clear()
        return True

    def gc(self, now: Optional[float] = None) -> None:
        current = now if now is not None else time.time()
        for token, card in list(self._by_token.items()):
            if current - card.created_at > LINK_LIFETIME_SECONDS:
                self._by_token.pop(token, None)
                self._by_id.pop(card.id, None)


def detect_lan_ip() -> str:
    """Best-effort LAN IP so a phone on the same Wi-Fi/Tailscale can open card links."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
    except OSError:
        pass
    return "127.0.0.1"


def gateway_public_url(host: str = "127.0.0.1", port: int = 8881) -> str:
    """Base URL for ``/c/<token>`` links sent to IM channels."""
    explicit = (
        os.getenv("AGENTICA_PUBLIC_URL")
        or os.getenv("PUBLIC_URL")
        or ""
    ).strip().rstrip("/")
    if explicit:
        return explicit
    shown_host = host
    if shown_host in ("0.0.0.0", "::", "[::]", "127.0.0.1", "localhost"):
        lan = detect_lan_ip()
        if shown_host in ("0.0.0.0", "::", "[::]") and lan != "127.0.0.1":
            shown_host = lan
        elif shown_host in ("127.0.0.1", "localhost"):
            shown_host = "127.0.0.1"
    return f"http://{shown_host}:{port}"
