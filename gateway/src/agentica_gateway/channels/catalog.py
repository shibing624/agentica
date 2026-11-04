# -*- coding: utf-8 -*-
"""Catalog of gateway entry points (web UI + IM channels).

The SPA Personal Assistant page renders this list. Status is live (what this
process actually registered); titles and how-to copy live in the frontend.
"""
from dataclasses import dataclass
from typing import Optional, Sequence

from ..services.channel_manager import ChannelManager

GUIDE_URL = "https://shibing624.github.io/agentica/advanced/gateway/"

_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})


@dataclass(frozen=True)
class ChannelSpec:
    id: str
    extra: Optional[str]
    env: tuple[tuple[str, str], ...]
    docs_anchor: str
    recommended: bool = False
    builtin: bool = False


# Order is the page order: web first, then the easiest IM, then the rest.
CATALOG: tuple[ChannelSpec, ...] = (
    ChannelSpec(
        id="web",
        extra=None,
        env=(),
        docs_anchor="#web-网页内置-ui",
        builtin=True,
    ),
    ChannelSpec(
        id="wechat",
        extra="wechat",
        env=(
            ("WECHAT_TOKEN_FILE", "~/.agentica/cache/wxbot_token.json"),
            ("WECHAT_ALLOWED_USERS", ""),
        ),
        docs_anchor="#个人微信wechat",
        recommended=True,
    ),
    ChannelSpec(
        id="wecom",
        extra="wecom",
        env=(
            ("WECOM_BOT_ID", ""),
            ("WECOM_SECRET", ""),
            ("WECOM_ALLOWED_USERS", ""),
        ),
        docs_anchor="#企业微信wecom",
    ),
    ChannelSpec(
        id="qq",
        extra="qq",
        env=(
            ("QQ_APP_ID", ""),
            ("QQ_APP_SECRET", ""),
            ("QQ_ALLOWED_USERS", ""),
        ),
        docs_anchor="#qqqq-开放平台官方机器人",
    ),
    ChannelSpec(
        id="feishu",
        extra=None,
        env=(
            ("FEISHU_APP_ID", ""),
            ("FEISHU_APP_SECRET", ""),
            ("FEISHU_ALLOWED_USERS", ""),
        ),
        docs_anchor="#飞书lark",
    ),
    ChannelSpec(
        id="telegram",
        extra="telegram",
        env=(
            ("TELEGRAM_BOT_TOKEN", ""),
            ("TELEGRAM_ALLOWED_USERS", ""),
        ),
        docs_anchor="#telegram",
    ),
    ChannelSpec(
        id="discord",
        extra="discord",
        env=(
            ("DISCORD_BOT_TOKEN", ""),
            ("DISCORD_ALLOWED_USERS", ""),
        ),
        docs_anchor="#discord",
    ),
    ChannelSpec(
        id="dingtalk",
        extra="dingtalk",
        env=(
            ("DINGTALK_CLIENT_ID", ""),
            ("DINGTALK_CLIENT_SECRET", ""),
            ("DINGTALK_ALLOWED_USERS", ""),
        ),
        docs_anchor="#钉钉dingtalk",
    ),
    ChannelSpec(
        id="slack",
        extra="slack",
        env=(
            ("SLACK_BOT_TOKEN", ""),
            ("SLACK_APP_TOKEN", ""),
            ("SLACK_ALLOWED_USERS", ""),
        ),
        docs_anchor="#slack",
    ),
)


def channel_overview(
    cm: Optional[ChannelManager],
    *,
    host: str,
    port: int,
    web_url: str,
) -> dict:
    """Payload for ``GET /api/channels``.

    ``channels`` / ``status`` stay the registered-only view they always were.
    ``catalog`` is every supported entry, including ones not configured yet.
    """
    status = cm.get_status() if cm else {}
    registered: Sequence[str] = cm.list_channels() if cm else []
    registered_set = set(registered)
    catalog = []
    for spec in CATALOG:
        if spec.builtin:
            configured, connected = True, True
        else:
            configured = spec.id in registered_set
            connected = bool((status.get(spec.id) or {}).get("connected"))
        catalog.append({
            "id": spec.id,
            "extra": spec.extra,
            "env": [{"name": name, "example": example} for name, example in spec.env],
            "recommended": spec.recommended,
            "docs_anchor": spec.docs_anchor,
            "configured": configured,
            "connected": connected,
        })
    return {
        "channels": list(registered),
        "status": status,
        "web_url": web_url,
        "listen": {
            "host": host,
            "port": port,
            "loopback": host in _LOOPBACK,
        },
        "guide_url": GUIDE_URL,
        "catalog": catalog,
    }
