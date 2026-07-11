# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
Personal WeChat channel implementation using an inline ``WxBotClient``.

Talks to ``https://ilinkai.weixin.qq.com`` (the official WeChat ClawBot /
iLink backend) via HTTP JSON and supports QR-code login. This is the
Python equivalent of Tencent's ``@tencent-weixin/openclaw-weixin`` Node
plugin — the iLink protocol is plain HTTP, so no external Node process is
required. The client is synchronous and runs ``run_loop`` in a background
daemon thread; inbound messages are dispatched to the main asyncio event
loop via ``call_soon_threadsafe``.

Protocol alignment follows the official openclaw-weixin reference:
* Every request carries ``base_info.channel_version`` and
  ``base_info.bot_agent`` (an observability UA — it lives in the JSON body,
  *not* an HTTP header).
* Typing indicator requires a ``typing_ticket`` from ``getconfig`` and is
  sent with ``ilink_user_id`` + ``status`` (1=typing, 2=cancel).
* Media (image/file/voice/video) is AES-128-ECB encrypted and uploaded to
  the CDN (POST of ``application/octet-stream``); the returned
  ``x-encrypted-param`` response header becomes the ``encrypt_query_param``
  of a ``CDNMedia`` reference attached to the outbound ``sendmessage``.
"""
import asyncio
import base64
import hashlib
import json
import os
import struct
import threading
import time
import uuid
import webbrowser
from collections import deque
from pathlib import Path
from typing import Optional, List, Callable
from urllib.parse import quote

import requests

from agentica.config import AGENTICA_CACHE_DIR
from agentica.utils.log import logger
from agentica.version import __version__

from .base import Channel, ChannelType, Message
from ..config import settings

# ── ilinkai protocol constants ──
_API = "https://ilinkai.weixin.qq.com"
# Cache the WeChat bot token under the shared Agentica cache root
# (~/.agentica/cache/) alongside every other regenerable scratch file,
# instead of a stray ~/.wxbot/ directory.
_DEFAULT_TOKEN_FILE = Path(AGENTICA_CACHE_DIR) / "wxbot_token.json"
_VERSION = __version__
_BOT_AGENT = f"Agentica/{__version__}"

# Message / item / media type enums (mirror openclaw-weixin types.ts)
_MSG_USER, _MSG_BOT = 1, 2
_ITEM_TEXT, _ITEM_IMAGE, _ITEM_VOICE, _ITEM_FILE, _ITEM_VIDEO = 1, 2, 3, 4, 5
_MEDIA_IMAGE, _MEDIA_VIDEO, _MEDIA_FILE, _MEDIA_VOICE = 1, 2, 3, 4
_STATE_FINISH = 2
_TYPING, _CANCEL = 1, 2

# ``qrcode`` is only needed when the token cache is empty; lazy-import to
# avoid pulling Pillow into core gateway installs. ``Crypto`` is only needed
# for media (AES-128-ECB); lazy-import so the gateway runs without
# pycryptodome unless media is actually used.
qrcode = None
Crypto = None
_AES = None
_pad = None
_unpad = None


def _ensure_qrcode():
    """Ensure the ``qrcode`` package has been imported (lazy)."""
    global qrcode
    if qrcode is None:
        try:
            import qrcode as _qr
            qrcode = _qr
        except ImportError:
            raise ImportError(
                "qrcode not installed. Run: pip install 'agentica[wechat]'"
            )


def _uin() -> str:
    """Generate a one-shot ``X-WECHAT-UIN`` header value."""
    return base64.b64encode(str(struct.unpack(">I", os.urandom(4))[0]).encode()).decode()


class WxBotClient:
    """WeChat bot client (inline ilinkai HTTP transport).

    Talks to the official ClawBot / iLink backend. Supports text and media:
    media is AES-128-ECB (PKCS7) encrypted and uploaded to the CDN, with the
    returned ``x-encrypted-param`` header reused as the ``encrypt_query_param``
    of a ``CDNMedia`` reference attached to ``sendmessage``.
    """

    def __init__(self, token: Optional[str] = None, token_file: Optional[str] = None):
        self._tf = Path(token_file) if token_file else _DEFAULT_TOKEN_FILE
        self._tf.parent.mkdir(parents=True, exist_ok=True)
        self.token = token
        self.bot_id: Optional[str] = None
        self._buf = ""
        if not self.token:
            self._load()

    def _load(self) -> None:
        """Load cached token from disk if present."""
        if self._tf.exists():
            d = json.loads(self._tf.read_text("utf-8"))
            self.token = d.get("bot_token", "")
            self.bot_id = d.get("ilink_bot_id", "")
            self._buf = d.get("updates_buf", "")

    def _save(self, **kw) -> None:
        """Persist current token state to disk."""
        d = {
            "bot_token": self.token or "",
            "ilink_bot_id": self.bot_id or "",
            "updates_buf": self._buf or "",
            **kw,
        }
        self._tf.write_text(json.dumps(d, ensure_ascii=False, indent=2), "utf-8")

    def _base_info(self) -> dict:
        """Request ``base_info`` per openclaw-weixin (channel_version + bot_agent)."""
        return {"channel_version": _VERSION, "bot_agent": _BOT_AGENT}

    def _post(self, ep: str, body: dict, timeout: int = 15) -> dict:
        """POST to an ilinkai endpoint with auth headers.

        ``base_info`` (carrying channel_version + bot_agent) is always
        attached, matching the official protocol.
        """
        h = {
            "Content-Type": "application/json",
            "AuthorizationType": "ilink_bot_token",
            "X-WECHAT-UIN": _uin(),
        }
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        body = dict(body)
        body["base_info"] = self._base_info()
        r = requests.post(f"{_API}/{ep}", json=body, headers=h, timeout=timeout)
        r.raise_for_status()
        return r.json()

    # ── crypto (AES-128-ECB, PKCS7) — lazy import pycryptodome ──
    def _ensure_crypto(self) -> None:
        """Import ``Crypto`` on first use; raise with an install hint if missing."""
        global Crypto, _AES, _pad, _unpad
        if Crypto is None:
            try:
                from Crypto.Cipher import AES as _aes
                from Crypto.Util.Padding import pad as _p, unpad as _u
            except ImportError:
                raise ImportError(
                    "pycryptodome not installed. Run: pip install 'agentica[wechat]'"
                )
            Crypto = True
            _AES = _aes
            _pad = _p
            _unpad = _u

    @staticmethod
    def _padded_size(n: int) -> int:
        """PKCS7 padded ciphertext length for ``n`` plaintext bytes (AES block 16)."""
        return ((n // 16) + 1) * 16

    def _aes_ecb_encrypt(self, plain: bytes, key: bytes) -> bytes:
        self._ensure_crypto()
        return _AES.new(key, _AES.MODE_ECB).encrypt(_pad(plain, 16))

    def _aes_ecb_decrypt(self, cipher: bytes, key: bytes) -> bytes:
        self._ensure_crypto()
        return _unpad(_AES.new(key, _AES.MODE_ECB).decrypt(cipher), 16)

    def login_qr(self, poll_interval: int = 2) -> dict:
        """Interactive QR-code login flow.

        Saves the QR PNG to ``<token_file_dir>/wx_qr.png``, opens it in the
        default web browser, and polls until the QR is confirmed or expires.
        Returns the final status payload on success.
        """
        _ensure_qrcode()
        r = requests.get(f"{_API}/ilink/bot/get_bot_qrcode", params={"bot_type": 3}, timeout=10)
        r.raise_for_status()
        d = r.json()
        qr_id, url = d["qrcode"], d.get("qrcode_img_content", "")
        logger.info(f"WeChat: QR login ID = {qr_id}")
        if url:
            qr = qrcode.QRCode(box_size=2, border=2)
            qr.add_data(url)
            qr.make(fit=True)

            # 1) ASCII QR printed to the terminal — works over SSH / headless
            #    sessions without any browser, so the operator can scan
            #    directly from the terminal.
            try:
                qr.print_ascii(invert=True)
            except Exception as e:  # pragma: no cover - terminal edge cases
                logger.warning(f"WeChat: ASCII QR render failed: {e}")

            # 2) PNG fallback — saved next to the token file and auto-opened
            #    in the default browser when one is available (needs Pillow).
            try:
                img = self._tf.parent / "wx_qr.png"
                qr.make_image().save(str(img))
                try:
                    webbrowser.open(str(img))
                except Exception:
                    pass
                logger.info(f"WeChat: scan QR at {img}")
            except Exception as e:  # pragma: no cover - Pillow missing
                logger.warning(f"WeChat: QR PNG save skipped (Pillow missing?): {e}")
        last = ""
        while True:
            time.sleep(poll_interval)
            try:
                s = requests.get(
                    f"{_API}/ilink/bot/get_qrcode_status",
                    params={"qrcode": qr_id},
                    timeout=60,
                ).json()
            except requests.exceptions.ReadTimeout:
                continue
            st = s.get("status", "")
            if st != last:
                logger.info(f"WeChat: QR status = {st}")
                last = st
            if st == "confirmed":
                self.token = s.get("bot_token", "")
                self.bot_id = s.get("ilink_bot_id", "")
                self._save(login_time=time.strftime("%Y-%m-%d %H:%M:%S"))
                logger.info(f"WeChat: QR login OK (bot_id={self.bot_id})")
                return s
            if st == "expired":
                raise RuntimeError("WeChat: QR code expired")

    def get_updates(self, timeout: int = 30) -> list:
        """Long-poll for new messages."""
        try:
            resp = self._post(
                "ilink/bot/getupdates",
                {"get_updates_buf": self._buf or "", "base_info": {"channel_version": _VERSION}},
                timeout=timeout + 5,
            )
        except requests.exceptions.ReadTimeout:
            return []
        if resp.get("errcode"):
            logger.warning(f"WeChat: getUpdates err {resp.get('errcode')} {resp.get('errmsg', '')}")
            if resp["errcode"] == -14:
                self._buf = ""
                self._save()
            return []
        nb = resp.get("get_updates_buf", "")
        if nb:
            self._buf = nb
            self._save()
        return resp.get("msgs") or []

    def send_text(self, to_user_id: str, text: str, context_token: str = "") -> dict:
        """Send a text message to a user."""
        msg = {
            "from_user_id": "",
            "to_user_id": to_user_id,
            "client_id": f"pyclient-{uuid.uuid4().hex[:16]}",
            "message_type": _MSG_BOT,
            "message_state": _STATE_FINISH,
            "item_list": [{"type": _ITEM_TEXT, "text_item": {"text": text}}],
        }
        if context_token:
            msg["context_token"] = context_token
        return self._post(
            "ilink/bot/sendmessage",
            {"msg": msg, "base_info": {"channel_version": _VERSION}},
        )

    def get_config(self, ilink_user_id: str, context_token: str = "") -> str:
        """Fetch account config (typing_ticket) for a user from ``getconfig``."""
        body = {"ilink_user_id": ilink_user_id}
        if context_token:
            body["context_token"] = context_token
        resp = self._post("ilink/bot/getconfig", body, timeout=10)
        return resp.get("typing_ticket", "") or ""

    def send_typing(
        self,
        ilink_user_id: str,
        typing_ticket: str = "",
        context_token: str = "",
        cancel: bool = False,
    ) -> dict:
        """Show / cancel the typing indicator.

        The official protocol requires a ``typing_ticket`` from ``getconfig``;
        it is fetched automatically when not supplied. The request uses
        ``ilink_user_id`` + ``status`` (1=typing, 2=cancel).
        """
        if not typing_ticket:
            typing_ticket = self.get_config(ilink_user_id, context_token)
        return self._post("ilink/bot/sendtyping", {
            "ilink_user_id": ilink_user_id,
            "typing_ticket": typing_ticket,
            "status": _CANCEL if cancel else _TYPING,
        }, timeout=10)

    # ── media (AES-128-ECB + CDN upload/download) ──
    def get_upload_url(
        self,
        *,
        filekey: str,
        media_type: int,
        to_user_id: str,
        raw_bytes: bytes,
        aeskey: bytes,
        context_token: str = "",
        need_thumb: bool = False,
    ) -> tuple:
        """Request a CDN pre-signed upload from ``getuploadurl``.

        Returns ``(upload_param, upload_full_url, thumb_upload_param)``.
        """
        rawsize = len(raw_bytes)
        body = {
            "filekey": filekey,
            "media_type": media_type,
            "to_user_id": to_user_id,
            "rawsize": rawsize,
            "rawfilemd5": hashlib.md5(raw_bytes).hexdigest(),
            "filesize": self._padded_size(rawsize),
            "no_need_thumb": not need_thumb,
            "aeskey": base64.b64encode(aeskey).decode(),
        }
        if context_token:
            body["context_token"] = context_token
        resp = self._post("ilink/bot/getuploadurl", body, timeout=15)
        return (
            resp.get("upload_param", ""),
            resp.get("upload_full_url", ""),
            resp.get("thumb_upload_param", ""),
        )

    @staticmethod
    def _cdn_upload_url(
        upload_param: str, upload_full_url: str, filekey: str, cdn_base_url: Optional[str] = None
    ) -> str:
        """Resolve the CDN upload URL (prioritise ``upload_full_url``)."""
        full = (upload_full_url or "").strip()
        if full:
            return full
        if upload_param and cdn_base_url:
            return (
                f"{cdn_base_url.rstrip('/')}/upload"
                f"?encrypted_query_param={quote(upload_param)}&filekey={quote(filekey)}"
            )
        raise ValueError("CDN upload URL missing (need upload_full_url or cdn_base_url)")

    @staticmethod
    def _cdn_download_url(
        encrypt_query_param: str, full_url: str, cdn_base_url: Optional[str] = None
    ) -> str:
        """Resolve the CDN download URL (prioritise ``full_url``)."""
        full = (full_url or "").strip()
        if full:
            return full
        if encrypt_query_param and cdn_base_url:
            return (
                f"{cdn_base_url.rstrip('/')}/download"
                f"?encrypted_query_param={quote(encrypt_query_param)}"
            )
        raise ValueError("CDN download URL missing (need full_url or cdn_base_url)")

    def send_media(
        self,
        to_user_id: str,
        media_type: int,
        data: bytes,
        filename: str = "",
        context_token: str = "",
        need_thumb: bool = False,
        cdn_base_url: Optional[str] = None,
    ) -> dict:
        """Encrypt + upload media to the CDN, then send it via ``sendmessage``.

        Flow: random 16-byte AES key -> ``getuploadurl`` -> POST encrypted
        bytes (application/octet-stream) to the CDN URL -> read the
        ``x-encrypted-param`` response header -> build a ``CDNMedia``
        reference -> ``sendmessage`` with the matching item.
        """
        aeskey = os.urandom(16)
        upload_param, upload_full_url, _ = self.get_upload_url(
            filekey=filename or f"{uuid.uuid4().hex}.bin",
            media_type=media_type,
            to_user_id=to_user_id,
            raw_bytes=data,
            aeskey=aeskey,
            context_token=context_token,
            need_thumb=need_thumb,
        )
        url = self._cdn_upload_url(upload_param, upload_full_url, filename, cdn_base_url)
        cipher = self._aes_ecb_encrypt(data, aeskey)
        h = {
            "Content-Type": "application/octet-stream",
            "AuthorizationType": "ilink_bot_token",
            "X-WECHAT-UIN": _uin(),
        }
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        r = requests.post(url, data=cipher, headers=h, timeout=30)
        r.raise_for_status()
        encrypt_query_param = r.headers.get("x-encrypted-param", "")
        cdn_media = {
            "encrypt_query_param": encrypt_query_param,
            "aes_key": base64.b64encode(aeskey).decode(),
        }
        if media_type == _MEDIA_IMAGE:
            item = {"type": _ITEM_IMAGE, "image_item": {"media": cdn_media}}
        elif media_type == _MEDIA_VIDEO:
            item = {"type": _ITEM_VIDEO, "video_item": {"media": cdn_media}}
        elif media_type == _MEDIA_VOICE:
            item = {"type": _ITEM_VOICE, "voice_item": {"media": cdn_media}}
        else:
            item = {
                "type": _ITEM_FILE,
                "file_item": {
                    "media": cdn_media,
                    "file_name": filename,
                    "md5": hashlib.md5(data).hexdigest(),
                    "len": len(data),
                },
            }
        msg = {
            "from_user_id": "",
            "to_user_id": to_user_id,
            "client_id": f"pyclient-{uuid.uuid4().hex[:16]}",
            "message_type": _MSG_BOT,
            "message_state": _STATE_FINISH,
            "item_list": [item],
        }
        if context_token:
            msg["context_token"] = context_token
        return self._post("ilink/bot/sendmessage", {"msg": msg})

    def download_media(self, cdn_media: dict, cdn_base_url: Optional[str] = None) -> bytes:
        """Download and AES-128-ECB decrypt a ``CDNMedia`` reference."""
        encrypt_query_param = cdn_media.get("encrypt_query_param", "")
        aes_key_b64 = cdn_media.get("aes_key", "")
        url = self._cdn_download_url(
            encrypt_query_param, cdn_media.get("full_url", ""), cdn_base_url
        )
        h = {
            "AuthorizationType": "ilink_bot_token",
            "X-WECHAT-UIN": _uin(),
        }
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        r = requests.get(url, headers=h, timeout=30)
        r.raise_for_status()
        key = base64.b64decode(aes_key_b64)
        return self._aes_ecb_decrypt(r.content, key)

    @staticmethod
    def extract_media(msg: dict) -> list:
        """Collect all ``CDNMedia`` references from an inbound message."""
        out = []
        for it in msg.get("item_list", []):
            for key in ("image_item", "file_item", "voice_item", "video_item"):
                sub = it.get(key)
                if not sub:
                    continue
                media = sub.get("media")
                if media:
                    out.append(media)
                if key == "video_item" and sub.get("thumb_media"):
                    out.append(sub["thumb_media"])
        return out

    @staticmethod
    def extract_text(msg: dict) -> str:
        """Concatenate all text items from an inbound message."""
        return "\n".join(
            it["text_item"].get("text", "")
            for it in msg.get("item_list", [])
            if it.get("type") == _ITEM_TEXT and it.get("text_item")
        )

    @staticmethod
    def is_user_msg(msg: dict) -> bool:
        return msg.get("message_type") == _MSG_USER

    def run_loop(self, on_message: Callable[["WxBotClient", dict], None], poll_timeout: int = 30) -> None:
        """Blocking inbound message loop. Run in a background thread.

        ``on_message(client, msg)`` is invoked once per new user message.
        Errors raised by the callback are logged but do not stop the loop.
        """
        logger.info(f"WeChat: listening (bot_id={self.bot_id})")
        seen: set = set()
        while True:
            try:
                for msg in self.get_updates(poll_timeout):
                    mid = msg.get("message_id", 0)
                    if not self.is_user_msg(msg) or mid in seen:
                        continue
                    seen.add(mid)
                    if len(seen) > 5000:
                        seen = set(list(seen)[-2000:])
                    try:
                        on_message(self, msg)
                    except Exception as e:
                        logger.error(f"WeChat: callback error: {e}")
            except KeyboardInterrupt:
                logger.info("WeChat: loop interrupted")
                break
            except Exception as e:
                logger.error(f"WeChat: loop error: {e}, retry in 5s")
                time.sleep(5)


class WeChatChannel(Channel):
    """Personal WeChat messaging channel.

    Wraps :class:`WxBotClient` (sync, blocking) inside the async
    ``Channel`` ABC. Inbound messages are received on a daemon thread and
    dispatched to the main event loop via ``call_soon_threadsafe``.
    """

    SPLIT_LIMIT = 1800

    def __init__(
        self,
        token_file: Optional[str] = None,
        allowed_users: Optional[List[str]] = None,
    ):
        super().__init__(allowed_users=allowed_users or settings.wechat_allowed_users or [])
        self.token_file = token_file or settings.wechat_token_file
        self._bot: Optional[WxBotClient] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None
        self._processed_ids: deque = deque(maxlen=2000)
        # Cache the latest ``context_token`` per user so replies stay in
        # the same WeChat conversation thread.
        self._user_ctx: dict = {}

    @property
    def channel_type(self) -> ChannelType:
        return ChannelType.WECHAT

    async def connect(self) -> bool:
        """Initialize WxBotClient (running QR login if no token)."""
        try:
            self._bot = WxBotClient(token_file=self.token_file)
            self._main_loop = asyncio.get_running_loop()

            if not self._bot.token:
                logger.warning("WeChat: no cached token, running QR login (interactive)...")
                # login_qr() blocks until QR is scanned; run off the event loop
                await asyncio.to_thread(self._bot.login_qr)

            self._loop_thread = threading.Thread(
                target=self._bot.run_loop,
                args=(self._on_native_message,),
                daemon=True,
                name="wechat-poll",
            )
            self._loop_thread.start()

            self._connected = True
            logger.info(f"WeChat: Connected (bot_id={self._bot.bot_id})")
            return True

        except ImportError as e:
            logger.error(f"WeChat: dependency missing: {e}")
            return False
        except Exception as e:
            logger.error(f"WeChat: Connect failed: {e}")
            return False

    async def disconnect(self):
        """Mark disconnected. The polling thread is daemon and exits with the process."""
        self._connected = False
        logger.info("WeChat: Disconnected")

    async def send(self, channel_id: str, content: str, **kwargs) -> bool:  # noqa: ARG002
        """Send a text reply to a WeChat user.

        ``channel_id`` is the WeChat ``user_id``. The cached
        ``context_token`` (if any) is included to keep the reply in the
        same conversation thread.
        """
        if not self._bot:
            logger.warning("WeChat: Not connected")
            return False

        ctx = self._user_ctx.get(channel_id, "")
        try:
            for chunk in self.split_text(content, self.SPLIT_LIMIT):
                await asyncio.to_thread(self._bot.send_text, channel_id, chunk, ctx)
            return True
        except Exception as e:
            logger.error(f"WeChat: Send error: {e} channel_id={channel_id}")
            return False

    async def send_typing(self, channel_id: str, context_token: str = "", cancel: bool = False) -> bool:
        """Show / cancel the typing indicator for a WeChat user.

        ``channel_id`` is the WeChat ``user_id``. The ``typing_ticket`` is
        fetched from ``getconfig`` automatically by the underlying client.
        """
        if not self._bot:
            logger.warning("WeChat: Not connected")
            return False
        try:
            await asyncio.to_thread(self._bot.send_typing, channel_id, "", context_token, cancel)
            return True
        except Exception as e:
            logger.error(f"WeChat: send_typing error: {e} channel_id={channel_id}")
            return False

    async def send_media(
        self,
        channel_id: str,
        media_type: int,
        data: bytes,
        filename: str = "",
        context_token: Optional[str] = None,
        need_thumb: bool = False,
        cdn_base_url: Optional[str] = None,
    ) -> bool:
        """Encrypt + upload media to a WeChat user (channel_id = user_id)."""
        if not self._bot:
            logger.warning("WeChat: Not connected")
            return False
        ctx = context_token if context_token is not None else self._user_ctx.get(channel_id, "")
        try:
            await asyncio.to_thread(
                self._bot.send_media,
                channel_id,
                media_type,
                data,
                filename,
                ctx,
                need_thumb,
                cdn_base_url,
            )
            return True
        except Exception as e:
            logger.error(f"WeChat: send_media error: {e} channel_id={channel_id}")
            return False

    def _on_native_message(self, bot: WxBotClient, msg: dict) -> None:
        """Sync callback (runs in poll thread). Dispatches to main loop."""
        try:
            mid = msg.get("message_id")
            if mid in self._processed_ids:
                return
            if mid:
                self._processed_ids.append(mid)

            text = bot.extract_text(msg).strip()
            media = WxBotClient.extract_media(msg)
            if not text and not media:
                return

            uid = msg.get("from_user_id", "") or "unknown"
            ctx = msg.get("context_token", "")
            if ctx:
                self._user_ctx[uid] = ctx

            if not self.check_allowlist(uid):
                logger.debug(f"WeChat: User {uid} not in allowlist")
                return

            message = Message(
                channel=ChannelType.WECHAT,
                channel_id=uid,
                sender_id=uid,
                sender_name=uid,
                content=text,
                message_id=str(mid) if mid else "",
                metadata={"context_token": ctx, "media": media},
            )

            if self._message_handler and self._main_loop:
                def _dispatch():
                    fut = asyncio.ensure_future(self._emit_message(message))
                    fut.add_done_callback(WeChatChannel._log_dispatch_error)

                self._main_loop.call_soon_threadsafe(_dispatch)
        except Exception as e:
            logger.error(f"WeChat: native message error: {e}")

    @staticmethod
    def _log_dispatch_error(fut: asyncio.Future) -> None:
        """Surface errors from dispatched message tasks."""
        if fut.exception():
            logger.error(f"WeChat: Dispatch error: {fut.exception()}")
