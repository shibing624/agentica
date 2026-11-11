# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Publish a file or an HTML page to Tencent Cloud COS and return a public link.

Uses native async ``httpx.AsyncClient`` + COS XML V5 HMAC-SHA1 signing (no external
SDK required at runtime; ``_client`` hook retained for custom/mock clients).
Requires these environment variables:

- ``COS_SECRET_ID`` / ``COS_SECRET_KEY`` / ``COS_REGION`` / ``COS_BUCKET``
- ``COS_PUBLIC_BASE_URL`` (optional): the bucket's custom domain, e.g.
  ``https://share.example.com``. Buckets created after 2024-01-01 force every
  object served from the default ``*.myqcloud.com`` domain to download
  (``Content-Disposition: attachment``), so a page only opens in a browser
  through a custom domain. Without it the default domain is used.
- ``COS_KEY_PREFIX`` (optional): the bucket directory ``COS_PUBLIC_BASE_URL``
  serves as its root, e.g. ``public`` when ``https://cdn.example.com/public/``
  maps to ``<bucket>/public/``. Objects are written under it.

The bucket must be public-read: the returned link carries no signature.
"""
import asyncio
import hashlib
import hmac
import html as html_lib
import inspect
import mimetypes
import os
import re
import time
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import aiofiles
import httpx
from markdown_it import MarkdownIt

from agentica.tools.base import Tool
from agentica.tools.helpers import tool_error, tool_result

_ENV_KEYS = ("COS_SECRET_ID", "COS_SECRET_KEY", "COS_REGION", "COS_BUCKET")
_SUBDIR = "agentica"
_client = None
_http_client: Optional[httpx.AsyncClient] = None
_http_client_loop: Optional[asyncio.AbstractEventLoop] = None
_http_client_cls = None

_XML_CODE_RE = re.compile(r"<Code>([^<]+)</Code>")
_XML_MSG_RE = re.compile(r"<Message>([^<]+)</Message>")

_PAGE_CSS = (
    "body{margin:0;background:#fff;color:#1f2328;font:16px/1.7 -apple-system,BlinkMacSystemFont,"
    "'PingFang SC','Helvetica Neue',sans-serif}"
    "main{max-width:760px;margin:0 auto;padding:20px 16px 48px}"
    "h1,h2,h3{line-height:1.35;margin:1.4em 0 .5em}h1{font-size:1.5em}h2{font-size:1.25em}"
    "pre{background:#f6f8fa;padding:12px;border-radius:8px;overflow:auto;font-size:13px}"
    "code{font-family:ui-monospace,Menlo,monospace;font-size:.9em}"
    ":not(pre)>code{background:#f0f2f4;padding:1px 5px;border-radius:4px}"
    "table{border-collapse:collapse;display:block;overflow:auto;font-size:14px}"
    "th,td{border:1px solid #d0d7de;padding:6px 10px}th{background:#f6f8fa}"
    "blockquote{margin:0;padding:0 12px;border-left:3px solid #d0d7de;color:#59636e}"
    "a{color:#0969da}"
    "@media(prefers-color-scheme:dark){body{background:#0d1117;color:#e6edf3}"
    "pre,th{background:#161b22}:not(pre)>code{background:#21262d}th,td{border-color:#30363d}a{color:#4493f8}}"
)


def cos_configured() -> bool:
    """All four required COS variables are set."""
    return all(os.getenv(k) for k in _ENV_KEYS)


def cos_pages_open_in_browser() -> bool:
    """Published pages open as pages (custom domain), not as downloads."""
    return cos_configured() and bool(os.getenv("COS_PUBLIC_BASE_URL", "").strip())


def _encode_cos_path(path: str) -> str:
    """Percent-encode an object key for URL paths while preserving ``/`` separators."""
    encoded = quote(path.lstrip("/").encode("utf-8"), safe="/-_.~")
    return encoded.replace("./", ".%2F")


def _cos_authorization(
    secret_id: str,
    secret_key: str,
    method: str,
    key: str,
    host: str,
    now: Optional[int] = None,
) -> str:
    """Build a Tencent Cloud COS XML V5 Authorization header (HMAC-SHA1).

    Matches ``qcloud_cos.cos_auth.CosS3Auth`` byte-for-byte: ``uri_path`` uses
    the unencoded ``/<key>`` while header values are percent-encoded with
    ``safe='-_.~'``.
    """
    ts = int(now if now is not None else time.time())
    key_time = f"{ts - 60};{ts + 3600}"
    sign_key = hmac.new(
        secret_key.encode("utf-8"),
        key_time.encode("utf-8"),
        hashlib.sha1,
    ).hexdigest()
    uri_path = "/" + key.lstrip("/")
    encoded_host = quote(host.strip().encode("utf-8"), safe="-_.~")
    http_string = f"{method.lower()}\n{uri_path}\n\nhost={encoded_host}\n"
    http_sha1 = hashlib.sha1(http_string.encode("utf-8")).hexdigest()
    string_to_sign = f"sha1\n{key_time}\n{http_sha1}\n"
    signature = hmac.new(
        sign_key.encode("utf-8"),
        string_to_sign.encode("utf-8"),
        hashlib.sha1,
    ).hexdigest()
    return (
        f"q-sign-algorithm=sha1"
        f"&q-ak={secret_id}"
        f"&q-sign-time={key_time}"
        f"&q-key-time={key_time}"
        f"&q-header-list=host"
        f"&q-url-param-list="
        f"&q-signature={signature}"
    )


def _format_cos_http_error(resp: httpx.Response) -> str:
    """Extract ``<Code>`` and ``<Message>`` from a COS XML error response if present."""
    body = (resp.text or "").strip()
    code_match = _XML_CODE_RE.search(body)
    msg_match = _XML_MSG_RE.search(body)
    if code_match or msg_match:
        code = code_match.group(1).strip() if code_match else str(resp.status_code)
        msg = msg_match.group(1).strip() if msg_match else body[:200]
        return f"HTTP {resp.status_code} ({code}): {msg}"
    if body:
        return f"HTTP {resp.status_code}: {body[:300]}"
    return f"HTTP {resp.status_code}"


def _get_http_client() -> httpx.AsyncClient:
    """Return a loop-scoped shared ``httpx.AsyncClient`` for COS uploads."""
    global _http_client, _http_client_loop, _http_client_cls
    loop = asyncio.get_running_loop()
    if (
        _http_client is None
        or _http_client_loop is not loop
        or _http_client_cls is not httpx.AsyncClient
        or getattr(_http_client, "is_closed", False)
    ):
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
        _http_client_loop = loop
        _http_client_cls = httpx.AsyncClient
    return _http_client


async def upload(body: bytes, suffix: str, content_type: str) -> str:
    """Put ``body`` under a random key asynchronously and return its public URL."""
    bucket = os.environ["COS_BUCKET"]
    region = os.environ["COS_REGION"]
    rel = f"{_SUBDIR}/{uuid.uuid4().hex}{suffix}"
    prefix = os.getenv("COS_KEY_PREFIX", "").strip("/")
    key = f"{prefix}/{rel}" if prefix else rel
    base = os.getenv("COS_PUBLIC_BASE_URL", "").strip().rstrip("/")
    encoded_rel = _encode_cos_path(rel)
    encoded_key = _encode_cos_path(key)

    if _client is not None:
        put_fn = _client.put_object
        if inspect.iscoroutinefunction(put_fn):
            await put_fn(Bucket=bucket, Key=key, Body=body, ContentType=content_type)
        else:
            res = await asyncio.to_thread(
                put_fn, Bucket=bucket, Key=key, Body=body, ContentType=content_type
            )
            if inspect.isawaitable(res):
                await res
        if base:
            return f"{base}/{encoded_rel}"
        url_fn = _client.get_object_url
        if inspect.iscoroutinefunction(url_fn):
            return await url_fn(Bucket=bucket, Key=key)
        url_res = await asyncio.to_thread(url_fn, Bucket=bucket, Key=key)
        return (await url_res) if inspect.isawaitable(url_res) else url_res

    host = f"{bucket}.cos.{region}.myqcloud.com"
    default_url = f"https://{host}/{encoded_key}"
    auth = _cos_authorization(
        secret_id=os.environ["COS_SECRET_ID"],
        secret_key=os.environ["COS_SECRET_KEY"],
        method="PUT",
        key=key,
        host=host,
    )
    headers = {
        "Host": host,
        "Content-Type": content_type,
        "Authorization": auth,
    }
    client = _get_http_client()
    last_exc: Optional[Exception] = None
    for attempt in range(2):
        try:
            resp = await client.put(default_url, content=body, headers=headers)
        except httpx.TransportError as exc:
            last_exc = exc
            if attempt == 0:
                continue
            raise
        if resp.status_code >= 500 and attempt == 0:
            continue
        if resp.status_code >= 400:
            raise RuntimeError(_format_cos_http_error(resp))
        break
    if base:
        return f"{base}/{encoded_rel}"
    return default_url


def markdown_page(markdown: str, title: str) -> str:
    """A self-contained, phone-friendly HTML page for a Markdown answer."""
    md = MarkdownIt("commonmark", {"html": False, "linkify": False}).enable(["table", "strikethrough"])
    return (
        "<!doctype html><html lang=\"zh\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>{html_lib.escape(title)}</title><style>{_PAGE_CSS}</style></head>"
        f"<body><main>{md.render(markdown)}</main></body></html>"
    )


async def publish_markdown(markdown: str, title: str) -> str:
    """Render Markdown to a page, upload it, return the URL."""
    return await upload(markdown_page(markdown, title).encode("utf-8"), ".html", "text/html; charset=utf-8")


class CosPublishTool(Tool):
    """Upload to COS and hand back a URL anyone can open."""

    SYSTEM_PROMPT = (
        "## Public share links (publish_markdown / publish_html / publish_file)\n\n"
        "These upload to a public bucket and return a URL. `publish_markdown` renders "
        "Markdown into a phone-friendly page (tables, code, dark mode); `publish_html` "
        "takes a self-contained page you wrote (inline CSS/JS, no local file "
        "references) for charts or custom layouts; `publish_file` uploads a local file.\n\n"
        "You decide when a page reads better than chat text. Typical cases: the user "
        "asks for a page, report or something to share; or, on an IM channel, the "
        "answer is long or leans on tables, long code or many sections. Then reply "
        "with the key conclusion in plain text plus the link, not the whole answer "
        "twice. Short or conversational answers stay as text. Anyone with the link "
        "can read it, so never publish secrets, credentials or private data."
    )

    def __init__(self) -> None:
        super().__init__(name="cos_publish_tool")
        self.register(self.publish_markdown, concurrency_safe=True)
        self.register(self.publish_html, concurrency_safe=True)
        self.register(self.publish_file, concurrency_safe=True)

    def get_system_prompt(self) -> Optional[str]:
        return self.SYSTEM_PROMPT

    @staticmethod
    def _result(url: str, **extra) -> str:
        if not cos_pages_open_in_browser():
            extra["note"] = (
                "No COS_PUBLIC_BASE_URL (custom domain) is set: browsers download "
                "this link instead of displaying it. Tell the user."
            )
        return tool_result(url=url, **extra)

    async def publish_markdown(self, markdown: str, title: str) -> str:
        """Render Markdown into a phone-friendly web page and return its public URL.

        Args:
            markdown: The page body in Markdown (tables, code blocks, headings).
                Raw HTML in it is shown as text, not rendered.
            title: Page title shown in the browser tab.

        Returns:
            JSON with `url`.
        """
        if not cos_configured():
            return tool_error("COS is not configured (COS_SECRET_ID/COS_SECRET_KEY/COS_REGION/COS_BUCKET).")
        try:
            url = await publish_markdown(markdown, title)
        except Exception as e:
            return tool_error(f"COS upload failed: {e}")
        return self._result(url, title=title)

    async def publish_html(self, html: str, title: str = "") -> str:
        """Publish an HTML page and return its public URL.

        Args:
            html: A complete, self-contained HTML document (inline CSS, no
                local file references). Include
                `<meta name="viewport" content="width=device-width,initial-scale=1">`
                so it reads well on a phone.
            title: Short label for the page, echoed back in the result.

        Returns:
            JSON with `url`.
        """
        if not cos_configured():
            return tool_error("COS is not configured (COS_SECRET_ID/COS_SECRET_KEY/COS_REGION/COS_BUCKET).")
        try:
            url = await upload(html.encode("utf-8"), ".html", "text/html; charset=utf-8")
        except Exception as e:
            return tool_error(f"COS upload failed: {e}")
        return self._result(url, title=title)

    async def publish_file(self, file_path: str) -> str:
        """Upload a local file (image, PDF, CSV, HTML …) and return its public URL.

        Args:
            file_path: Absolute path of the file to upload.

        Returns:
            JSON with `url`, `name` and `size`.
        """
        if not cos_configured():
            return tool_error("COS is not configured (COS_SECRET_ID/COS_SECRET_KEY/COS_REGION/COS_BUCKET).")
        path = Path(os.path.expanduser(file_path))
        if not path.is_file():
            return tool_error(f"Not a file: {file_path}")
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/"):
            content_type += "; charset=utf-8"
        try:
            async with aiofiles.open(path, "rb") as fh:
                data = await fh.read()
        except OSError as e:
            return tool_error(f"Cannot read file {file_path}: {e}")
        try:
            url = await upload(data, path.suffix.lower(), content_type)
        except Exception as e:
            return tool_error(f"COS upload failed: {e}")
        return self._result(url, name=path.name, size=len(data))
