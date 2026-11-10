# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Publish a file or an HTML page to Tencent Cloud COS and return a public link.

Requires ``pip install cos-python-sdk-v5`` and these environment variables:

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
import html as html_lib
import mimetypes
import os
import uuid
from pathlib import Path
from typing import Optional

from markdown_it import MarkdownIt

from agentica.tools.base import Tool
from agentica.tools.helpers import tool_error, tool_result

_ENV_KEYS = ("COS_SECRET_ID", "COS_SECRET_KEY", "COS_REGION", "COS_BUCKET")
_SUBDIR = "agentica"
_client = None

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


def _cos():
    global _client
    if _client is None:
        from qcloud_cos import CosConfig, CosS3Client

        _client = CosS3Client(CosConfig(
            Region=os.environ["COS_REGION"],
            SecretId=os.environ["COS_SECRET_ID"],
            SecretKey=os.environ["COS_SECRET_KEY"],
            Scheme="https",
        ))
    return _client


def upload(body: bytes, suffix: str, content_type: str) -> str:
    """Put ``body`` under a random key and return its public URL."""
    bucket = os.environ["COS_BUCKET"]
    rel = f"{_SUBDIR}/{uuid.uuid4().hex}{suffix}"
    prefix = os.getenv("COS_KEY_PREFIX", "").strip("/")
    key = f"{prefix}/{rel}" if prefix else rel
    _cos().put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)
    base = os.getenv("COS_PUBLIC_BASE_URL", "").strip().rstrip("/")
    if base:
        return f"{base}/{rel}"
    return _cos().get_object_url(Bucket=bucket, Key=key)


def markdown_page(markdown: str, title: str) -> str:
    """A self-contained, phone-friendly HTML page for a Markdown answer."""
    md = MarkdownIt("commonmark", {"html": False, "linkify": False}).enable(["table", "strikethrough"])
    return (
        "<!doctype html><html lang=\"zh\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>{html_lib.escape(title)}</title><style>{_PAGE_CSS}</style></head>"
        f"<body><main>{md.render(markdown)}</main></body></html>"
    )


def publish_markdown(markdown: str, title: str) -> str:
    """Render Markdown to a page, upload it, return the URL."""
    return upload(markdown_page(markdown, title).encode("utf-8"), ".html", "text/html; charset=utf-8")


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
        self.register(self.publish_markdown)
        self.register(self.publish_html)
        self.register(self.publish_file)

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

    def publish_markdown(self, markdown: str, title: str) -> str:
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
            url = publish_markdown(markdown, title)
        except Exception as e:
            return tool_error(f"COS upload failed: {e}")
        return self._result(url, title=title)

    def publish_html(self, html: str, title: str = "") -> str:
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
            url = upload(html.encode("utf-8"), ".html", "text/html; charset=utf-8")
        except Exception as e:
            return tool_error(f"COS upload failed: {e}")
        return self._result(url, title=title)

    def publish_file(self, file_path: str) -> str:
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
            url = upload(path.read_bytes(), path.suffix.lower(), content_type)
        except Exception as e:
            return tool_error(f"COS upload failed: {e}")
        return self._result(url, name=path.name, size=path.stat().st_size)
