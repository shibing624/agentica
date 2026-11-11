"""CosPublishTool: native async uploads land in the bucket with the right type and come back as public links."""
import asyncio
import inspect
import json
import time

import aiofiles
import httpx
import pytest

from agentica.tools import cos_tool


class _FakeCos:
    def __init__(self):
        self.puts = []

    def put_object(self, Bucket, Key, Body, ContentType):
        self.puts.append({"bucket": Bucket, "key": Key, "body": Body, "type": ContentType})

    def get_object_url(self, Bucket, Key):
        return f"https://{Bucket}.cos.ap-beijing.myqcloud.com/{Key}"


@pytest.fixture
def cos(monkeypatch):
    for k, v in {"COS_SECRET_ID": "id", "COS_SECRET_KEY": "key",
                 "COS_REGION": "ap-beijing", "COS_BUCKET": "bkt-1"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("COS_PUBLIC_BASE_URL", raising=False)
    monkeypatch.delenv("COS_KEY_PREFIX", raising=False)
    fake = _FakeCos()
    monkeypatch.setattr(cos_tool, "_client", fake)
    return fake


def test_cos_publish_methods_are_native_async_and_concurrency_safe():
    tool = cos_tool.CosPublishTool()
    for name in ("publish_markdown", "publish_html", "publish_file"):
        fn = tool.functions[name]
        assert inspect.iscoroutinefunction(fn.entrypoint)
        assert fn.concurrency_safe is True


def test_cos_authorization_matches_official_sdk_known_answer():
    """Byte-for-byte known-answer vector verified against qcloud_cos.CosS3Auth."""
    header = cos_tool._cos_authorization(
        secret_id="AKIDexample",
        secret_key="secretkey",
        method="PUT",
        key="agentica/abc123.html",
        host="mybkt-125000.cos.ap-shanghai.myqcloud.com",
        now=1700000000,
    )
    expected = (
        "q-sign-algorithm=sha1"
        "&q-ak=AKIDexample"
        "&q-sign-time=1699999940;1700003600"
        "&q-key-time=1699999940;1700003600"
        "&q-header-list=host"
        "&q-url-param-list="
        "&q-signature=30a57dd291798ccd04aa883db8c7edda8b77bf7f"
    )
    assert header == expected


@pytest.mark.asyncio
async def test_publish_html_uploads_as_html_and_warns_without_custom_domain(cos):
    out = json.loads(await cos_tool.CosPublishTool().publish_html("<h1>hi</h1>", title="t"))
    put = cos.puts[0]
    assert put["type"] == "text/html; charset=utf-8"
    assert put["body"] == "<h1>hi</h1>".encode()
    assert put["key"].startswith("agentica/") and put["key"].endswith(".html")
    assert out["url"] == f"https://bkt-1.cos.ap-beijing.myqcloud.com/{put['key']}"
    assert "COS_PUBLIC_BASE_URL" in out["note"]


@pytest.mark.asyncio
async def test_custom_domain_is_used_for_the_link(cos, monkeypatch):
    monkeypatch.setenv("COS_PUBLIC_BASE_URL", "https://share.example.com/")
    out = json.loads(await cos_tool.CosPublishTool().publish_html("<p>x</p>"))
    assert out["url"] == f"https://share.example.com/{cos.puts[0]['key']}"
    assert "note" not in out


@pytest.mark.asyncio
async def test_key_prefix_is_the_directory_the_public_url_serves(cos, monkeypatch):
    """``https://cdn/public/`` serves ``<bucket>/public/``: the object goes under
    the prefix, the link does not repeat it."""
    monkeypatch.setenv("COS_PUBLIC_BASE_URL", "https://cdn.example.com/public/")
    monkeypatch.setenv("COS_KEY_PREFIX", "/public/")
    out = json.loads(await cos_tool.CosPublishTool().publish_html("<p>x</p>"))
    key = cos.puts[0]["key"]
    assert key.startswith("public/agentica/")
    assert out["url"] == "https://cdn.example.com/public/" + key[len("public/"):]


@pytest.mark.asyncio
async def test_publish_file_keeps_suffix_and_type(cos, tmp_path):
    f = tmp_path / "report.CSV"
    f.write_text("a,b\n1,2\n")
    out = json.loads(await cos_tool.CosPublishTool().publish_file(str(f)))
    put = cos.puts[0]
    assert put["key"].endswith(".csv")
    assert put["type"] == "text/csv; charset=utf-8"
    assert out["name"] == "report.CSV" and out["size"] == f.stat().st_size


@pytest.mark.asyncio
async def test_publish_file_reports_local_read_error_distinct_from_cos_upload(cos, tmp_path, monkeypatch):
    f = tmp_path / "unreadable.txt"
    f.write_text("hello")

    def _fail_open(*args, **kwargs):
        raise PermissionError("permission denied")

    monkeypatch.setattr(aiofiles, "open", _fail_open)
    out = json.loads(await cos_tool.CosPublishTool().publish_file(str(f)))
    assert "Cannot read file" in out["error"]
    assert "COS upload failed" not in out["error"]


@pytest.mark.asyncio
async def test_unconfigured_is_an_error_not_an_upload(monkeypatch):
    for k in ("COS_SECRET_ID", "COS_SECRET_KEY", "COS_REGION", "COS_BUCKET"):
        monkeypatch.delenv(k, raising=False)
    out = json.loads(await cos_tool.CosPublishTool().publish_html("<p>x</p>"))
    assert "error" in out


def test_markdown_page_renders_tables_and_escapes_raw_html():
    page = cos_tool.markdown_page("| a | b |\n|---|---|\n| 1 | 2 |\n\n<script>x</script>", "标题<")
    assert "<table>" in page and "<td>1</td>" in page
    assert "<script>" not in page
    assert "<title>标题&lt;</title>" in page
    assert 'name="viewport"' in page


@pytest.mark.asyncio
async def test_publish_markdown_tool_uploads_a_rendered_page(cos, monkeypatch):
    monkeypatch.setenv("COS_PUBLIC_BASE_URL", "https://share.example.com")
    out = json.loads(await cos_tool.CosPublishTool().publish_markdown("| a |\n|---|\n| 1 |", title="报告"))
    put = cos.puts[0]
    assert put["type"] == "text/html; charset=utf-8"
    assert b"<table>" in put["body"]
    assert out["url"].endswith(".html") and out["title"] == "报告"


@pytest.mark.asyncio
async def test_real_httpx_transport_encodes_special_key_and_retries_5xx(monkeypatch):
    """Real httpx.AsyncClient + MockTransport: # and ? in COS_KEY_PREFIX are percent-encoded
    in the HTTP request path, and a transient 503 retries once."""
    for k, v in {
        "COS_SECRET_ID": "ak-123",
        "COS_SECRET_KEY": "sk-456",
        "COS_REGION": "ap-shanghai",
        "COS_BUCKET": "mybkt-125000",
        "COS_KEY_PREFIX": "public#reports?v1",
    }.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("COS_PUBLIC_BASE_URL", raising=False)
    monkeypatch.setattr(cos_tool, "_client", None)

    requests_seen = []

    def _handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        if len(requests_seen) == 1:
            return httpx.Response(503, text="<Error><Code>SlowDown</Code></Error>", request=request)
        return httpx.Response(200, request=request)

    real_client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    monkeypatch.setattr(cos_tool, "_get_http_client", lambda: real_client)
    try:
        out = json.loads(await cos_tool.CosPublishTool().publish_html("<p>native</p>", title="native"))
        assert len(requests_seen) == 2
        req = requests_seen[-1]
        # # and ? must be percent-encoded in raw_path, never split into fragment or query
        assert req.url.raw_path.startswith(b"/public%23reports%3Fv1/agentica/")
        assert req.url.query == b""
        assert "q-sign-algorithm=sha1&q-ak=ak-123" in req.headers["Authorization"]
        assert out["url"].startswith("https://mybkt-125000.cos.ap-shanghai.myqcloud.com/public%23reports%3Fv1/agentica/")
    finally:
        await real_client.aclose()


@pytest.mark.asyncio
async def test_cos_xml_error_code_and_message_are_surfaced(monkeypatch):
    for k, v in {
        "COS_SECRET_ID": "ak-123",
        "COS_SECRET_KEY": "sk-456",
        "COS_REGION": "ap-shanghai",
        "COS_BUCKET": "mybkt-125000",
    }.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("COS_KEY_PREFIX", raising=False)
    monkeypatch.setattr(cos_tool, "_client", None)

    xml_err = (
        "<?xml version='1.0' encoding='utf-8'?>"
        "<Error><Code>AccessDenied</Code><Message>Access Denied to bucket.</Message></Error>"
    )
    real_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(403, text=xml_err, request=req))
    )
    monkeypatch.setattr(cos_tool, "_get_http_client", lambda: real_client)
    try:
        out = json.loads(await cos_tool.CosPublishTool().publish_html("<p>x</p>"))
        assert "HTTP 403 (AccessDenied): Access Denied to bucket." in out["error"]
    finally:
        await real_client.aclose()


@pytest.mark.asyncio
async def test_sync_client_runs_off_event_loop_and_async_client_is_awaited(monkeypatch):
    """A blocking sync _client runs in a worker thread without stalling the loop;
    an async _client is awaited directly."""
    for k, v in {
        "COS_SECRET_ID": "id",
        "COS_SECRET_KEY": "key",
        "COS_REGION": "ap-beijing",
        "COS_BUCKET": "bkt-1",
    }.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("COS_PUBLIC_BASE_URL", raising=False)

    class _SlowSyncCos:
        def put_object(self, **kwargs):
            time.sleep(0.15)

        def get_object_url(self, Bucket, Key):
            return f"https://{Bucket}.cos.ap-beijing.myqcloud.com/{Key}"

    monkeypatch.setattr(cos_tool, "_client", _SlowSyncCos())
    ticks = []
    done = False

    async def _ticker():
        while not done:
            ticks.append(time.monotonic())
            await asyncio.sleep(0.01)

    ticker_task = asyncio.create_task(_ticker())
    await cos_tool.CosPublishTool().publish_html("<p>sync</p>")
    done = True
    await ticker_task
    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    assert gaps and max(gaps) < 0.08, f"event loop stalled: max gap {max(gaps):.3f}s"

    class _AsyncCos:
        def __init__(self):
            self.called = False

        async def put_object(self, **kwargs):
            await asyncio.sleep(0.01)
            self.called = True

        async def get_object_url(self, Bucket, Key):
            return f"https://async.example.com/{Key}"

    async_fake = _AsyncCos()
    monkeypatch.setattr(cos_tool, "_client", async_fake)
    out = json.loads(await cos_tool.CosPublishTool().publish_html("<p>async</p>"))
    assert async_fake.called is True
    assert out["url"].startswith("https://async.example.com/")
