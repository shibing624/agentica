"""CosPublishTool: uploads land in the bucket with the right type and come back as public links."""
import json

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


def test_publish_html_uploads_as_html_and_warns_without_custom_domain(cos):
    out = json.loads(cos_tool.CosPublishTool().publish_html("<h1>hi</h1>", title="t"))
    put = cos.puts[0]
    assert put["type"] == "text/html; charset=utf-8"
    assert put["body"] == "<h1>hi</h1>".encode()
    assert put["key"].startswith("agentica/") and put["key"].endswith(".html")
    assert out["url"] == f"https://bkt-1.cos.ap-beijing.myqcloud.com/{put['key']}"
    # The default domain forces a download for buckets made after 2024: say so.
    assert "COS_PUBLIC_BASE_URL" in out["note"]


def test_custom_domain_is_used_for_the_link(cos, monkeypatch):
    monkeypatch.setenv("COS_PUBLIC_BASE_URL", "https://share.example.com/")
    out = json.loads(cos_tool.CosPublishTool().publish_html("<p>x</p>"))
    assert out["url"] == f"https://share.example.com/{cos.puts[0]['key']}"
    assert "note" not in out


def test_key_prefix_is_the_directory_the_public_url_serves(cos, monkeypatch):
    """``https://cdn/public/`` serves ``<bucket>/public/``: the object goes under
    the prefix, the link does not repeat it."""
    monkeypatch.setenv("COS_PUBLIC_BASE_URL", "https://cdn.example.com/public/")
    monkeypatch.setenv("COS_KEY_PREFIX", "/public/")
    out = json.loads(cos_tool.CosPublishTool().publish_html("<p>x</p>"))
    key = cos.puts[0]["key"]
    assert key.startswith("public/agentica/")
    assert out["url"] == "https://cdn.example.com/public/" + key[len("public/"):]


def test_publish_file_keeps_suffix_and_type(cos, tmp_path):
    f = tmp_path / "report.CSV"
    f.write_text("a,b\n1,2\n")
    out = json.loads(cos_tool.CosPublishTool().publish_file(str(f)))
    put = cos.puts[0]
    assert put["key"].endswith(".csv")
    assert put["type"] == "text/csv; charset=utf-8"
    assert out["name"] == "report.CSV" and out["size"] == f.stat().st_size


def test_unconfigured_is_an_error_not_an_upload(monkeypatch):
    for k in ("COS_SECRET_ID", "COS_SECRET_KEY", "COS_REGION", "COS_BUCKET"):
        monkeypatch.delenv(k, raising=False)
    out = json.loads(cos_tool.CosPublishTool().publish_html("<p>x</p>"))
    assert "error" in out


def test_markdown_page_renders_tables_and_escapes_raw_html():
    page = cos_tool.markdown_page("| a | b |\n|---|---|\n| 1 | 2 |\n\n<script>x</script>", "标题<")
    assert "<table>" in page and "<td>1</td>" in page
    assert "<script>" not in page
    assert "<title>标题&lt;</title>" in page
    assert 'name="viewport"' in page


def test_publish_markdown_tool_uploads_a_rendered_page(cos, monkeypatch):
    monkeypatch.setenv("COS_PUBLIC_BASE_URL", "https://share.example.com")
    out = json.loads(cos_tool.CosPublishTool().publish_markdown("| a |\n|---|\n| 1 |", title="报告"))
    put = cos.puts[0]
    assert put["type"] == "text/html; charset=utf-8"
    body = put["body"].decode()
    assert "<td>1</td>" in body and "<title>报告</title>" in body
    assert out["url"] == f"https://share.example.com/{put['key']}"
