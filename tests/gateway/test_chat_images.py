# -*- coding: utf-8 -*-
"""Web chat pictures survive a reload: stored by hash, named by a message marker."""
import pytest

pytest.importorskip("agentica_gateway", reason="requires agentica-gateway")

from agentica_gateway.services import chat_images

PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 16


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(chat_images, "AGENTICA_HOME", str(tmp_path))


def test_save_is_content_addressed_and_resolvable():
    name = chat_images.save(PNG, "image/png")
    assert name == chat_images.save(PNG, "image/png")
    assert name.endswith(".png")
    assert chat_images.resolve(name).read_bytes() == PNG
    assert chat_images.save(PNG, "image/jpeg").endswith(".jpg")


@pytest.mark.parametrize("name", ["../x.png", "a" * 64 + ".svg", "a" * 63 + ".png", "", "a" * 64 + ".png"])
def test_resolve_rejects_malformed_or_missing_names(name):
    assert chat_images.resolve(name) is None


def test_strip_markers_cuts_marker_and_description_only():
    name = chat_images.save(PNG)
    text = "看下这个" + chat_images.marker([name]) + "\n\n[图片内容]\n一张猫"
    assert chat_images.strip_markers(text) == "看下这个"
    assert chat_images.strip_markers(chat_images.marker([name]).lstrip()) == ""
    assert chat_images.strip_markers("提到 [图片: x] 的文字") == "提到 [图片: x] 的文字"
