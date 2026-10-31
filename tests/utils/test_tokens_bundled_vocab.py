# -*- coding: utf-8 -*-
"""Token counting must not depend on the network or tiktoken's temp-dir cache.

``tiktoken.get_encoding`` downloads its vocabulary with no timeout and caches it
under the system temp dir; after a reboot cleared that dir, the CLI sat for
minutes before showing its prompt. The encoding is now built from the
vocabulary shipped in the package.
"""
import gzip
import hashlib
import socket

import requests

from agentica.utils import tokens

# tiktoken_ext.openai_public: expected_hash of o200k_base.tiktoken
O200K_BASE_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"


def test_bundled_vocab_is_tiktokens_o200k_base():
    with gzip.open(tokens._O200K_BASE_PATH, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == O200K_BASE_SHA256


def test_encoding_builds_without_network_or_cache(monkeypatch, tmp_path):
    def no_network(*args, **kwargs):
        raise AssertionError("token counting tried to reach the network")

    monkeypatch.setattr(requests, "get", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))
    tokens._get_tiktoken_encoding.cache_clear()
    try:
        # Known o200k_base ids, so a wrong vocab or regex fails here.
        enc = tokens._get_tiktoken_encoding()
        assert enc.encode("hello world") == [24912, 2375]
        assert enc.n_vocab == 200019
        assert tokens.count_text_tokens("上下文压缩在网关侧计费", model_id="gpt-4") == 10
        assert tokens.count_text_tokens("x <|endoftext|>") == len(
            enc.encode("x <|endoftext|>", disallowed_special=())
        )
    finally:
        tokens._get_tiktoken_encoding.cache_clear()
    assert list(tmp_path.iterdir()) == []


def test_every_model_id_uses_the_same_encoding():
    text = "def f(x):\n    return x"
    assert {
        tokens.count_text_tokens(text, model_id=m)
        for m in ("gpt-4", "gpt-4o", "deepseek-v4-flash", "claude-opus-4")
    } == {len(tokens._get_tiktoken_encoding().encode(text))}

