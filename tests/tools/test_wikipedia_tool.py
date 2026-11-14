# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: WikipediaTool's payload shape, which `Document.to_dict()` used to define.

1.6.0 deleted `agentica/knowledge/`, and this tool was the one caller of the
`Document` port beyond RAG: `Document(name=..., content=...).to_dict()` produced
`{"name", "meta_data", "content"}` because `meta_data` defaults to `{}` and
`exclude_none=True` keeps it. Replacing that with a dict literal silently dropped
the key, which is a wire change for anything that reads these results. Pin the
whole payload rather than just "it returns JSON".
"""
import json
import sys
import types
from unittest.mock import patch

from agentica.tools.wikipedia_tool import WikipediaTool


def _fake_wikipedia(summary):
    module = types.ModuleType("wikipedia")
    module.summary = summary
    return module


class TestWikipediaPayload:
    def test_payload_keeps_the_keys_document_to_dict_emitted(self):
        with patch.dict(sys.modules, {"wikipedia": _fake_wikipedia(lambda q: "SUMMARY")}):
            raw = WikipediaTool().search_wikipedia_single_query("Ada Lovelace")

        assert json.loads(raw) == {
            "name": "Ada Lovelace",
            "meta_data": {},
            "content": "SUMMARY",
        }

    def test_summary_text_is_passed_through_verbatim(self):
        """Chinese text must not be escaped, and nothing may be truncated."""
        body = "第一台通用计算机的设想者。"
        with patch.dict(sys.modules, {"wikipedia": _fake_wikipedia(lambda q: body)}):
            raw = WikipediaTool().search_wikipedia_single_query("q")

        assert "灵感" not in raw
        assert body in raw

    def test_a_single_query_is_routed_to_the_single_query_path(self):
        seen = []

        def summary(query):
            seen.append(query)
            return "S"

        with patch.dict(sys.modules, {"wikipedia": _fake_wikipedia(summary)}):
            WikipediaTool().search_wikipedia_single_query("only one")

        assert seen == ["only one"]
