# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for ZhipuWebSearchTool against the /paas/v4/web_search endpoint.
"""
import asyncio
import json
import logging
import os
import sys
from unittest.mock import patch

import httpx
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.tools.zhipu_web_search_tool import (
    DEFAULT_SEARCH_ENGINE,
    MAX_COUNT,
    MAX_QUERY_CHARS,
    MIN_COUNT,
    RECENCY_FILTERS,
    SEARCH_ENGINES,
    ZHIPU_WEB_SEARCH_URL,
    ZhipuWebSearchTool,
)

# Verbatim from Zhipu's official docs response example.
OFFICIAL_RESPONSE = {
    "created": 1748261757,
    "id": "20250526201557dda85ca6801b467b",
    "request_id": "20250526201557dda85ca6801b467b",
    "search_intent": [
        {"intent": "SEARCH_ALL", "keywords": "2025年4月 财经新闻", "query": "搜索2025年4月的财经新闻"}
    ],
    "search_result": [
        {
            "content": "1-4月我国对外直接投资575.4亿美元。",
            "icon": "https://sfile.chatglm.cn/searchImage/sohu_icon_new.jpg",
            "link": "https://www.sohu.com/a/897879632_121123890",
            "media": "搜狐",
            "publish_date": "2025-05-23",
            "refer": "ref_1",
            "title": "2025年5月23日财经早资讯",
        }
    ],
}

_real_client = httpx.AsyncClient


class FakeZhipu:
    """Patches httpx so the tool talks to a fake server, recording requests."""

    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self.body = OFFICIAL_RESPONSE if body is None else body
        self.requests = []

    def _handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if isinstance(self.body, str):
            return httpx.Response(self.status_code, text=self.body)
        return httpx.Response(self.status_code, json=self.body)

    def __enter__(self):
        def factory(**kwargs):
            return _real_client(transport=httpx.MockTransport(self._handler), **kwargs)

        self._patch = patch("httpx.AsyncClient", factory)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        return False

    @property
    def payload(self):
        return json.loads(self.requests[-1].content)


class TestZhipuWebSearchRequest:
    def test_posts_to_web_search_endpoint_with_bearer_auth(self):
        with FakeZhipu() as server:
            asyncio.run(ZhipuWebSearchTool(api_key="k").zhipu_web_search("财经新闻"))
        request = server.requests[-1]
        assert request.method == "POST"
        assert str(request.url) == ZHIPU_WEB_SEARCH_URL
        assert request.headers["authorization"] == "Bearer k"

    def test_max_results_is_pushed_down_as_count(self):
        with FakeZhipu() as server:
            asyncio.run(ZhipuWebSearchTool(api_key="k").zhipu_web_search("q", max_results=3))
        assert server.payload["count"] == 3

    def test_count_falls_back_to_instance_default(self):
        with FakeZhipu() as server:
            asyncio.run(ZhipuWebSearchTool(api_key="k", count=12).zhipu_web_search("q", max_results=0))
        assert server.payload["count"] == 12

    def test_count_is_capped_at_api_maximum(self):
        with FakeZhipu() as server:
            asyncio.run(ZhipuWebSearchTool(api_key="k").zhipu_web_search("q", max_results=999))
        assert server.payload["count"] == MAX_COUNT

    def test_count_is_clamped_to_api_minimum(self):
        # The API documents 1 <= count <= 50; a negative would be rejected.
        with FakeZhipu() as server:
            asyncio.run(ZhipuWebSearchTool(api_key="k").zhipu_web_search("q", max_results=-5))
        assert server.payload["count"] == MIN_COUNT

    def test_user_id_is_sent_only_when_set(self):
        with FakeZhipu() as server:
            asyncio.run(ZhipuWebSearchTool(api_key="k").zhipu_web_search("q"))
            assert "user_id" not in server.payload
            asyncio.run(ZhipuWebSearchTool(api_key="k", user_id="tenant-42").zhipu_web_search("q"))
        assert server.payload["user_id"] == "tenant-42"

    def test_long_query_warns_but_still_searches(self, caplog):
        long_query = "光" * (MAX_QUERY_CHARS + 5)
        with caplog.at_level(logging.WARNING):
            with FakeZhipu() as server:
                raw = asyncio.run(ZhipuWebSearchTool(api_key="k").zhipu_web_search(long_query))
        assert server.payload["search_query"] == long_query, "must not silently truncate the query"
        assert json.loads(raw), "a long query still returns results"

    def test_defaults_to_search_pro_without_intent_recognition(self):
        with FakeZhipu() as server:
            asyncio.run(ZhipuWebSearchTool(api_key="k").zhipu_web_search("q"))
        assert server.payload["search_engine"] == DEFAULT_SEARCH_ENGINE == "search_pro"
        assert server.payload["search_intent"] is False

    @pytest.mark.parametrize("engine", list(SEARCH_ENGINES))
    def test_every_engine_reaches_the_wire(self, engine):
        with FakeZhipu() as server:
            asyncio.run(ZhipuWebSearchTool(api_key="k", search_engine=engine).zhipu_web_search("q"))
        assert server.payload["search_engine"] == engine

    def test_optional_filters_are_omitted_when_unset(self):
        with FakeZhipu() as server:
            asyncio.run(ZhipuWebSearchTool(api_key="k").zhipu_web_search("q"))
        payload = server.payload
        for field in ("search_domain_filter", "search_recency_filter", "content_size"):
            assert field not in payload

    def test_optional_filters_are_sent_when_set(self):
        with FakeZhipu() as server:
            tool = ZhipuWebSearchTool(
                api_key="k",
                search_recency_filter="oneWeek",
                search_domain_filter="www.sohu.com",
                content_size="high",
            )
            asyncio.run(tool.zhipu_web_search("q"))
        payload = server.payload
        assert payload["search_recency_filter"] == "oneWeek"
        assert payload["search_domain_filter"] == "www.sohu.com"
        assert payload["content_size"] == "high"

    def test_request_id_meets_api_length_requirement(self):
        with FakeZhipu() as server:
            asyncio.run(ZhipuWebSearchTool(api_key="k").zhipu_web_search("q"))
        assert 6 <= len(server.payload["request_id"]) <= 64


class TestZhipuWebSearchResponse:
    def test_parses_official_response_shape(self):
        with FakeZhipu():
            raw = asyncio.run(ZhipuWebSearchTool(api_key="k").zhipu_web_search("q"))
        results = json.loads(raw)
        assert len(results) == 1
        assert results[0]["title"] == "2025年5月23日财经早资讯"
        assert results[0]["link"] == "https://www.sohu.com/a/897879632_121123890"
        assert results[0]["media"] == "搜狐"
        assert results[0]["publish_date"] == "2025-05-23"

    def test_strips_fields_that_only_waste_context(self):
        with FakeZhipu():
            raw = asyncio.run(ZhipuWebSearchTool(api_key="k").zhipu_web_search("q"))
        assert set(json.loads(raw)[0]) == {"title", "content", "link", "media", "publish_date"}

    def test_drops_empty_values(self):
        body = {"search_result": [{"title": "t", "link": "u", "media": "", "publish_date": None}]}
        with FakeZhipu(body=body):
            raw = asyncio.run(ZhipuWebSearchTool(api_key="k").zhipu_web_search("q"))
        assert set(json.loads(raw)[0]) == {"title", "link"}

    def test_truncates_when_the_engine_overshoots_count(self):
        # search_pro_sogou snaps count up to the nearest of 10/20/30/40/50, and
        # the others overshoot on some queries: count is a hint, not a promise.
        body = {"search_result": [{"title": f"t{i}", "link": f"u{i}"} for i in range(10)]}
        with FakeZhipu(body=body) as server:
            raw = asyncio.run(
                ZhipuWebSearchTool(api_key="k", search_engine="search_pro_sogou").zhipu_web_search(
                    "q", max_results=3
                )
            )
        assert server.payload["count"] == 3
        results = json.loads(raw)
        assert len(results) == 3, "max_results must be honoured even when the API ignores count"
        assert [r["title"] for r in results] == ["t0", "t1", "t2"]

    def test_truncation_uses_instance_count_when_max_results_omitted(self):
        body = {"search_result": [{"title": f"t{i}", "link": f"u{i}"} for i in range(10)]}
        with FakeZhipu(body=body):
            raw = asyncio.run(ZhipuWebSearchTool(api_key="k", count=4).zhipu_web_search("q", max_results=0))
        assert len(json.loads(raw)) == 4

    def test_missing_search_result_yields_empty_list(self):
        with FakeZhipu(body={"id": "x"}):
            raw = asyncio.run(ZhipuWebSearchTool(api_key="k").zhipu_web_search("q"))
        assert json.loads(raw) == []

    def test_multiple_queries_are_grouped_by_query(self):
        with FakeZhipu() as server:
            raw = asyncio.run(ZhipuWebSearchTool(api_key="k").zhipu_web_search(["上海天气", "北京天气"]))
        grouped = json.loads(raw)
        assert set(grouped) == {"上海天气", "北京天气"}
        assert len(server.requests) == 2
        assert json.loads(grouped["上海天气"])[0]["title"] == "2025年5月23日财经早资讯"

    def test_search_intent_enabled_returns_intent_alongside_results(self):
        with FakeZhipu() as server:
            raw = asyncio.run(ZhipuWebSearchTool(api_key="k", search_intent=True).zhipu_web_search("q"))
        assert server.payload["search_intent"] is True
        out = json.loads(raw)
        assert set(out) == {"search_intent", "search_result"}
        assert out["search_intent"][0]["intent"] == "SEARCH_ALL"

    def test_search_intent_disabled_returns_a_bare_list(self):
        with FakeZhipu():
            raw = asyncio.run(ZhipuWebSearchTool(api_key="k").zhipu_web_search("q"))
        assert isinstance(json.loads(raw), list)


class TestZhipuWebSearchErrors:
    def test_api_error_body_is_surfaced_not_swallowed(self):
        body = {"error": {"code": "1113", "message": "余额不足或无可用资源包,请充值。"}}
        with FakeZhipu(status_code=429, body=body):
            with pytest.raises(RuntimeError) as exc:
                asyncio.run(ZhipuWebSearchTool(api_key="k").zhipu_web_search("q"))
        # A bare "429" reads as rate limiting when the real cause is an empty account.
        assert "余额不足" in str(exc.value)
        assert "429" in str(exc.value)

    def test_non_json_error_body_still_surfaces(self):
        with FakeZhipu(status_code=502, body="<html>bad gateway</html>"):
            with pytest.raises(RuntimeError, match="502"):
                asyncio.run(ZhipuWebSearchTool(api_key="k").zhipu_web_search("q"))

    def test_missing_api_key_returns_message_without_calling_api(self):
        with patch.dict(os.environ, {}, clear=True):
            tool = ZhipuWebSearchTool()
        with FakeZhipu() as server:
            result = asyncio.run(tool.zhipu_web_search("q"))
        assert result == "Please set the ZAI_API_KEY"
        assert server.requests == []

    def test_unknown_engine_fails_at_construction_with_choices(self):
        with pytest.raises(ValueError) as exc:
            ZhipuWebSearchTool(api_key="k", search_engine="search_turbo")
        assert "search_turbo" in str(exc.value)
        for engine in SEARCH_ENGINES:
            assert engine in str(exc.value)

    def test_unknown_recency_filter_fails_at_construction(self):
        with pytest.raises(ValueError) as exc:
            ZhipuWebSearchTool(api_key="k", search_recency_filter="lastWeek")
        assert "lastWeek" in str(exc.value)
        for value in RECENCY_FILTERS:
            assert value in str(exc.value)

    def test_unknown_content_size_fails_at_construction(self):
        with pytest.raises(ValueError, match="huge"):
            ZhipuWebSearchTool(api_key="k", content_size="huge")

    @pytest.mark.parametrize("user_id", ["abc", "x" * 129])
    def test_out_of_range_user_id_fails_at_construction(self, user_id):
        with pytest.raises(ValueError, match="user_id must be 6-128 characters"):
            ZhipuWebSearchTool(api_key="k", user_id=user_id)


class TestZhipuBackendWiring:
    def test_registered_as_the_zhipu_web_search_provider(self):
        from agentica.tools.builtin.web_tools import BuiltinWebSearchTool

        with patch.dict(os.environ, {"ZAI_API_KEY": "k"}, clear=False):
            tool = BuiltinWebSearchTool(provider="zhipu")
        assert tool.provider == "zhipu"
        # The model must always see the same tool name regardless of engine.
        assert list(tool.functions) == ["web_search"]
        assert tool._search_fn.__name__ == "zhipu_web_search"
        assert isinstance(tool._search_fn.__self__, ZhipuWebSearchTool)

    def test_engine_tier_is_selectable_by_env_var(self):
        from agentica.tools.builtin.web_tools import BuiltinWebSearchTool

        env = {"ZAI_API_KEY": "k", "AGENTICA_ZHIPU_SEARCH_ENGINE": "search_pro"}
        with patch.dict(os.environ, env, clear=False):
            tool = BuiltinWebSearchTool(provider="zhipu")
        assert tool._search_fn.__self__.search_engine == "search_pro"

    def test_engine_defaults_to_search_pro_without_env_var(self):
        from agentica.tools.builtin.web_tools import BuiltinWebSearchTool

        env = dict(os.environ)
        env.pop("AGENTICA_ZHIPU_SEARCH_ENGINE", None)
        env["ZAI_API_KEY"] = "k"
        with patch.dict(os.environ, env, clear=True):
            tool = BuiltinWebSearchTool(provider="zhipu")
        assert tool._search_fn.__self__.search_engine == DEFAULT_SEARCH_ENGINE

    def test_cli_registry_points_at_the_renamed_module(self):
        from agentica.cli.runtime import TOOL_REGISTRY, _get_tool_import_path

        assert "web_search_pro" not in TOOL_REGISTRY
        path = _get_tool_import_path("zhipu_web_search")
        assert path == "agentica.tools.zhipu_web_search_tool.ZhipuWebSearchTool"
