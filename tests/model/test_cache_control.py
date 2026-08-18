"""Tests for Anthropic-style prompt caching on OpenAI-compatible proxies."""

from typing import Any, Dict, List
from unittest.mock import patch

import pytest
from openai.types.completion_usage import CompletionUsage

from agentica import OpenAIChat
from agentica.cost_tracker import CostTracker
from agentica.model.message import Message
from agentica.model.metrics import Metrics
from agentica.model.openai.chat import (
    _request_has_cache_control,
    _tag_content_block_cache_control,
    _tag_tools_cache_control,
)
from agentica.model.usage import TokenDetails, Usage


def _rk(tools: Any = None, extra_headers: Any = None) -> Dict[str, Any]:
    kw: Dict[str, Any] = {}
    if tools is not None:
        kw["tools"] = tools
    if extra_headers is not None:
        kw["extra_headers"] = extra_headers
    return kw


# ── module-level helpers ──────────────────────────────────────────────────────


def test_tag_content_block_str_wraps_and_caches():
    out = _tag_content_block_cache_control("hello")
    assert out == [{"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}]


def test_tag_content_block_list_tags_last_block_only():
    blocks = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    out = _tag_content_block_cache_control(blocks)
    assert "cache_control" not in out[0]
    assert out[1]["cache_control"] == {"type": "ephemeral"}


def test_tag_tools_caches_last_tool_only():
    tools = [{"type": "function", "function": {"name": "a"}}, {"type": "function", "function": {"name": "b"}}]
    out = _tag_tools_cache_control(tools)
    assert "cache_control" not in out[0]
    assert out[-1]["cache_control"] == {"type": "ephemeral"}
    # original list not mutated
    assert "cache_control" not in tools[-1]


def test_request_has_cache_control_detects_messages_and_tools():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}]}]
    assert _request_has_cache_control(msgs, None) is True
    assert _request_has_cache_control([{"role": "user", "content": "plain"}], None) is False
    tools = [{"type": "function", "function": {"name": "a"}, "cache_control": {"type": "ephemeral"}}]
    assert _request_has_cache_control([], tools) is True


# ── OpenAIChat._apply_cache_control ───────────────────────────────────────────


def _msgs(n: int) -> List[Dict[str, Any]]:
    out = [{"role": "system", "content": "sys"}]
    for i in range(n):
        out.append({"role": "user", "content": f"u{i}"})
    return out


def test_disabled_is_noop_and_warns_on_manual_blocks(caplog):
    model = OpenAIChat(id="m", api_key="fake_openai_key", enable_cache_control=False)
    formatted = [{"role": "user", "content": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}]}]
    with patch("agentica.model.openai.chat.logger.warning") as warn:
        out, kw = model._apply_cache_control(formatted, _rk())
    # unchanged, no injection
    assert out == formatted
    assert "cache_control" not in (kw.get("tools") or [])
    warn.assert_called_once()


def test_disabled_no_warn_without_manual_blocks():
    model = OpenAIChat(id="m", api_key="fake_openai_key", enable_cache_control=False)
    with patch("agentica.model.openai.chat.logger.warning") as warn:
        model._apply_cache_control(_msgs(2), _rk())
    warn.assert_not_called()


def test_enabled_tags_system_as_cached_block():
    model = OpenAIChat(id="m", api_key="fake_openai_key", enable_cache_control=True)
    out, _ = model._apply_cache_control(_msgs(1), _rk())
    sys = out[0]
    assert sys["role"] == "system"
    assert sys["content"] == [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]


def test_enabled_tags_last_k_messages_default_3():
    model = OpenAIChat(id="m", api_key="fake_openai_key", enable_cache_control=True)
    out, _ = model._apply_cache_control(_msgs(5), _rk())
    # system(1) + last 3 of 5 user msgs = 4 breakpoints; first 2 users uncached.
    assert "cache_control" not in out[1]["content"]
    assert "cache_control" not in out[2]["content"]
    # out indices: 0=system,1..5=user0..4 ; last 3 = user2,user3,user4 -> out[3],[4],[5]
    for idx in (3, 4, 5):
        assert out[idx]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_enabled_with_tools_reduces_message_budget():
    model = OpenAIChat(id="m", api_key="fake_openai_key", enable_cache_control=True)
    tools = [{"type": "function", "function": {"name": "a"}}, {"type": "function", "function": {"name": "b"}}]
    out, kw = model._apply_cache_control(_msgs(5), _rk(tools=tools))
    # tools(1) + system(1) = 2 used -> 2 message breakpoints remain
    assert kw["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    cached_user = [m for m in out if m["role"] == "user" and isinstance(m["content"], list)
                   and m["content"][-1].get("cache_control")]
    assert len(cached_user) == 2


def test_tool_result_tagged_per_proxy_example():
    # Proxy cache example: tool results carry cache_control in block-list form.
    model = OpenAIChat(id="m", api_key="fake_openai_key", enable_cache_control=True, cache_control_messages=2)
    formatted = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "x", "type": "function", "function": {"name": "f"}}]},
        {"role": "tool", "content": "tool result", "tool_call_id": "x"},
        {"role": "user", "content": "u2"},
    ]
    out, _ = model._apply_cache_control(formatted, _rk())
    # system tagged
    assert out[0]["content"][-1].get("cache_control") == {"type": "ephemeral"}
    # tool result wrapped into a cached block list (proxy format), string preserved
    assert out[3]["content"] == [{"type": "text", "text": "tool result", "cache_control": {"type": "ephemeral"}}]
    # last user tagged
    assert out[4]["content"][-1].get("cache_control") == {"type": "ephemeral"}


def test_assistant_with_tool_calls_not_tagged():
    # Assistant tool_calls live in a separate field, not content blocks, so no
    # cache_control is attached and budget is not consumed on them.
    model = OpenAIChat(id="m", api_key="fake_openai_key", enable_cache_control=True, cache_control_messages=3)
    formatted = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "x", "type": "function", "function": {"name": "f"}}]},
        {"role": "tool", "content": "r", "tool_call_id": "x"},
        {"role": "user", "content": "u"},
    ]
    out, _ = model._apply_cache_control(formatted, _rk())
    asst = [m for m in out if m["role"] == "assistant"][0]
    assert asst["content"] is None
    assert "tool_calls" in asst
    # budget: system(1) + 3 message slots, but only tool+user are cacheable -> 2 tagged
    tagged = [m for m in out if isinstance(m.get("content"), list)
              and m["content"][-1].get("cache_control")]
    assert len(tagged) == 3  # system + tool + user


def test_empty_content_messages_skipped_without_consuming_budget():
    # Empty user/tool content must not consume a breakpoint slot (a breakpoint
    # on empty content is invalid). A later non-empty message still gets cached.
    model = OpenAIChat(id="m", api_key="fake_openai_key", enable_cache_control=True, cache_control_messages=1)
    formatted = [
        {"role": "system", "content": "sys"},
        {"role": "tool", "content": "", "tool_call_id": "x"},
        {"role": "user", "content": "real"},
    ]
    out, _ = model._apply_cache_control(formatted, _rk())
    # empty tool content unchanged (no block, no cache_control)
    tool = [m for m in out if m["role"] == "tool"][0]
    assert tool["content"] == ""
    # the one message breakpoint went to the non-empty user, not the empty tool
    user = [m for m in out if m["role"] == "user"][0]
    assert user["content"][-1].get("cache_control") == {"type": "ephemeral"}


def test_breakpoint_budget_never_exceeds_four():
    model = OpenAIChat(id="m", api_key="fake_openai_key", enable_cache_control=True, cache_control_messages=10)
    tools = [{"type": "function", "function": {"name": "a"}}]
    out, kw = model._apply_cache_control(_msgs(8), _rk(tools=tools))
    count = 0
    count += 1  # tools
    if out[0]["role"] == "system" and isinstance(out[0]["content"], list):
        count += 1
    count += sum(1 for m in out if m["role"] != "system" and isinstance(m["content"], list)
                 and m["content"][-1].get("cache_control"))
    assert count <= 4


def test_default_off_even_for_known_proxy_base_url():
    # No whitelist: caching is off by default regardless of base_url. The user
    # must opt in explicitly (enable_cache_control=True / CLI flag / profile).
    model = OpenAIChat(id="m", api_key="fake_openai_key", base_url="https://proxy.example/v1")
    out, kw = model._apply_cache_control(_msgs(1), _rk())
    assert out == _msgs(1)
    assert "cache_control" not in (kw.get("tools") or [])
    assert "extra_headers" not in kw


def test_explicit_false_stays_off():
    model = OpenAIChat(id="m", api_key="fake_openai_key", base_url="https://proxy.example/v1", enable_cache_control=False)
    out, _ = model._apply_cache_control(_msgs(1), _rk())
    assert out == _msgs(1)


def test_session_header_stable_across_calls():
    model = OpenAIChat(
        id="m", api_key="fake_openai_key", enable_cache_control=True,
        cache_control_session_header="X-Session-Id",
    )
    _, kw1 = model._apply_cache_control(_msgs(1), _rk())
    _, kw2 = model._apply_cache_control(_msgs(1), _rk())
    h1 = kw1["extra_headers"]["X-Session-Id"]
    h2 = kw2["extra_headers"]["X-Session-Id"]
    assert h1 == h2 and h1.startswith("agentica-cache-")


def test_session_header_merges_into_existing_headers():
    model = OpenAIChat(
        id="m", api_key="fake_openai_key", enable_cache_control=True,
        cache_control_session_header="X-Session-Id",
    )
    _, kw = model._apply_cache_control(_msgs(1), _rk(extra_headers={"X-Foo": "bar"}))
    assert kw["extra_headers"]["X-Foo"] == "bar"
    assert "X-Session-Id" in kw["extra_headers"]


def test_no_session_header_when_unset():
    model = OpenAIChat(id="m", api_key="fake_openai_key", enable_cache_control=True)
    _, kw = model._apply_cache_control(_msgs(1), _rk())
    assert "extra_headers" not in kw


# ── usage parsing + cost tracker ──────────────────────────────────────────────


def _proxy_usage() -> CompletionUsage:
    return CompletionUsage.model_validate({
        "prompt_tokens": 7256,
        "completion_tokens": 6,
        "total_tokens": 7262,
        "prompt_tokens_details": {"cache_read_tokens": 7237, "cache_creation_tokens": 19, "cached_tokens": 0},
    })


def test_update_usage_metrics_parses_cache_read_and_creation():
    model = OpenAIChat(id="claude-3-5-sonnet", api_key="fake_openai_key")
    model._cost_tracker = CostTracker()
    metrics = Metrics()
    model.add_response_usage_to_metrics(metrics=metrics, response_usage=_proxy_usage())
    assistant = Message(role="assistant", content="ok")
    model.update_usage_metrics(assistant, metrics, _proxy_usage())

    details: TokenDetails = model.usage.input_tokens_details
    assert details.cache_read_tokens == 7237
    assert details.cache_creation_tokens == 19

    stat = model._cost_tracker.model_usage["claude-3-5-sonnet"]
    assert stat.input_tokens == 0
    assert stat.cache_read_tokens == 7237
    assert stat.cache_write_tokens == 19


def test_usage_merge_accumulates_cache_fields():
    a = Usage()
    a.add(type("E", (), {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                         "response_time": 0.1, "input_tokens_details": TokenDetails(
                             cache_read_tokens=100, cache_creation_tokens=20),
                         "output_tokens_details": None})())
    a.add(type("E", (), {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4,
                         "response_time": 0.1, "input_tokens_details": TokenDetails(
                             cache_read_tokens=50, cache_creation_tokens=5),
                         "output_tokens_details": None})())
    assert a.input_tokens_details.cache_read_tokens == 150
    assert a.input_tokens_details.cache_creation_tokens == 25


# ── volatile system split ─────────────────────────────────────────────────────


def test_system_split_at_volatile_marker():
    from agentica.model.message import VOLATILE_SYSTEM_MARKER
    from agentica.model.openai.chat import _tag_system_content_cache_control

    out = _tag_system_content_cache_control(f"stable part\n{VOLATILE_SYSTEM_MARKER}\nvolatile tail")
    assert isinstance(out, list) and len(out) == 2
    assert out[0] == {"type": "text", "text": "stable part", "cache_control": {"type": "ephemeral"}}
    assert out[1] == {"type": "text", "text": "volatile tail"}
    # the marker itself is a routing hint and must never reach the model
    assert VOLATILE_SYSTEM_MARKER not in out[0]["text"]
    assert VOLATILE_SYSTEM_MARKER not in out[1]["text"]


def test_system_split_marker_as_last_line_yields_single_block():
    from agentica.model.message import VOLATILE_SYSTEM_MARKER
    from agentica.model.openai.chat import _tag_system_content_cache_control

    out = _tag_system_content_cache_control(f"stable part\n{VOLATILE_SYSTEM_MARKER}")
    assert out == [{"type": "text", "text": "stable part", "cache_control": {"type": "ephemeral"}}]


def test_system_no_marker_single_tagged_block():
    from agentica.model.openai.chat import _tag_system_content_cache_control

    out = _tag_system_content_cache_control("plain system")
    assert out == [{"type": "text", "text": "plain system", "cache_control": {"type": "ephemeral"}}]


def test_apply_cache_control_splits_volatile_system():
    from agentica.model.message import VOLATILE_SYSTEM_MARKER

    model = OpenAIChat(id="m", api_key="fake_openai_key", enable_cache_control=True)
    msgs = [
        {"role": "system", "content": f"S\n{VOLATILE_SYSTEM_MARKER}\nV"},
        {"role": "user", "content": "u"},
    ]
    out, _ = model._apply_cache_control(msgs, _rk())
    sys_blocks = out[0]["content"]
    assert len(sys_blocks) == 2
    assert sys_blocks[0].get("cache_control") == {"type": "ephemeral"}
    assert "cache_control" not in sys_blocks[1]
    # breakpoint budget still counts the system tag exactly once
    assert out[1]["content"][-1].get("cache_control") == {"type": "ephemeral"}


def test_apply_cache_control_strips_marker_when_caching_off():
    from agentica.model.message import VOLATILE_SYSTEM_MARKER

    # Default config (enable_cache_control=False): the marker must not reach
    # the model, since the split that consumes it never runs.
    model = OpenAIChat(id="m", api_key="fake_openai_key")
    msgs = [
        {"role": "system", "content": f"S\n{VOLATILE_SYSTEM_MARKER}\nV"},
        {"role": "user", "content": "u"},
    ]
    out, _ = model._apply_cache_control(msgs, _rk())
    assert VOLATILE_SYSTEM_MARKER not in out[0]["content"]
    assert out[0]["content"] == "S\n\nV"


# ── persisted sticky routing id ───────────────────────────────────────────────


def test_persistent_session_id_stable_per_base_url(tmp_path, monkeypatch):
    from agentica.model.openai import chat as chat_mod

    monkeypatch.setattr(chat_mod.Path, "home", staticmethod(lambda: tmp_path))
    a = chat_mod._persistent_cache_session_id("https://x.example/v1")
    b = chat_mod._persistent_cache_session_id("https://x.example/v1")
    c = chat_mod._persistent_cache_session_id("https://y.example/v1")
    assert a == b and a.startswith("agentica-cache-")
    assert c != a


def test_persistent_session_id_tolerates_corrupt_file(tmp_path, monkeypatch):
    from agentica.model.openai import chat as chat_mod

    monkeypatch.setattr(chat_mod.Path, "home", staticmethod(lambda: tmp_path))
    f = tmp_path / ".agentica" / "cache" / "cache_routing.json"
    f.parent.mkdir(parents=True)
    f.write_text("not json at all")
    sid = chat_mod._persistent_cache_session_id("https://x.example/v1")
    assert sid.startswith("agentica-cache-")


def test_session_header_shared_across_instances(tmp_path, monkeypatch):
    from agentica.model.openai import chat as chat_mod

    monkeypatch.setattr(chat_mod.Path, "home", staticmethod(lambda: tmp_path))
    kwargs = dict(id="m", api_key="fake_openai_key", enable_cache_control=True,
                  cache_control_session_header="X-Session-Id", base_url="https://x.example/v1")
    m1 = OpenAIChat(**kwargs)
    m2 = OpenAIChat(**kwargs)
    _, kw1 = m1._apply_cache_control(_msgs(1), _rk())
    _, kw2 = m2._apply_cache_control(_msgs(1), _rk())
    assert kw1["extra_headers"]["X-Session-Id"] == kw2["extra_headers"]["X-Session-Id"]


# ── cache keep-alive ──────────────────────────────────────────────────────────


def test_apply_cache_control_captures_payload_for_keepalive():
    model = OpenAIChat(id="m", api_key="fake_openai_key", enable_cache_control=True)
    assert model._last_cache_payload is None
    model._apply_cache_control(_msgs(1), _rk())
    assert model._last_cache_payload is not None
    messages, request_kwargs = model._last_cache_payload
    assert messages[0]["role"] == "system"


def test_keepalive_not_armed_when_disabled_or_no_payload():
    model = OpenAIChat(id="m", api_key="fake_openai_key", enable_cache_control=True, cache_keepalive=False)
    model._last_cache_payload = ([], {})
    model._arm_cache_keepalive()
    assert model._cache_keepalive_generation == 0

    model2 = OpenAIChat(id="m", api_key="fake_openai_key", enable_cache_control=False)
    model2._arm_cache_keepalive()
    assert model2._cache_keepalive_generation == 0


def test_keepalive_worker_stops_when_superseded():
    model = OpenAIChat(id="m", api_key="fake_openai_key", enable_cache_control=True,
                       cache_keepalive_interval=0, cache_keepalive_max_pings=5)
    model._last_cache_payload = ([{"role": "system", "content": "S"}], {})
    calls = []

    async def fake_ping():
        calls.append(1)
        return 0

    model._cache_keepalive_ping = fake_ping
    # supersede immediately: generation 0 worker must exit without pinging
    model._cache_keepalive_loop(generation=-1)
    assert calls == []


def test_cache_off_strips_volatile_marker():
    from agentica.model.message import VOLATILE_SYSTEM_MARKER

    model = OpenAIChat(id="m", api_key="fake_openai_key")  # cache off by default
    msgs = [
        {"role": "system", "content": f"S\n{VOLATILE_SYSTEM_MARKER}\nV"},
        {"role": "user", "content": "u"},
    ]
    out, _ = model._apply_cache_control(msgs, _rk())
    assert out[0]["content"] == "S\n\nV"
    assert VOLATILE_SYSTEM_MARKER not in out[0]["content"]
    assert out[1] == {"role": "user", "content": "u"}  # non-system untouched


def test_keepalive_not_armed_without_sticky_header():
    # Without backend affinity a ping would land on a random backend and become
    # a full-price cache write — the arm must refuse (code-review Important #2).
    model = OpenAIChat(id="m", api_key="fake_openai_key", enable_cache_control=True,
                       cache_keepalive_interval=3600)
    model._apply_cache_control(_msgs(1), _rk())
    model._arm_cache_keepalive()
    assert model._cache_keepalive_generation == 0


def test_keepalive_armed_with_sticky_header():
    model = OpenAIChat(id="m", api_key="fake_openai_key", enable_cache_control=True,
                       cache_control_session_header="X-Session-Id", cache_keepalive_interval=3600)
    model._apply_cache_control(_msgs(1), _rk())
    model._arm_cache_keepalive()
    assert model._cache_keepalive_generation == 1


def test_keepalive_ping_request_shape(monkeypatch):
    from agentica.model.openai import chat as chat_mod

    sent: dict = {}

    class _FakeCompletions:
        async def create(self, **kwargs):
            sent.update(kwargs)

            class _U:
                prompt_tokens_details = None

            class _R:
                usage = _U()

            return _R()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self, **kwargs):
            sent["client_params"] = kwargs

        @property
        def chat(self):
            return _FakeChat()

    monkeypatch.setattr(chat_mod, "AsyncOpenAIClient", _FakeClient)
    model = OpenAIChat(id="m", api_key="fake_openai_key", enable_cache_control=True,
                       cache_control_session_header="X-Session-Id", base_url="https://x.example/v1")
    sys_blocks = [{"type": "text", "text": "S", "cache_control": {"type": "ephemeral"}}]
    model._last_cache_payload = (
        [{"role": "system", "content": sys_blocks}, {"role": "user", "content": "u"}],
        {"tools": [{"type": "function"}], "extra_headers": {"X-Session-Id": "sid"}},
    )
    import asyncio

    asyncio.run(model._cache_keepalive_ping())
    assert sent["max_tokens"] == 1
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]  # no conversation replay
    assert sent["tools"] == [{"type": "function"}]
    assert sent["extra_headers"] == {"X-Session-Id": "sid"}
    assert "http_client" in sent["client_params"]  # throwaway client, not the shared one


# ── prompt_cache_key (OpenAI-side routing affinity) ───────────────────────────


def test_prompt_cache_key_is_the_session_id():
    model = OpenAIChat(id="m", api_key="fake_openai_key")
    model.session_id = "sess-abc"
    assert model.request_kwargs["prompt_cache_key"] == "sess-abc"


def test_prompt_cache_key_is_independent_of_cache_control():
    """Anthropic breakpoints and OpenAI routing affinity are separate features."""
    model = OpenAIChat(id="m", api_key="fake_openai_key", enable_cache_control=False)
    model.session_id = "sess-abc"
    assert model.request_kwargs["prompt_cache_key"] == "sess-abc"


def test_prompt_cache_key_can_be_turned_off():
    model = OpenAIChat(id="m", api_key="fake_openai_key", enable_prompt_cache_key=False)
    model.session_id = "sess-abc"
    assert "prompt_cache_key" not in model.request_kwargs


def test_prompt_cache_key_falls_back_to_sticky_routing_id(monkeypatch):
    """A bare SDK Agent sets no session; an arbitrary stable key still beats none."""
    from agentica.model.openai import chat as chat_mod

    monkeypatch.setattr(chat_mod, "_persistent_cache_session_id", lambda _base: "sticky-1")
    model = OpenAIChat(id="m", api_key="fake_openai_key")
    assert model.session_id is None
    assert model.request_kwargs["prompt_cache_key"] == "sticky-1"


def test_explicit_prompt_cache_key_wins():
    model = OpenAIChat(id="m", api_key="fake_openai_key",
                       request_params={"prompt_cache_key": "mine"})
    model.session_id = "sess-abc"
    assert model.request_kwargs["prompt_cache_key"] == "mine"


def test_prompt_cache_key_is_stable_across_turns():
    """The whole point: one key for the life of the conversation."""
    model = OpenAIChat(id="m", api_key="fake_openai_key")
    model.session_id = "sess-abc"
    assert model.request_kwargs["prompt_cache_key"] == model.request_kwargs["prompt_cache_key"]


def test_prompt_cache_key_absent_from_to_dict():
    """to_dict describes the model's configuration, not per-session routing."""
    model = OpenAIChat(id="m", api_key="fake_openai_key")
    model.session_id = "sess-abc"
    assert "prompt_cache_key" not in model.to_dict()


# ── marker must not leak to non-caching providers ─────────────────────────────


def test_litellm_format_strips_marker():
    litellm = pytest.importorskip("litellm", reason="litellm not installed")
    if not hasattr(litellm, "completion"):
        pytest.skip("litellm importable only as a namespace stub (broken install)")
    from agentica.model.litellm.chat import LiteLLMChat
    from agentica.model.message import VOLATILE_SYSTEM_MARKER, Message

    model = LiteLLMChat(id="gpt-test", api_key="fake")
    out = model.format_message(Message(role="system", content=f"S\n{VOLATILE_SYSTEM_MARKER}\nV"))
    assert VOLATILE_SYSTEM_MARKER not in out["content"]


def test_ollama_format_strips_marker():
    from agentica.model.ollama.chat import Ollama
    from agentica.model.message import VOLATILE_SYSTEM_MARKER, Message

    model = Ollama(id="llama-test")
    out = model.format_message(Message(role="system", content=f"S\n{VOLATILE_SYSTEM_MARKER}\nV"))
    assert VOLATILE_SYSTEM_MARKER not in out["content"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
