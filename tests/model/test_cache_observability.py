# -*- coding: utf-8 -*-
"""P3-1 cache observability: hit ratio, prefix-break attribution, resume warmth."""
import json
import time
from types import SimpleNamespace

import pytest

from agentica.memory.session_log import SessionLog
from agentica.model.message import Message
from agentica.model.usage import RequestUsage, TokenDetails, Usage
from agentica.runner.compress import _prefix_digests, _first_prefix_break, CompressMixin


# ---------------------------------------------------------------------------
# Usage.cache_hit_ratio
# ---------------------------------------------------------------------------

class TestCacheHitRatio:
    def test_openai_inclusive_convention(self):
        # OpenAI: prompt_tokens INCLUDES cached share.
        ru = RequestUsage(input_tokens=1000, input_tokens_details=TokenDetails(cached_tokens=900))
        assert ru.cache_hit_ratio() == pytest.approx(0.9)

    def test_anthropic_exclusive_convention(self):
        # Anthropic: input EXCLUDES cache; true prompt = 100 + 800 + 100.
        ru = RequestUsage(
            input_tokens=100,
            input_tokens_details=TokenDetails(cache_read_tokens=800, cache_creation_tokens=100),
        )
        assert ru.cache_hit_ratio() == pytest.approx(0.8)

    def test_no_cache_data_returns_none_not_zero(self):
        assert RequestUsage(input_tokens=1000).cache_hit_ratio() is None
        ru = RequestUsage(input_tokens=1000, input_tokens_details=TokenDetails())
        assert ru.cache_hit_ratio() is None

    def test_aggregate_usage(self):
        u = Usage()
        u.add(RequestUsage(input_tokens=1000, input_tokens_details=TokenDetails(cached_tokens=900)))
        u.add(RequestUsage(input_tokens=1000, input_tokens_details=TokenDetails(cached_tokens=500)))
        # (900+500) / (fresh 600 + hit 1400) = 0.7
        assert u.cache_hit_ratio() == pytest.approx(0.7)

    def test_aggregate_usage_normalises_each_request_before_merging(self):
        u = Usage()
        # OpenAI-compatible inclusive: prompt includes cached tokens.
        u.add(RequestUsage(
            input_tokens=1000,
            input_tokens_details=TokenDetails(cached_tokens=900, cache_read_tokens=900),
        ))
        # Anthropic exclusive: cache_read is outside input_tokens.
        u.add(RequestUsage(
            input_tokens=100,
            input_tokens_details=TokenDetails(cache_read_tokens=800),
        ))
        assert u.cache_hit_ratio() == pytest.approx((900 + 800) / (1000 + 100 + 800))


# ---------------------------------------------------------------------------
# Prefix digests / break attribution
# ---------------------------------------------------------------------------

def _msgs(*texts):
    return [Message(role=("user" if i % 2 == 0 else "assistant"), content=t) for i, t in enumerate(texts)]


class TestPrefixBreak:
    def test_append_only_tail_is_no_break(self):
        prev = _prefix_digests(_msgs("a", "b"))
        curr = _prefix_digests(_msgs("a", "b", "c"))
        assert _first_prefix_break(prev, curr) is None

    def test_mid_change_reports_index(self):
        prev = _prefix_digests(_msgs("a", "b", "c"))
        curr = _prefix_digests(_msgs("a", "B-CHANGED", "c"))
        assert _first_prefix_break(prev, curr) == 1

    def test_shrink_is_break_at_cut(self):
        prev = _prefix_digests(_msgs("a", "b", "c"))
        curr = _prefix_digests(_msgs("a"))
        assert _first_prefix_break(prev, curr) == 1

    def test_local_fields_do_not_count_as_drift(self):
        m1 = _msgs("a", "b")
        m2 = _msgs("a", "b")
        m2[0].metrics = {"tokens": 999}
        m2[1].provider_data = {"blob": "x"}
        assert _prefix_digests(m1) == _prefix_digests(m2)


class TestEmitContextUsage:
    def _emit(self, prev_usage_ratio, messages, model_extra=None):
        events = []
        agent = SimpleNamespace(_event_callback=events.append, name="a", _parent_run_id=None)
        usage = Usage()
        if prev_usage_ratio is not None:
            usage.add(RequestUsage(
                input_tokens=1000, input_tokens_details=TokenDetails(cached_tokens=prev_usage_ratio)
            ))
        model = SimpleNamespace(tools=None, context_window=128000, id="gpt-4o", usage=usage)
        if model_extra:
            for k, v in model_extra.items():
                setattr(model, k, v)
        CompressMixin._emit_context_usage(agent, model, messages)
        return events[0], model

    def test_event_carries_ratio_and_break_fields(self):
        event, _ = self._emit(800, _msgs("a"))
        assert event["type"] == "context.usage"
        assert event["cache_hit_ratio"] == pytest.approx(0.8)
        assert event["prefix_break_index"] is None  # first request: nothing to diff

    def test_second_request_reports_break_index(self):
        try:
            ratio = None
            events, agent_msgs = [], _msgs("a", "b")
            agent = SimpleNamespace(_event_callback=events.append, name="a", _parent_run_id=None)
            model = SimpleNamespace(tools=None, context_window=128000, id="gpt-4o", usage=Usage())
            CompressMixin._emit_context_usage(agent, model, agent_msgs)
            changed = _msgs("a", "B-CHANGED")
            CompressMixin._emit_context_usage(agent, model, changed)
        finally:
            pass
        assert events[1]["prefix_break_index"] == 1

    def test_no_usage_no_ratio(self):
        event, _ = self._emit(None, _msgs("a"))
        assert event["cache_hit_ratio"] is None


# ---------------------------------------------------------------------------
# SessionLog.cache_warmth_hint
# ---------------------------------------------------------------------------

class TestCacheWarmthHint:
    def _log(self, tmp_path, with_boundary=True, model="m-a", age_sec=0):
        log = SessionLog("s-warm", base_dir=str(tmp_path))
        log.append("user", "q")
        if with_boundary:
            log.append_compact_boundary("sum", model=model)
        log.append("assistant", "a")
        if age_sec:
            # age every entry by rewriting timestamps into the past
            lines = []
            for line in log.path.read_text(encoding="utf-8").splitlines():
                entry = json.loads(line)
                entry["timestamp"] = time.strftime(
                    "%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(time.time() - age_sec)
                )
                lines.append(json.dumps(entry, ensure_ascii=False))
            log.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return log

    def test_warm_when_boundary_matches(self, tmp_path):
        log = self._log(tmp_path)
        assert log.cache_warmth_hint("m-a") == "warm"

    def test_cold_on_lineage_mismatch(self, tmp_path):
        log = self._log(tmp_path)
        assert log.cache_warmth_hint("m-b") == "cold"

    def test_unknown_without_boundary(self, tmp_path):
        log = self._log(tmp_path, with_boundary=False)
        assert log.cache_warmth_hint("m-a") == "unknown"

    def test_unknown_without_file(self, tmp_path):
        log = SessionLog("s-nonexistent", base_dir=str(tmp_path))
        assert log.cache_warmth_hint("m-a") == "unknown"

    def test_cold_past_ttl(self, tmp_path):
        log = self._log(tmp_path, age_sec=600)
        assert log.cache_warmth_hint("m-a", ttl_seconds=300) == "cold"

    def test_warm_within_ttl(self, tmp_path):
        log = self._log(tmp_path)
        assert log.cache_warmth_hint("m-a", ttl_seconds=300) == "warm"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
