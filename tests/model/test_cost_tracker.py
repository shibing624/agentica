# -*- coding: utf-8 -*-
"""Tests for agentica.cost_tracker — per-run LLM cost accounting."""
import unittest

from agentica.cost_tracker import CostTracker, ModelUsageStat, MODEL_PRICING


class TestCostTrackerNormalise(unittest.TestCase):
    """_normalise strips provider prefixes and lowercases."""

    def test_openai_prefix(self):
        self.assertEqual(CostTracker._normalise("openai/gpt-4o"), "gpt-4o")

    def test_anthropic_prefix(self):
        self.assertEqual(CostTracker._normalise("anthropic/claude-3-haiku"), "claude-3-haiku")

    def test_groq_prefix(self):
        self.assertEqual(CostTracker._normalise("groq/llama3-70b-8192"), "llama3-70b-8192")

    def test_together_ai_prefix(self):
        self.assertEqual(CostTracker._normalise("together_ai/mixtral"), "mixtral")

    def test_no_prefix(self):
        self.assertEqual(CostTracker._normalise("gpt-4o-mini"), "gpt-4o-mini")

    def test_uppercase_lowered(self):
        self.assertEqual(CostTracker._normalise("GPT-4O"), "gpt-4o")

    def test_whitespace_stripped(self):
        self.assertEqual(CostTracker._normalise("  gpt-4o  "), "gpt-4o")


class TestCostTrackerLookupPricing(unittest.TestCase):
    """_lookup_pricing: exact match → prefix match → family match → zero."""

    def test_exact_match(self):
        ct = CostTracker()
        pricing = ct._lookup_pricing("gpt-4o-mini")
        self.assertGreater(pricing["input"], 0)
        self.assertGreater(pricing["output"], 0)

    def test_prefix_match(self):
        ct = CostTracker()
        pricing = ct._lookup_pricing("gpt-4o-2024-11-20")
        self.assertGreater(pricing["input"], 0)
        self.assertGreater(pricing["output"], 0)

    def test_family_match(self):
        ct = CostTracker()
        # "claude-unknown-version" → family "claude" matches "claude-opus-4"
        pricing = ct._lookup_pricing("claude-unknown-version")
        self.assertIn("input", pricing)
        self.assertGreater(pricing["input"], 0)

    def test_unknown_model_returns_zero(self):
        ct = CostTracker()
        pricing = ct._lookup_pricing("totally-unknown-model-xyz")
        self.assertEqual(pricing["input"], 0.0)
        self.assertEqual(pricing["output"], 0.0)
        self.assertTrue(ct.has_unknown_model)


class TestCostTrackerRecord(unittest.TestCase):
    """record() calculates cost and accumulates stats."""

    def test_record_returns_cost(self):
        ct = CostTracker()
        cost = ct.record("gpt-4o-mini", input_tokens=1000, output_tokens=500)
        # gpt-4o-mini: input 0.15/M, output 0.60/M
        expected = 1000 * 0.15 / 1_000_000 + 500 * 0.60 / 1_000_000
        self.assertAlmostEqual(cost, expected, places=8)

    def test_record_accumulates_totals(self):
        ct = CostTracker()
        ct.record("gpt-4o-mini", input_tokens=100, output_tokens=50)
        ct.record("gpt-4o-mini", input_tokens=200, output_tokens=100)
        self.assertEqual(ct.total_input_tokens, 300)
        self.assertEqual(ct.total_output_tokens, 150)
        self.assertEqual(ct.turns, 2)
        self.assertGreater(ct.total_cost_usd, 0)

    def test_record_with_cache_tokens(self):
        ct = CostTracker()
        cost = ct.record("gpt-4o", input_tokens=100, output_tokens=50,
                         cache_read_tokens=200, cache_write_tokens=50)
        self.assertGreater(cost, 0)
        stat = ct.model_usage["gpt-4o"]
        self.assertEqual(stat.cache_read_tokens, 200)
        self.assertEqual(stat.cache_write_tokens, 50)

    def test_record_per_model_breakdown(self):
        ct = CostTracker()
        ct.record("gpt-4o-mini", input_tokens=100, output_tokens=50)
        ct.record("gpt-4o", input_tokens=100, output_tokens=50)
        self.assertIn("gpt-4o-mini", ct.model_usage)
        self.assertIn("gpt-4o", ct.model_usage)
        self.assertEqual(ct.model_usage["gpt-4o-mini"].requests, 1)
        self.assertEqual(ct.model_usage["gpt-4o"].requests, 1)

    def test_record_unknown_model(self):
        ct = CostTracker()
        cost = ct.record("unknown-model", input_tokens=1000, output_tokens=500)
        self.assertEqual(cost, 0.0)
        self.assertTrue(ct.has_unknown_model)


class TestCostTrackerSummary(unittest.TestCase):
    """summary() generates human-readable output."""

    def test_summary_format(self):
        ct = CostTracker()
        ct.record("gpt-4o-mini", input_tokens=1000, output_tokens=500)
        s = ct.summary()
        self.assertIn("Total cost:", s)
        self.assertIn("Total tokens:", s)
        self.assertIn("API calls:", s)
        self.assertIn("gpt-4o-mini", s)

    def test_summary_unknown_model_warning(self):
        ct = CostTracker()
        ct.record("unknown-xyz", input_tokens=100, output_tokens=50)
        s = ct.summary()
        self.assertIn("unknown model", s)

    def test_summary_empty(self):
        ct = CostTracker()
        s = ct.summary()
        self.assertIn("$0.0000", s)


class TestModelUsageStat(unittest.TestCase):
    """ModelUsageStat defaults."""

    def test_defaults(self):
        stat = ModelUsageStat()
        self.assertEqual(stat.input_tokens, 0)
        self.assertEqual(stat.output_tokens, 0)
        self.assertEqual(stat.cost_usd, 0.0)
        self.assertEqual(stat.requests, 0)


class TestPricingCache(unittest.TestCase):
    """Cache load/save via file mtime TTL."""

    def test_cache_round_trip(self):
        """Write cache, load it back — pricing should match."""
        import tempfile, json, os
        from agentica.cost_tracker import _load_cached, _parse_catalog, _CACHE_TTL

        pricing = {"gpt-test": {"input": 1.0, "output": 2.0, "cache_read": 0.0, "cache_write": 0.0}}

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "model_pricing_cache.json")
            with unittest.mock.patch("agentica.cost_tracker._get_cache_path", return_value=cache_path):
                with open(cache_path, "w") as f:
                    json.dump(pricing, f)

                loaded = _load_cached()
                self.assertIsNotNone(loaded)
                self.assertIn("gpt-test", loaded)
                self.assertEqual(loaded["gpt-test"]["input"], 1.0)

    def test_cache_expired_returns_none(self):
        """Stale cache (mtime older than TTL) should return None."""
        import tempfile, json, os
        from agentica.cost_tracker import _load_cached, _CACHE_TTL

        pricing = {"gpt-stale": {"input": 1.0, "output": 2.0, "cache_read": 0.0, "cache_write": 0.0}}

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "model_pricing_cache.json")
            with open(cache_path, "w") as f:
                json.dump(pricing, f)

            stale_time = os.path.getmtime(cache_path) - _CACHE_TTL - 100
            os.utime(cache_path, (stale_time, stale_time))

            with unittest.mock.patch("agentica.cost_tracker._get_cache_path", return_value=cache_path):
                loaded = _load_cached()
                self.assertIsNone(loaded)

    def test_missing_cache_returns_none(self):
        """Non-existent cache file should return None."""
        with unittest.mock.patch("agentica.cost_tracker._get_cache_path", return_value="/nonexistent/path.json"):
            from agentica.cost_tracker import _load_cached
            self.assertIsNone(_load_cached())

    def test_parse_catalog_extracts_pricing(self):
        """_parse_catalog should convert models.dev format to flat pricing dict."""
        from agentica.cost_tracker import _parse_catalog

        catalog = {
            "openai": {
                "models": {
                    "gpt-test-model": {
                        "cost": {"input": 5.0, "output": 15.0, "cache_read": 0.5},
                        "limit": {"context": 128000},
                    }
                }
            }
        }
        result = _parse_catalog(catalog)
        self.assertIn("gpt-test-model", result)
        self.assertEqual(result["gpt-test-model"]["input"], 5.0)
        self.assertEqual(result["gpt-test-model"]["output"], 15.0)
        self.assertEqual(result["gpt-test-model"]["cache_read"], 0.5)
        self.assertEqual(result["gpt-test-model"]["cache_write"], 0.0)
        self.assertEqual(result["gpt-test-model"]["context_window"], 128000)

    def test_parse_catalog_official_provider_takes_priority(self):
        """Official provider (openai) should override third-party (302ai)."""
        from agentica.cost_tracker import _parse_catalog

        catalog = {
            "302ai": {
                "models": {
                    "gpt-4o": {
                        "cost": {"input": 2.50, "output": 10.00, "cache_read": 0.0},
                        "limit": {"context": 64000},
                    }
                }
            },
            "openai": {
                "models": {
                    "gpt-4o": {
                        "cost": {"input": 2.50, "output": 10.00, "cache_read": 1.25},
                        "limit": {"context": 128000},
                    }
                }
            },
        }
        result = _parse_catalog(catalog)
        self.assertEqual(result["gpt-4o"]["cache_read"], 1.25)
        self.assertEqual(result["gpt-4o"]["context_window"], 128000)

    def test_parse_catalog_third_party_does_not_override_official(self):
        """Once official is recorded, third-party entries are ignored."""
        from agentica.cost_tracker import _parse_catalog

        catalog = {
            "openai": {
                "models": {
                    "gpt-4o": {
                        "cost": {"input": 2.50, "output": 10.00, "cache_read": 1.25},
                        "limit": {"context": 128000},
                    }
                }
            },
            "some-reseller": {
                "models": {
                    "gpt-4o": {
                        "cost": {"input": 5.00, "output": 20.00, "cache_read": 0.0},
                        "limit": {"context": 32000},
                    }
                }
            },
        }
        result = _parse_catalog(catalog)
        self.assertEqual(result["gpt-4o"]["cache_read"], 1.25)
        self.assertEqual(result["gpt-4o"]["context_window"], 128000)


class TestGetModelContextWindow(unittest.TestCase):
    """get_model_context_window: catalog lookup for context limits."""

    def test_exact_match_from_fallback(self):
        from agentica.cost_tracker import get_model_context_window
        cw = get_model_context_window("gpt-4o")
        self.assertEqual(cw, 128000)

    def test_exact_match_glm5(self):
        from agentica.cost_tracker import get_model_context_window
        cw = get_model_context_window("glm-5")
        self.assertEqual(cw, 204800)

    def test_prefix_match(self):
        from agentica.cost_tracker import get_model_context_window
        cw = get_model_context_window("gpt-4o-2024-11-20")
        self.assertGreater(cw, 0)

    def test_unknown_returns_default(self):
        from agentica.cost_tracker import get_model_context_window
        cw = get_model_context_window("totally-unknown-xyz", default=32000)
        self.assertEqual(cw, 32000)

    def test_qwen_max_context(self):
        from agentica.cost_tracker import get_model_context_window
        cw = get_model_context_window("qwen-max")
        self.assertEqual(cw, 32768)


if __name__ == "__main__":
    unittest.main()
