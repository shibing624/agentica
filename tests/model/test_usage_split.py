# -*- coding: utf-8 -*-
"""Cached tokens must be counted once, whichever convention the provider uses.

OpenAI reports prompt_tokens INCLUSIVE of cached_tokens; Anthropic reports input
EXCLUSIVE of the cache. Both land in the same CostTracker, which adds the parts —
so the adapter has to make them disjoint first.

All tests mock LLM API keys — no real API usage.
"""
import unittest

from agentica.cost_tracker import CostTracker
from agentica.model.usage import split_prompt_usage


class TestSplitPromptUsage(unittest.TestCase):

    def test_openai_cached_tokens_are_carved_out_of_prompt_tokens(self):
        fresh, read, write = split_prompt_usage(10_000, {"cached_tokens": 8_000})
        self.assertEqual((fresh, read, write), (2_000, 8_000, 0))
        self.assertEqual(fresh + read + write, 10_000, "parts must sum to the prompt")

    def test_anthropic_style_keys_are_already_disjoint(self):
        fresh, read, write = split_prompt_usage(
            120, {"cache_read_tokens": 50_000, "cache_creation_tokens": 300}
        )
        self.assertEqual((fresh, read, write), (120, 50_000, 300))

    def test_proxy_reporting_both_namings_is_treated_as_exclusive(self):
        """Claude-fronting proxies echo the OpenAI alias alongside their own."""
        fresh, read, _ = split_prompt_usage(
            120, {"cached_tokens": 50_000, "cache_read_tokens": 50_000}
        )
        self.assertEqual((fresh, read), (120, 50_000))

    def test_no_cache_reported(self):
        self.assertEqual(split_prompt_usage(900, None), (900, 0, 0))
        self.assertEqual(split_prompt_usage(900, {"cached_tokens": 0}), (900, 0, 0))

    def test_none_valued_keys_do_not_crash(self):
        """Providers routinely send explicit nulls in prompt_tokens_details."""
        self.assertEqual(
            split_prompt_usage(900, {"cached_tokens": None, "cache_read_tokens": None}),
            (900, 0, 0),
        )

    def test_never_goes_negative_on_inconsistent_reporting(self):
        fresh, read, _ = split_prompt_usage(100, {"cached_tokens": 500})
        self.assertEqual(fresh, 0)
        self.assertEqual(read, 500)


class TestNoDoubleCounting(unittest.TestCase):
    """The end-to-end effect the split exists to prevent."""

    def test_openai_prompt_is_neither_double_charged_nor_double_sized(self):
        prompt_tokens = 10_000
        details = {"cached_tokens": 8_000}

        ct = CostTracker()
        fresh, read, write = split_prompt_usage(prompt_tokens, details)
        ct.record("gpt-4o", input_tokens=fresh, output_tokens=100,
                  cache_read_tokens=read, cache_write_tokens=write)
        self.assertEqual(ct.context_input_tokens, prompt_tokens)
        self.assertEqual(ct.total_prompt_tokens, prompt_tokens)

        naive = CostTracker()
        naive.record("gpt-4o", input_tokens=prompt_tokens, output_tokens=100,
                     cache_read_tokens=8_000)
        self.assertEqual(naive.context_input_tokens, 18_000)
        self.assertLess(ct.total_cost_usd, naive.total_cost_usd,
                        "the naive path bills the cached prefix twice")

    def test_anthropic_prompt_still_counts_the_cached_prefix(self):
        ct = CostTracker()
        fresh, read, write = split_prompt_usage(
            120, {"cache_read_tokens": 50_000, "cache_creation_tokens": 300}
        )
        ct.record("claude-sonnet-4", input_tokens=fresh, output_tokens=50,
                  cache_read_tokens=read, cache_write_tokens=write)
        self.assertEqual(ct.context_input_tokens, 50_420)


if __name__ == "__main__":
    unittest.main()
