# -*- coding: utf-8 -*-
"""
Tests for the Usage model and cross-request token aggregation.
"""
import unittest

from agentica.model.usage import Usage, RequestUsage, TokenDetails


class TestTokenDetails(unittest.TestCase):
    """Test TokenDetails model."""

    def test_defaults(self):
        td = TokenDetails()
        self.assertEqual(td.cached_tokens, 0)
        self.assertEqual(td.reasoning_tokens, 0)

    def test_custom_values(self):
        td = TokenDetails(cached_tokens=100, reasoning_tokens=50)
        self.assertEqual(td.cached_tokens, 100)
        self.assertEqual(td.reasoning_tokens, 50)


class TestRequestUsage(unittest.TestCase):
    """Test RequestUsage model."""

    def test_defaults(self):
        ru = RequestUsage()
        self.assertEqual(ru.request_index, 0)
        self.assertEqual(ru.input_tokens, 0)
        self.assertEqual(ru.output_tokens, 0)
        self.assertEqual(ru.total_tokens, 0)
        self.assertIsNone(ru.input_tokens_details)
        self.assertIsNone(ru.output_tokens_details)
        self.assertIsNone(ru.response_time)

    def test_with_details(self):
        ru = RequestUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            response_time=1.5,
            input_tokens_details=TokenDetails(cached_tokens=30),
            output_tokens_details=TokenDetails(reasoning_tokens=20),
        )
        self.assertEqual(ru.input_tokens, 100)
        self.assertEqual(ru.output_tokens, 50)
        self.assertEqual(ru.total_tokens, 150)
        self.assertEqual(ru.response_time, 1.5)
        self.assertEqual(ru.input_tokens_details.cached_tokens, 30)
        self.assertEqual(ru.output_tokens_details.reasoning_tokens, 20)


class TestUsage(unittest.TestCase):
    """Test Usage aggregation model."""

    def test_empty_usage(self):
        usage = Usage()
        self.assertEqual(usage.input_tokens, 0)
        self.assertEqual(usage.output_tokens, 0)
        self.assertEqual(usage.total_tokens, 0)
        self.assertEqual(usage.requests, 0)
        self.assertEqual(len(usage.request_usage_entries), 0)
        self.assertEqual(usage.input_tokens_details.cached_tokens, 0)
        self.assertEqual(usage.output_tokens_details.reasoning_tokens, 0)

    def test_add_single_entry(self):
        usage = Usage()
        entry = RequestUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        usage.add(entry)

        self.assertEqual(usage.input_tokens, 100)
        self.assertEqual(usage.output_tokens, 50)
        self.assertEqual(usage.total_tokens, 150)
        self.assertEqual(usage.requests, 1)
        self.assertEqual(len(usage.request_usage_entries), 1)
        self.assertEqual(usage.request_usage_entries[0].request_index, 0)

    def test_add_multiple_entries(self):
        usage = Usage()
        usage.add(RequestUsage(input_tokens=100, output_tokens=50, total_tokens=150))
        usage.add(RequestUsage(input_tokens=200, output_tokens=80, total_tokens=280))
        usage.add(RequestUsage(input_tokens=150, output_tokens=60, total_tokens=210))

        self.assertEqual(usage.input_tokens, 450)
        self.assertEqual(usage.output_tokens, 190)
        self.assertEqual(usage.total_tokens, 640)
        self.assertEqual(usage.requests, 3)
        self.assertEqual(len(usage.request_usage_entries), 3)
        # Check request_index auto-assigned
        self.assertEqual(usage.request_usage_entries[0].request_index, 0)
        self.assertEqual(usage.request_usage_entries[1].request_index, 1)
        self.assertEqual(usage.request_usage_entries[2].request_index, 2)

    def test_add_with_details(self):
        usage = Usage()
        usage.add(RequestUsage(
            input_tokens=100, output_tokens=50, total_tokens=150,
            input_tokens_details=TokenDetails(cached_tokens=30),
            output_tokens_details=TokenDetails(reasoning_tokens=20),
        ))
        usage.add(RequestUsage(
            input_tokens=200, output_tokens=80, total_tokens=280,
            input_tokens_details=TokenDetails(cached_tokens=50),
            output_tokens_details=TokenDetails(reasoning_tokens=10),
        ))

        self.assertEqual(usage.input_tokens_details.cached_tokens, 80)
        self.assertEqual(usage.output_tokens_details.reasoning_tokens, 30)

    def test_merge(self):
        parent = Usage()
        parent.add(RequestUsage(input_tokens=100, output_tokens=50, total_tokens=150))

        child = Usage()
        child.add(RequestUsage(
            input_tokens=200, output_tokens=80, total_tokens=280,
            input_tokens_details=TokenDetails(cached_tokens=40),
        ))
        child.add(RequestUsage(input_tokens=150, output_tokens=60, total_tokens=210))

        parent.merge(child)

        self.assertEqual(parent.input_tokens, 450)
        self.assertEqual(parent.output_tokens, 190)
        self.assertEqual(parent.total_tokens, 640)
        self.assertEqual(parent.requests, 3)
        self.assertEqual(len(parent.request_usage_entries), 3)
        self.assertEqual(parent.input_tokens_details.cached_tokens, 40)

    def test_merge_details_aggregation(self):
        parent = Usage()
        parent.add(RequestUsage(
            input_tokens=100, output_tokens=50, total_tokens=150,
            input_tokens_details=TokenDetails(cached_tokens=10),
            output_tokens_details=TokenDetails(reasoning_tokens=5),
        ))

        child = Usage()
        child.add(RequestUsage(
            input_tokens=200, output_tokens=80, total_tokens=280,
            input_tokens_details=TokenDetails(cached_tokens=20),
            output_tokens_details=TokenDetails(reasoning_tokens=15),
        ))

        parent.merge(child)

        self.assertEqual(parent.input_tokens_details.cached_tokens, 30)
        self.assertEqual(parent.output_tokens_details.reasoning_tokens, 20)

    def test_merge_empty(self):
        parent = Usage()
        parent.add(RequestUsage(input_tokens=100, output_tokens=50, total_tokens=150))
        empty = Usage()
        parent.merge(empty)

        self.assertEqual(parent.input_tokens, 100)
        self.assertEqual(parent.requests, 1)

    def test_serialization(self):
        """Test that Usage can be serialized to dict/JSON."""
        usage = Usage()
        usage.add(RequestUsage(
            input_tokens=100, output_tokens=50, total_tokens=150,
            response_time=1.5,
            input_tokens_details=TokenDetails(cached_tokens=30),
        ))
        d = usage.model_dump()
        self.assertEqual(d["input_tokens"], 100)
        self.assertEqual(d["requests"], 1)
        self.assertEqual(len(d["request_usage_entries"]), 1)
        self.assertEqual(d["request_usage_entries"][0]["input_tokens"], 100)
        self.assertEqual(d["input_tokens_details"]["cached_tokens"], 30)

    def test_deserialization(self):
        """Test that Usage can be deserialized from dict."""
        d = {
            "input_tokens": 300,
            "output_tokens": 130,
            "total_tokens": 430,
            "requests": 2,
            "input_tokens_details": {"cached_tokens": 50, "reasoning_tokens": 0},
            "output_tokens_details": {"cached_tokens": 0, "reasoning_tokens": 25},
            "request_usage_entries": [
                {"request_index": 0, "input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
                {"request_index": 1, "input_tokens": 200, "output_tokens": 80, "total_tokens": 280},
            ],
        }
        usage = Usage(**d)
        self.assertEqual(usage.input_tokens, 300)
        self.assertEqual(usage.requests, 2)
        self.assertEqual(len(usage.request_usage_entries), 2)
        self.assertEqual(usage.input_tokens_details.cached_tokens, 50)
        self.assertEqual(usage.output_tokens_details.reasoning_tokens, 25)


class TestUsageOnModel(unittest.TestCase):
    """Test that Usage is properly integrated into Model."""

    def test_model_has_usage_field(self):
        """Test that Model base class has a usage field."""
        from dataclasses import fields as dc_fields
        from agentica.model.base import Model
        # Model is now a dataclass, check the field exists
        field_names = {f.name for f in dc_fields(Model)}
        self.assertIn("usage", field_names)

    def test_model_clear_resets_usage(self):
        """Test that Model.clear() resets usage."""
        from agentica.model.openai.chat import OpenAIChat
        model = OpenAIChat(id="gpt-4o")
        # Simulate adding usage
        model.usage.add(RequestUsage(input_tokens=100, output_tokens=50, total_tokens=150))
        self.assertEqual(model.usage.requests, 1)
        model.clear()
        self.assertEqual(model.usage.requests, 0)
        self.assertEqual(model.usage.input_tokens, 0)

    def test_run_response_has_usage_field(self):
        """Test that RunResponse has a usage field."""
        from agentica.run_response import RunResponse
        self.assertIn("usage", RunResponse.model_fields)

    def test_run_response_usage_serialization(self):
        """Test RunResponse with usage serializes correctly."""
        from agentica.run_response import RunResponse
        usage = Usage()
        usage.add(RequestUsage(input_tokens=100, output_tokens=50, total_tokens=150))
        rr = RunResponse(content="hello", usage=usage)
        d = rr.model_dump(exclude_none=True)
        self.assertEqual(d["usage"]["requests"], 1)
        self.assertEqual(d["usage"]["total_tokens"], 150)


class TestUsageImports(unittest.TestCase):
    """Test that Usage is properly exported from agentica package."""

    def test_import_from_package(self):
        from agentica import Usage, RequestUsage, TokenDetails
        self.assertIsNotNone(Usage)
        self.assertIsNotNone(RequestUsage)
        self.assertIsNotNone(TokenDetails)

    def test_import_from_model_usage(self):
        from agentica.model.usage import Usage, RequestUsage, TokenDetails
        self.assertIsNotNone(Usage)
        self.assertIsNotNone(RequestUsage)
        self.assertIsNotNone(TokenDetails)


if __name__ == "__main__":
    unittest.main()
