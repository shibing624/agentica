# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for the pluggable LLM provider registry.
"""
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import provider_registry as pr


class _DummyModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class TestProviderRegistry(unittest.TestCase):
    def tearDown(self):
        # Clean up any test registrations.
        for slug in ("myllm", "myllm2", "dupe"):
            pr.unregister_provider(slug)

    def test_register_and_create(self):
        pr.register_provider("myllm", lambda **kw: _DummyModel(**kw))
        model = pr.create_provider("myllm", id="m1", api_key="k")
        self.assertIsInstance(model, _DummyModel)
        self.assertEqual(model.kwargs["id"], "m1")
        self.assertEqual(model.kwargs["api_key"], "k")

    def test_slug_case_insensitive(self):
        pr.register_provider("MyLLM", lambda **kw: _DummyModel(**kw))
        self.assertIsNotNone(pr.get_provider_factory("myllm"))
        pr.unregister_provider("myllm")

    def test_builtins_seeded(self):
        providers = pr.list_providers()
        for expected in ("openai", "deepseek", "zhipuai", "claude", "azure"):
            self.assertIn(expected, providers)

    def test_unknown_provider_raises(self):
        with self.assertRaises(ValueError):
            pr.create_provider("does-not-exist-xyz")

    def test_duplicate_without_overwrite_raises(self):
        pr.register_provider("dupe", lambda **kw: _DummyModel(**kw))
        with self.assertRaises(ValueError):
            pr.register_provider("dupe", lambda **kw: _DummyModel(**kw))

    def test_overwrite_allowed(self):
        pr.register_provider("dupe", lambda **kw: _DummyModel(tag="a", **kw))
        pr.register_provider("dupe", lambda **kw: _DummyModel(tag="b", **kw), overwrite=True)
        self.assertEqual(pr.create_provider("dupe").kwargs["tag"], "b")

    def test_user_registration_not_clobbered_by_seeding(self):
        # Register BEFORE any seeding-triggering call, then trigger seeding.
        pr.register_provider("myllm2", lambda **kw: _DummyModel(custom=True, **kw))
        _ = pr.list_providers()  # triggers _ensure_seeded
        self.assertTrue(pr.create_provider("myllm2").kwargs["custom"])

    def test_empty_slug_rejected(self):
        with self.assertRaises(ValueError):
            pr.register_provider("  ", lambda **kw: _DummyModel(**kw))

    def test_non_callable_rejected(self):
        with self.assertRaises(ValueError):
            pr.register_provider("bad", "not-callable")


class TestPublicExport(unittest.TestCase):
    def test_importable_from_agentica(self):
        from agentica import (
            register_provider, create_provider, list_providers,
            get_provider_factory, unregister_provider,
        )
        self.assertTrue(callable(register_provider))
        self.assertTrue(callable(create_provider))


if __name__ == "__main__":
    unittest.main()
