# tests/test_json_parse.py
# -*- coding: utf-8 -*-
"""Unit tests for agentica.utils.json_parse."""
import unittest

from agentica.utils.json_parse import extract_json_object, extract_json_array


class TestExtractJsonObject(unittest.TestCase):
    def test_pure_json_object(self):
        self.assertEqual(extract_json_object('{"a": 1}'), {"a": 1})

    def test_json_object_with_prose(self):
        text = 'Sure, here is the result:\n{"a": 1, "b": "x"}\nDone.'
        self.assertEqual(extract_json_object(text), {"a": 1, "b": "x"})

    def test_json_object_in_code_fence(self):
        text = '```json\n{"a": 1}\n```'
        self.assertEqual(extract_json_object(text), {"a": 1})

    def test_json_object_in_unlabeled_fence(self):
        text = '```\n{"a": 1}\n```'
        self.assertEqual(extract_json_object(text), {"a": 1})

    def test_returns_none_for_non_object(self):
        self.assertIsNone(extract_json_object("not json at all"))

    def test_returns_none_for_array_input(self):
        # Array at top level — extract_json_object should return None.
        self.assertIsNone(extract_json_object("[1, 2, 3]"))

    def test_handles_nested_braces(self):
        text = 'prefix {"outer": {"inner": 42}} suffix'
        self.assertEqual(extract_json_object(text), {"outer": {"inner": 42}})


class TestExtractJsonArray(unittest.TestCase):
    def test_pure_json_array(self):
        self.assertEqual(extract_json_array("[1, 2, 3]"), [1, 2, 3])

    def test_array_with_prose(self):
        text = "Here you go: [1, 2, 3] -- enjoy."
        self.assertEqual(extract_json_array(text), [1, 2, 3])

    def test_array_in_code_fence(self):
        text = '```json\n[{"a": 1}]\n```'
        self.assertEqual(extract_json_array(text), [{"a": 1}])

    def test_returns_none_for_non_array(self):
        self.assertIsNone(extract_json_array('{"a": 1}'))

    def test_returns_none_for_garbage(self):
        self.assertIsNone(extract_json_array("not json"))

    def test_handles_nested_arrays(self):
        text = 'prefix [[1,2],[3,4]] suffix'
        self.assertEqual(extract_json_array(text), [[1, 2], [3, 4]])


if __name__ == "__main__":
    unittest.main()
