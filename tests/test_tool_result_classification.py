# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for tool result classification + storage integration.
"""
import base64
import os
import sys
import tempfile
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.compression.tool_result_classification import (
    ToolResultClass, classify_tool_result, describe_media,
)
from agentica.compression.tool_result_storage import maybe_persist_result, PREVIEW_CHARS


class TestClassify(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(classify_tool_result("   "), ToolResultClass.EMPTY)

    def test_error_flag(self):
        self.assertEqual(classify_tool_result("anything", is_error=True), ToolResultClass.ERROR)

    def test_normal_text(self):
        self.assertEqual(classify_tool_result("hello world"), ToolResultClass.NORMAL)

    def test_large_text(self):
        big = "x" * 5000
        self.assertEqual(classify_tool_result(big, large_threshold=1000), ToolResultClass.LARGE)

    def test_data_image_uri(self):
        content = "data:image/png;base64," + base64.b64encode(b"\x89PNG fake" * 200).decode()
        self.assertEqual(classify_tool_result(content), ToolResultClass.IMAGE)

    def test_bare_base64_blob(self):
        blob = base64.b64encode(b"some binary payload " * 200).decode()
        self.assertEqual(classify_tool_result(blob), ToolResultClass.IMAGE)

    def test_binary_nul(self):
        self.assertEqual(classify_tool_result("abc\x00\x01\x02def"), ToolResultClass.BINARY)


class TestDescribeMedia(unittest.TestCase):
    def test_image_descriptor(self):
        content = "data:image/png;base64,AAAA"
        out = describe_media(content, ToolResultClass.IMAGE)
        self.assertIn("image/png", out)
        self.assertIn("omitted", out)

    def test_binary_descriptor(self):
        out = describe_media("\x00\x01", ToolResultClass.BINARY)
        self.assertIn("binary", out)


class TestStorageIntegration(unittest.TestCase):
    def test_large_image_diverted_to_descriptor(self):
        # A big base64 image under the 50K size threshold should still be
        # replaced with a compact descriptor (not dumped into context).
        big_image = "data:image/png;base64," + base64.b64encode(b"x" * 8000).decode()
        self.assertGreater(len(big_image), PREVIEW_CHARS)
        with tempfile.TemporaryDirectory() as tmp:
            result = maybe_persist_result(
                tool_name="screenshot",
                tool_use_id="call_1",
                content=big_image,
                session_id="s1",
                cwd=tmp,
                max_result_size_chars=50_000,  # image is UNDER this
            )
        self.assertIn("<media", result)
        self.assertIn("image", result)
        self.assertNotIn(big_image[:200], result)  # raw base64 must be gone

    def test_normal_small_text_passes_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = maybe_persist_result(
                tool_name="read_file", tool_use_id="c2",
                content="short normal output", session_id="s1", cwd=tmp,
                max_result_size_chars=50_000,
            )
        self.assertEqual(result, "short normal output")

    def test_large_text_still_persists(self):
        big_text = "line\n" * 20000
        with tempfile.TemporaryDirectory() as tmp:
            result = maybe_persist_result(
                tool_name="execute", tool_use_id="c3",
                content=big_text, session_id="s1", cwd=tmp,
                max_result_size_chars=1000,
            )
        self.assertIn("persisted-output", result)


if __name__ == "__main__":
    unittest.main()
