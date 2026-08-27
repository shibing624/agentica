# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for V4A apply_diff / parse_patch_envelope.
"""
import os
import sys
import unittest
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.tools.patch_tool import apply_diff, parse_patch_envelope


class TestApplyDiff(unittest.TestCase):
    """Test cases for the apply_diff function."""

    def test_create_mode_simple(self):
        """Test creating a new file with V4A create mode."""
        diff = """+line1
+line2
+line3"""
        result = apply_diff("", diff, mode="create")
        self.assertEqual(result, "line1\nline2\nline3")

    def test_create_mode_with_code(self):
        """Test creating a Python file with V4A create mode."""
        diff = """+def hello():
+    print("Hello, World!")
+
+if __name__ == "__main__":
+    hello()"""
        result = apply_diff("", diff, mode="create")
        expected = """def hello():
    print("Hello, World!")

if __name__ == "__main__":
    hello()"""
        self.assertEqual(result, expected)

    def test_update_mode_simple_replace(self):
        """Test simple line replacement."""
        original = """def hello():
    print("Hello")

def world():
    print("World")"""

        diff = """@@
 def hello():
-    print("Hello")
+    print("Hello, Universe!")"""

        result = apply_diff(original, diff, mode="default")
        self.assertIn('print("Hello, Universe!")', result)
        self.assertIn('def world():', result)

    def test_update_mode_add_lines(self):
        """Test adding new lines."""
        original = """class Calculator:
    def add(self, a, b):
        return a + b"""

        diff = """@@
 class Calculator:
     def add(self, a, b):
-        return a + b
+        # Add two numbers
+        result = a + b
+        return result"""

        result = apply_diff(original, diff, mode="default")
        self.assertIn("# Add two numbers", result)
        self.assertIn("result = a + b", result)

    def test_update_mode_delete_lines(self):
        """Test deleting lines."""
        original = """def func():
    # This is a comment
    # Another comment
    return 42"""

        diff = """@@
 def func():
-    # This is a comment
-    # Another comment
     return 42"""

        result = apply_diff(original, diff, mode="default")
        self.assertNotIn("# This is a comment", result)
        self.assertIn("return 42", result)

    def test_exact_match_rejects_trailing_whitespace_mismatch(self):
        original = "def hello():   \n    print('hi')"
        diff = """@@
 def hello():
-    print('hi')
+    print('hello')"""

        with self.assertRaisesRegex(ValueError, "Hunk 1: context not found"):
            apply_diff(original, diff, mode="default")

    def test_exact_match_rejects_quote_normalization(self):
        original = 'title = “keep”\nvalue = ‘old’'
        diff = """@@
 title = “keep”
-value = 'old'
+value = 'new'"""

        with self.assertRaisesRegex(ValueError, "Hunk 1: context not found"):
            apply_diff(original, diff, mode="default")

    def test_quote_mismatch_rejects_ambiguous_context(self):
        original = "value = ‘old’\nseparator\nvalue = ‘old’"
        diff = """@@
-value = 'old'
+value = 'new'"""

        with self.assertRaisesRegex(ValueError, "Hunk 1: context not found"):
            apply_diff(original, diff, mode="default")

    def test_quote_normalization_preserves_leading_whitespace(self):
        original = "def update():\n    value = ‘old’"
        diff = """@@
-value = 'old'
+value = 'new'"""

        with self.assertRaisesRegex(ValueError, "Hunk 1: context not found"):
            apply_diff(original, diff, mode="default")

        self.assertEqual(original, "def update():\n    value = ‘old’")

    def test_trailing_whitespace_mismatch_is_context_not_found(self):
        original = "value = ‘old’   "
        diff = """@@
-value = 'old'
+value = 'new'"""

        with self.assertRaisesRegex(ValueError, "Hunk 1: context not found"):
            apply_diff(original, diff, mode="default")

    def test_reports_all_missing_hunks_in_one_file(self):
        original = "FIRST = 1\nMIDDLE = True\nSECOND = 2"
        diff = """@@
-STALE_FIRST = 1
+FIRST = 10
@@
-STALE_SECOND = 2
+SECOND = 20"""

        with self.assertRaises(ValueError) as exc:
            apply_diff(original, diff, mode="default")

        message = str(exc.exception)
        self.assertIn("Hunk 1: context not found", message)
        self.assertIn("Hunk 2: context not found", message)
        self.assertNotIn("Expected context:", message)
        self.assertNotIn("Actual from line", message)
        self.assertNotIn("STALE_FIRST = 1", message)

    def test_unprefixed_source_line_is_malformed_not_context_mismatch(self):
        original = "def max_matching(n_left, n_right, adj):\n    return 0\n"
        diff = """@@
def max_matching(n_left, n_right, adj):
-    return 0
+    return 1"""

        with self.assertRaises(ValueError) as exc:
            apply_diff(original, diff, mode="default")

        message = str(exc.exception)
        self.assertIn("Malformed patch", message)
        self.assertIn("def max_matching", message)
        self.assertNotIn("context not found", message)
        self.assertNotIn("Expected context:", message)


class TestParsePatchEnvelope(unittest.TestCase):
    """Tests for strict multi-file patch envelope parsing."""

    def test_parses_multiple_file_operations(self):
        patch = """*** Begin Patch
*** Update File: app.py
@@
-OLD = 1
+NEW = 1
*** Add File: test_app.py
+def test_app():
+    pass
*** Delete File: obsolete.py
*** End Patch"""

        operations = parse_patch_envelope(patch)

        self.assertEqual(
            [(op.action, op.path) for op in operations],
            [("update", "app.py"), ("add", "test_app.py"), ("delete", "obsolete.py")],
        )
        self.assertIn("-OLD = 1", operations[0].diff)
        self.assertIn("+def test_app():", operations[1].diff)
        self.assertEqual(operations[2].diff, "")

    def test_rejects_duplicate_file_operations(self):
        patch = """*** Begin Patch
*** Update File: app.py
@@
-a
+b
*** Delete File: app.py
*** End Patch"""

        with self.assertRaisesRegex(ValueError, "Duplicate file operation"):
            parse_patch_envelope(patch)

    def test_rejects_text_outside_envelope(self):
        with self.assertRaisesRegex(ValueError, "must start"):
            parse_patch_envelope("prefix\n*** Begin Patch\n*** Delete File: a.py\n*** End Patch")



if __name__ == '__main__':
    unittest.main()
