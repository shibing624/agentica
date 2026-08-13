# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for PatchTool with V4A diff format support.
"""
import os
import sys
import unittest
import tempfile
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.tools.patch_tool import PatchTool, apply_diff, parse_patch_envelope


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

    def test_fuzzy_matching_whitespace(self):
        """Test fuzzy matching with trailing whitespace."""
        original = "def hello():   \n    print('hi')"
        diff = """@@
 def hello():
-    print('hi')
+    print('hello')"""

        result = apply_diff(original, diff, mode="default")
        self.assertIn("print('hello')", result)

    def test_quote_normalization_applies_only_unique_context(self):
        original = 'title = “keep”\nvalue = ‘old’'
        diff = """@@
 title = “keep”
-value = 'old'
+value = 'new'"""

        result = apply_diff(original, diff, mode="default")

        self.assertEqual(result, 'title = “keep”\nvalue = \'new\'')

    def test_quote_normalization_rejects_ambiguous_context(self):
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

    def test_quote_normalization_tolerates_trailing_whitespace(self):
        original = "value = ‘old’   "
        diff = """@@
-value = 'old'
+value = 'new'"""

        result = apply_diff(original, diff, mode="default")

        self.assertEqual(result, "value = 'new'")

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
        self.assertIn("STALE_FIRST = 1", message)
        self.assertIn("Hunk 2: context not found", message)
        self.assertIn("STALE_SECOND = 2", message)
        self.assertIn("None of the expected lines appear in the file.", message)

    def test_stale_hunk_shows_matching_region_not_file_header(self):
        original = "\n".join([
            "# -*- coding: utf-8 -*-",
            '"""module docstring"""',
            "",
            "class StreamDisplayManager:",
            "    @staticmethod",
            "    def _fmt_elapsed(elapsed):",
            '        """Format elapsed seconds."""',
            "        if elapsed < 10:",
            "            return ''",
            "        return f'({elapsed:.1f}s)'",
        ])
        diff = """@@
     @staticmethod
     def _fmt_elapsed(elapsed):
         \"\"\"Format elapsed seconds; fast calls render nothing.\"\"\"
-        if elapsed < 1:
+        if elapsed < 10:
             return ''
         return f'({elapsed:.1f}s)'"""

        with self.assertRaises(ValueError) as exc:
            apply_diff(original, diff, mode="default")

        message = str(exc.exception)
        self.assertIn("def _fmt_elapsed(elapsed):", message)
        self.assertIn("Actual from line 5:", message)
        self.assertIn('"""Format elapsed seconds."""', message)
        self.assertNotIn("# -*- coding: utf-8 -*-", message)
        self.assertNotIn("from line 1.", message)

    def test_stale_hunk_says_when_expected_lines_are_absent(self):
        original = "# -*- coding: utf-8 -*-\nprint('hello')\n"
        diff = """@@
 def test_chat_accepts_cron_source_override(self, tmp_path):
     from agentica.gateway.services.agent_service import AgentService"""

        with self.assertRaises(ValueError) as exc:
            apply_diff(original, diff, mode="default")

        message = str(exc.exception)
        self.assertIn("None of the expected lines appear in the file.", message)
        self.assertNotIn("# -*- coding: utf-8 -*-", message)
        self.assertNotIn("Actual from line", message)

    def test_stale_hunk_shows_mismatch_past_the_preview_limit(self):
        """A hunk that matches 6 lines and diverges on the 7th must show line 7.

        The preview used to cut both blocks at 6 lines, so this failure printed
        two byte-identical previews with the only difference folded away.
        """
        original = "\n".join([
            "- v1.4.10 entry",
            "- v1.4.9 entry",
            "",
            "<details>",
            "<summary>Older releases</summary>",
            "",
            "- v1.4.7 entry with a very long tail that the model shortened",
        ])
        diff = """@@
 - v1.4.10 entry
-- v1.4.9 entry
 
 <details>
 <summary>Older releases</summary>
 
+- v1.4.9 entry
 - v1.4.7 entry"""

        with self.assertRaises(ValueError) as exc:
            apply_diff(original, diff, mode="default")

        message = str(exc.exception)
        self.assertIn("> - v1.4.7 entry", message)
        self.assertIn(
            "> - v1.4.7 entry with a very long tail that the model shortened",
            message,
        )
        self.assertIn("First difference at context line 7 (file line 7)", message)
        self.assertIn("earlier line", message)
        self.assertNotIn("more context lines", message)

    def test_short_hunk_preview_has_no_window_offset_noise(self):
        original = "ALPHA = 1\nBETA = 2\n"
        diff = """@@
-GAMMA = 3
+GAMMA = 4"""

        with self.assertRaises(ValueError) as exc:
            apply_diff(original, diff, mode="default")

        message = str(exc.exception)
        self.assertIn("  GAMMA = 3", message)
        self.assertNotIn("earlier line", message)
        self.assertNotIn("First difference", message)

    def test_mismatch_note_reports_absolute_file_line(self):
        original = "\n".join([f"line{index}" for index in range(1, 21)])
        diff = """@@ line10
 line11
 line12
-line13changed
+line13new"""

        with self.assertRaises(ValueError) as exc:
            apply_diff(original, diff, mode="default")

        message = str(exc.exception)
        self.assertIn("> line13changed", message)
        self.assertIn("> line13", message)
        self.assertIn("First difference at context line 3 (file line 13)", message)

    def test_eof_hunk_shorter_file_region_is_reported(self):
        original = "alpha\nbeta"
        diff = """@@
 alpha
 beta
 gamma
-delta
+DELTA
*** End of File"""

        with self.assertRaises(ValueError) as exc:
            apply_diff(original, diff, mode="default")

        message = str(exc.exception)
        self.assertIn("the file region ends before the hunk does", message)


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


class TestPatchTool(unittest.TestCase):
    """Test cases for the PatchTool class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.tool = PatchTool(work_dir=self.test_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_apply_patch_v4a_format(self):
        """Test applying V4A format patch."""
        file_path = os.path.join(self.test_dir, "test.py")
        with open(file_path, 'w') as f:
            f.write("def hello():\n    print('Hello')\n")

        v4a_patch = """@@
 def hello():
-    print('Hello')
+    print('Hello, World!')"""

        result = self.tool.apply_patch("test.py", v4a_patch)
        self.assertIn("Successfully patched", result)
        
        with open(file_path, 'r') as f:
            content = f.read()
            self.assertIn("Hello, World!", content)

    def test_apply_patch_full_v4a_wrapper(self):
        """Test applying full V4A format with wrapper."""
        file_path = os.path.join(self.test_dir, "test.py")
        with open(file_path, 'w') as f:
            f.write("class Calc:\n    def add(self, a, b):\n        return a + b\n")

        full_v4a = """*** Begin Patch
*** Update File: test.py
@@
 class Calc:
     def add(self, a, b):
-        return a + b
+        # Add numbers
+        return a + b
*** End Patch"""

        result = self.tool.apply_patch("test.py", full_v4a)
        self.assertIn("Successfully patched", result)
        
        with open(file_path, 'r') as f:
            content = f.read()
            self.assertIn("# Add numbers", content)

    def test_apply_patch_unified_format(self):
        """Test applying unified diff format patch."""
        file_path = os.path.join(self.test_dir, "test.py")
        with open(file_path, 'w') as f:
            f.write("line1\nline2\nline3\n")

        unified_patch = """@@ -1,3 +1,3 @@
 line1
-line2
+modified_line2
 line3"""

        result = self.tool.apply_patch("test.py", unified_patch)
        self.assertIn("Successfully patched", result)
        
        with open(file_path, 'r') as f:
            content = f.read()
            self.assertIn("modified_line2", content)

    def test_compare_files(self):
        """Test comparing two files."""
        file1 = os.path.join(self.test_dir, "file1.py")
        file2 = os.path.join(self.test_dir, "file2.py")
        
        with open(file1, 'w') as f:
            f.write("line1\nline2\nline3")
        with open(file2, 'w') as f:
            f.write("line1\nmodified\nline3")

        result = self.tool.compare_files("file1.py", "file2.py")
        self.assertIn("-line2", result)
        self.assertIn("+modified", result)

    def test_compare_files_identical(self):
        """Test comparing identical files."""
        file1 = os.path.join(self.test_dir, "file1.py")
        file2 = os.path.join(self.test_dir, "file2.py")
        
        with open(file1, 'w') as f:
            f.write("same content")
        with open(file2, 'w') as f:
            f.write("same content")

        result = self.tool.compare_files("file1.py", "file2.py")
        self.assertEqual(result, "Files are identical.")

    def test_detect_diff_format_v4a(self):
        """Test V4A format detection."""
        v4a_patches = [
            "*** Begin Patch\n*** Update File: test.py\n@@\n content",
            "@@\n-old\n+new",
            "@@ def hello():\n-old\n+new",
        ]
        for patch in v4a_patches:
            result = self.tool._detect_diff_format(patch)
            self.assertEqual(result, "v4a", f"Failed for: {patch[:30]}...")

    def test_detect_diff_format_unified(self):
        """Test unified diff format detection."""
        unified_patch = "@@ -1,3 +1,4 @@\n context\n-old\n+new"
        result = self.tool._detect_diff_format(unified_patch)
        self.assertEqual(result, "unified")

    def test_file_not_found(self):
        """Test error handling for non-existent file."""
        with self.assertRaises((FileNotFoundError, ValueError)):
            self.tool.apply_patch("nonexistent.py", "@@\n-old\n+new")


if __name__ == '__main__':
    unittest.main()
