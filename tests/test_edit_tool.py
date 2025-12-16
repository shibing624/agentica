# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for EditTool with V4A diff format support.
"""
import os
import sys
import unittest
import tempfile
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.tools.edit_tool import EditTool, apply_diff


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


class TestEditTool(unittest.TestCase):
    """Test cases for the EditTool class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.editor = EditTool(work_dir=self.test_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_edit_file_create(self):
        """Test creating a new file."""
        result = self.editor.edit_file("test.py", "print('Hello')")
        self.assertIn("Created", result)
        
        file_path = os.path.join(self.test_dir, "test.py")
        self.assertTrue(os.path.exists(file_path))
        with open(file_path, 'r') as f:
            self.assertEqual(f.read(), "print('Hello')")

    def test_edit_file_update(self):
        """Test updating an existing file."""
        file_path = os.path.join(self.test_dir, "test.py")
        with open(file_path, 'w') as f:
            f.write("print('Old')")

        result = self.editor.edit_file("test.py", "print('New')")
        self.assertIn("Updated", result)
        
        with open(file_path, 'r') as f:
            self.assertEqual(f.read(), "print('New')")

    def test_search_replace_literal(self):
        """Test literal search and replace."""
        file_path = os.path.join(self.test_dir, "test.py")
        with open(file_path, 'w') as f:
            f.write("Hello World\nHello Python")

        result = self.editor.search_replace("test.py", "Hello", "Hi")
        self.assertIn("Replaced 2", result)
        
        with open(file_path, 'r') as f:
            content = f.read()
            self.assertIn("Hi World", content)
            self.assertIn("Hi Python", content)

    def test_search_replace_regex(self):
        """Test regex search and replace."""
        file_path = os.path.join(self.test_dir, "test.py")
        with open(file_path, 'w') as f:
            f.write("foo123bar\nfoo456bar")

        result = self.editor.search_replace("test.py", r"foo\d+bar", "replaced", use_regex=True)
        self.assertIn("Replaced 2", result)
        
        with open(file_path, 'r') as f:
            content = f.read()
            self.assertEqual(content, "replaced\nreplaced")

    def test_search_replace_no_match(self):
        """Test search and replace with no matches."""
        file_path = os.path.join(self.test_dir, "test.py")
        with open(file_path, 'w') as f:
            f.write("Hello World")

        result = self.editor.search_replace("test.py", "NotFound", "Replaced")
        self.assertIn("No occurrences", result)

    def test_apply_patch_v4a_format(self):
        """Test applying V4A format patch."""
        file_path = os.path.join(self.test_dir, "test.py")
        with open(file_path, 'w') as f:
            f.write("def hello():\n    print('Hello')\n")

        v4a_patch = """@@
 def hello():
-    print('Hello')
+    print('Hello, World!')"""

        result = self.editor.apply_patch("test.py", v4a_patch)
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

        result = self.editor.apply_patch("test.py", full_v4a)
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

        result = self.editor.apply_patch("test.py", unified_patch)
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

        result = self.editor.compare_files("file1.py", "file2.py")
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

        result = self.editor.compare_files("file1.py", "file2.py")
        self.assertEqual(result, "Files are identical.")

    def test_detect_diff_format_v4a(self):
        """Test V4A format detection."""
        v4a_patches = [
            "*** Begin Patch\n*** Update File: test.py\n@@\n content",
            "@@\n-old\n+new",
            "@@ def hello():\n-old\n+new",
        ]
        for patch in v4a_patches:
            result = self.editor._detect_diff_format(patch)
            self.assertEqual(result, "v4a", f"Failed for: {patch[:30]}...")

    def test_detect_diff_format_unified(self):
        """Test unified diff format detection."""
        unified_patch = "@@ -1,3 +1,4 @@\n context\n-old\n+new"
        result = self.editor._detect_diff_format(unified_patch)
        self.assertEqual(result, "unified")

    def test_file_not_found(self):
        """Test error handling for non-existent file."""
        result = self.editor.search_replace("nonexistent.py", "a", "b")
        self.assertIn("Error", result)

    def test_invalid_regex(self):
        """Test error handling for invalid regex."""
        file_path = os.path.join(self.test_dir, "test.py")
        with open(file_path, 'w') as f:
            f.write("content")

        result = self.editor.search_replace("test.py", "[invalid", "replacement", use_regex=True)
        self.assertIn("Error", result)


if __name__ == '__main__':
    unittest.main()
