# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: What apply_patch tells the model when a hunk fails to match.

Production logs (6,951 tool results across two long sessions) put apply_patch at
6.5% errors, the worst rate of any frequently-used tool. All 17
"context not found" failures returned one thing: the model's own line echoed
back. Rebuilt from the logged patches, they were: 5 hunks whose block was in the
file verbatim but above the previous hunk (hunks out of file order), 2 wrong
indentation, 1 squeezed-out spaces, 9 lines the file did not have. Echoing the
line cannot tell those apart, and each needs a different fix.

The tool must name the *class* of mismatch without quoting the file: an earlier
attempt dumped Expected/Actual and was removed (a8b305c7) because it handed the
model lines it had not read.
"""

import asyncio
import tempfile
import unittest
from pathlib import Path

from agentica.tools.builtin.file_tool import BuiltinFileTool

SRC = """def alpha():
    return 1


def beta(value):
    if value:
        return _iso_to_epoch(value) if value else None
    return None
"""


def _apply(patch: str, src: str = SRC) -> str:
    """Run apply_patch in a scratch dir and return what the caller/model sees."""
    directory = Path(tempfile.mkdtemp())
    (directory / "m.py").write_text(src)
    tool = BuiltinFileTool(work_dir=str(directory))
    try:
        return str(asyncio.run(tool.apply_patch(patch=patch)))
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


class TestContextMissIsDiagnosable(unittest.TestCase):
    """A miss must say which repair to make, distinguishing the cases."""

    def test_wrong_indentation_is_named_as_indentation(self):
        out = _apply("""*** Begin Patch
*** Update File: m.py
@@
-  return _iso_to_epoch(value) if value else None
+  return convert(value)
*** End Patch""")
        # The file has that line at a different indent; saying "not found" is
        # false and sends the model looking for a line that is right there.
        self.assertIn("at line 7 with different indentation or spacing", out)
        self.assertNotIn("No line in the file matches", out)

    def test_line_absent_from_file_is_named_absent(self):
        out = _apply("""*** Begin Patch
*** Update File: m.py
@@
-    return something_never_in_file()
+    return 2
*** End Patch""")
        self.assertIn("No line in the file matches", out)
        self.assertNotIn("different indentation or spacing", out)

    def test_dropped_internal_spaces_is_named_as_spacing(self):
        """The real production case: the model squeezed the spaces out."""
        src = "- New combined dev should be written under campaign/inputs/\n"
        # Every character is present in the file; only the spacing differs.
        squeezed = "- Newcombineddevshouldbe writtenundercampaign/inputs/"
        out = _apply(
            "*** Begin Patch\n*** Update File: m.py\n@@\n"
            " " + squeezed + "\n"
            "*** End Patch",
            src=src,
        )
        self.assertIn("at line 1 with different indentation or spacing", out)

    def test_multi_line_drift_shows_where_the_block_diverges(self):
        """Every hunk line exists, but not in this order — a block that cannot line up."""
        src = "alpha\nbeta\ngamma\n"
        out = _apply(
            "*** Begin Patch\n*** Update File: m.py\n@@\n"
            " beta\n"
            " alpha\n"
            " gamma\n"
            "*** End Patch",
            src=src,
        )
        # Each line is in the file, so "not found" would be the wrong answer.
        self.assertIn("contiguous block", out)
        self.assertNotIn("No line in the file matches", out)

    def test_a_real_match_still_applies(self):
        directory = Path(tempfile.mkdtemp())
        (directory / "m.py").write_text(SRC)
        tool = BuiltinFileTool(work_dir=str(directory))
        asyncio.run(tool.apply_patch(patch="""*** Begin Patch
*** Update File: m.py
@@
-    return 1
+    return 42
*** End Patch"""))
        self.assertIn("return 42", (directory / "m.py").read_text())


class TestHunksOutOfFileOrder(unittest.TestCase):
    """The largest class in the logs: hunk 2 aims above where hunk 1 matched.

    Every line was in the file verbatim, and the tool still said "not found",
    because matching only searched forward from the previous hunk.
    """

    SRC = (
        "import logging\n"
        "\n"
        'logger = logging.getLogger("m")\n'
        "\n"
        "\n"
        "async def write():\n"
        "    if err is not None:\n"
        "        raise err\n"
        "    raise Stale()\n"
    )
    # Body edit first, module constant second: the order a model writes it in.
    PATCH = (
        "*** Begin Patch\n*** Update File: m.py\n"
        "@@\n"
        "     if err is not None:\n"
        "         raise err\n"
        "+    retry()\n"
        "@@\n"
        ' logger = logging.getLogger("m")\n'
        "+\n"
        "+_RETRIES = 2\n"
        "*** End Patch"
    )

    def test_block_found_once_above_the_previous_hunk_is_applied(self):
        directory = Path(tempfile.mkdtemp())
        (directory / "m.py").write_text(self.SRC)
        tool = BuiltinFileTool(work_dir=str(directory))
        asyncio.run(tool.apply_patch(patch=self.PATCH))
        self.assertEqual(
            (directory / "m.py").read_text(),
            self.SRC.replace(
                'logger = logging.getLogger("m")\n',
                'logger = logging.getLogger("m")\n\n_RETRIES = 2\n',
            ).replace("        raise err\n", "        raise err\n    retry()\n"),
        )

    def test_ambiguous_block_above_the_previous_hunk_names_the_order(self):
        # Two places above hunk 1 match, so where hunk 2 goes is a guess.
        out = _apply(
            "*** Begin Patch\n*** Update File: m.py\n"
            "@@\n b\n+c\n@@\n a\n+x\n*** End Patch",
            src="a\na\nb\n",
        )
        self.assertIn("Hunk 2: context not found: 'a'.", out)
        self.assertIn("at line 1 and elsewhere, all above where the previous hunk matched", out)

    def test_out_of_order_hunks_that_overlap_fail_instead_of_writing(self):
        out = _apply(
            "*** Begin Patch\n*** Update File: m.py\n"
            "@@\n-q\n+Q1\n r\n@@\n p\n-q\n+Q2\n*** End Patch",
            src="p\nq\nr\n",
        )
        self.assertIn("overlapping chunk", out)


class TestDiagnosisNeverQuotesTheFile(unittest.TestCase):
    """Naming the mismatch class must not leak lines the model has not read.

    Exception, on purpose (db165e12): when every hunk line exists in the file
    and only the block's order is wrong, the model has already read that span,
    so naming the breaking line answers its actual question. That path is left
    alone; the ones below are the cases where the model read nothing.
    """

    # The line the model's hunk expects, and a DIFFERENT line the file really
    # has there. If the preview dumps file text, UNREAD_LINE shows up.
    UNREAD_LINE = "SECRET_FILE_CONTENT_NEVER_READ"

    def test_indentation_miss_does_not_leak_the_file_line(self):
        src = f"def f():\n        {self.UNREAD_LINE} = 1\n"
        out = _apply(
            "*** Begin Patch\n*** Update File: m.py\n@@\n"
            "-    placeholder = 2\n"
            "*** End Patch",
            src=src,
        )
        self.assertNotIn(self.UNREAD_LINE, out)

    def test_absent_line_does_not_leak_surrounding_file_text(self):
        src = f"def f():\n    {self.UNREAD_LINE} = 1\n    return 2\n"
        out = _apply(
            "*** Begin Patch\n*** Update File: m.py\n@@\n"
            "-    nonexistent_thing = 3\n"
            "*** End Patch",
            src=src,
        )
        self.assertNotIn(self.UNREAD_LINE, out)

    def test_spacing_miss_does_not_leak_the_file_line(self):
        # The model never sends UNREAD_LINE: its hunk is a short, plainly wrong
        # line, and the file's own line is what must not come back.
        src = f"- {self.UNREAD_LINE} goes here\n"
        out = _apply(
            "*** Begin Patch\n*** Update File: m.py\n@@\n"
            "- some other line entirely\n"
            "*** End Patch",
            src=src,
        )
        self.assertNotIn(self.UNREAD_LINE, out)


class TestProductionFailureShapes(unittest.TestCase):
    """The shapes found in the session logs, reproduced."""

    NOTES = (
        "# Session notes\n"
        "\n"
        "- New combined dev should be written under campaign/inputs/ with old ret + new TS dev prefix IDs\n"
        "- Original ret file parent campaign means put single-ret outputs at eval root\n"
    )

    def test_spliced_and_squeezed_prose_line_is_absent(self):
        """The notes.md failure in the logs: spaces dropped and text spliced on.

        Nothing in the file equals it, even ignoring whitespace. The right
        repair is to re-read the line, not a guess about which part to cut.
        """
        spliced = (
            "- Newcombineddevshouldbe writtenunder campaign/inputs/ witholdret+newTSdevprefixIDs "
            "thenoutputsoutSIDELINGSoutsideinputdir"
        )
        out = _apply(
            "*** Begin Patch\n*** Update File: m.py\n@@\n " + spliced + "\n*** End Patch",
            src=self.NOTES,
        )
        self.assertIn("No line in the file matches", out)

    def test_respaced_line_is_named_as_spacing(self):
        clipped = "-Originalretfileparentcampaignmeansputsingle-retoutputsatevalroot"
        # A context line keeps the leading space; the file's own "-" follows it.
        out = _apply(
            "*** Begin Patch\n*** Update File: m.py\n@@\n " + clipped + "\n*** End Patch",
            src=self.NOTES,
        )
        self.assertIn("at line 4 with different indentation or spacing", out)

    def test_truly_absent_line_is_the_only_case_called_absent(self):
        absent = "-this line was never in this file at all"
        out = _apply(
            "*** Begin Patch\n*** Update File: m.py\n@@\n " + absent + "\n*** End Patch",
            src=self.NOTES,
        )
        self.assertIn("No line in the file matches", out)


if __name__ == "__main__":
    unittest.main()
