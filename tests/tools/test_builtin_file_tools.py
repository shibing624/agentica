# -*- coding: utf-8 -*-
"""Tests for BuiltinFileTool (read / write / glob / grep)."""
import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-real")

import pytest

from agentica.tools.builtin import BuiltinFileTool


class _BlockingStream:
    def __init__(self, started: asyncio.Event):
        self._started = started

    async def readline(self):
        self._started.set()
        await asyncio.Future()

    async def read(self):
        self._started.set()
        await asyncio.Future()


class BlockingSubprocess:
    """Minimal subprocess double whose stdout readline blocks until cancelled."""

    def __init__(self):
        self.started = asyncio.Event()
        self.returncode = None
        self._transport = None
        self.stdout = _BlockingStream(self.started)
        self.stderr = _BlockingStream(self.started)

    async def communicate(self):
        self.started.set()
        await asyncio.Future()



class TestBuiltinFileToolReadFile:
    def test_empty_file_returns_empty_notice(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "empty.txt")
        Path(fp).touch()
        result = asyncio.run(file_tool.read_file(fp))
        assert "File is empty:" in result
        assert "<system-reminder>" not in result

    def test_read_simple_file(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "test.txt")
        p.write_text("line1\nline2\nline3\n")
        result = asyncio.run(file_tool.read_file(str(p)))
        assert "line1" in result
        assert "line2" in result
        assert "line3" in result
        assert "[File metadata:" not in result

    def test_read_file_with_offset_limit(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "lines.txt")
        p.write_text("\n".join(f"line{i}" for i in range(1, 21)))
        # Read lines 5-9 (offset=4, limit=5)
        result = asyncio.run(file_tool.read_file(str(p), offset=4, limit=5))
        assert "line5" in result
        assert "line9" in result
        # line1 should not be present since offset skips it
        assert "line1\t" not in result  # Use tab to avoid matching "line10"

    def test_read_nonexistent_file_returns_path_state(self, file_tool, tmp_dir):
        existing = Path(tmp_dir, "src")
        existing.mkdir()
        candidate = existing / "usage.py"
        candidate.write_text("x = 1\n")

        with pytest.raises(FileNotFoundError) as exc:
            asyncio.run(file_tool.read_file(str(existing / "usage_tracking.py")))

        msg = str(exc.value)
        assert "File not found:" in msg
        assert "Resolved path:" in msg
        assert f"Nearest existing parent: {existing}" in msg
        assert "Candidate paths:" not in msg
        assert str(candidate) not in msg
        assert "Next step:" not in msg

    def test_read_long_lines_truncated(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "long.txt")
        p.write_text("A" * 5000 + "\n")
        result = asyncio.run(file_tool.read_file(str(p)))
        # Default max_line_length is 2000, line should be truncated with "..."
        assert "..." in result

    def test_read_file_shows_line_numbers(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "numbered.txt")
        p.write_text("alpha\nbeta\ngamma\n")
        result = asyncio.run(file_tool.read_file(str(p)))
        # Should contain line numbers (cat -n format)
        assert "\t" in result  # tab separator between line number and content


class TestMissingPathErrors:
    """Missing-path errors expose path state without guessing candidates."""

    def test_grep_missing_path_does_not_suggest_candidates(self, file_tool, tmp_dir):
        existing = Path(tmp_dir, "src")
        existing.mkdir()
        (existing / "config.py").write_text("X = 1\n")

        with pytest.raises(FileNotFoundError) as exc:
            asyncio.run(file_tool.grep("class .*Config", path="missing-dir/config.py"))

        msg = str(exc.value)
        assert "Path not found:" in msg
        assert "Resolved path:" in msg
        assert "Nearest existing parent:" in msg
        assert "Did you mean:" not in msg
        assert "src/config.py" not in msg
        assert "Next step:" not in msg

    def test_glob_missing_path_does_not_suggest_candidates(self, file_tool, tmp_dir):
        Path(tmp_dir, "src").mkdir()
        Path(tmp_dir, "src", "config.py").write_text("X = 1\n")

        with pytest.raises(FileNotFoundError) as exc:
            asyncio.run(file_tool.glob("*.py", path="missing-dir/config.py"))

        msg = str(exc.value)
        assert "Directory not found:" in msg
        assert "Did you mean:" not in msg
        assert "run glob '**/config.py'" not in msg

    def test_apply_patch_missing_path_does_not_suggest_candidates(self, file_tool, tmp_dir):
        Path(tmp_dir, "src").mkdir()
        Path(tmp_dir, "src", "config.py").write_text("X = 1\n")

        with pytest.raises(FileNotFoundError) as exc:
            asyncio.run(file_tool.apply_patch(
                "*** Begin Patch\n"
                "*** Update File: missing-dir/config.py\n"
                "@@\n"
                "-a\n"
                "+b\n"
                "*** End Patch"
            ))

        msg = str(exc.value)
        assert "File not found:" in msg or "not found" in msg.lower()
        assert "Did you mean:" not in msg
        assert "src/config.py" not in msg


class TestBuiltinFileToolReadCorrectness:
    """read_file always reflects current disk content (no memoization)."""

    def test_second_read_returns_same_content(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "cached.txt")
        p.write_text("\n".join(f"line{i}" for i in range(1, 11)))
        r1 = asyncio.run(file_tool.read_file(str(p)))
        r2 = asyncio.run(file_tool.read_file(str(p)))
        assert r1 == r2  # no cache: each call re-reads, content identical

    def test_external_size_change_reflected(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "ext_size.txt")
        p.write_text("v1")
        asyncio.run(file_tool.read_file(str(p)))
        p.write_text("v2 with more bytes")
        r2 = asyncio.run(file_tool.read_file(str(p)))
        assert "v2 with more bytes" in r2

    def test_external_mtime_only_change_reflected(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "ext_mtime.txt")
        p.write_text("same-size-a")
        asyncio.run(file_tool.read_file(str(p)))
        p.write_text("same-size-b")  # identical size, different content
        st = p.stat()
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
        r2 = asyncio.run(file_tool.read_file(str(p)))
        assert "same-size-b" in r2

    def test_apply_patch_reflected_on_next_read(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "edit.txt")
        p.write_text("hello world\n")
        asyncio.run(file_tool.read_file(str(p)))
        asyncio.run(file_tool.apply_patch(
            "*** Begin Patch\n"
            "*** Update File: edit.txt\n"
            "@@\n"
            "-hello world\n"
            "+hello agentica\n"
            "*** End Patch"
        ))
        r = asyncio.run(file_tool.read_file(str(p)))
        assert "hello agentica" in r
        assert "hello world" not in r

    def test_write_file_reflected_on_next_read(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "w.txt")
        p.write_text("old content")
        asyncio.run(file_tool.read_file(str(p)))
        asyncio.run(file_tool.write_file(str(p), "new content"))
        r = asyncio.run(file_tool.read_file(str(p)))
        assert "new content" in r
        assert "old content" not in r

    def test_offset_and_limit_select_distinct_windows(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "pages.txt")
        p.write_text("\n".join(f"line{i}" for i in range(1, 21)))
        r1 = asyncio.run(file_tool.read_file(str(p), offset=0, limit=5))
        r2 = asyncio.run(file_tool.read_file(str(p), offset=5, limit=5))
        assert r1 != r2
        assert "line6" in r2 and "line6" not in r1

    def test_read_file_tail_last_lines(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "tail.txt")
        p.write_text("\n".join(f"line{i}" for i in range(1, 11)))
        result = asyncio.run(file_tool.read_file(str(p), tail=3))
        assert "line8" in result and "line10" in result
        assert "line7" not in result
        assert "line1\t" not in result

    def test_read_file_negative_offset_is_tail_window(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "neg.txt")
        p.write_text("\n".join(f"line{i}" for i in range(1, 11)))
        result = asyncio.run(file_tool.read_file(str(p), offset=-4, limit=4))
        assert "line7" in result and "line10" in result
        assert "line6" not in result

    def test_read_file_negative_offset_limit_takes_oldest_of_window(self, file_tool, tmp_dir):
        """offset=-4 opens a 4-line window on the end; limit=2 keeps its
        oldest lines (7,8 of 10), matching positive-offset semantics."""
        p = Path(tmp_dir, "win.txt")
        p.write_text("\n".join(f"line{i}" for i in range(1, 11)))
        result = asyncio.run(file_tool.read_file(str(p), offset=-4, limit=2))
        assert "line7" in result and "line8" in result
        assert "line9" not in result and "line10" not in result

    def test_effective_tail_treats_zero_as_omit(self):
        from agentica.tools.builtin.file_tool import _effective_tail

        assert _effective_tail(None) is None
        assert _effective_tail(0) is None
        assert _effective_tail("0") is None
        assert _effective_tail(3) == 3
        assert _effective_tail(-3) == 3

    def test_read_file_tail_zero_reads_from_start(self, file_tool, tmp_dir):
        """tail=0 is omit, not an error — models send it for a top-of-file page."""
        p = Path(tmp_dir, "zero.txt")
        p.write_text("a\nb\nc\n")
        result = asyncio.run(file_tool.read_file(str(p), tail=0))
        assert "a" in result and "b" in result

    def test_read_file_negative_tail_is_last_n_lines(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "neg_tail.txt")
        p.write_text("\n".join(f"line{i}" for i in range(1, 11)))
        result = asyncio.run(file_tool.read_file(str(p), tail=-3))
        assert "line8" in result and "line10" in result
        assert "line7" not in result

    def test_read_file_tail_larger_than_file_returns_all(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "short.txt")
        p.write_text("\n".join(f"line{i}" for i in range(1, 4)))
        result = asyncio.run(file_tool.read_file(str(p), tail=700))
        assert "line1" in result and "line3" in result

    def test_read_file_tail_beats_large_file_guard(self, file_tool, tmp_dir):
        """tail holds a bounded number of lines on any file size — the 256KB
        guard must not reject the call its own error message recommends."""
        p = Path(tmp_dir, "big.log")
        n = 5000
        p.write_text("\n".join(f"row{i} {'x' * 80}" for i in range(1, n + 1)))
        assert p.stat().st_size > file_tool.MAX_FILE_SIZE_BYTES

        result = asyncio.run(file_tool.read_file(str(p), tail=5))
        assert f"row{n}" in result and f"row{n - 4}" in result
        assert f"row{n - 5}" not in result

        result = asyncio.run(file_tool.read_file(str(p), offset=-3))
        assert f"row{n}" in result and f"row{n - 2}" in result

        # front-paged reads on the same file still hit the guard
        with pytest.raises(ValueError, match="Use tail to read the end") as exc:
            asyncio.run(file_tool.read_file(str(p)))
        assert "offset/limit" not in str(exc.value)

    def test_read_file_tail_times_out_on_slow_scan(self, file_tool, tmp_dir):
        p = Path(tmp_dir, "slow.log")
        p.write_text("x\n")

        async def hang(*args, **kwargs):
            await asyncio.sleep(1)
            return "", 0, 1, 0

        with patch.object(file_tool, "_read_from_end", hang), \
             patch("agentica.tools.builtin.file_tool._READ_TIMEOUT", 0.05):
            with pytest.raises(TimeoutError, match="from the end"):
                asyncio.run(file_tool.read_file(str(p), tail=1))


class TestBuiltinFileToolWriteFile:
    def test_write_result_carries_actual_file_change(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "existing.txt")
        Path(fp).write_text("before\n")

        result = asyncio.run(file_tool.write_file(fp, "after\n"))

        assert result.display_meta == {
            "files": [{
                "path": str(Path(fp).resolve()),
                "action": "update",
                "before": "before\n",
                "after": "after\n",
            }]
        }

    def test_write_new_file(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "new.txt")
        result = asyncio.run(file_tool.write_file(fp, "hello world"))
        assert "Created" in result
        assert Path(fp).read_text() == "hello world"

    def test_write_overwrite_file(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "existing.txt")
        Path(fp).write_text("old")
        result = asyncio.run(file_tool.write_file(fp, "new content"))
        assert "Updated" in result
        assert Path(fp).read_text() == "new content"

    def test_write_creates_parent_dirs(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "a", "b", "c.txt")
        result = asyncio.run(file_tool.write_file(fp, "nested"))
        assert "Created" in result
        assert Path(fp).read_text() == "nested"

    def test_write_returns_absolute_path(self, file_tool, tmp_dir):
        fp = os.path.join(tmp_dir, "abs.txt")
        result = asyncio.run(file_tool.write_file(fp, "test"))
        # Result should contain the absolute path
        assert tmp_dir in result


class TestBuiltinFileToolGlob:
    def test_glob_py_files(self, file_tool, tmp_dir):
        Path(tmp_dir, "a.py").write_text("")
        Path(tmp_dir, "b.py").write_text("")
        Path(tmp_dir, "c.txt").write_text("")
        result = asyncio.run(file_tool.glob("*.py", tmp_dir))
        files = json.loads(result)
        assert len(files) == 2
        assert all(f.endswith(".py") for f in files)

    def test_glob_manages_own_timeout(self, file_tool):
        from agentica.tools.builtin.file_tool import _GLOB_TIMEOUT
        fn = file_tool.functions["glob"]
        assert fn.manages_own_timeout is True
        assert _GLOB_TIMEOUT == 20

    def test_glob_default_timeout_still_bounds(self, tmp_dir):
        """When the walk hangs, the hardcoded default still bounds it."""
        import time as _time
        tool = BuiltinFileTool(work_dir=tmp_dir)

        def slow_glob(self, pattern):
            _time.sleep(0.2)
            return iter(())

        with patch("agentica.tools.builtin.file_tool._GLOB_TIMEOUT", 0.05), \
             patch.object(Path, "glob", slow_glob):
            with pytest.raises(TimeoutError, match=r"Narrow `path`"):
                asyncio.run(tool.glob("**/*", str(tmp_dir)))

    def test_glob_recursive(self, file_tool, tmp_dir):
        sub = Path(tmp_dir, "sub")
        sub.mkdir()
        Path(sub, "deep.py").write_text("")
        Path(tmp_dir, "top.py").write_text("")
        result = asyncio.run(file_tool.glob("**/*.py", tmp_dir))
        files = json.loads(result)
        assert len(files) == 2

    def test_glob_no_matches(self, file_tool, tmp_dir):
        result = asyncio.run(file_tool.glob("*.xyz", tmp_dir))
        files = json.loads(result)
        assert files == []

    def test_glob_nonexistent_dir(self, file_tool):
        with pytest.raises(FileNotFoundError):
            asyncio.run(file_tool.glob("*", "/nonexistent_dir_xyz"))


class TestBuiltinFileToolGrep:
    def test_grep_schema_omits_timeout_and_multiline(self):
        import inspect
        from agentica.tools.builtin.execute_tool import BuiltinExecuteTool

        grep_params = inspect.signature(BuiltinFileTool.grep).parameters
        glob_params = inspect.signature(BuiltinFileTool.glob).parameters
        wait_params = inspect.signature(BuiltinExecuteTool.wait).parameters
        execute_params = inspect.signature(BuiltinExecuteTool.execute).parameters
        assert "timeout" not in grep_params
        assert "multiline" not in grep_params
        assert "include" not in grep_params
        assert "timeout" not in glob_params
        assert "timeout" in wait_params
        assert "timeout" in execute_params

    def test_grep_default_returns_content_with_line_numbers(self, file_tool, tmp_dir):
        Path(tmp_dir, "a.txt").write_text("hello world\n")
        Path(tmp_dir, "b.txt").write_text("goodbye world\n")
        Path(tmp_dir, "c.txt").write_text("nothing here\n")
        result = asyncio.run(file_tool.grep("hello", tmp_dir))
        assert "a.txt" in result
        assert "hello world" in result, "default mode must include matched line content"
        assert "c.txt" not in result

    def test_grep_long_line_returns_bounded_context_with_position(self, file_tool, tmp_dir):
        long_line = "A" * 5000 + '"timeout": 30' + "B" * 5000
        Path(tmp_dir, "data.jsonl").write_text(long_line + "\n")
        result = asyncio.run(file_tool.grep(r'"timeout": 30', tmp_dir))
        assert len(result) < 1000
        assert '"timeout": 30' in result
        assert "line_len=10013" in result
        assert "col=5001" in result

    def test_grep_fallback_long_line_matches_rg_bounded_format(self, file_tool, tmp_dir):
        long_line = "A" * 5000 + '"timeout": 30' + "B" * 5000
        Path(tmp_dir, "data.jsonl").write_text(long_line + "\n")
        with patch("agentica.tools.builtin.file_tool.shutil.which", return_value=None):
            result = asyncio.run(file_tool.grep(r'"timeout": 30', tmp_dir))
        assert len(result) < 1000
        assert '"timeout": 30' in result
        assert "line_len=10013" in result
        assert "col=5001" in result

    def test_format_rg_output_does_not_treat_time_as_line_number(self):
        from agentica.tools.builtin.file_tool import _format_rg_output

        content = "A" * 3000 + '{"timeout": 30, "ts": "12:30:45"}' + "B" * 3000
        raw = f"/tmp/data.jsonl:1:{content}"
        out = _format_rg_output(raw, r'"timeout": 30')
        assert out.startswith("/tmp/data.jsonl:1: col=")
        assert '"timeout": 30' in out
        assert f"line_len={len(content)}" in out
        assert "12:30: col=" not in out

    def test_format_rg_output_uses_column_when_rg_provides_it(self):
        from agentica.tools.builtin.file_tool import _format_rg_output

        content = "A" * 3000 + '{"timeout": 30, "ts": "12:30:45"}' + "B" * 3000
        raw = f"/tmp/data.jsonl:1:3001:{content}"
        out = _format_rg_output(raw, r'"timeout": 30', with_column=True)
        assert out.startswith("/tmp/data.jsonl:1: col=3001")
        assert '"timeout": 30' in out
        assert f"line_len={len(content)}" in out

    def test_grep_long_line_with_timestamp_keeps_match_window(self, file_tool, tmp_dir):
        payload = '{"timeout": 30, "ts": "12:30:45"}'
        long_line = "A" * 3000 + payload + "B" * 3000
        Path(tmp_dir, "data.jsonl").write_text(long_line + "\n")
        result = asyncio.run(file_tool.grep(r'"timeout": 30', tmp_dir))
        assert "data.jsonl:1: col=" in result
        assert '"timeout": 30' in result
        assert f"line_len={len(long_line)}" in result
        assert "12:30: col=" not in result

    def test_grep_short_line_stays_file_line_content(self, file_tool, tmp_dir):
        Path(tmp_dir, "log.txt").write_text("hello 12:30:45 timeout\n")
        result = asyncio.run(file_tool.grep("hello", tmp_dir))
        assert "col=" not in result
        assert "line_len=" not in result
        assert "hello 12:30:45 timeout" in result

    def test_grep_content_mode(self, file_tool, tmp_dir):
        Path(tmp_dir, "code.py").write_text("def foo():\n    pass\ndef bar():\n    pass\n")
        result = asyncio.run(file_tool.grep("def", tmp_dir))
        assert "def foo" in result
        assert "def bar" in result

    def test_grep_content_limit_is_global_across_files(self, file_tool, tmp_dir):
        """limit is a total line cap, not per-file --max-count."""
        for name in ("a.py", "b.py", "c.py"):
            Path(tmp_dir, name).write_text("HIT one\nHIT two\nHIT three\n")
        result = asyncio.run(
            file_tool.grep("HIT", tmp_dir, limit=4)
        )
        hits = [ln for ln in result.splitlines() if "HIT" in ln]
        assert len(hits) == 4

    def test_grep_content_limit_caps_within_one_file(self, file_tool, tmp_dir):
        """The global cap binds within a single file too — exactly `limit`
        lines, not limit-per-file."""
        Path(tmp_dir, "many.py").write_text("HIT\n" * 10)
        result = asyncio.run(
            file_tool.grep("HIT", tmp_dir, limit=3)
        )
        hits = [ln for ln in result.splitlines() if "HIT" in ln]
        assert len(hits) == 3

    def test_collect_rg_output_kills_process_at_limit(self):
        """limit is an I/O bound: stop reading and reap rg, don't buffer the rest."""
        from agentica.tools.builtin.file_tool import _collect_rg_output

        class _Stdout:
            def __init__(self, lines):
                self._lines = list(lines)
                self.reads = 0

            async def readline(self):
                self.reads += 1
                if not self._lines:
                    return b""
                return self._lines.pop(0)

        class _Proc:
            def __init__(self):
                self.stdout = _Stdout([b"a\n", b"b\n", b"c\n", b"d\n", b"e\n"])
                self.stderr = None
                self.returncode = None
                self.waited = False

            async def wait(self):
                self.waited = True
                self.returncode = 0

        proc = _Proc()
        killed = []

        async def fake_term(p, **kwargs):
            killed.append(p)
            p.returncode = -9

        async def run():
            with patch(
                "agentica.tools.builtin.file_tool.terminate_subprocess",
                fake_term,
            ):
                return await _collect_rg_output(proc, 2)

        stdout, stderr, hit_cap = asyncio.run(run())
        assert hit_cap is True
        assert stdout == b"a\nb\n"
        assert stderr == b""
        assert killed == [proc]
        # 2 payload lines + 1 lookahead line that proves the tree kept going
        assert proc.stdout.reads == 3
        assert proc.waited is False

    def test_collect_rg_output_exact_boundary_no_false_cap(self):
        """A tree with exactly `limit` matches must not report truncation:
        EOF arrives before the lookahead line, so hit_cap stays False and the
        process is waited, not killed."""
        from agentica.tools.builtin.file_tool import _collect_rg_output

        class _Stdout:
            def __init__(self, lines):
                self._lines = list(lines)

            async def readline(self):
                if not self._lines:
                    return b""
                return self._lines.pop(0)

        class _Proc:
            class _Stderr:
                async def read(self):
                    return b""

            def __init__(self):
                self.stdout = _Stdout([b"a\n", b"b\n"])
                self.stderr = self._Stderr()
                self.returncode = None
                self.waited = False

            async def wait(self):
                self.waited = True
                self.returncode = 0

        proc = _Proc()

        async def run():
            return await _collect_rg_output(proc, 2)

        stdout, _stderr, hit_cap = asyncio.run(run())
        assert hit_cap is False
        assert stdout == b"a\nb\n"
        assert proc.waited is True
        assert proc.returncode == 0

    def test_grep_timeout_does_not_forbid_execute(self, file_tool):
        doc = BuiltinFileTool.grep.__doc__ or ""
        assert "switch to execute" not in doc

    def test_grep_no_matches(self, file_tool, tmp_dir):
        Path(tmp_dir, "empty.txt").write_text("nothing\n")
        result = asyncio.run(file_tool.grep("zzzzz", tmp_dir))
        assert "No matches" in result

    def test_grep_nonexistent_path(self, file_tool):
        with pytest.raises(FileNotFoundError, match="Path not found"):
            asyncio.run(file_tool.grep("test", "/nonexistent_xyz"))

    def test_grep_accepts_file_path(self, file_tool, tmp_dir):
        fp = Path(tmp_dir, "single.py")
        fp.write_text("gate_passed = True\n")
        result = asyncio.run(file_tool.grep("gate_passed", str(fp)))
        assert "gate_passed = True" in result

    def test_grep_fallback_accepts_file_path(self, file_tool, tmp_dir):
        fp = Path(tmp_dir, "single.py")
        fp.write_text("commit_pass = True\n")
        with patch("agentica.tools.builtin.file_tool.shutil.which", return_value=None):
            result = asyncio.run(file_tool.grep("commit_pass", str(fp)))
        assert "commit_pass = True" in result

    def test_grep_manages_own_timeout(self, file_tool):
        """grep must self-limit so the outer 120s executor wrapper is skipped."""
        from agentica.tools.builtin.file_tool import _GREP_TIMEOUT
        fn = file_tool.functions["grep"]
        assert fn.manages_own_timeout is True
        assert _GREP_TIMEOUT == 20

    def test_grep_fallback_times_out(self, tmp_dir):
        """When rg is unavailable, the pure-Python fallback still hard-times-out
        instead of running up to the outer 120s executor limit."""
        import time as _time
        tool = BuiltinFileTool(work_dir=tmp_dir)

        # Slow sync fallback worker; the real _run_grep_fallback wraps it with
        # asyncio.wait_for, so the timeout fires well before this returns.
        def slow_worker(*args, **kwargs):
            _time.sleep(0.2)
            return "should not reach"
        tool._grep_fallback = slow_worker

        with patch("agentica.tools.builtin.file_tool.shutil.which", return_value=None), \
             patch("agentica.tools.builtin.file_tool._GREP_TIMEOUT", 0.05):
            with pytest.raises(TimeoutError, match=r"Narrow `path`"):
                asyncio.run(tool.grep("x", str(tmp_dir)))

    def test_grep_default_timeout_still_bounds(self, tmp_dir):
        """When no timeout arg is passed, the module default still bounds the
        search (a timeout must always be set — bad disk / regex hang)."""
        import time as _time
        tool = BuiltinFileTool(work_dir=tmp_dir)

        def slow_worker(*args, **kwargs):
            _time.sleep(0.2)
            return "should not reach"
        tool._grep_fallback = slow_worker

        with patch("agentica.tools.builtin.file_tool._GREP_TIMEOUT", 0.05), \
             patch("agentica.tools.builtin.file_tool.shutil.which", return_value=None):
            with pytest.raises(TimeoutError, match=r"grep timed out after 0.05 seconds"):
                asyncio.run(tool.grep("x", str(tmp_dir)))

    def test_grep_cancellation_cleans_up_subprocess(self, tmp_dir):
        async def cancel_running_grep():
            process = BlockingSubprocess()
            cleanup = AsyncMock()
            with patch(
                "agentica.tools.builtin.file_tool.asyncio.create_subprocess_exec",
                new=AsyncMock(return_value=process),
            ), patch(
                "agentica.tools.builtin.file_tool.terminate_subprocess",
                cleanup,
            ), patch(
                "agentica.tools.builtin.file_tool.shutil.which",
                return_value="/usr/bin/rg",
            ):
                tool = BuiltinFileTool(work_dir=tmp_dir)
                task = asyncio.create_task(tool.grep("needle", str(tmp_dir)))
                await process.started.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

            cleanup.assert_awaited_once_with(process)

        asyncio.run(cancel_running_grep())



class TestFileToolRegistrationGuard:
    """End-to-end guard: every BuiltinFileTool function must be visible in
    the final tool schema sent to the LLM.

    This test exists because a past bug placed self.register() calls outside
    __init__(), causing read_file / glob to silently disappear from the
    model's tool list while execute remained available — the model then fell
    back to shell commands.
    """

    EXPECTED_FUNCTIONS = {"read_file", "write_file",
                          "apply_patch", "glob", "grep"}

    def test_file_tool_functions_in_tool_dict(self):
        """Tool.functions dict must contain all file operations after init."""
        tool = BuiltinFileTool(work_dir="/tmp")
        registered = set(tool.functions.keys())
        missing = self.EXPECTED_FUNCTIONS - registered
        assert not missing, f"Functions missing from Tool.functions: {missing}"

    def test_file_tool_functions_reach_model_api_schema(self):
        """After Agent.update_model(), every file function must appear in
        Model.get_tools_for_api() — the payload actually sent to the LLM."""
        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat

        file_tool = BuiltinFileTool(work_dir="/tmp")
        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            tools=[file_tool],
        )
        agent.update_model()

        api_tools = agent.model.get_tools_for_api()
        api_names = {t["function"]["name"] for t in api_tools}
        missing = self.EXPECTED_FUNCTIONS - api_names
        assert not missing, (
            f"Functions missing from Model.get_tools_for_api(): {missing}. "
            f"Likely cause: self.register() not called in __init__."
        )
        by_name = {
            t["function"]["name"]: t["function"]["parameters"]["properties"]
            for t in api_tools
        }
        assert "timeout" not in by_name["grep"]
        assert "multiline" not in by_name["grep"]
        assert "timeout" not in by_name["glob"]

    def test_auto_mode_schema_does_not_include_request_path_access(self):
        from agentica.agent import Agent
        from agentica.agent.config import ToolConfig
        from agentica.model.openai import OpenAIChat

        file_tool = BuiltinFileTool(work_dir="/tmp")
        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            tools=[file_tool],
            tool_config=ToolConfig(permission_mode="auto"),
        )
        agent.update_model()
        api_names = {t["function"]["name"] for t in agent.model.get_tools_for_api()}
        assert "request_path_access" not in api_names
        assert "ls" not in api_names
        assert "edit_file" not in api_names
        assert "write_file" in api_names

