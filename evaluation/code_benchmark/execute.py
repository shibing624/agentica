# -*- coding: utf-8 -*-
"""Subprocess-based local judges. Not a sandbox — do not feed untrusted code."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


@dataclass
class ExecResult:
    ok: bool
    returncode: int
    output: str
    timed_out: bool = False
    stderr: str = ""


def _decode(blob) -> str:
    if blob is None:
        return ""
    if isinstance(blob, bytes):
        return blob.decode("utf-8", errors="replace")
    return str(blob)


def _clip(text: str, max_output: int) -> str:
    if max_output and len(text) > max_output:
        return text[-max_output:]
    return text


def run_command(
    command: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    timeout: int = 60,
    stdin: Optional[str] = None,
    env: Optional[dict] = None,
    combine_output: bool = True,
    max_output: int = 8000,
) -> ExecResult:
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd) if cwd else None,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _decode(exc.stdout)
        stderr = _decode(exc.stderr)
        blob = (stdout + stderr) if combine_output else stdout
        return ExecResult(ok=False, returncode=-1, output=_clip(blob, max_output), timed_out=True, stderr=_clip(stderr, max_output))
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    blob = (stdout + stderr) if combine_output else stdout
    return ExecResult(
        ok=completed.returncode == 0,
        returncode=completed.returncode,
        output=_clip(blob, max_output),
        stderr=_clip(stderr, max_output),
    )


def run_python_source(
    source: str,
    *,
    timeout: int = 20,
    stdin: Optional[str] = None,
    extra_files: Optional[dict] = None,
) -> ExecResult:
    with tempfile.TemporaryDirectory(prefix="codebench-") as tmp:
        root = Path(tmp)
        script = root / "main.py"
        script.write_text(source, encoding="utf-8")
        if extra_files:
            for name, content in extra_files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
        return run_command(
            [sys.executable, str(script)],
            cwd=root,
            timeout=timeout,
            stdin=stdin,
        )


def run_pytest(cwd: Path, test_files: List[str], timeout: int = 180) -> ExecResult:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        f"--rootdir={cwd}",
        "--noconftest",
    ]
    command.extend(test_files)
    return run_command(command, cwd=cwd, timeout=timeout)


def judge_check_function(
    code: str,
    test: str,
    entry_point: str,
    *,
    timeout: int = 20,
) -> ExecResult:
    """HumanEval-style: exec code + test, then call check(entry_point)."""
    source = (
        code.rstrip()
        + "\n\n"
        + test.rstrip()
        + f"\n\ncheck({entry_point})\nprint('PASSED')\n"
    )
    return run_python_source(source, timeout=timeout)


def judge_plus_inputs(
    prompt: str,
    completion: str,
    canonical: str,
    entry_point: str,
    plus_inputs,
    *,
    timeout: int = 20,
) -> ExecResult:
    """Compare completion vs canonical on EvalPlus extra inputs."""
    payload = json.dumps(plus_inputs)
    source = (
        "import json\n"
        f"prompt = {prompt!r}\n"
        f"completion = {completion!r}\n"
        f"canonical = {canonical!r}\n"
        f"entry_point = {entry_point!r}\n"
        f"plus_inputs = json.loads({payload!r})\n"
        "ns_gt = {}\n"
        "ns_pred = {}\n"
        "exec(prompt + canonical, ns_gt)\n"
        "exec(prompt + completion, ns_pred)\n"
        "fn_gt = ns_gt[entry_point]\n"
        "fn_pred = ns_pred[entry_point]\n"
        "for i, args in enumerate(plus_inputs):\n"
        "    if not isinstance(args, (list, tuple)):\n"
        "        args = [args]\n"
        "    got = fn_pred(*args)\n"
        "    expected = fn_gt(*args)\n"
        "    if got != expected:\n"
        "        raise AssertionError(f'plus case {i}: {got!r} != {expected!r}')\n"
        "print('PASSED')\n"
    )
    return run_python_source(source, timeout=timeout)


def judge_unittest_module(solution: str, test: str, *, timeout: int = 30) -> ExecResult:
    """BigCodeBench-style: solution module + unittest.TestCase in test."""
    source = (
        "import unittest\n"
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parent))\n"
        "import solution  # noqa: F401\n"
        "import testmod\n"
        "suite = unittest.defaultTestLoader.loadTestsFromModule(testmod)\n"
        "result = unittest.TextTestRunner(verbosity=0).run(suite)\n"
        "if result.failures or result.errors:\n"
        "    sys.exit(1)\n"
        "print('PASSED')\n"
    )
    return run_python_source(
        source,
        timeout=timeout,
        extra_files={"solution.py": solution, "testmod.py": test},
    )
