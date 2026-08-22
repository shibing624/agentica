#!/usr/bin/env python3
"""Fail if a test imports optional extras at module top without importorskip.

GitHub Actions is a bare ``pip install -e .``. A collection-time ImportError
aborts the whole pytest run (exit 2). This has bitten gateway tests more than
once: ``from agentica.gateway...`` loads ``gateway/__init__.py``, which
imports fastapi.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"

# Imported-module prefix → package that importorskip must name first.
EXTRA_IMPORTS = {
    "agentica.gateway": "fastapi",
    "agentica.acp": "websockets",
}


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _call_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    return ""


def _importorskip_pkg(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return None
    if _call_name(node.value.func) not in {"importorskip", "pytest.importorskip"}:
        return None
    if not node.value.args:
        return None
    arg0 = node.value.args[0]
    if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
        return arg0.value.split(".", 1)[0]
    return None


def _imported_module(node: ast.AST) -> str | None:
    if isinstance(node, ast.ImportFrom) and node.module:
        return node.module
    if isinstance(node, ast.Import):
        return node.names[0].name if node.names else None
    return None


def check_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    skipped: set[str] = set()
    errors: list[str] = []
    for node in tree.body:
        pkg = _importorskip_pkg(node)
        if pkg:
            skipped.add(pkg)
            continue
        mod = _imported_module(node)
        if not mod:
            continue
        for prefix, extra in EXTRA_IMPORTS.items():
            if mod == prefix or mod.startswith(prefix + "."):
                if extra not in skipped:
                    loc = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
                    errors.append(
                        f"{loc}:{node.lineno}: "
                        f"module-level import of {mod} needs "
                        f"pytest.importorskip({extra!r}) first"
                    )
    return errors


def main() -> int:
    errors: list[str] = []
    for path in sorted(TESTS.rglob("*.py")):
        errors.extend(check_file(path))
    if errors:
        print("Bare CI would fail collection on these tests:\n", file=sys.stderr)
        for line in errors:
            print(line, file=sys.stderr)
        return 1
    print(f"ok: {sum(1 for _ in TESTS.rglob('*.py'))} test files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
