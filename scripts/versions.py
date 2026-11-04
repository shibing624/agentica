#!/usr/bin/env python3
"""One version for every artifact this repo releases.

agentica and agentica-gateway pin each other exactly (the gateway uses agentica
internals), and Desktop / the TS SDK are released from the same tag, so they
all carry one version. ``agentica/version.py`` is the source of truth.

    python scripts/versions.py check        # CI: fail if anything drifted
    python scripts/versions.py set 1.5.1    # bump everything, then commit
    python scripts/versions.py check --tag v1.5.1   # release: tag must match

Edits are textual and anchored, so comments and formatting survive.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SEMVER = r"\d+\.\d+\.\d+(?:[-.]?(?:a|b|rc|dev|post)\.?\d+)?"


def _sub(pattern: str, repl_fmt: str, *, count: int):
    """A (read, write) pair for every match of ``pattern`` in one file.

    ``pattern`` has one group, the version; ``repl_fmt`` gets ``{v}``.
    ``count`` is how many matches the file must contain, so a refactor that
    drops or duplicates a pin fails loudly instead of being skipped.
    """
    rx = re.compile(pattern, re.M)

    def read(text: str) -> list[str]:
        found = [m.group(1) for m in rx.finditer(text)]
        if len(found) != count:
            raise SystemExit(f"expected {count} match(es) of {pattern!r}, found {len(found)}")
        return found

    def write(text: str, v: str) -> str:
        read(text)
        return rx.sub(lambda m: m.group(0).replace(m.group(1), v), text)

    return read, write


def _json_version(path: Path):
    def read(text: str) -> list[str]:
        return [json.loads(text)["version"]]

    def write(text: str, v: str) -> str:
        # Only the top-level "version" key, which npm always writes first-level
        # with two-space indent; avoids reformatting the whole file.
        new, n = re.subn(r'^(  "version": ")[^"]+(")', rf"\g<1>{v}\g<2>", text, count=1, flags=re.M)
        if n != 1:
            raise SystemExit(f"{path}: no top-level version key")
        return new

    return read, write


def _lock_version(path: Path):
    """package-lock.json: the root ``version`` and ``packages[""].version``."""

    def read(text: str) -> list[str]:
        data = json.loads(text)
        return [data["version"], data["packages"][""]["version"]]

    def write(text: str, v: str) -> str:
        new, n = re.subn(r'^(  "version": ")[^"]+(")', rf"\g<1>{v}\g<2>", text, count=1, flags=re.M)
        new, m = re.subn(r'^(      "version": ")[^"]+(")', rf"\g<1>{v}\g<2>", new, count=1, flags=re.M)
        if n != 1 or m != 1:
            raise SystemExit(f"{path}: version keys not where npm puts them")
        return new

    return read, write


def targets() -> dict[Path, tuple]:
    t = {
        ROOT / "agentica/version.py": _sub(rf'^__version__ = "({_SEMVER})"', "", count=1),
        ROOT / "pyproject.toml": _sub(rf'^version = "({_SEMVER})"', "", count=1),
        ROOT / "gateway/pyproject.toml": _sub(rf'^version = "({_SEMVER})"', "", count=1),
        # gateway → agentica, exact.
        ROOT / "gateway/pyproject.toml#pin": _sub(rf'"agentica==({_SEMVER})"', "", count=1),
        # agentica[gateway|telegram|...] → agentica-gateway, exact (8 extras).
        ROOT / "pyproject.toml#extras": _sub(rf'"agentica-gateway\[[a-z]+\]==({_SEMVER})"', "", count=8),
    }
    for pkg in ("desktop", "sdk-ts"):
        t[ROOT / pkg / "package.json"] = _json_version(ROOT / pkg / "package.json")
        lock = ROOT / pkg / "package-lock.json"
        if lock.is_file():
            t[lock] = _lock_version(lock)
    return t


def _file(key: Path) -> Path:
    return Path(str(key).split("#", 1)[0])


def current() -> dict[str, list[str]]:
    out = {}
    for key, (read, _) in targets().items():
        out[str(key.relative_to(ROOT))] = read(_file(key).read_text(encoding="utf-8"))
    return out


def cmd_check(tag: str | None) -> int:
    found = current()
    truth = found["agentica/version.py"][0]
    want = truth
    errors = []
    if tag is not None:
        tv = tag[1:] if tag.startswith("v") else tag
        if tv != truth:
            errors.append(f"tag {tag} != agentica/version.py {truth}")
    for where, versions in found.items():
        for v in versions:
            if v != want:
                errors.append(f"{where}: {v} (want {want})")
    if errors:
        print("version drift:", file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
        print(f"fix: python scripts/versions.py set {want}", file=sys.stderr)
        return 1
    print(f"ok: {truth} in {sum(len(v) for v in found.values())} places")
    return 0


def cmd_set(v: str) -> int:
    if not re.fullmatch(_SEMVER, v):
        raise SystemExit(f"not a version: {v}")
    by_file: dict[Path, list] = {}
    for key, (_, write) in targets().items():
        by_file.setdefault(_file(key), []).append(write)
    for path, writers in by_file.items():
        text = path.read_text(encoding="utf-8")
        for w in writers:
            text = w(text, v)
        path.write_text(text, encoding="utf-8")
        print(f"  {path.relative_to(ROOT)}")
    return cmd_check(None)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--tag")
    s = sub.add_parser("set")
    s.add_argument("version")
    a = p.parse_args(argv)
    return cmd_check(a.tag) if a.cmd == "check" else cmd_set(a.version)


if __name__ == "__main__":
    sys.exit(main())
