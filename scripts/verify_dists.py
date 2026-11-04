#!/usr/bin/env python3
"""Check the built agentica + agentica-gateway distributions before they ship.

    python scripts/verify_dists.py dist/

Asserts what users would otherwise find out after a release:
* exactly one wheel and one sdist of each, all at the repo version;
* the Web UI (index.html + KaTeX woff2 + favicon) is in the gateway wheel AND
  its sdist (so a wheel built from the sdist, without Node, still has it), and
  in neither agentica artifact;
* the gateway pins ``agentica==<version>``, and ``agentica[gateway]`` pins
  ``agentica-gateway[feishu]==<version>``;
* the console scripts land in the right distribution.
"""
from __future__ import annotations

import re
import sys
import tarfile
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def _version() -> str:
    text = (ROOT / "agentica" / "version.py").read_text(encoding="utf-8")
    return re.search(r'__version__ = "([^"]+)"', text).group(1)


def _one(dist: Path, pattern: str) -> Path:
    found = sorted(dist.glob(pattern))
    if len(found) != 1:
        raise SystemExit(f"want exactly one {pattern} in {dist}, got {[p.name for p in found]}")
    return found[0]


def _wheel_meta(whl: Path, name: str) -> str:
    with ZipFile(whl) as z:
        return z.read(next(n for n in z.namelist() if n.endswith(f".dist-info/{name}"))).decode()


def main(argv: list[str]) -> int:
    dist = Path(argv[1] if len(argv) > 1 else "dist")
    v = _version()
    core_whl = _one(dist, f"agentica-{v}-*.whl")
    core_sd = _one(dist, f"agentica-{v}.tar.gz")
    gw_whl = _one(dist, f"agentica_gateway-{v}-*.whl")
    gw_sd = _one(dist, f"agentica_gateway-{v}.tar.gz")
    errors: list[str] = []

    gw_names = ZipFile(gw_whl).namelist()
    for need in ("agentica_gateway/ui/index.html", "agentica_gateway/ui/favicon.png"):
        if need not in gw_names:
            errors.append(f"{gw_whl.name}: missing {need}")
    if not any(n.startswith("agentica_gateway/ui/assets/") and n.endswith(".woff2") for n in gw_names):
        errors.append(f"{gw_whl.name}: no KaTeX .woff2 under ui/assets (the SPA's math fonts would 404)")
    if not any(n.endswith("/src/agentica_gateway/ui/index.html") for n in tarfile.open(gw_sd).getnames()):
        errors.append(f"{gw_sd.name}: no src/agentica_gateway/ui/index.html (a wheel built from it would lack the UI)")

    core_names = ZipFile(core_whl).namelist()
    if any("/ui/assets/" in n for n in core_names):
        errors.append(f"{core_whl.name}: carries Web UI files; they belong to agentica-gateway")
    if any(n.startswith("agentica_gateway/") for n in core_names):
        errors.append(f"{core_whl.name}: contains agentica_gateway")
    if any("/ui/assets/" in n for n in tarfile.open(core_sd).getnames()):
        errors.append(f"{core_sd.name}: carries Web UI files")

    gw_meta = _wheel_meta(gw_whl, "METADATA")
    if f"Requires-Dist: agentica=={v}" not in gw_meta:
        errors.append(f"{gw_whl.name}: does not pin agentica=={v}")
    core_meta = _wheel_meta(core_whl, "METADATA")
    if f'Requires-Dist: agentica-gateway[feishu]=={v}; extra == "gateway"' not in core_meta:
        errors.append(f"{core_whl.name}: agentica[gateway] does not pin agentica-gateway[feishu]=={v}")

    core_ep = _wheel_meta(core_whl, "entry_points.txt")
    gw_ep = _wheel_meta(gw_whl, "entry_points.txt")
    # The core keeps a forwarding copy of the command for uv tool installs.
    if "agentica-gateway = agentica.cli.gateway_command:main" not in core_ep:
        errors.append(f"{core_whl.name}: agentica-gateway must forward via agentica.cli.gateway_command")
    if "agentica-gateway = agentica_gateway.main:main" not in gw_ep:
        errors.append(f"{gw_whl.name}: missing the agentica-gateway command")
    if "agentica = agentica.cli.main:main" not in core_ep:
        errors.append(f"{core_whl.name}: missing the agentica command")

    if errors:
        print("distribution check failed:", file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
        return 1
    ui = sum(n.startswith("agentica_gateway/ui/") for n in gw_names)
    print(f"ok: {v}: {core_whl.name}, {gw_whl.name} ({ui} UI files), and both sdists")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
