# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Where the gateway finds the built Web UI.

The UI is built by ``web/`` (``npm run build`` → ``web/dist``) and never
written into this package's source tree. Three places can hold it, first hit
wins:

1. ``$AGENTICA_WEB_DIST`` — an explicit dist directory (debugging a build,
   or a host that ships the UI elsewhere).
2. ``agentica_gateway/ui`` inside the installed package — the wheel's copy,
   put there at build time by ``gateway/hatch_build.py``. This is what every
   ``pip install agentica-gateway`` user gets.
3. ``<repo>/web/dist`` — a source checkout. An editable install has no copy
   in the package, so without this a developer would get the 503 page after
   every ``npm run build``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

ENV_VAR = "AGENTICA_WEB_DIST"
_PACKAGE_DIR = Path(__file__).resolve().parent


def candidate_dirs() -> Iterable[Path]:
    env = os.environ.get(ENV_VAR, "").strip()
    if env:
        yield Path(env).expanduser()
    yield _PACKAGE_DIR / "ui"
    # gateway/src/agentica_gateway/ → repo root is three levels up.
    yield _PACKAGE_DIR.parents[2] / "web" / "dist"


def resolve_ui_dir() -> Optional[Path]:
    """The first candidate holding an ``index.html``, else None."""
    for d in candidate_dirs():
        if (d / "index.html").is_file():
            return d
    return None
