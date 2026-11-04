"""Put the built Web UI into the agentica-gateway wheel and sdist.

The UI is built by ``web/`` into ``web/dist`` and never written into this
package's source tree. This hook copies it in at build time:

* wheel  → ``agentica_gateway/ui``      (what the gateway serves)
* sdist  → ``src/agentica_gateway/ui``  (so a wheel built from the sdist, e.g.
  ``pip install agentica-gateway`` on a platform without our wheel, still has
  it — without Node)

Source lookup: ``$AGENTICA_WEB_DIST``, then ``../web/dist`` (repo checkout),
then ``src/agentica_gateway/ui`` (we are building from an unpacked sdist).

A release must never ship a gateway without its UI, so a missing dist fails
the build. Editable installs skip the copy: at runtime the gateway finds
``web/dist`` in the checkout itself (``agentica_gateway/ui_dist.py``).
Set ``AGENTICA_GATEWAY_ALLOW_NO_UI=1`` to build a UI-less wheel on purpose
(the gateway then answers /chat with a 503 explaining how to build).
"""
from __future__ import annotations

import os
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_DEST = {"wheel": "agentica_gateway/ui", "sdist": "src/agentica_gateway/ui"}


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def _from_sdist(self) -> bool:
        # An unpacked sdist has PKG-INFO; a git checkout never does.
        return (Path(self.root) / "PKG-INFO").is_file()

    def _find_dist(self) -> Path | None:
        root = Path(self.root)
        env = os.environ.get("AGENTICA_WEB_DIST", "").strip()
        candidates = [Path(env).expanduser()] if env else []
        candidates.append(root.parent / "web" / "dist")
        if self._from_sdist():
            candidates.append(root / "src" / "agentica_gateway" / "ui")
        for c in candidates:
            if (c / "index.html").is_file():
                return c.resolve()
        return None

    def initialize(self, version: str, build_data: dict) -> None:
        if self.target_name == "wheel" and version == "editable":
            return
        dist = self._find_dist()
        if dist is None:
            if os.environ.get("AGENTICA_GATEWAY_ALLOW_NO_UI") == "1":
                self.app.display_warning("agentica-gateway: building WITHOUT the Web UI")
                return
            raise RuntimeError(
                "agentica-gateway: Web UI not built (no web/dist/index.html). "
                "Run `cd web && npm ci && npm run build` first, point "
                "AGENTICA_WEB_DIST at a built dist, or set "
                "AGENTICA_GATEWAY_ALLOW_NO_UI=1 to build without it."
            )
        # Building from the sdist: the UI already sits inside the package tree,
        # and `packages = ["src/agentica_gateway"]` picks it up as-is.
        if dist == (Path(self.root) / "src" / "agentica_gateway" / "ui").resolve():
            return
        dest = _DEST[self.target_name]
        build_data.setdefault("force_include", {})[str(dist)] = dest
        self.app.display_info(f"agentica-gateway: Web UI {dist} -> {dest}")
