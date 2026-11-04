# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: The ``agentica-gateway`` command as shipped by the ``agentica``
distribution.

The gateway lives in its own distribution (``agentica-gateway``) since 1.5.1,
which declares the same command. This copy exists for uv: ``uv tool install
"agentica[gateway]"`` — the documented install, and what ``uv tool upgrade
agentica`` re-runs for everyone who used it — only exposes executables of the
package named on the command line, so without it the command would vanish on
upgrade. It hands straight to the real entry point, or says what to install.
"""
from __future__ import annotations

import sys

_HINT = (
    "agentica-gateway is not installed. It ships separately since agentica 1.5.1:\n\n"
    "    uv tool install --force \"agentica[gateway]\"   # CLI + Web UI, both commands\n"
    "    pip install agentica-gateway                  # or into the current environment\n"
)


def main() -> int:
    try:
        from agentica_gateway.main import main as gateway_main
    except ImportError as e:
        # Only "the package is absent" earns the install hint; a broken import
        # inside an installed gateway must surface as itself.
        if getattr(e, "name", None) not in ("agentica_gateway", "agentica_gateway.main", "fastapi"):
            raise
        sys.stderr.write(_HINT)
        return 1
    return gateway_main()


if __name__ == "__main__":
    sys.exit(main())
