# -*- coding: utf-8 -*-
"""The package's public surface, and the CLI's startup import.

The CLI installs the egress from ``cli/interactive/app.py`` as
``from agentica.shell_hooks import install_hook_egress``. That import is the only
thing standing between a configured hook and it working at startup, and nothing
else in the suite exercises it, so it is pinned here.
"""

from __future__ import annotations


def test_the_cli_startup_import_resolves():
    from agentica.shell_hooks import install_hook_egress

    assert callable(install_hook_egress)


def test_importing_the_package_starts_nothing():
    """Import is not installation: no thread, no process, no half-wired channel."""
    import agentica.shell_hooks as pkg

    assert pkg.get_hook_egress() is None


def test_the_public_names_are_importable():
    from agentica.shell_hooks import (
        HookRequest,
        ShellHooksConfig,
        approval_payload,
        get_hook_egress,
        hook_egress_dispatch,
        install_hook_egress,
        load_shell_hooks_config,
        question_payload,
        reset_hook_egress_for_tests,
        start_hook_request,
    )

    for obj in (
        HookRequest,
        ShellHooksConfig,
        approval_payload,
        get_hook_egress,
        hook_egress_dispatch,
        install_hook_egress,
        load_shell_hooks_config,
        question_payload,
        reset_hook_egress_for_tests,
        start_hook_request,
    ):
        assert obj is not None
