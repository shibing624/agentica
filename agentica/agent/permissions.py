# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Unified 3-tier tool permission model.

Shared by the SDK (Agent/DeepAgent), the CLI, and the Gateway so all three
surfaces expose the exact same vocabulary and behavior. The names match the
product labels (Read Only / Workspace Write / Full access). Tools stay in the
schema in every tier — hiding a tool is never how a tier works — and the only
thing a tier changes is whether the runner parks before ``fc.execute()`` (see
``agentica.agent.approvals``).

  - "read-only" : Read Only. Reads (including outside the work directory),
                  read-only ``execute``, memory, ``task``, skills, other
                  builtins, and network tools (``web_search`` / ``fetch_url``)
                  run without prompting. Parking: every ``write_file`` /
                  ``apply_patch``, non-read-only ``execute``, and hard-unsafe
                  commands/paths (``rm -rf /``, ``/etc``, ``~/.ssh``).
  - "auto"      : Workspace Write. Reads, in-workspace writes (the platform
                  temp directories count as inside), ordinary ``execute``,
                  network tools, and every builtin / skill / third-party tool
                  run without prompting. File writes outside the work
                  directory or to a sensitive path and hard-unsafe execute
                  would park; with the auto reviewer on (default) the agent's
                  own model decides first — allow runs the call unasked, and
                  deny / a failed review falls back to asking a human.
  - "allow-all" : Full access. Never parks and never denies, including
                  hard-unsafe commands/paths and project deny-similar grants
                  (those apply only in read-only/auto). Warns and records the
                  override. The process runs with the OS user's own
                  privileges — if that user can sudo/root, so can execute.

Network is not a permission tier: every tier lets the web tools run. A file
sandbox mode does not govern network in the reference model, and asking one
mode's approval card about a search was the mismatch this removes.

The OS sandbox (``SandboxConfig.os_sandbox``) is a second, independent axis:
it confines every ``execute`` in all three tiers. ``execute(escalate=True)``
— run outside it — always parks in read-only/auto, before any allow-similar
grant or the read-only fast path; allow-all runs it like everything else.

Callers should not construct their own mode strings — always compare against
``PERMISSION_MODES`` / use ``validate_permission_mode`` so a typo fails loud
instead of silently falling through to the most permissive behavior.
"""
# Ordered most- to least-restrictive. ``cronjob`` relies on this order for its
# "a job may only be stricter than the session that created it" check.
PERMISSION_MODES = ("read-only", "auto", "allow-all")

# Where ``execute`` commands run, independent of the permission tier:
#   "off"             : on the host, as the OS user (default for SDK and CLI).
#   "workspace-write" : inside the OS sandbox (``agentica.tools.os_sandbox``) —
#                       writes only in the work dir / its worktrees / temp /
#                       caches, credential paths unreadable, network off.
SANDBOX_MODES = ("off", "workspace-write")


def validate_permission_mode(mode: str) -> None:
    """Raise ValueError if `mode` is not one of PERMISSION_MODES.

    Raises on the retired ``"ask"`` value too — the tier was renamed to
    ``"read-only"`` and a silent fallback would change what a saved preference
    means.
    """
    if mode not in PERMISSION_MODES:
        raise ValueError(f"Invalid permission mode: {mode!r}. Must be one of {PERMISSION_MODES}.")


def validate_sandbox_mode(mode: str) -> None:
    """Raise ValueError if `mode` is not one of SANDBOX_MODES.

    Loud on purpose: a caller who asked for a sandbox and silently got the
    host is the failure that matters.
    """
    if mode not in SANDBOX_MODES:
        raise ValueError(f"Invalid sandbox mode: {mode!r}. Must be one of {SANDBOX_MODES}.")


def sandbox_should_be_enabled(mode: str) -> bool:
    """Whether file writes should be path-restricted for `mode`."""
    return mode != "allow-all"
