# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Rich rendering for `agentica doctor`.

Thin presentation layer over agentica.diagnostics.run_doctor(); the SDK function
holds all the logic so it stays usable headless.
"""
from agentica.diagnostics import run_doctor, OK, WARN, FAIL

_ICON = {OK: "[green]\u2713[/green]", WARN: "[yellow]\u26a0[/yellow]", FAIL: "[red]\u2717[/red]"}


def show_doctor(console, **doctor_kwargs) -> bool:
    """Render the doctor report. Returns True if no check failed."""
    report = run_doctor(**doctor_kwargs)
    console.print()
    console.print("  [bold]Agentica doctor[/bold] — environment health check")
    console.print()
    for c in report.checks:
        icon = _ICON.get(c.status, "?")
        detail = f"  [dim]{c.detail}[/dim]" if c.detail else ""
        console.print(f"  {icon} {c.name}{detail}")
    console.print()
    counts = report.counts()
    style = "green" if report.ok else "red"
    console.print(f"  [{style}]{report.summary()}[/{style}]")
    if counts[FAIL]:
        console.print("  [dim]Fix the \u2717 items above, then re-run `agentica doctor`.[/dim]")
    console.print()
    return report.ok
