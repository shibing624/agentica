# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Edit-time LSP diagnostics for coding agents.

Wraps the LSP client in ``tools/lsp_tool.py`` to give the agent real semantic
feedback (type errors, undefined names, bad imports) right after it edits a
file — and crucially returns ONLY the diagnostics newly introduced by the edit,
so the model isn't flooded with the project's pre-existing issues.

This is an opt-in SDK primitive. It never starts automatically; the caller
constructs an ``LspDiagnosticsChecker`` and either uses it directly or passes
it to ``BuiltinFileTool(diagnostics_checker=...)``.

    checker = LspDiagnosticsChecker(work_dir=".", servers=["pyright"])
    checker.snapshot_before("app.py")            # cheap: cached after first call
    ... edit app.py ...
    print(checker.report_after("app.py"))        # "" if no new problems

The checker caches a per-file baseline, so editing the same file repeatedly
costs ONE diagnostics round-trip per edit (the post-edit fetch) instead of two.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from agentica.tools.lsp_tool import LspServerManager, DEFAULT_LSP_SERVERS
from agentica.utils.log import logger

_SEVERITY = {1: "error", 2: "warning", 3: "information", 4: "hint"}


@dataclass(frozen=True)
class Diagnostic:
    """A normalized LSP diagnostic."""
    file: str
    line: int          # 1-based for display
    character: int     # 1-based for display
    severity: str
    message: str
    source: Optional[str] = None
    code: Optional[str] = None

    @property
    def identity(self) -> tuple:
        """Stable key used to diff baseline vs. post-edit.

        Keyed on (severity, message, character) and deliberately ignores the
        line number, so inserting/deleting lines elsewhere does not make an
        unchanged pre-existing diagnostic look "new". Caveat: a re-indentation
        that shifts ``character`` can make an old diagnostic appear as new — an
        accepted false-positive in exchange for ignoring line drift.
        """
        return (self.severity, self.message, self.character)


def _parse(file_path: str, raw: dict) -> Diagnostic:
    rng = raw.get("range", {}).get("start", {})
    code = raw.get("code")
    return Diagnostic(
        file=file_path,
        line=int(rng.get("line", 0)) + 1,
        character=int(rng.get("character", 0)) + 1,
        severity=_SEVERITY.get(raw.get("severity", 1), "error"),
        message=str(raw.get("message", "")).strip(),
        source=raw.get("source"),
        code=str(code) if code is not None else None,
    )


def format_diagnostics(diags: List[Diagnostic], header: Optional[str] = None) -> str:
    """Render diagnostics as a compact, model-friendly block."""
    if not diags:
        return ""
    lines = [header] if header else []
    for d in diags:
        src = f" [{d.source}{':' + d.code if d.code else ''}]" if d.source else ""
        lines.append(f"  {d.severity} {Path(d.file).name}:{d.line}:{d.character} {d.message}{src}")
    return "\n".join(lines)


class LspDiagnosticsChecker:
    """Opt-in edit-time diagnostics over one or more language servers."""

    def __init__(
            self,
            work_dir: Optional[str] = None,
            servers: Optional[List[str]] = None,
            timeout: float = 3.0,
            errors_only: bool = False,
    ):
        """
        Args:
            work_dir: Workspace root for the language servers.
            servers: LSP server names (default ["pyright"]). Unavailable servers
                are skipped with a warning — the checker degrades to a no-op.
            timeout: Max seconds to wait for a publishDiagnostics batch. Kept low
                (3s) because this sits on the hot edit path; a slow/unresponsive
                server should degrade to "no feedback" rather than stall the edit.
            errors_only: When True, only severity=="error" diagnostics are returned.
        """
        self.workspace_path = Path(work_dir) if work_dir else Path.cwd()
        self.timeout = timeout
        self.errors_only = errors_only
        # Per-file baseline cache (resolved path -> last known diagnostics), so
        # repeated edits to a file don't each re-fetch a pre-edit baseline.
        self._baselines: Dict[str, List[Diagnostic]] = {}
        self.manager = LspServerManager(self.workspace_path)
        for name in (servers or ["pyright"]):
            if name in DEFAULT_LSP_SERVERS:
                try:
                    self.manager.register_server(DEFAULT_LSP_SERVERS[name])
                except RuntimeError as e:
                    logger.warning(f"LSP diagnostics: server '{name}' unavailable: {e}")

    def available(self) -> bool:
        """True if any language server actually started."""
        return self.manager.has_any()

    def has_client(self, file_path: str) -> bool:
        """True if a server can handle this file's language."""
        return self.manager.get_client(file_path) is not None

    def diagnostics(self, file_path: str) -> List[Diagnostic]:
        """Current diagnostics for a file (empty if no server handles it)."""
        client = self.manager.get_client(file_path)
        if client is None:
            return []
        path = Path(file_path).resolve()
        if not path.exists():
            return []
        try:
            raw = client.get_diagnostics(path, timeout=self.timeout)
        except Exception as e:
            logger.warning(f"LSP diagnostics failed for {file_path}: {e}")
            return []
        diags = [_parse(str(path), d) for d in raw]
        if self.errors_only:
            diags = [d for d in diags if d.severity == "error"]
        return diags

    def snapshot_before(self, file_path: str) -> None:
        """Ensure a pre-edit baseline exists for ``file_path``.

        Only fetches diagnostics on the FIRST call for a file (cache miss);
        later edits reuse the baseline refreshed by ``report_after``, so the
        steady-state cost is one round-trip per edit, not two. No-op when no
        server handles the file.
        """
        if not self.has_client(file_path):
            return
        key = str(Path(file_path).resolve())
        if key not in self._baselines:
            self._baselines[key] = self.diagnostics(file_path)

    def report_after(self, file_path: str) -> str:
        """Formatted diagnostics introduced since the cached baseline.

        Refreshes the cached baseline to the post-edit state, so a subsequent
        edit diffs against this result (and the same problem isn't re-reported).
        Returns "" when there are no new diagnostics or no server handles it.
        """
        if not self.has_client(file_path):
            return ""
        key = str(Path(file_path).resolve())
        baseline = self._baselines.get(key, [])
        current = self.diagnostics(file_path)
        base_keys = {d.identity for d in baseline}
        new_diags = [d for d in current if d.identity not in base_keys]
        self._baselines[key] = current
        return format_diagnostics(
            new_diags, header="New diagnostics introduced by this edit:"
        )

    def shutdown(self) -> None:
        self.manager.shutdown_all()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
