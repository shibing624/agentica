# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Environment health check ("doctor") for Agentica.

A focused preflight that surfaces the things that actually make a run fail or
degrade — Python version, writable home/checkpoint dirs, a configured provider
+ its API key, and optional tooling (pyright for LSP, MCP config). It does NOT
make network calls (no cost, no hang); "configured" means the key is present,
not that the endpoint answered.

SDK use:
    from agentica.diagnostics import run_doctor
    report = run_doctor()
    print(report.ok, report.summary())

CLI use:
    agentica doctor
"""
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from typing import List

from agentica.config import AGENTICA_HOME, AGENTICA_DOTENV_PATH

OK = "ok"
WARN = "warn"
FAIL = "fail"


@dataclass
class DoctorCheck:
    """One environment check result."""
    name: str
    status: str          # OK | WARN | FAIL
    detail: str = ""


@dataclass
class DoctorReport:
    checks: List[DoctorCheck] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append(DoctorCheck(name=name, status=status, detail=detail))

    @property
    def ok(self) -> bool:
        """True when no check failed (warnings are tolerated)."""
        return all(c.status != FAIL for c in self.checks)

    def counts(self) -> dict:
        out = {OK: 0, WARN: 0, FAIL: 0}
        for c in self.checks:
            out[c.status] = out.get(c.status, 0) + 1
        return out

    def summary(self) -> str:
        c = self.counts()
        return f"{c[OK]} ok, {c[WARN]} warning(s), {c[FAIL]} failure(s)"


def _dir_writable(path: str) -> bool:
    """Can we create (and remove) a file under ``path``? Creates it if missing."""
    try:
        os.makedirs(path, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path)
        os.close(fd)
        os.unlink(tmp)
        return True
    except OSError:
        return False


def _check_python(report: DoctorReport) -> None:
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= (3, 10):
        report.add("Python version", OK, ver)
    else:
        report.add("Python version", FAIL, f"{ver} (Agentica requires >= 3.10)")


def _check_version(report: DoctorReport) -> None:
    try:
        from agentica.version import __version__
        report.add("Agentica version", OK, __version__)
    except Exception as e:
        report.add("Agentica version", WARN, f"unknown ({e})")


def _check_dirs(report: DoctorReport) -> None:
    if _dir_writable(AGENTICA_HOME):
        report.add("Home dir writable", OK, AGENTICA_HOME)
    else:
        report.add("Home dir writable", FAIL, f"cannot write to {AGENTICA_HOME}")

    ckpt_root = os.path.join(AGENTICA_HOME, "checkpoints")
    if _dir_writable(ckpt_root):
        report.add("Checkpoint dir writable", OK, ckpt_root)
    else:
        report.add("Checkpoint dir writable", WARN, f"cannot write to {ckpt_root}")

    if os.path.exists(AGENTICA_DOTENV_PATH):
        report.add(".env file", OK, AGENTICA_DOTENV_PATH)
    else:
        report.add(".env file", WARN, f"not found at {AGENTICA_DOTENV_PATH} (using process env only)")


def _check_provider(report: DoctorReport) -> None:
    from agentica.cli.setup import load_cli_config, provider_env_var, has_api_key, DEFAULT_PROVIDER, DEFAULT_MODEL

    saved = load_cli_config()
    provider = saved.get("model_provider") or DEFAULT_PROVIDER
    model = saved.get("model_name") or DEFAULT_MODEL
    report.add("Configured provider", OK, f"{provider}/{model}")

    env_var = provider_env_var(provider)
    if has_api_key(provider):
        report.add("API key", OK, f"{env_var} is set")
    else:
        report.add("API key", FAIL, f"{env_var} not set — run `agentica setup` or export it")


def _check_lsp(report: DoctorReport) -> None:
    # pyright ships the server as `pyright-langserver` (or `pyright`).
    found = shutil.which("pyright-langserver") or shutil.which("pyright")
    if found:
        report.add("LSP (pyright)", OK, found)
    else:
        report.add("LSP (pyright)", WARN, "not on PATH — enable_diagnostics will no-op (npm i -g pyright)")


def _check_mcp(report: DoctorReport) -> None:
    from agentica.mcp.config import MCPConfig

    try:
        cfg = MCPConfig()
    except Exception as e:
        report.add("MCP config", FAIL, f"failed to load: {e}")
        return
    if not cfg.config_path:
        report.add("MCP config", OK, "none found (optional)")
    elif cfg.servers:
        report.add("MCP config", OK, f"{len(cfg.servers)} server(s) in {cfg.config_path}")
    else:
        report.add("MCP config", WARN, f"{cfg.config_path} has no usable servers")


def run_doctor() -> DoctorReport:
    """Run all environment checks and return a structured report.

    Never raises: a check that blows up is recorded as a FAIL entry rather than
    propagating, since the whole point of the doctor is to not crash. No network
    calls.
    """
    report = DoctorReport()
    checks = [
        ("Python version", _check_python),
        ("Agentica version", _check_version),
        ("Filesystem", _check_dirs),
        ("Provider/credentials", _check_provider),
        ("LSP", _check_lsp),
        ("MCP config", _check_mcp),
    ]
    for label, fn in checks:
        try:
            fn(report)
        except Exception as e:  # diagnostic boundary: degrade, don't crash
            report.add(f"{label} (check error)", FAIL, str(e))
    return report


if __name__ == "__main__":
    r = run_doctor()
    for c in r.checks:
        print(f"[{c.status.upper():4}] {c.name}: {c.detail}")
    print(r.summary())
