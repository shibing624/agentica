# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: The gateway as its own distribution: where it finds the Web UI,
and the deprecated ``agentica.gateway`` alias.
"""
import subprocess
import sys
import warnings

import pytest

pytest.importorskip("agentica_gateway", reason="requires agentica-gateway")

from agentica_gateway import ui_dist  # noqa: E402


def _dist(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "index.html").write_text("<!doctype html>")
    return path


def test_env_var_wins_over_package_and_checkout(tmp_path, monkeypatch):
    env = _dist(tmp_path / "env")
    pkg = _dist(tmp_path / "pkg" / "agentica_gateway" / "ui")
    monkeypatch.setattr(ui_dist, "_PACKAGE_DIR", pkg.parent)
    monkeypatch.setenv(ui_dist.ENV_VAR, str(env))
    assert ui_dist.resolve_ui_dir() == env


def test_packaged_copy_wins_over_checkout(tmp_path, monkeypatch):
    # <root>/gateway/src/agentica_gateway/ui  vs  <root>/web/dist
    pkg_dir = tmp_path / "gateway" / "src" / "agentica_gateway"
    pkg_ui = _dist(pkg_dir / "ui")
    _dist(tmp_path / "web" / "dist")
    monkeypatch.setattr(ui_dist, "_PACKAGE_DIR", pkg_dir)
    monkeypatch.delenv(ui_dist.ENV_VAR, raising=False)
    assert ui_dist.resolve_ui_dir() == pkg_ui


def test_editable_checkout_falls_back_to_web_dist(tmp_path, monkeypatch):
    pkg_dir = tmp_path / "gateway" / "src" / "agentica_gateway"
    pkg_dir.mkdir(parents=True)
    web = _dist(tmp_path / "web" / "dist")
    monkeypatch.setattr(ui_dist, "_PACKAGE_DIR", pkg_dir)
    monkeypatch.delenv(ui_dist.ENV_VAR, raising=False)
    assert ui_dist.resolve_ui_dir() == web


def test_directory_without_index_is_skipped(tmp_path, monkeypatch):
    empty = tmp_path / "env"
    empty.mkdir()
    pkg_dir = tmp_path / "gateway" / "src" / "agentica_gateway"
    pkg_dir.mkdir(parents=True)
    monkeypatch.setattr(ui_dist, "_PACKAGE_DIR", pkg_dir)
    monkeypatch.setenv(ui_dist.ENV_VAR, str(empty))
    assert ui_dist.resolve_ui_dir() is None


def test_old_import_path_is_the_same_module_and_warns():
    import agentica_gateway.deps as new_deps
    import agentica_gateway.routes.settings as new_settings

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        from agentica.gateway import deps as old_deps
        from agentica.gateway.routes import settings as old_settings

    # One module object, not a copy: singletons and monkeypatch targets stay shared.
    assert old_deps is new_deps
    assert old_settings is new_settings


def test_old_import_path_emits_deprecation_in_a_fresh_process():
    out = subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c", "import agentica.gateway"],
        capture_output=True, text=True, timeout=120,
    )
    assert out.returncode != 0
    assert "agentica_gateway" in out.stderr


def test_python_dash_m_old_entry_still_runs():
    # Older Desktop builds spawn `python -m agentica.gateway.main`.
    out = subprocess.run(
        [sys.executable, "-W", "ignore", "-m", "agentica.gateway.main", "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr
    assert "agentica-gateway" in out.stdout
