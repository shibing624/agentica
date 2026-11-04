# -*- coding: utf-8 -*-
"""Sidebar workspaces live in the account's prefs: a folder plus an optional label."""
from types import SimpleNamespace

import pytest

pytest.importorskip("agentica_gateway", reason="requires agentica-gateway")

from fastapi import HTTPException  # noqa: E402

from agentica_gateway.routes import settings as settings_routes  # noqa: E402


def _request(body, user_id="u1"):
    async def _json():
        return body

    return SimpleNamespace(
        state=SimpleNamespace(principal=SimpleNamespace(user_id=user_id)),
        json=_json,
    )


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_routes, "AGENTICA_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(settings_routes.deps, "agent_service", None)
    return tmp_path


def test_clean_workspaces_dedupes_trims_and_drops_junk():
    cleaned = settings_routes._clean_workspaces([
        {"dir": "/a/b/", "name": "  Alpha  "},
        {"dir": "/a/b"},
        {"dir": "  "},
        {"dir": 3},
        "nope",
        {"dir": "/c", "name": "x" * 500},
    ])
    assert cleaned[0] == {"dir": "/a/b", "name": "Alpha"}
    assert len(cleaned) == 2
    assert len(cleaned[1]["name"]) == settings_routes._MAX_WORKSPACE_NAME
    assert settings_routes._clean_workspaces("not a list") == []


@pytest.mark.asyncio
async def test_new_workspace_must_exist_and_be_a_directory(home):
    folder = home / "proj"
    folder.mkdir()
    afile = home / "file.txt"
    afile.write_text("x")

    out = await settings_routes.put_prefs(_request({"workspaces": [{"dir": str(folder), "name": "P"}]}))
    assert out["workspaces"] == [{"dir": str(folder), "name": "P"}]

    with pytest.raises(HTTPException) as missing:
        await settings_routes.put_prefs(_request({"workspaces": [{"dir": str(home / "nope")}]}))
    assert missing.value.status_code == 400

    with pytest.raises(HTTPException) as notdir:
        await settings_routes.put_prefs(_request({"workspaces": [{"dir": str(afile)}]}))
    assert notdir.value.status_code == 400


@pytest.mark.asyncio
async def test_rename_and_remove_do_not_recheck_the_folder(home):
    folder = home / "proj"
    folder.mkdir()
    await settings_routes.put_prefs(_request({"workspaces": [{"dir": str(folder)}]}))
    folder.rmdir()

    # Already known: renaming a workspace whose folder has since vanished still works,
    # and so does removing it — otherwise a stale entry could never be cleaned up.
    out = await settings_routes.put_prefs(_request({"workspaces": [{"dir": str(folder), "name": "Renamed"}]}))
    assert out["workspaces"][0]["name"] == "Renamed"
    out = await settings_routes.put_prefs(_request({"workspaces": []}))
    assert out["workspaces"] == []


@pytest.mark.asyncio
async def test_other_prefs_survive_a_workspace_update(home):
    await settings_routes.put_prefs(_request({"theme": "dark"}))
    out = await settings_routes.put_prefs(_request({"workspaces": []}))
    assert out["theme"] == "dark"
    assert (await settings_routes.get_prefs(_request({})))["theme"] == "dark"
