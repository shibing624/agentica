# -*- coding: utf-8 -*-
"""The one session list: which sessions exist, which the default view shows, and what each row says."""
import os
import time
from unittest.mock import patch

import pytest

from agentica.session import catalog as catalog_mod
from agentica.session.catalog import LiveState, list_catalog
from agentica.session.log import SessionLog

DAY = 86400


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTICA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("AGENTICA_PROJECTS_DIR", str(tmp_path / "home" / "projects"))
    catalog_mod._preview_cache.clear()


@pytest.fixture
def proj(tmp_path):
    d = tmp_path / "proj"
    d.mkdir()
    return d


def _session(sid, work_dir, *, source="gateway", age_days=0.0, users=("hello",), ts=None):
    log = SessionLog(sid, work_dir=str(work_dir), user_id="default")
    for text in users:
        log.append("user", text, **({"timestamp": ts} if ts else {}))
        log.append("assistant", f"re: {text}")
    if source:
        log.set_source(source)
    if age_days:
        when = time.time() - age_days * DAY
        os.utime(log.path, (when, when))
    return log


def _ids(catalog_or_rows):
    rows = catalog_or_rows.rows if hasattr(catalog_or_rows, "rows") else catalog_or_rows
    return {r["session_id"] for r in rows}


class TestRecentView:
    def test_old_sessions_fold_whatever_channel_they_came_from(self, proj):
        """The Web used to exempt gateway/IM sessions from the 30-day window,
        so IM showed every file ("共有 116 个会话"). One rule now, every source."""
        _session("new-web", proj)
        _session("old-web", proj, age_days=40)
        _session("old-wx", proj, source="wechat", age_days=40)
        cat = list_catalog(user_id="default", scope="recent")
        assert _ids(cat) == {"new-web"}
        assert {f["session_id"] for f in cat.folded} == {"old-web", "old-wx"}
        assert cat.hidden == 2

    def test_pinned_archived_and_live_old_sessions_stay(self, proj):
        _session("old-pinned", proj, age_days=40).set_pinned(True)
        _session("old-archived", proj, age_days=40).set_archived(True)
        _session("old-running", proj, age_days=40)
        _session("old-plain", proj, age_days=40)
        cat = list_catalog(user_id="default", scope="recent", live=LiveState(running={"old-running"}))
        assert _ids(cat) == {"old-pinned", "old-archived", "old-running"}
        # The default view leaves out archived ones and counts them as hidden.
        assert {r["session_id"] for r in cat.active} == {"old-pinned", "old-running"}
        assert cat.hidden == 2

    def test_all_scope_folds_nothing_and_deletes_nothing(self, proj):
        for i in range(5):
            _session(f"old-{i}", proj, age_days=40 + i)
        cat = list_catalog(user_id="default", scope="all")
        assert len(cat.rows) == 5 and cat.folded == []
        assert len(list(SessionLog.list_sessions(work_dir=str(proj), user_id="default"))) == 5

    def test_recent_days_is_the_window(self, proj):
        _session("ten-days", proj, age_days=10)
        assert _ids(list_catalog(user_id="default", recent_days=7)) == set()
        assert _ids(list_catalog(user_id="default", recent_days=30)) == {"ten-days"}

    def test_folded_sessions_are_not_previewed(self, proj):
        _session("old", proj, age_days=40)
        with patch.object(SessionLog, "session_preview") as preview:
            list_catalog(user_id="default", scope="recent")
        preview.assert_not_called()


class TestListed:
    def test_cli_only_on_request_legacy_and_sdk_never(self, proj):
        _session("web", proj)
        _session("cli", proj, source="cli")
        _session("sdk", proj, source="sdk")
        _session("legacy", proj, source=None)
        assert _ids(list_catalog(user_id="default", scope="all")) == {"web"}
        assert _ids(list_catalog(user_id="default", scope="all", include_cli=True)) == {"web", "cli"}

    def test_stubs_and_legacy_cron_sessions_are_not_sessions(self, proj):
        _session("real", proj)
        SessionLog("stub", work_dir=str(proj), user_id="default").path.touch()
        _session("scheduled_job1", proj)
        named = SessionLog("titled-before-run", work_dir=str(proj), user_id="default")
        named.path.touch()
        named.set_name("每日巡检")
        named.set_source("cron")
        assert _ids(list_catalog(user_id="default", scope="all")) == {"real", "titled-before-run"}

    def test_one_project_only_when_work_dir_given(self, tmp_path, proj):
        other = tmp_path / "other"
        other.mkdir()
        _session("here", proj)
        _session("there", other)
        assert _ids(list_catalog(user_id="default", work_dir=str(proj))) == {"here"}
        assert _ids(list_catalog(user_id="default")) == {"here", "there"}


class TestRow:
    def test_rows_are_newest_created_first(self, proj):
        _session("older", proj, ts="2026-01-01T00:00:00.000Z")
        _session("newer", proj, ts="2026-01-02T00:00:00.000Z")
        rows = list_catalog(user_id="default", scope="all").rows
        assert [r["session_id"] for r in rows] == ["newer", "older"]

    def test_title_source_origin_and_live_fields(self, proj):
        log = _session("agent:main:feishu:c7", proj, users=("看下缓存",))
        rows = {r["session_id"]: r for r in list_catalog(
            user_id="default", scope="all",
            live=LiveState(running={"agent:main:feishu:c7"}, approval_modes={"agent:main:feishu:c7": "read-only"}),
        ).rows}
        row = rows["agent:main:feishu:c7"]
        assert (row["name"], row["source"], row["origin"]) == ("看下缓存", "gateway", "feishu")
        assert (row["running"], row["unread"], row["approval_mode"]) == (True, False, "read-only")
        assert row["size_bytes"] == log.path.stat().st_size

    def test_preview_is_cached_by_file_version(self, proj):
        log = _session("s1", proj)
        with patch.object(SessionLog, "session_preview", wraps=SessionLog.session_preview) as preview:
            list_catalog(user_id="default", scope="all")
            list_catalog(user_id="default", scope="all")
            assert preview.call_count == 1
            log.append("user", "more")
            list_catalog(user_id="default", scope="all")
            assert preview.call_count == 2
