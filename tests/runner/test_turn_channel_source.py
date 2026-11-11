# -*- coding: utf-8 -*-
"""A turn's entry point (RunConfig.source / .channel) is what the session sidecar records."""
from types import SimpleNamespace

from agentica.memory.session_log import SessionLog
from agentica.run.context import RunSource
from agentica.runner.persist import PersistMixin


def _agent(log, source, channel, input_meta=None):
    return SimpleNamespace(
        _session_log=log,
        run_context=SimpleNamespace(source=source),
        _run_channel=channel,
        _run_input_meta=input_meta,
    )


def _turn(log, source, channel, text):
    log._turn_user_uuid = None
    PersistMixin._persist_turn_user_message(_agent(log, source, channel), text, [])


def test_channel_overrides_the_coarse_source_and_origin_stays_put(tmp_path):
    log = SessionLog("s1", base_dir=str(tmp_path))
    _turn(log, RunSource.gateway, "wechat", "from wechat")
    assert (log.get_source(), log.get_origin()) == ("wechat", "wechat")

    _turn(log, RunSource.gateway, None, "from the web page")
    assert (log.get_source(), log.get_origin()) == ("gateway", "wechat")

    _turn(log, RunSource.cli, None, "from the terminal")
    assert (log.get_source(), log.get_origin()) == ("cli", "wechat")


def test_input_meta_is_written_on_the_turns_user_entry(tmp_path):
    """RunConfig.input_meta is how a scheduled run marks its turn (job id/name),
    so a transcript can show it as that job's run rather than as typed text."""
    log = SessionLog("s2", base_dir=str(tmp_path))
    log._turn_user_uuid = None
    PersistMixin._persist_turn_user_message(
        _agent(log, RunSource.cron, None, {"scheduled": {"job_id": "j1", "name": "Daily"}}),
        "run it", [],
    )
    users = [e for e in log.iter_raw_entries() if e.get("type") == "user"]
    assert users[-1]["content"] == "run it"
    assert users[-1]["scheduled"] == {"job_id": "j1", "name": "Daily"}
