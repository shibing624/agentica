# -*- coding: utf-8 -*-
"""A turn's entry point (RunConfig.source / .channel) is what the session sidecar records."""
from types import SimpleNamespace

from agentica.memory.session_log import SessionLog
from agentica.run.context import RunSource
from agentica.runner.persist import PersistMixin


def _agent(log, source, channel):
    return SimpleNamespace(
        _session_log=log,
        run_context=SimpleNamespace(source=source),
        _run_channel=channel,
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
