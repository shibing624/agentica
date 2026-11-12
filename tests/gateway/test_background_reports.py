# -*- coding: utf-8 -*-
"""A finished background command reports back on its own, as in the CLI.

Web, Desktop and IM had no way to learn that a detached command ended: the
agent only found out when someone asked. Now the session's registry is drained
like the CLI's and each report runs as an automated turn once the session is
idle — in the IM chat bound to it, else as a live web run.
"""
import asyncio
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("agentica_gateway", reason="requires agentica-gateway")

from agentica_gateway.services import live_turn
from agentica_gateway.services import agent_service as svc_mod
from agentica_gateway.services.agent_service import AgentService


def _svc(tmp_path, handler):
    svc = AgentService(workspace_path=str(tmp_path / "ws"))
    svc.background_report_handler = handler
    return svc


def _run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=20))


def test_a_finished_command_is_reported_with_its_output(tmp_path):
    got = []

    async def scenario():
        done = asyncio.Event()

        async def handler(session_id, owner, report):
            got.append((session_id, owner, report))
            done.set()

        svc = _svc(tmp_path, handler)
        await svc._ensure_initialized()
        registry = svc._background_registry("s1", "alice")
        assert registry.delivers_results is True
        registry.start(f"{sys.executable} -c 'print(\"eval done: 0.91\"); raise SystemExit(2)'", cwd=str(tmp_path))
        await asyncio.wait_for(done.wait(), timeout=15)
        svc.stop_all_background_jobs()

    _run(scenario())
    (session_id, owner, report), = got
    assert (session_id, owner) == ("s1", "alice")
    assert "failed: exit 2" in report
    assert "eval done: 0.91" in report


def test_a_command_the_user_stopped_is_not_reported(tmp_path):
    got = []

    async def scenario():
        async def handler(*args):
            got.append(args)

        svc = _svc(tmp_path, handler)
        await svc._ensure_initialized()
        registry = svc._background_registry("s1", None)
        item = registry.start(f"{sys.executable} -c 'import time; time.sleep(60)'", cwd=str(tmp_path))
        await asyncio.to_thread(registry.stop, item.id)
        await asyncio.sleep(2.0)

    _run(scenario())
    assert got == []


def test_the_report_waits_until_the_session_is_idle(tmp_path):
    """Starting a second run on a busy session would be refused; the report
    is held until the turn in flight ends, then runs."""
    times = {}

    async def scenario():
        done = asyncio.Event()

        async def handler(*_):
            times["reported"] = asyncio.get_running_loop().time()
            done.set()

        svc = _svc(tmp_path, handler)
        await svc._ensure_initialized()
        lock = svc._get_session_lock("s1", None)
        await lock.acquire()
        registry = svc._background_registry("s1", None)
        registry.start(f"{sys.executable} -c 'pass'", cwd=str(tmp_path))
        await asyncio.sleep(2.5)
        assert not done.is_set()
        times["released"] = asyncio.get_running_loop().time()
        lock.release()
        await asyncio.wait_for(done.wait(), timeout=10)

    _run(scenario())
    assert times["reported"] >= times["released"]


def test_without_a_handler_the_tool_says_reports_do_not_come_back(tmp_path):
    svc = AgentService(workspace_path=str(tmp_path / "ws"))
    assert svc._background_registry("s1", None).delivers_results is False


def test_deleting_the_chat_while_its_report_waits_voids_the_report(tmp_path):
    """The report waited for a busy session; the user deleted the chat. It
    must not run and rebuild the deleted log."""
    got = []

    async def scenario():
        async def handler(*args):
            got.append(args)

        svc = _svc(tmp_path, handler)
        await svc._ensure_initialized()
        lock = svc._get_session_lock("s1", None)
        await lock.acquire()
        registry = svc._background_registry("s1", None)
        registry.start(f"{sys.executable} -c 'pass'", cwd=str(tmp_path))
        await asyncio.sleep(2.0)
        svc.delete_session("s1")
        await asyncio.sleep(2.5)

    _run(scenario())
    assert got == []


def test_a_failing_handler_is_logged_and_later_reports_still_arrive(tmp_path):
    seen = []

    async def scenario():
        second = asyncio.Event()

        async def handler(session_id, owner, report):
            seen.append(report)
            if len(seen) == 1:
                raise RuntimeError("channel down")
            second.set()

        svc = _svc(tmp_path, handler)
        await svc._ensure_initialized()
        registry = svc._background_registry("s1", None)
        registry.start(f"{sys.executable} -c 'print(1)'", cwd=str(tmp_path))
        await asyncio.sleep(2.0)
        registry.start(f"{sys.executable} -c 'print(2)'", cwd=str(tmp_path))
        await asyncio.wait_for(second.wait(), timeout=15)

    _run(scenario())
    assert len(seen) == 2


def test_an_automated_turn_does_not_name_the_chat(tmp_path, monkeypatch):
    svc = AgentService(workspace_path=str(tmp_path / "ws"))
    named = []
    monkeypatch.setattr(svc, "_schedule_title", lambda *a: named.append(a))

    class Agent:
        tools = []

        async def run_stream(self, *_a, **_k):
            if False:
                yield None

    async def scenario():
        monkeypatch.setattr(svc, "_get_agent", lambda *a, **k: asyncio.sleep(0, Agent()))
        monkeypatch.setattr(svc_mod, "usage_payload", lambda *a, **k: asyncio.sleep(0, None))
        monkeypatch.setattr(svc_mod, "turn_usage_payload", lambda *a: None)
        monkeypatch.setattr(svc_mod, "extract_metrics", lambda *a: {})
        await svc.chat_stream("[Background terminal #1 ...]", "s1", automated=True)
        await svc.chat_stream("what changed?", "s1")

    _run(scenario())
    assert [a[2] for a in named] == ["what changed?"]


# ---------------------------------------------------------------- IM side

class _FakeService:
    def __init__(self, reply="评测跑完了：0.91，比基线高 3 点。"):
        self.calls = []
        self.reply = reply

    def set_session_work_dir(self, *a, **k): ...
    def set_session_approval_mode(self, *a, **k): ...
    def is_session_active(self, *a, **k): return False
    def take_undelivered_steer(self, *a, **k): return []
    def list_background_jobs(self, *a, **k): return []
    def background_job_output(self, *a, **k): return None
    async def stop_background_job(self, *a, **k): return None
    async def stop_all_session_background_jobs(self, *a, **k): return []

    async def chat_stream(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(content=self.reply, tool_calls=0, tools_used=[], media_notes=[], turn_usage=None)


class _FakeChannels:
    def __init__(self, connected=("wechat",)):
        self.sent = []
        self.connected = set(connected)

    def get_channel(self, channel):
        return SimpleNamespace(is_connected=True) if channel.value in self.connected else None

    async def send(self, channel, channel_id, text):
        self.sent.append((channel_id, text))
        return True


def _bot(tmp_path, svc, channels, chats=(("k1", "u1"),), session="s1"):
    from agentica_gateway.channels.base import ChannelType
    from agentica_gateway.services.im_bot import ImBotController, ImChatState

    bot = ImBotController(channel_manager=channels, agent_service=svc, state_file=tmp_path / "im.json",
                          user_id_provider=lambda: "home")
    for key, cid in chats:
        bot._states[key] = ImChatState(chat_key=key, channel=ChannelType.WECHAT, channel_id=cid,
                                       work_dir=str(tmp_path), session_id=session)
    bot.find_live_cli_peer = lambda sid: None
    return bot


def test_the_im_chats_bound_to_the_session_get_the_report_answer(tmp_path):
    svc, channels = _FakeService(), _FakeChannels()

    async def scenario():
        bot = _bot(tmp_path, svc, channels, chats=(("k1", "u1"), ("k2", "u2")))
        assert await bot.deliver_background_report("home", "s1", "[Background terminal #1 ...]") is True
        await asyncio.wait_for(bot._turn_tasks["s1"], timeout=10)

    _run(scenario())
    (kw,) = svc.calls
    assert kw["automated"] is True and kw["message"].startswith("[Background terminal #1")
    assert sorted(cid for cid, _ in channels.sent) == ["u1", "u2"]
    assert all(text.startswith("🔔 后台任务已结束") for _, text in channels.sent)
    assert live_turn.active("s1", "home") is None


def test_another_accounts_report_never_reaches_the_im_chat(tmp_path):
    """IM chats belong to the home account; alice's session id may collide."""
    svc, channels = _FakeService(), _FakeChannels()

    async def scenario():
        bot = _bot(tmp_path, svc, channels)
        assert await bot.deliver_background_report("alice", "s1", "secret output") is False

    _run(scenario())
    assert svc.calls == [] and channels.sent == []


def test_a_gateway_without_the_channel_hands_the_report_back(tmp_path):
    """A standby gateway loads the same chat bindings but has no WeChat; it
    must say no so the report runs on the web instead of vanishing."""
    svc, channels = _FakeService(), _FakeChannels(connected=())

    async def scenario():
        bot = _bot(tmp_path, svc, channels)
        assert await bot.deliver_background_report("home", "s1", "report") is False

    _run(scenario())
    assert svc.calls == []


def test_a_session_the_terminal_took_over_gets_the_report_through_its_mailbox(tmp_path, monkeypatch):
    from agentica_gateway.services import im_bot as im_mod

    svc, channels, mailed = _FakeService(), _FakeChannels(), []
    monkeypatch.setattr(im_mod, "send_peer_message", lambda peer, **kw: mailed.append((peer.name, kw)))

    async def scenario():
        bot = _bot(tmp_path, svc, channels)
        bot.find_live_cli_peer = lambda sid: SimpleNamespace(name="cli-1")
        assert await bot.deliver_background_report("home", "s1", "report text") is True

    _run(scenario())
    assert svc.calls == []
    assert mailed == [("cli-1", {"text": "report text", "from_name": "gateway",
                                 "from_peer_id": "gateway", "from_kind": "agent"})]
    assert "终端 [cli-1]" in channels.sent[0][1]


def test_a_report_with_no_text_answer_still_tells_the_phone(tmp_path):
    svc, channels = _FakeService(reply=""), _FakeChannels()

    async def scenario():
        bot = _bot(tmp_path, svc, channels)
        await bot.deliver_background_report("home", "s1", "report")
        await asyncio.wait_for(bot._turn_tasks["s1"], timeout=10)

    _run(scenario())
    assert len(channels.sent) == 1 and channels.sent[0][1].startswith("🔔 后台任务已结束")


def test_a_report_queued_behind_user_lines_runs_as_its_own_turn(tmp_path):
    from agentica_gateway.services.im_bot import QueuedTurn

    svc, channels = _FakeService(), _FakeChannels()

    async def scenario():
        bot = _bot(tmp_path, svc, channels)
        state = bot._states["k1"]
        q = asyncio.Queue()
        for item in (
            QueuedTurn(state, str(tmp_path), "s1", "line one"),
            QueuedTurn(state, str(tmp_path), "s1", "line two"),
            QueuedTurn(state, str(tmp_path), "s1", "REPORT", automated=True),
            QueuedTurn(state, str(tmp_path), "s1", "line three"),
        ):
            q.put_nowait(item)
        await bot._drain_session_queue("s1", q)
        await asyncio.wait_for(q.join(), timeout=5)

    _run(scenario())
    assert [(c["message"], c["automated"]) for c in svc.calls] == [
        ("line one\n\nline two", False), ("REPORT", True), ("line three", False),
    ]


# ------------------------------------------------------------ main wiring

def test_main_runs_the_report_on_the_web_when_no_im_chat_takes_it(monkeypatch):
    from agentica_gateway import deps, main
    from agentica_gateway.routes import chat

    started = []
    monkeypatch.setattr(chat, "start_background_report_run", lambda svc, sid, owner, rep: started.append((sid, owner, rep)))

    class Bot:
        def __init__(self, takes):
            self.takes, self.asked = takes, []

        async def deliver_background_report(self, owner, sid, rep):
            self.asked.append((owner, sid))
            return self.takes

    svc = SimpleNamespace()
    deliver = main._background_report_handler(svc)
    for takes in (True, False):
        monkeypatch.setattr(deps, "im_bot", Bot(takes))
        _run(deliver("s1", "alice", "r"))
        assert deps.im_bot.asked == [("alice", "s1")]
    assert started == [("s1", "alice", "r")]


def test_a_gateway_that_loses_the_im_lock_never_connects_a_channel(monkeypatch):
    """The one line that stops duplicate IM replies: no claim, no channels."""
    from agentica_gateway import main, runtime

    connected = []
    monkeypatch.setattr(runtime, "claim_im_channels", lambda: False)
    monkeypatch.setattr(main, "_setup_channels", lambda: asyncio.sleep(0, connected.append(1)))
    assert _run(main._claim_and_start_im()) is False
    assert connected == []


def test_a_standby_gateway_takes_over_im_when_the_owner_exits(monkeypatch):
    from agentica_gateway import main, runtime

    claims = iter([False, False, True])
    connected = []
    monkeypatch.setattr(runtime, "claim_im_channels", lambda: next(claims))
    monkeypatch.setattr(main, "_setup_channels", lambda: asyncio.sleep(0, connected.append(1)))
    monkeypatch.setattr(main, "IM_TAKEOVER_POLL_S", 0.01)
    _run(main._take_over_im_when_free())
    assert connected == [1]
