# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for the gateway agent's own peer identity (gateway/services/agent_peers.py).

What is under test is one promise: a plain sentence typed into IM ("让三个会话都
把改动提交了") reaches this machine's live CLI sessions, and their answers come
back to the same chat. The `@`-command path is peer_bridge.py's; this is the
path where the gateway's own agent does the addressing.
"""
import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest

# Gateway tests require fastapi etc. Skip cleanly if not installed.
pytest.importorskip("fastapi", reason="Gateway tests require agentica[gateway]")

from agentica import peers
from agentica.peers import PeerSession, list_live_peers, unread_count
from agentica.gateway.channels.base import Channel, ChannelType, Message
from agentica.gateway.services.agent_peers import GatewayAgentPeers
from agentica.gateway.services.peer_bridge import PeerBridge


@pytest.fixture(autouse=True)
def isolated_peers_root(tmp_path, monkeypatch):
    """Keep live records and mailboxes out of the real cache dir."""
    monkeypatch.setattr(peers, "AGENTICA_CACHE_DIR", str(tmp_path))
    yield tmp_path


class FakeChannel(Channel):
    """A channel that records what it was asked to send."""

    def __init__(self):
        super().__init__()
        self.sent = []

    @property
    def channel_type(self):
        return ChannelType.WECOM

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def send(self, channel_id, content, **kwargs):
        self.sent.append((channel_id, content))
        return True


class FakeChannelManager:
    def __init__(self, channel):
        self.channel = channel

    def get_channel(self, channel_type):
        return self.channel if channel_type == self.channel.channel_type else None

    async def send(self, channel_type, channel_id, content, **kwargs):
        return await self.channel.send(channel_id, content)


def _cli(name, *, cwd="/tmp/proj"):
    session = PeerSession(name=name, cwd=cwd)
    session.publish()
    return session


def _gateway_peers(*, live=(), busy=()):
    """A GatewayAgentPeers whose liveness/busy answers are fixed sets."""
    channel = FakeChannel()
    peers_service = GatewayAgentPeers(
        channel_manager=FakeChannelManager(channel),
        is_live=lambda sid: sid in live,
        is_busy=lambda sid: sid in busy,
    )
    return peers_service, channel


class TestPresence:
    """The gateway agent is a peer like any other, and says which one it is."""

    def test_a_chat_session_is_published_where_a_cli_can_find_it(self):
        service, _ = _gateway_peers(live={"web-1"})

        session = service.session_for("web-1", cwd="/tmp/proj")

        published = list_live_peers()
        assert [p.peer_id for p in published] == [session.peer_id]
        assert published[0].session_id == "web-1"
        assert published[0].cwd == os.path.realpath("/tmp/proj")

    def test_the_name_is_channel_folder_id_like_a_cli_session(self):
        """CLI is ``<folder>-<xx>``; gateway is ``<channel>-<folder>-<xx>``.

        match_peers() treats a name prefix as an address, so this must not
        collide with a CLI's ``agentica-41`` or a bridge endpoint's
        ``wechat-<openid>``. The WeChat sender id must not appear — that is
        what made ``send_message`` targets unusable.
        """
        service, _ = _gateway_peers(live={"s1"})
        service.note_route("s1", ChannelType.WECHAT, "o9cq8035jyckmmlzta33-mkm")

        session = service.session_for("s1", cwd="/tmp/agentica")

        assert session.name == f"wechat-agentica-{session.peer_id[:2]}"
        assert "o9cq" not in session.name

    def test_a_web_session_uses_the_web_channel_and_the_cwd_folder(self):
        service, _ = _gateway_peers(live={"web-1"})

        session = service.session_for("web-1", cwd="/tmp/proj")

        assert session.name == f"web-proj-{session.peer_id[:2]}"

    def test_note_turn_publishes_what_the_session_is_working_on(self):
        service, _ = _gateway_peers(live={"s1"})
        service.session_for("s1")

        service.note_turn("s1", "让三个会话都把改动提交了\n第二行也算")

        info = list_live_peers()[0]
        assert info.task == "让三个会话都把改动提交了 第二行也算"
        assert info.busy is True

    def test_note_turn_on_an_unregistered_session_creates_nothing(self):
        """A cron run has no peer identity; a turn is not a reason to mint one."""
        service, _ = _gateway_peers(live={"scheduled_job1"})

        service.note_turn("scheduled_job1", "nightly report")

        assert list_live_peers() == []


class TestSending:
    """What the whole feature exists for: reaching a CLI session."""

    def test_the_gateway_agent_can_message_a_live_cli_session(self):
        cli = _cli("payments-a1")
        service, _ = _gateway_peers(live={"s1"})
        gw = service.session_for("s1")

        message = gw.send("payments-a1", "commit your changes")

        assert message.to_peer_id == cli.peer_id
        # Not the user speaking: the CLI applies the unprivileged policy.
        assert message.from_kind == "agent"
        assert [m.text for m in cli.drain()] == ["commit your changes"]


class TestReplies:
    """A CLI's answer has to reach whoever asked, from wherever they asked."""

    def test_a_reply_is_pushed_to_the_im_conversation(self):
        cli = _cli("payments-a1")
        service, channel = _gateway_peers(live={"s1"})
        service.note_route("s1", ChannelType.WECOM, "chat-1")
        gw = service.session_for("s1")
        cli.send(gw.peer_id, "committed 4f2a1c9")

        asyncio.run(service._tick())

        assert channel.sent == [("chat-1", "payments-a1 ›\ncommitted 4f2a1c9")]
        assert unread_count(gw.peer_id) == 0

    def test_mid_turn_the_mailbox_is_left_for_the_running_agent(self):
        """The Runner drains it between tool batches, so the model acts on it —
        strictly better than telling the user something the agent it is talking
        to has not seen."""
        cli = _cli("payments-a1")
        service, channel = _gateway_peers(live={"s1"}, busy={"s1"})
        service.note_route("s1", ChannelType.WECOM, "chat-1")
        gw = service.session_for("s1")
        cli.send(gw.peer_id, "committed 4f2a1c9")

        asyncio.run(service._tick())

        assert channel.sent == []
        assert unread_count(gw.peer_id) == 1

    def test_a_web_session_keeps_its_mail_until_its_next_turn(self):
        """No IM route to push to; draining here would consume a message with
        nowhere to put it."""
        cli = _cli("payments-a1")
        service, channel = _gateway_peers(live={"web-1"})
        gw = service.session_for("web-1")
        cli.send(gw.peer_id, "committed 4f2a1c9")

        asyncio.run(service._tick())

        assert channel.sent == []
        assert unread_count(gw.peer_id) == 1


class TestLifecycle:
    """A live record must not outlive the session it describes."""

    def test_a_session_the_agent_cache_forgot_stops_being_addressable(self):
        service, _ = _gateway_peers(live={"s1"})
        service.session_for("s1")

        service._is_live = lambda sid: False
        asyncio.run(service._tick())

        assert list_live_peers() == []
        assert service.peer_ids() == set()

    def test_a_mid_turn_session_survives_a_liveness_miss(self):
        """The agent is built after the first note; a turn in flight is proof
        enough that the session exists."""
        service, _ = _gateway_peers(live=(), busy={"s1"})
        service.session_for("s1")

        asyncio.run(service._tick())

        assert [p.session_id for p in list_live_peers()] == ["s1"]

    def test_stop_unpublishes_everything(self):
        service, _ = _gateway_peers(live={"s1", "s2"})
        service.session_for("s1")
        service.session_for("s2")

        asyncio.run(service.stop())

        assert list_live_peers() == []

    def test_forget_unpublishes_one_session(self):
        service, _ = _gateway_peers(live={"s1", "s2"})
        service.session_for("s1")
        kept = service.session_for("s2")

        service.forget("s1")

        assert [p.peer_id for p in list_live_peers()] == [kept.peer_id]


class TestBridgeExclusion:
    """The phone must not be able to relay a line into the agent answering it."""

    def _bridge_with_gateway(self):
        service, _ = _gateway_peers(live={"s1"})
        service.note_route("s1", ChannelType.WECOM, "chat-1")
        gw = service.session_for("s1")
        channel = FakeChannel()
        bridge = PeerBridge(
            FakeChannelManager(channel),
            gateway_peer_ids=service.peer_ids,
        )
        return bridge, channel, gw

    def _im(self, text):
        return Message(
            channel=ChannelType.WECOM,
            channel_id="chat-1",
            sender_id="xuming",
            sender_name="Xu Ming",
            content=text,
            message_id="m1",
        )

    def test_at_list_does_not_offer_the_gateway_agent(self):
        bridge, channel, gw = self._bridge_with_gateway()
        _cli("payments-a1")

        asyncio.run(bridge.handle(self._im("@list")))

        listing = channel.sent[0][1]
        assert "payments-a1" in listing
        assert gw.name not in listing

    def test_the_gateway_agent_cannot_be_addressed_with_at(self):
        bridge, channel, gw = self._bridge_with_gateway()
        _cli("payments-a1")

        asyncio.run(bridge.handle(self._im(f"@{gw.name} hello")))

        assert f"No live session matches '{gw.name}'" in channel.sent[0][1]


class TestAgentServiceWiring:
    """The tools have to actually be on the agent the gateway runs."""

    def _service(self, tmp_path, peers_service):
        from agentica.gateway.services.agent_service import AgentService

        svc = AgentService(workspace_path=str(tmp_path))
        svc._workspace = None
        svc.agent_peers = peers_service
        return svc

    def _build(self, svc, session_id):
        with patch("agentica.gateway.services.agent_service.create_model") as create:
            create.return_value = MagicMock()
            return svc._build_agent(session_id)

    def test_an_interactive_agent_gets_peer_messaging_and_a_mailbox(self):
        from agentica.tools.peer_tool import PeerMessagingTool

        service, _ = _gateway_peers(live={"s1"})
        svc = self._service("/tmp", service)

        agent = self._build(svc, "s1")

        assert any(isinstance(tool, PeerMessagingTool) for tool in agent.tools)
        # Set so the Runner drains replies between tool batches, as in the CLI.
        assert agent.peer_session is service.session_for("s1")

    def test_a_cron_agent_gets_neither(self):
        """A cron run's agent is never cached, so a published mailbox for it
        would stop being read the moment the job ends."""
        from agentica.tools.peer_tool import PeerMessagingTool

        service, _ = _gateway_peers(live={"scheduled_job1"})
        svc = self._service("/tmp", service)

        agent = self._build(svc, "scheduled_job1")

        assert not any(isinstance(tool, PeerMessagingTool) for tool in agent.tools)
        assert agent.peer_session is None
        assert list_live_peers() == []

    def test_a_service_without_the_peer_channel_still_builds(self):
        """PEER_BRIDGE=false, and every SDK/test AgentService: no peer tools,
        no crash."""
        from agentica.tools.peer_tool import PeerMessagingTool

        svc = self._service("/tmp", None)

        agent = self._build(svc, "s1")

        assert not any(isinstance(tool, PeerMessagingTool) for tool in agent.tools)
        assert agent.peer_session is None

    def test_deleting_a_session_unpublishes_its_peer_at_once(self, tmp_path):
        service, _ = _gateway_peers(live={"s1"})
        svc = self._service(tmp_path, service)
        self._build(svc, "s1")

        svc.delete_session("s1")

        assert list_live_peers() == []

    def test_liveness_probes_do_not_reorder_the_agent_cache(self):
        """The peer loop asks about every session it holds; if that touched the
        LRU it would keep stale sessions alive and evict the live one."""
        svc = self._service("/tmp", None)
        svc._cache.max_size = 2
        svc._cache.put("a", MagicMock())
        svc._cache.put("b", MagicMock())

        assert svc.has_cached_session("a")
        svc._cache.put("c", MagicMock())

        assert not svc.has_cached_session("a")
        assert svc.has_cached_session("b")
        assert svc.has_cached_session("c")


class TestLifespanWiring:
    """The unit above is only worth anything if startup actually connects it."""

    def test_startup_hands_the_agent_service_a_peer_channel(self):
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        from agentica.gateway import deps
        from agentica.gateway.config import settings
        from agentica.gateway.main import app

        with patch.object(settings, "peer_bridge_enabled", True):
            with TestClient(app, raise_server_exceptions=False):
                assert deps.agent_peers is not None
                assert deps.agent_service.agent_peers is deps.agent_peers
                # And the bridge knows which peers are the gateway's own, so
                # `@list` never offers the agent that answers unprefixed lines.
                assert deps.peer_bridge._gateway_peer_ids == deps.agent_peers.peer_ids
