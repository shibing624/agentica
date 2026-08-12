# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for the IM -> local CLI peer bridge (gateway/services/peer_bridge.py).
"""
import asyncio

import pytest

# Gateway tests require fastapi etc. Skip cleanly if not installed.
pytest.importorskip("fastapi", reason="Gateway tests require agentica[gateway]")

from agentica import peers
from agentica.peers import PeerSession, drain_inbox
from agentica.gateway.channels.base import Channel, ChannelType, Message
from agentica.gateway.services.peer_bridge import (
    PeerBridge,
    parse_bridge_line,
)


@pytest.fixture(autouse=True)
def isolated_peers_root(tmp_path, monkeypatch):
    """Keep live records and mailboxes out of the real cache dir."""
    monkeypatch.setattr(peers, "AGENTICA_CACHE_DIR", str(tmp_path))
    yield tmp_path


class FakeChannel(Channel):
    """A channel that records what it was asked to send."""

    def __init__(self, allowed_users=None):
        super().__init__(allowed_users=allowed_users)
        self.sent = []
        self._connected = True

    @property
    def channel_type(self):
        return ChannelType.WECOM

    async def connect(self):
        return True

    async def disconnect(self):
        self._connected = False

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


def _im(text, *, sender_id="xuming"):
    return Message(
        channel=ChannelType.WECOM,
        channel_id="chat-1",
        sender_id=sender_id,
        sender_name="Xu Ming",
        content=text,
        message_id="m1",
    )


def _cli(name, *, cwd="/tmp/proj"):
    session = PeerSession(name=name, cwd=cwd)
    session.publish()
    return session


def _bridge(allowed_users=("xuming",)):
    channel = FakeChannel(allowed_users=list(allowed_users))
    return PeerBridge(FakeChannelManager(channel)), channel


class TestParsing:
    """One IM line in, one intent out — the whole inbound syntax."""

    def test_at_list_and_its_synonyms_ask_for_the_listing(self):
        for line in ("@list", "@ls", "@agents", "@", "@peers"):
            assert parse_bridge_line(line).kind == "list", line

    def test_at_name_with_text_sends(self):
        cmd = parse_bridge_line("@nlp-f1 rerun arm 3")
        assert (cmd.kind, cmd.target, cmd.text) == ("send", "nlp-f1", "rerun arm 3")

    def test_at_name_alone_only_picks_the_target(self):
        cmd = parse_bridge_line("@nlp-f1")
        assert (cmd.kind, cmd.target) == ("pin", "nlp-f1")

    def test_an_unprefixed_line_belongs_to_the_gateway_agent(self):
        """Enabling the bridge must not take the gateway's own agent away from
        someone who never types ``@``."""
        assert parse_bridge_line("what is the weather") is None

    def test_an_unprefixed_line_goes_to_the_pinned_session(self):
        cmd = parse_bridge_line("rerun arm 4", pinned="nlp-f1")
        assert (cmd.kind, cmd.target, cmd.text) == ("send", "nlp-f1", "rerun arm 4")

    def test_at_off_releases_the_pin(self):
        assert parse_bridge_line("@off", pinned="nlp-f1").kind == "off"


class TestRelaying:
    def test_a_message_lands_in_the_cli_mailbox_as_the_user(self):
        """The user is typing it, from a phone, so the receiving CLI must treat
        it with the authority of a line typed in that terminal."""
        cli = _cli("nlp-f1")
        bridge, channel = _bridge()

        handled = asyncio.run(bridge.handle(_im("@nlp-f1 rerun arm 3")))

        assert handled is True
        delivered = drain_inbox(cli.peer_id)
        assert [m.text for m in delivered] == ["rerun arm 3"]
        assert delivered[0].from_kind == "user"
        assert "Sent to nlp-f1" in channel.sent[-1][1]

    def test_the_target_sticks_so_the_next_bare_line_lands_there_too(self):
        cli = _cli("nlp-f1")
        bridge, _ = _bridge()

        asyncio.run(bridge.handle(_im("@nlp-f1 rerun arm 3")))
        assert asyncio.run(bridge.handle(_im("and arm 4"))) is True

        assert [m.text for m in drain_inbox(cli.peer_id)] == ["rerun arm 3", "and arm 4"]

    def test_off_hands_plain_lines_back_to_the_gateway_agent(self):
        _cli("nlp-f1")
        bridge, _ = _bridge()

        asyncio.run(bridge.handle(_im("@nlp-f1 hello")))
        asyncio.run(bridge.handle(_im("@off")))

        assert asyncio.run(bridge.handle(_im("what is the weather"))) is False

    def test_an_unknown_name_is_answered_with_the_live_listing(self):
        """On a phone the next thing needed is the name to retry with."""
        _cli("nlp-f1")
        bridge, channel = _bridge()

        asyncio.run(bridge.handle(_im("@nlp-f9 hello")))

        reply = channel.sent[-1][1]
        assert "No live session matches 'nlp-f9'" in reply
        assert "nlp-f1" in reply

    def test_with_nothing_running_the_reply_names_the_directory_searched(self):
        """An AGENTICA_HOME / OS-user mismatch looks exactly like having no
        session open, so the only cure is saying where it looked."""
        bridge, channel = _bridge()

        asyncio.run(bridge.handle(_im("@list")))

        assert str(peers.peers_root()) in channel.sent[-1][1]

    def test_a_refusal_from_the_channel_is_passed_through_verbatim(self):
        cli = _cli("nlp-f1")
        bridge, channel = _bridge()
        for i in range(peers.MAX_UNREAD):
            peers.send_message(
                cli.info, text=f"filler {i}", from_name="other", from_peer_id="ffff"
            )

        asyncio.run(bridge.handle(_im("@nlp-f1 rerun arm 3")))

        reply = channel.sent[-1][1]
        assert "Not delivered:" in reply
        assert "it is not reading them" in reply


class TestGuards:
    def test_it_refuses_to_relay_for_a_channel_open_to_everyone(self):
        """An empty allowlist means "everyone" — fine for chatting with the
        gateway's agent, not for typing into this machine's terminals."""
        cli = _cli("nlp-f1")
        bridge, channel = _bridge(allowed_users=())

        handled = asyncio.run(bridge.handle(_im("@nlp-f1 rm -rf /")))

        assert handled is True
        assert drain_inbox(cli.peer_id) == []
        reply = channel.sent[-1][1]
        assert "WECOM_ALLOWED_USERS" in reply
        assert "xuming" in reply

    def test_a_sender_outside_the_allowlist_is_refused(self):
        cli = _cli("nlp-f1")
        bridge, channel = _bridge(allowed_users=("xuming",))

        asyncio.run(bridge.handle(_im("@nlp-f1 hello", sender_id="stranger")))

        assert drain_inbox(cli.peer_id) == []
        assert "not in wecom's allowed_users" in channel.sent[-1][1]

    def test_the_bridge_never_lists_or_addresses_its_own_endpoints(self):
        """Each IM user is published as a peer so a CLI can answer it; nobody
        wants to send their own phone a message."""
        _cli("nlp-f1")
        bridge, channel = _bridge()

        asyncio.run(bridge.handle(_im("@list")))
        listing = channel.sent[-1][1]
        asyncio.run(bridge.handle(_im("@wecom-xuming hello")))

        assert "wecom-xuming" not in listing
        assert "No live session matches 'wecom-xuming'" in channel.sent[-1][1]


class TestReplies:
    def test_a_cli_reply_is_pushed_back_to_the_chat_it_came_from(self):
        cli = _cli("nlp-f1")
        bridge, channel = _bridge()
        asyncio.run(bridge.handle(_im("@nlp-f1 rerun arm 3")))

        cli.send("wecom-xuming", "arm 3 done: win rate 0.63")
        asyncio.run(bridge._drain_once())

        chat_id, body = channel.sent[-1]
        assert chat_id == "chat-1"
        assert "nlp-f1" in body
        assert "win rate 0.63" in body

    def test_shutdown_takes_the_endpoint_out_of_every_cli_listing(self):
        cli = _cli("nlp-f1")
        bridge, _ = _bridge()
        asyncio.run(bridge.handle(_im("@nlp-f1 hello")))
        assert [p.name for p in cli.list_peers()] == ["wecom-xuming"]

        asyncio.run(bridge.stop())

        assert cli.list_peers() == []
