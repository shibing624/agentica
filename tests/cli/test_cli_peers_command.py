# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Sending a message into a live session from a process that is not
itself a session — the `agentica peers send` path.

The two things worth pinning are the ones a consumer gets wrong silently:
the message must arrive as *the user* (authority), and an unresolvable target
must fail loudly instead of going nowhere.
"""

from __future__ import annotations

import pytest

from agentica import peers
from agentica.cli.peers_cmd import _default_from_name, _render_peers, run_peers_command
from agentica.peers import (
    DELIVERY_QUEUE,
    DELIVERY_STEER,
    PeerMessage,
    PeerMessageRefused,
    PeerSession,
    format_for_model,
    list_live_peers,
    send_to_live_peer,
)


@pytest.fixture(autouse=True)
def isolated_peers_root(tmp_path, monkeypatch):
    monkeypatch.setattr(peers, "AGENTICA_CACHE_DIR", str(tmp_path))
    yield tmp_path


def _session(name, *, cwd="/tmp/proj"):
    session = PeerSession(name=name, cwd=cwd)
    session.publish()
    return session


class _Args:
    """The argparse result `agentica peers send` produces."""

    def __init__(self, **kw):
        self.command = "peers"
        self.peers_command = kw.pop("peers_command", "send")
        self.to = kw.pop("to", None)
        self.text = kw.pop("text", None)
        self.text_flag = kw.pop("text_flag", None)
        self.delivery = kw.pop("delivery", DELIVERY_STEER)
        self.from_name = kw.pop("from_name", None)
        for key, value in kw.items():
            setattr(self, key, value)


class _Con:
    def __init__(self):
        self.lines = []

    def print(self, *args, **kwargs):
        self.lines.append(" ".join(str(a) for a in args))

    @property
    def text(self):
        return "\n".join(self.lines)


class TestSendToALivePeer:
    def test_the_message_lands_in_the_targets_mailbox(self):
        receiver = _session("tmux-cli")

        sent = send_to_live_peer("tmux-cli", "run the tests", from_name="vpet")

        assert sent.to_peer_id == receiver.peer_id
        assert [m.text for m in receiver.drain()] == ["run the tests"]

    def test_it_arrives_as_the_user_not_as_an_agent(self):
        """The receiving session grants a `user` message the human's authority.

        An external surface the person is driving (a pet, a script they ran) must
        not be delivered as another agent's information: that would silently
        demote their instruction to a suggestion.
        """
        receiver = _session("tmux-cli")

        send_to_live_peer("tmux-cli", "stop and commit", from_name="vpet")

        drained = receiver.drain()
        assert drained[0].from_user
        assert "treat as their instruction" in format_for_model(drained)

    def test_the_sender_a_peer_id_resolves_can_address_a_reply(self):
        """The /n, send_message-tool and gateway paths all look like this."""
        _session("dispatcher")
        receiver = _session("tmux-cli")

        send_to_live_peer(
            "tmux-cli", "do X", from_name="dispatcher", from_peer_id="abcd1234"
        )

        rendered = format_for_model(receiver.drain())
        assert "report back with send_message to dispatcher" in rendered

    def test_case_insensitive_name(self):
        _session("Tmux-CLI")
        receiver = peers.resolve_peer("tmux-cli")
        assert receiver is not None

    def test_a_peer_id_works_as_the_target(self):
        receiver = _session("tmux-cli")

        send_to_live_peer(receiver.peer_id, "by id", from_name="vpet")

        assert [m.text for m in receiver.drain()] == ["by id"]

    def test_a_session_id_prefix_works_as_the_target(self):
        receiver = PeerSession(name="tmux-cli", cwd="/tmp/proj",
                               session_id="e549ef44-4836-47e4-81e5-92fe60e561f7")
        receiver.publish()

        send_to_live_peer("e549ef44", "by session prefix", from_name="vpet")

        assert [m.text for m in receiver.drain()] == ["by session prefix"]


class TestTheAddressingErrors:
    """Loud, and different per cause: "no such session" and "be more specific"
    have different fixes."""

    def test_an_unknown_target_names_the_way_to_look(self):
        _session("tmux-cli")

        with pytest.raises(PeerMessageRefused) as exc:
            send_to_live_peer("nothing-by-that-name", "hi", from_name="vpet")

        assert "no live session matches" in str(exc.value)
        assert "peers list" in str(exc.value)

    def test_an_ambiguous_prefix_lists_the_candidates(self):
        _session("proj-a")
        _session("proj-b")

        with pytest.raises(PeerMessageRefused) as exc:
            send_to_live_peer("proj", "hi", from_name="vpet")

        message = str(exc.value)
        assert "matches 2 live sessions" in message
        assert "proj-a" in message and "proj-b" in message
        # The fix is a more precise address, so say so.
        assert "peer id" in message

    def test_an_empty_target_is_refused_not_sent_to_everyone(self):
        _session("tmux-cli")

        with pytest.raises(PeerMessageRefused):
            send_to_live_peer("", "hi", from_name="vpet")

    def test_a_missing_sender_name_is_refused(self):
        """An unattributed instruction is worse than none: the receiving session
        cannot tell who is talking."""
        _session("tmux-cli")

        with pytest.raises(PeerMessageRefused) as exc:
            send_to_live_peer("tmux-cli", "hi", from_name="   ")

        assert "from_name is required" in str(exc.value)

    def test_empty_text_is_refused(self):
        _session("tmux-cli")

        with pytest.raises(PeerMessageRefused):
            send_to_live_peer("tmux-cli", "   ", from_name="vpet")

    def test_a_dead_session_is_not_a_target(self, monkeypatch):
        """A live record whose process is gone is reaped, so the sender gets the
        unknown-target error rather than a message into a mailbox nobody reads."""
        session = _session("tmux-cli")
        session.info.pid = 2 ** 22  # no such process
        session.publish()

        with pytest.raises(PeerMessageRefused):
            send_to_live_peer("tmux-cli", "hi", from_name="vpet")

    def test_the_targets_own_limits_still_apply(self):
        """The message-file limits belong to peers.py, not to this entry point,
        so a second implementation of them cannot drift."""
        from agentica.peers import MAX_MESSAGE_CHARS

        _session("tmux-cli")

        with pytest.raises(PeerMessageRefused) as exc:
            send_to_live_peer("tmux-cli", "x" * (MAX_MESSAGE_CHARS + 1), from_name="vpet")

        assert "over the" in str(exc.value)


class TestDeliveryAndTheWireFormat:
    def test_steer_is_the_default(self):
        receiver = _session("tmux-cli")
        sent = send_to_live_peer("tmux-cli", "now", from_name="vpet")
        assert sent.delivery == DELIVERY_STEER
        assert receiver.drain(delivery=DELIVERY_STEER)

    def test_queue_waits_for_the_current_run(self):
        receiver = _session("tmux-cli")
        send_to_live_peer("tmux-cli", "after this", from_name="vpet",
                          delivery=DELIVERY_QUEUE)

        # A running turn takes only steer mail; the queued one stays behind.
        assert receiver.drain(delivery=DELIVERY_STEER) == []
        assert [m.text for m in receiver.drain()] == ["after this"]

    def test_the_file_is_the_one_peers_py_owns(self):
        """No second writer of the format: this must round-trip through parse()."""
        receiver = _session("tmux-cli")

        sent = send_to_live_peer("tmux-cli", "round trip", from_name="vpet-desktop")

        box = peers.mailbox_dir(receiver.peer_id)
        raw = next(box.glob("*.md")).read_text(encoding="utf-8")
        parsed = PeerMessage.parse(raw)
        assert parsed is not None
        assert parsed.text == "round trip"
        assert parsed.from_kind == "user"
        assert parsed.from_peer_id == sent.from_peer_id == ""

    def test_the_mailbox_is_private(self):
        receiver = _session("tmux-cli")
        send_to_live_peer("tmux-cli", "secret", from_name="vpet")

        box = peers.mailbox_dir(receiver.peer_id)
        assert (box.stat().st_mode & 0o777) == 0o700
        assert (next(box.glob("*.md")).stat().st_mode & 0o777) == 0o600

    def test_a_user_message_releases_the_send_brakes(self):
        """A relayed user instruction must not be refused by a brake built for an
        unattended agent loop (PeerSession.drain documents this)."""
        receiver = _session("tmux-cli")
        send_to_live_peer("tmux-cli", "typed by the human", from_name="vpet")

        before = receiver._recent_sends.get("somewhere", [])
        receiver.drain()
        assert receiver._recent_sends.get("somewhere", []) == before
        assert receiver.unread_count() == 0


class TestAnUnaddressableSenderIsNotAskedToBeRepliedTo:
    """`from_peer_id` empty means the sender is not a session on this machine.

    Naming it in a "report back with send_message to X" instruction sends the
    model to an address that never resolves.
    """

    def test_a_non_session_user_message_does_not_invite_a_reply(self):
        message = PeerMessage(
            text="run the tests",
            from_name="vpet-desktop",
            from_peer_id="",
            to_peer_id="beef",
            from_kind="user",
        )

        rendered = format_for_model([message])

        assert "send_message to" not in rendered
        assert "no reply address" in rendered
        assert "vpet-desktop" in rendered
        # The authority claim is what makes it an instruction; keep that.
        assert "treat as their instruction" in rendered

    def test_a_non_session_agent_message_does_not_invite_a_reply(self):
        message = PeerMessage(
            text="fyi", from_name="some-script", from_peer_id="", to_peer_id="beef"
        )

        rendered = format_for_model([message])

        assert "send_message to" not in rendered
        assert "no reply address" in rendered

    def test_an_addressable_sender_is_still_asked_to_be_replied_to(self):
        """The regression guard: the fix must not silence the reply instruction
        for senders that can actually be reached."""
        from_user = PeerMessage(
            text="go", from_name="wechat-41", from_peer_id="abcd1234",
            to_peer_id="beef", from_kind="user",
        )
        from_agent = PeerMessage(
            text="done", from_name="alpha", from_peer_id="abcd1234", to_peer_id="beef"
        )

        assert "report back with send_message to wechat-41" in format_for_model([from_user])
        assert "report the outcome back with send_message to alpha" in format_for_model([from_agent])


class TestTheCliCommand:
    def test_list_names_the_sessions_and_how_to_send(self):
        _session("tmux-cli")

        con = _Con()
        assert run_peers_command(_Args(peers_command="list"), con) == 0
        assert "tmux-cli" in con.text
        assert "agentica peers send" in con.text

    def test_list_with_nothing_running_says_so(self):
        con = _Con()
        assert run_peers_command(_Args(peers_command="list"), con) == 0
        assert "No live agentica sessions" in con.text

    def test_send_delivers_and_reports_the_resolved_name(self):
        receiver = _session("tmux-cli")

        con = _Con()
        code = run_peers_command(
            _Args(to="tmux-cli", text_flag="run the tests", from_name="vpet"), con
        )

        assert code == 0
        assert "tmux-cli" in con.text
        assert [m.text for m in receiver.drain()] == ["run the tests"]

    def test_send_accepts_the_text_positionally(self):
        receiver = _session("tmux-cli")

        con = _Con()
        assert run_peers_command(_Args(to="tmux-cli", text="positional"), con) == 0
        assert [m.text for m in receiver.drain()] == ["positional"]

    def test_an_unknown_target_exits_non_zero_with_the_reason(self):
        _session("tmux-cli")

        con = _Con()
        code = run_peers_command(_Args(to="nope", text_flag="hi"), con)

        assert code == 1
        assert "Not sent" in con.text
        assert "no live session matches" in con.text

    def test_an_ambiguous_target_exits_non_zero(self):
        _session("proj-a")
        _session("proj-b")

        con = _Con()
        code = run_peers_command(_Args(to="proj", text_flag="hi"), con)

        assert code == 1
        assert "matches 2 live sessions" in con.text

    def test_no_text_is_a_usage_error(self):
        _session("tmux-cli")
        con = _Con()
        assert run_peers_command(_Args(to="tmux-cli"), con) == 2
        assert "No message text" in con.text

    def test_both_text_forms_together_is_refused(self):
        """Picking one silently would drop the other; joining them would send a
        message nobody wrote."""
        _session("tmux-cli")
        con = _Con()
        code = run_peers_command(_Args(to="tmux-cli", text="a", text_flag="b"), con)
        assert code == 2
        assert "not both" in con.text

    def test_the_message_is_attributed_to_the_user(self):
        receiver = _session("tmux-cli")

        run_peers_command(_Args(to="tmux-cli", text_flag="go", from_name="vpet"), _Con())

        drained = receiver.drain()
        assert drained[0].from_kind == "user"

    def test_a_text_that_looks_like_a_flag_is_sendable(self):
        receiver = _session("tmux-cli")

        run_peers_command(
            _Args(to="tmux-cli", text_flag="--not-a-flag, just my message"), _Con()
        )

        assert [m.text for m in receiver.drain()] == ["--not-a-flag, just my message"]

    def test_the_default_sender_is_the_hostname(self):
        assert _default_from_name() not in ("", "external")
        receiver = _session("tmux-cli")

        run_peers_command(_Args(to="tmux-cli", text_flag="no name given"), _Con())

        assert receiver.drain()[0].from_name == _default_from_name()


class TestTheParser:
    """Drive the real argparse setup, since the flags are the public contract."""

    def _parse(self, argv):
        import sys
        from agentica.cli.runtime import parse_args

        original = sys.argv
        try:
            sys.argv = ["agentica", "peers", *argv]
            return parse_args()
        finally:
            sys.argv = original

    def test_send_flags(self):
        args = self._parse(["send", "--to", "tmux-cli", "--text", "hello"])
        assert args.command == "peers"
        assert args.peers_command == "send"
        assert args.to == "tmux-cli"
        assert args.text_flag == "hello"
        assert args.delivery == DELIVERY_STEER

    def test_delivery_choices_come_from_peers_py(self):
        args = self._parse(["send", "--to", "x", "--text", "y", "--delivery", "queue"])
        assert args.delivery == DELIVERY_QUEUE

    def test_an_unknown_delivery_is_rejected(self):
        with pytest.raises(SystemExit):
            self._parse(["send", "--to", "x", "--text", "y", "--delivery", "later"])

    def test_to_is_required(self):
        with pytest.raises(SystemExit):
            self._parse(["send", "--text", "hello"])

    def test_list_needs_no_arguments(self):
        args = self._parse(["list"])
        assert args.peers_command == "list"

    def test_a_subcommand_is_required(self):
        with pytest.raises(SystemExit):
            self._parse([])


class TestRenderPeers:
    def test_a_busy_session_says_a_message_still_lands(self):
        """`busy` reads as the price of sending, not as capacity — a running
        session takes a message between tool calls."""
        session = _session("tmux-cli")
        session.publish(busy=True, task="refactoring the parser")

        rendered = _render_peers(list_live_peers())

        assert "mid-turn" in rendered
        assert "refactoring the parser" in rendered

    def test_the_peer_id_survives_a_rich_console(self):
        """The id is in brackets, and rich reads brackets as a style tag.

        Found by scripts/verify_peers_send_e2e.py, which greps the real command's
        output for the id and could not find it: the listing showed the name and
        silently dropped the one address that always works. Pinned through the
        real ``Console`` because ``_render_peers`` alone was never wrong.
        """
        from io import StringIO

        from rich.console import Console

        session = _session("tmux-cli")
        buf = StringIO()

        run_peers_command(_Args(peers_command="list"), Console(file=buf, width=200))

        printed = buf.getvalue()
        assert session.peer_id in printed
        assert f"[peer={session.peer_id}]" in printed

    def test_the_send_receipt_survives_a_rich_console(self):
        """Same trap on the other command: the receipt names the resolved peer."""
        from io import StringIO

        from rich.console import Console

        session = _session("tmux-cli")
        buf = StringIO()

        run_peers_command(
            _Args(to="tmux-cli", text_flag="hello", from_name="vpet"),
            Console(file=buf, width=200),
        )

        assert session.peer_id in buf.getvalue()

    def test_a_path_with_brackets_is_not_eaten(self):
        """Same cause, different data: cwd is user data too."""
        from io import StringIO

        from rich.console import Console

        _session("odd", cwd="/tmp/[weird] place")
        buf = StringIO()

        run_peers_command(_Args(peers_command="list"), Console(file=buf, width=200))

        assert "[weird] place" in buf.getvalue()
