# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: How an arriving peer message is rendered in the receiving terminal.
"""
import os
from unittest.mock import MagicMock, patch

from rich.console import Console

from agentica.cli.commands.context import PendingQueue
from agentica.cli.display.messages import display_peer_messages
from agentica.cli.interactive.attachments import (
    queue_item_preview,
    unpack_queue_payload,
)
from agentica.cli.interactive.btw import hand_to_agent
from agentica.cli.interactive.session_state import SessionState
from agentica.peers import PeerMessage

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")


def _render(messages) -> str:
    console = Console(width=100, force_terminal=False, no_color=True)
    with patch("agentica.cli.display.messages.get_console", return_value=console):
        with console.capture() as captured:
            display_peer_messages(messages)
    return captured.get()


def _message(text, *, from_kind="agent", from_name="nlp-f1") -> PeerMessage:
    return PeerMessage(
        text=text,
        from_name=from_name,
        from_peer_id="abcd1234",
        to_peer_id="beef5678",
        from_kind=from_kind,
    )


class TestPeerMessageRendering:
    def test_an_agent_message_is_labelled_with_the_sender(self):
        out = _render([_message("方案A全量+四臂完成")])

        assert "nlp-f1" in out
        assert "💬" in out
        assert "方案A全量+四臂完成" in out

    def test_an_agent_message_does_not_borrow_the_human_prompt_marker(self):
        """``❯`` means "the user said this"; agent traffic must not wear it."""
        out = _render([_message("I refactored the loader")])

        assert "❯" not in out

    def test_a_relayed_user_message_is_shown_as_the_user_speaking(self):
        out = _render([_message("ship it", from_kind="user")])

        assert "❯" in out
        assert "via nlp-f1" in out
        assert "ship it" in out

    def test_the_model_facing_header_is_not_what_the_user_reads(self):
        """The bracketed authority header belongs in the model's context only."""
        out = _render([_message("done")])

        assert "reply with send_message" not in out
        assert "[Message from another agent session" not in out

    def test_a_long_body_is_not_truncated_with_an_ellipsis(self):
        body = " ".join(f"item-{i}" for i in range(80))
        out = _render([_message(body)])

        assert "item-79" in out
        assert "…" not in out


class TestRelayedTurnsAreNotEchoedTwice:
    def test_queued_relayed_text_is_tagged_so_the_echo_can_be_skipped(self):
        state = SessionState()
        state.agent_running = False
        state.current_agent = MagicMock()
        pending = PendingQueue()

        hand_to_agent(state, pending, "[Message from another agent session] hi")

        assert pending.peek_all() == [
            ("__RELAYED__", "[Message from another agent session] hi")
        ]

    def test_the_queue_bar_previews_the_text_not_the_marker(self):
        preview = queue_item_preview(("__RELAYED__", "schema migration finished"))

        assert preview == "schema migration finished"


class TestQueuePayloadClassification:
    """``unpack_queue_payload`` is the one place that answers "who typed this".

    The process loop reads ``is_relayed`` to skip the echo and to refuse slash
    command dispatch; both call sites are inside ``run_interactive``'s closure
    and are covered by this classification plus the guards being a single
    expression each.
    """

    def test_a_typed_line_is_neither_btw_nor_relayed(self):
        queued = unpack_queue_payload("/compact")

        assert queued.text == "/compact"
        assert not queued.is_btw
        assert not queued.is_relayed
        assert queued.images == []

    def test_a_typed_line_keeps_its_image_attachments(self):
        queued = unpack_queue_payload(("look at this", ["/tmp/a.png"]))

        assert queued.text == "look at this"
        assert queued.images == ["/tmp/a.png"]

    def test_a_btw_payload_is_marked_ephemeral(self):
        queued = unpack_queue_payload(("__BTW__", "what is the cwd"))

        assert queued.is_btw
        assert not queued.is_relayed

    def test_a_relayed_slash_command_stays_text(self):
        """PEER_MESSAGING_POLICY promises the sender that a slash command in a
        peer message is plain text; this is what makes that true."""
        queued = unpack_queue_payload(("__RELAYED__", "/compact now please"))

        assert queued.is_relayed
        assert queued.text == "/compact now please"
