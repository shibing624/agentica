"""Regression tests for orphan ``tool_use`` (assistant tool call with no result).

An assistant message carrying a tool call must be answered by a result for
every id it issued. When a run is interrupted between "tool_use appended" and
"tool_result appended", the orphan stays in the in-memory array and every later
request in that run is rejected:

  - Anthropic: 400 "`tool_use` ids were found without `tool_result` blocks"
  - OpenAI chat: an assistant message with tool_calls must be followed by tool
    messages responding to each tool_call_id
  - OpenAI Responses: no tool output found for function call

These cover the three fixes: the runner keeps the round atomic, the Anthropic
formatter pairs by id and pads gaps, and the error is classified as a
transcript-shape error (repairable) rather than a transient (blindly retried).
"""

import asyncio
import unittest

from agentica.model.anthropic.claude import Claude
from agentica.model.loop_state import LoopState
from agentica.model.message import Message
from agentica.runner.persist import PersistMixin


ANTHROPIC_ORPHAN_ERROR = (
    "Anthropic Claude bad request: messages.231: `tool_use` ids were found "
    "without `tool_result` blocks immediately after: toolu_01Ju3. Each "
    "`tool_use` block must have a corresponding `tool_result` block in the "
    "next message."
)


def _tool_result_ids(message):
    if message.role == "tool" and message.tool_call_id:
        return {message.tool_call_id}
    if message.role != "user" or not isinstance(message.content, list):
        return set()
    return {
        b.get("tool_use_id")
        for b in message.content
        if isinstance(b, dict) and b.get("type") == "tool_result"
    }


class TestAnthropicResultPairing(unittest.TestCase):
    """``format_function_call_results`` must pair by id and pad gaps."""

    def setUp(self):
        self.model = Claude(id="claude-opus-5", api_key="fake-key")

    def test_pairs_by_id_not_by_position(self):
        """A result list ordered differently from tool_ids must not be mislabelled.

        Parallel tools can finish out of order. Zipping by index would attach
        tool A's output to tool B's id -- silently wrong, and never detected by
        the provider because the shape is still valid.
        """
        messages = []
        self.model.format_function_call_results(
            function_call_results=[
                Message(role="tool", tool_call_id="id_b", content="B output"),
                Message(role="tool", tool_call_id="id_a", content="A output"),
            ],
            tool_ids=["id_a", "id_b"],
            messages=messages,
        )
        self.assertEqual(len(messages), 1)
        blocks = messages[0].content
        by_id = {b["tool_use_id"]: b["content"] for b in blocks}
        self.assertEqual(by_id["id_a"], "A output")
        self.assertEqual(by_id["id_b"], "B output")

    def test_pads_missing_result_so_no_id_is_orphaned(self):
        """An interrupted round still answers every id it issued."""
        messages = []
        self.model.format_function_call_results(
            function_call_results=[
                Message(role="tool", tool_call_id="id_a", content="A output"),
            ],
            tool_ids=["id_a", "id_b"],
            messages=messages,
        )
        blocks = messages[0].content
        self.assertEqual(
            {b["tool_use_id"] for b in blocks},
            {"id_a", "id_b"},
            "every issued id must be answered, even when its tool never ran",
        )
        padded = [b for b in blocks if b["tool_use_id"] == "id_b"][0]
        self.assertTrue(padded.get("is_error"))
        self.assertIn("interrupted", padded["content"])

    def test_no_results_at_all_still_answers_every_id(self):
        """Cancelled before any tool finished: the whole round is padded."""
        messages = []
        self.model.format_function_call_results(
            function_call_results=[],
            tool_ids=["id_a", "id_b"],
            messages=messages,
        )
        self.assertEqual(len(messages), 1)
        self.assertEqual(
            {b["tool_use_id"] for b in messages[0].content}, {"id_a", "id_b"}
        )

    def test_nothing_expected_and_nothing_produced_appends_nothing(self):
        messages = []
        self.model.format_function_call_results([], [], messages)
        self.assertEqual(messages, [])

    def test_error_results_are_flagged_is_error(self):
        messages = []
        self.model.format_function_call_results(
            function_call_results=[
                Message(
                    role="tool",
                    tool_call_id="id_a",
                    content="boom",
                    tool_call_error=True,
                ),
            ],
            tool_ids=["id_a"],
            messages=messages,
        )
        self.assertTrue(messages[0].content[0].get("is_error"))


class TestRunnerRoundAtomicity(unittest.TestCase):
    """A cancel during tool execution must not leave the tool_use unanswered."""

    def test_cancel_mid_execution_still_appends_results(self):
        from agentica.runner.loop import LoopMixin

        class FakeModel(Claude):
            pass

        model = FakeModel(id="claude-opus-5", api_key="fake-key")
        assistant = Message(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "toolu_orphan",
                    "type": "function",
                    "function": {"name": "execute", "arguments": "{}"},
                }
            ],
        )
        messages = [Message(role="user", content="go"), assistant]

        runner = LoopMixin.__new__(LoopMixin)

        # parse_tool_calls needs a resolvable function; stub the pieces the
        # method under test depends on so the test stays about the window.
        def fake_parse(assistant_message, msgs, tool_role="tool"):
            return ["sentinel_call"], {"tool_ids": ["toolu_orphan"], "tool_role": "tool"}

        model.parse_tool_calls = fake_parse  # type: ignore[assignment]

        async def exploding_execute(**kwargs):
            # Simulate the long-running tool being cancelled mid-flight, which
            # is what a 34-minute execute + consumer teardown produced.
            raise asyncio.CancelledError()
            yield  # pragma: no cover

        runner._execute_tool_calls = lambda **kw: exploding_execute(**kw)  # type: ignore[assignment]

        async def drive():
            agen = runner._handle_tool_calls_in_runner_stream(messages, object(), model)
            async for _ in agen:
                pass

        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(drive())

        answered = set()
        for m in messages:
            answered |= _tool_result_ids(m)
        self.assertIn(
            "toolu_orphan",
            answered,
            "the tool_use must be answered even when execution was cancelled, "
            "otherwise every later request in the run is rejected",
        )


class TestOrphanErrorClassification(unittest.TestCase):
    """The provider rejection must be treated as a shape error, not a transient."""

    def setUp(self):
        self.state = LoopState()
        self.err = ANTHROPIC_ORPHAN_ERROR.lower()

    def test_recognised_as_tool_history_error(self):
        self.assertTrue(
            any(h in self.err for h in self.state.TOOL_HISTORY_HINTS),
            "an unanswered tool_use must route to the repair path",
        )

    def test_not_treated_as_retryable_transient(self):
        """Blindly re-sending the same array reproduces the 400 identically."""
        self.assertFalse(
            any(r in self.err for r in self.state.RETRYABLE_SUBSTRINGS),
            "re-issuing an unrepaired transcript turns a recoverable drop into "
            "a hard failure",
        )
        self.assertFalse(any(r in self.err for r in self.state.FALLBACK_ONLY_SUBSTRINGS))
        self.assertFalse(any(r in self.err for r in self.state.PROMPT_TOO_LONG_HINTS))

    def test_openai_wordings_also_recognised(self):
        for msg in (
            "an assistant message with 'tool_calls' must be followed by tool "
            "messages responding to each 'tool_call_id'",
            "no tool output found for function call call_abc123",
        ):
            self.assertTrue(
                any(h in msg.lower() for h in self.state.TOOL_HISTORY_HINTS),
                f"unrecognised: {msg}",
            )


class TestRepairUnansweredToolCalls(unittest.TestCase):
    """The repair pads missing results wherever the orphan sits."""

    def test_repairs_orphan_deep_in_history_keeping_transcript(self):
        messages = [
            Message(role="user", content="start"),
            Message(
                role="assistant",
                content="",
                tool_calls=[{"id": "old_orphan", "function": {"name": "execute"}}],
            ),
            # no result for old_orphan -- this is the poisoned round
            Message(role="assistant", content="", tool_calls=[{"id": "good", "function": {"name": "read"}}]),
            Message(
                role="user",
                content=[{"type": "tool_result", "tool_use_id": "good", "content": "ok"}],
            ),
            Message(role="assistant", content="done"),
        ]
        before = len(messages)
        repaired = PersistMixin.repair_unanswered_tool_calls(messages, anthropic_blocks=True)
        self.assertEqual(repaired, 1)
        self.assertEqual(len(messages), before + 1, "repair inserts, never deletes")

        answered = set()
        for m in messages:
            answered |= _tool_result_ids(m)
        self.assertIn("old_orphan", answered)
        self.assertIn("good", answered)
        # The transcript survives: the nuclear strip would have removed these.
        self.assertTrue(any(m.role == "assistant" and m.content == "done" for m in messages))

    def test_openai_shape_uses_tool_role_messages(self):
        messages = [
            Message(
                role="assistant",
                content="",
                tool_calls=[{"id": "call_1", "function": {"name": "execute"}}],
            ),
        ]
        repaired = PersistMixin.repair_unanswered_tool_calls(messages, anthropic_blocks=False)
        self.assertEqual(repaired, 1)
        self.assertEqual(messages[1].role, "tool")
        self.assertEqual(messages[1].tool_call_id, "call_1")

    def test_clean_history_is_untouched(self):
        messages = [
            Message(role="assistant", content="", tool_calls=[{"id": "a", "function": {"name": "x"}}]),
            Message(role="user", content=[{"type": "tool_result", "tool_use_id": "a", "content": "ok"}]),
        ]
        snapshot = list(messages)
        self.assertEqual(
            PersistMixin.repair_unanswered_tool_calls(messages, anthropic_blocks=True), 0
        )
        self.assertEqual(messages, snapshot)

    def test_partially_answered_round_pads_only_the_gap(self):
        messages = [
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    {"id": "a", "function": {"name": "x"}},
                    {"id": "b", "function": {"name": "y"}},
                ],
            ),
            Message(role="user", content=[{"type": "tool_result", "tool_use_id": "a", "content": "ok"}]),
        ]
        self.assertEqual(
            PersistMixin.repair_unanswered_tool_calls(messages, anthropic_blocks=True), 1
        )
        answered = set()
        for m in messages:
            answered |= _tool_result_ids(m)
        self.assertEqual(answered, {"a", "b"})


if __name__ == "__main__":
    unittest.main()
