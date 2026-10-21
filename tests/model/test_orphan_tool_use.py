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
        repaired = PersistMixin.repair_unanswered_tool_calls(messages)
        self.assertEqual(repaired, 1)
        self.assertEqual(len(messages), before + 1, "repair inserts, never deletes")

        answered = set()
        for m in messages:
            answered |= _tool_result_ids(m)
        self.assertIn("old_orphan", answered)
        self.assertIn("good", answered)
        # The transcript survives: the nuclear strip would have removed these.
        self.assertTrue(any(m.role == "assistant" and m.content == "done" for m in messages))

    def test_pad_is_a_tool_role_message_on_every_provider(self):
        messages = [
            Message(
                role="assistant",
                content="",
                tool_calls=[{"id": "call_1", "function": {"name": "execute"}}],
            ),
        ]
        repaired = PersistMixin.repair_unanswered_tool_calls(messages)
        self.assertEqual(repaired, 1)
        self.assertEqual(messages[1].role, "tool")
        self.assertEqual(messages[1].tool_call_id, "call_1")

    def test_clean_history_is_untouched(self):
        messages = [
            Message(role="assistant", content="", tool_calls=[{"id": "a", "function": {"name": "x"}}]),
            Message(role="user", content=[{"type": "tool_result", "tool_use_id": "a", "content": "ok"}]),
        ]
        snapshot = list(messages)
        self.assertEqual(PersistMixin.repair_unanswered_tool_calls(messages), 0)
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
        self.assertEqual(PersistMixin.repair_unanswered_tool_calls(messages), 1)
        answered = set()
        for m in messages:
            answered |= _tool_result_ids(m)
        self.assertEqual(answered, {"a", "b"})


class TestSanitizeMessagesUnderstandsBlockShape(unittest.TestCase):
    """``sanitize_messages`` must not call an answered Anthropic round unanswered.

    It scanned only for ``role="tool"`` replies, but Claude answers a tool call
    with ``role="user"`` carrying ``tool_result`` blocks. Every Claude tool
    result therefore looked missing, and a placeholder reading "execution may
    have been interrupted" was inserted *before* the real output -- telling the
    model its successful tool had failed, on every single call.
    """

    @staticmethod
    def _placeholders(msgs):
        from agentica.model.base import Model

        out = Model.sanitize_messages(list(msgs))
        return out, [
            m
            for m in out
            if m.role == "tool" and "did not return a response" in (m.content or "")
        ]

    def test_answered_anthropic_round_gets_no_placeholder(self):
        out, injected = self._placeholders(
            [
                Message(role="assistant", content="", tool_calls=[{"id": "t1", "function": {"name": "execute"}}]),
                Message(role="user", content=[{"type": "tool_result", "tool_use_id": "t1", "content": "real output"}]),
            ]
        )
        self.assertEqual(
            injected,
            [],
            "the result is right there in the next message; claiming it was "
            "interrupted feeds the model a false failure",
        )
        self.assertEqual(len(out), 2, "nothing should be inserted")

    def test_genuinely_unanswered_anthropic_round_still_padded(self):
        _out, injected = self._placeholders(
            [
                Message(role="assistant", content="", tool_calls=[{"id": "t1", "function": {"name": "execute"}}]),
                Message(role="assistant", content="next turn"),
            ]
        )
        self.assertEqual(len(injected), 1, "a real orphan must still be padded")

    def test_partially_answered_anthropic_round_pads_only_the_gap(self):
        _out, injected = self._placeholders(
            [
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {"id": "t1", "function": {"name": "a"}},
                        {"id": "t2", "function": {"name": "b"}},
                    ],
                ),
                Message(role="user", content=[{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]),
            ]
        )
        self.assertEqual(len(injected), 1)
        self.assertEqual(injected[0].tool_call_id, "t2")

    def test_openai_shape_is_unaffected(self):
        _out, injected = self._placeholders(
            [
                Message(role="assistant", content="", tool_calls=[{"id": "c1", "function": {"name": "x"}}]),
                Message(role="tool", tool_call_id="c1", content="ok"),
            ]
        )
        self.assertEqual(injected, [])

        _out, injected = self._placeholders(
            [
                Message(role="assistant", content="", tool_calls=[{"id": "c1", "function": {"name": "x"}}]),
            ]
        )
        self.assertEqual(len(injected), 1)

class TestPaddingHoldsOnEveryProvider(unittest.TestCase):
    """``format_tool_results`` must answer every issued id on ALL paths.

    The Runner calls this from a ``finally`` and its comment asserts the round
    is atomic. That guarantee was only true for Claude: the default
    implementation did a bare ``messages.extend`` and ``parse_tool_calls``
    returned no ``tool_ids``, so an interrupted round left orphans on the
    OpenAI/Ollama paths. They happened to survive because the per-request
    ``sanitize_messages`` bridged the gap later -- so removing or narrowing
    that pass would have broken them silently. Pinned here so the invariant
    the comment claims is the invariant the code has.
    """

    def _answered(self, model):
        assistant = Message(
            role="assistant",
            content="",
            tool_calls=[
                {"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
                {"id": "c3", "type": "function", "function": {"name": "c", "arguments": "{}"}},
            ],
        )
        _fc, meta = model.parse_tool_calls(assistant, [assistant], tool_role="tool")
        messages = [assistant]
        # Only the first call came back: the shape an interrupt produces.
        model.format_tool_results(
            [Message(role="tool", tool_call_id="c1", content="ok")], messages, meta
        )
        answered = set()
        for m in messages:
            answered |= _tool_result_ids(m)
        return answered

    def test_openai_chat_pads_interrupted_round(self):
        from agentica.model.openai.chat import OpenAIChat

        self.assertEqual(
            self._answered(OpenAIChat(id="gpt-4o", api_key="fake")), {"c1", "c2", "c3"}
        )

    def test_openai_responses_pads_interrupted_round(self):
        from agentica.model.openai.responses import OpenAIResponses

        self.assertEqual(
            self._answered(OpenAIResponses(id="gpt-4o", api_key="fake")), {"c1", "c2", "c3"}
        )

    def test_ollama_pads_interrupted_round(self):
        from agentica.model.ollama.chat import Ollama

        self.assertEqual(self._answered(Ollama(id="llama3")), {"c1", "c2", "c3"})

    def test_claude_pads_interrupted_round(self):
        self.assertEqual(
            self._answered(Claude(id="claude-opus-5", api_key="fake")), {"c1", "c2", "c3"}
        )

    def test_parse_tool_calls_carries_tool_ids_on_base_path(self):
        """Padding is impossible without the ids; pin them into the metadata."""
        from agentica.model.openai.chat import OpenAIChat

        assistant = Message(
            role="assistant",
            content="",
            tool_calls=[{"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{}"}}],
        )
        _fc, meta = OpenAIChat(id="gpt-4o", api_key="fake").parse_tool_calls(
            assistant, [assistant], tool_role="tool"
        )
        self.assertEqual(meta.get("tool_ids"), ["c1"])

    def test_fully_answered_round_is_not_padded(self):
        from agentica.model.openai.chat import OpenAIChat

        model = OpenAIChat(id="gpt-4o", api_key="fake")
        assistant = Message(
            role="assistant",
            content="",
            tool_calls=[{"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{}"}}],
        )
        _fc, meta = model.parse_tool_calls(assistant, [assistant], tool_role="tool")
        messages = [assistant]
        model.format_tool_results(
            [Message(role="tool", tool_call_id="c1", content="real output")], messages, meta
        )
        self.assertEqual(len(messages), 2, "nothing to pad, nothing appended")
        self.assertEqual(messages[1].content, "real output")


if __name__ == "__main__":
    unittest.main()


class _StrictAnthropicClient:
    """Fake ``client.messages`` that validates the request the way Anthropic does.

    Records every request and raises the provider's shape errors instead of
    answering, so a test fails on exactly what the live API rejected.
    """

    def __init__(self, rounds):
        self._rounds = iter(rounds)
        self.requests = []
        self.messages = self

    @staticmethod
    def shape_errors(chat_messages):
        errors = []
        if chat_messages[-1]["role"] != "user":
            errors.append("This model does not support assistant message prefill.")
        for i, message in enumerate(chat_messages):
            if message["role"] != "assistant":
                continue
            ids = {
                b["id"] for b in message["content"]
                if isinstance(b, dict) and b.get("type") == "tool_use"
            }
            following = chat_messages[i + 1]["content"] if i + 1 < len(chat_messages) else []
            answered = {
                b.get("tool_use_id") for b in following
                if isinstance(b, dict) and b.get("type") == "tool_result"
            }
            if ids - answered:
                errors.append(
                    "`tool_use` ids were found without `tool_result` blocks "
                    f"immediately after: {sorted(ids - answered)}"
                )
        return errors

    def stream(self, **kwargs):
        self.requests.append(kwargs["messages"])
        errors = self.shape_errors(kwargs["messages"])
        if errors:
            raise RuntimeError("; ".join(errors))
        events = list(next(self._rounds))

        class _Mgr:
            async def __aenter__(self_inner):
                class _Iter:
                    def __aiter__(self_it):
                        return self_it

                    async def __anext__(self_it):
                        if not events:
                            raise StopAsyncIteration
                        return events.pop(0)

                return _Iter()

            async def __aexit__(self_inner, *_a):
                return None

        return _Mgr()


class TestClaudeRefusedToolCallAnswersItsId(unittest.TestCase):
    """A call refused before it runs must answer its own tool_use id.

    Live failure (session 692116c3, 2026-10-21): the model echoed an evicted
    ``apply_patch`` payload, the call was refused, and Claude appended the
    refusal as plain user text. The tool_use was left without a tool_result,
    Anthropic rejected the next request, and the strip-and-retry recovery
    then sent a request ending in an assistant turn — the run died on
    "assistant message prefill".
    """

    @staticmethod
    def _events(*blocks, stop_reason):
        from unittest.mock import MagicMock
        from anthropic.types import RawContentBlockDeltaEvent, TextBlock, TextDelta
        from anthropic.lib.streaming._types import (
            ParsedContentBlockStopEvent,
            ParsedMessageStopEvent,
        )

        msg = MagicMock()
        msg.usage = MagicMock(
            input_tokens=1, output_tokens=1,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        )
        msg.stop_reason = stop_reason
        events = []
        for block in blocks:
            if isinstance(block, TextBlock):
                events.append(RawContentBlockDeltaEvent.model_construct(
                    type="content_block_delta", index=0,
                    delta=TextDelta(type="text_delta", text=block.text),
                ))
            events.append(ParsedContentBlockStopEvent.model_construct(
                type="content_block_stop", index=0, content_block=block
            ))
        return events + [ParsedMessageStopEvent.model_construct(type="message_stop", message=msg)]

    def _run(self, tool_uses):
        import json
        import tempfile
        from anthropic.types import TextBlock, ToolUseBlock
        from agentica.agent import Agent

        def echo(text: str) -> str:
            """Echo the text back."""
            return f"echo:{text}"

        client = _StrictAnthropicClient([
            self._events(
                *[ToolUseBlock(type="tool_use", id=i, name=n, input=a) for i, n, a in tool_uses],
                stop_reason="tool_use",
            ),
            self._events(TextBlock(type="text", text="done"), stop_reason="end_turn"),
        ])
        model = Claude(id="claude-opus-4-6", api_key="fake")
        model.client = client
        with tempfile.TemporaryDirectory() as tmp:
            agent = Agent(
                name="t", model=model, tools=[echo],
                session_id="s-refused", session_base_dir=tmp,
            )

            async def _drive():
                async for _ in agent.run_stream("go"):
                    pass

            asyncio.run(_drive())
        self.assertEqual(agent.run_response.content, "done")
        for request in client.requests:
            self.assertEqual(client.shape_errors(request), [], json.dumps(request, default=str))
        self.assertEqual(len(client.requests), 2, "no recovery retry may be needed")
        return client.requests[1]

    def test_evicted_payload_is_answered_with_the_refusal(self):
        from agentica.compression.tool_call_args import omitted_tool_arg

        second = self._run([("toolu_evicted", "echo", {"text": omitted_tool_arg(900)})])
        self.assertEqual([m["role"] for m in second], ["user", "assistant", "user"])
        (result,) = second[2]["content"]
        self.assertEqual(result["type"], "tool_result")
        self.assertEqual(result["tool_use_id"], "toolu_evicted")
        self.assertTrue(result["is_error"])
        self.assertIn("were dropped from context", result["content"])

    def test_refused_and_executed_calls_share_one_result_message(self):
        second = self._run([
            ("toolu_unknown", "no_such_tool", {}),
            ("toolu_echo", "echo", {"text": "hi"}),
        ])
        self.assertEqual([m["role"] for m in second], ["user", "assistant", "user"])
        results = {b["tool_use_id"]: b for b in second[2]["content"]}
        self.assertEqual(results["toolu_unknown"]["content"], "Could not find function to call.")
        self.assertTrue(results["toolu_unknown"]["is_error"])
        self.assertEqual(results["toolu_echo"]["content"], "echo:hi")


class TestStripRecoveryNeverEndsOnAssistant(unittest.TestCase):
    def test_turn_ending_in_a_tool_round_restarts_from_the_user(self):
        from types import SimpleNamespace

        agent = SimpleNamespace(working_memory=SimpleNamespace(runs=[]))
        messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="do all"),
            Message(
                role="assistant",
                content="patching now",
                tool_calls=[{"id": "t1", "type": "function",
                             "function": {"name": "apply_patch", "arguments": "{}"}}],
            ),
            Message(role="tool", tool_call_id="t1", content="ok"),
        ]
        PersistMixin._sanitize_tool_history_after_error(agent, messages)
        self.assertEqual([(m.role, m.content) for m in messages],
                         [("system", "sys"), ("user", "do all")])
