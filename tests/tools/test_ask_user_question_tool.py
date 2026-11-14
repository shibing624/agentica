# -*- coding: utf-8 -*-
"""Tests for AskUserQuestionTool."""
import asyncio
import json


class TestAskUserQuestionTool:
    def test_manages_own_timeout(self):
        """ask_user_question must wait indefinitely for the user (CC/Cursor
        semantics), not be auto-passed by the outer ~120s tool-executor timeout."""
        from agentica.tools.ask_user_question_tool import AskUserQuestionTool

        tool = AskUserQuestionTool(input_callback=lambda p, o=None: "ok")
        assert tool.functions["ask_user_question"].manages_own_timeout is True

    def test_uses_callback(self):
        import asyncio
        from agentica.tools.ask_user_question_tool import AskUserQuestionTool

        captured = {}

        def cb(prompt, options=None):
            captured["prompt"] = prompt
            captured["options"] = options
            return "my answer"

        tool = AskUserQuestionTool(input_callback=cb)
        result = json.loads(asyncio.run(tool.ask_user_question(prompt="What now?")))
        assert result["response"] == "my answer"
        assert "What now?" in captured["prompt"]

    def test_callback_less_instance_uses_registered_default(self):
        """A callback-less AskUserQuestionTool (subagent / cron / regression)
        must route through the process-wide default callback registered by the
        TUI, instead of deadlocking on bare input() while prompt_toolkit owns
        stdin."""
        import asyncio
        from agentica.tools.ask_user_question_tool import (
            AskUserQuestionTool,
            set_default_ask_user_question_callback,
        )

        captured = {}

        def default_cb(prompt, options=None):
            captured["prompt"] = prompt
            captured["options"] = options
            return "default answer"

        set_default_ask_user_question_callback(default_cb)
        try:
            tool = AskUserQuestionTool()  # no explicit callback
            assert tool.input_callback is None
            result = json.loads(asyncio.run(tool.ask_user_question(prompt="Pick")))
            assert result["response"] == "default answer"
            assert "Pick" in captured["prompt"]
        finally:
            set_default_ask_user_question_callback(None)

    def test_no_callback_no_default_falls_back_to_input(self):
        """Without the TUI registered, a callback-less tool keeps the legacy
        bare-input() behavior (non-interactive scripts)."""
        from agentica.tools.ask_user_question_tool import (
            AskUserQuestionTool,
            set_default_ask_user_question_callback,
        )

        set_default_ask_user_question_callback(None)
        tool = AskUserQuestionTool()
        assert tool.input_callback is None

    def test_reply_is_returned_verbatim_with_the_options_offered(self):
        """The user's wording reaches the model untouched.

        A reply carrying a choice plus its rationale ("3 , 100题, …workers=10
        是ok的吧") is one thing to read in context, and the model doing the
        reading is already in this turn. Anything that rewrites it — an old
        isdigit/prefix rule that silently fell back to ``options[0]``, or a
        second LLM pass that "resolves" the wording — can only lose or invent
        detail. The options ride along so the reply is not an orphan label.
        """
        import asyncio
        from agentica.tools.ask_user_question_tool import AskUserQuestionTool

        options = [
            "100 题（~3.5 小时，足够验证 30 题结论）（推荐）",
            "198 题（与现有主表同口径）",
            "316 题（全量，~11 小时）",
        ]
        raw = "3 , 100题, qwen,bge服务支持100并发,我们的workers=10是ok的吧"
        tool = AskUserQuestionTool(input_callback=lambda p, o=None: raw)
        result = json.loads(asyncio.run(
            tool.ask_user_question(prompt="选哪个规模？", options=options)
        ))
        assert result["response"] == raw
        assert result["options"] == options

    def test_options_are_shown_in_the_order_given(self):
        """Ordering is the model's call — a recommendation is just the label it
        put first, so the tool must not reshuffle the list the user sees."""
        import asyncio
        from agentica.tools.ask_user_question_tool import AskUserQuestionTool

        shown = {}

        def cb(prompt, options=None):
            shown["options"] = options
            return "1"

        options = ["JSON (recommended)", "CSV", "XML"]
        tool = AskUserQuestionTool(input_callback=cb)
        result = json.loads(asyncio.run(
            tool.ask_user_question(prompt="format?", options=options)
        ))
        assert shown["options"] == options
        assert result["response"] == "1"

    def test_schema_asks_for_both_position_and_a_label(self):
        """Ordering alone does not survive the readback.

        The user picks by position, so "put your recommendation first" is not
        enough: when the answer arrives the model sees "1", not which choice it
        meant. The tool deliberately does not add or reorder anything (the
        label is the model's own words), so the schema has to ask for it —
        dropping "and say so in its label" is what let the recommended marker
        disappear from real option lists.
        """
        from agentica.tools.ask_user_question_tool import AskUserQuestionTool

        tool = AskUserQuestionTool(input_callback=lambda p, o=None: "1")
        fn = tool.functions["ask_user_question"]
        fn.process_entrypoint()
        description = fn.parameters["properties"]["options"]["description"]

        assert "first" in description
        # The marker must be asked for by example, and the examples must match
        # what this repo's own tests/prompts actually use.
        assert "label" in description
        assert "（推荐）" in description
        assert "(recommended)" in description

    def test_schema_says_options_is_an_array_not_a_string(self):
        """The first call used to stringify the list because the schema had
        no property description and the docstring showed a Python list."""
        from agentica.tools.ask_user_question_tool import AskUserQuestionTool

        tool = AskUserQuestionTool(input_callback=lambda p, o=None: "1")
        fn = tool.functions["ask_user_question"]
        fn.process_entrypoint()
        options = fn.parameters["properties"]["options"]
        assert options["type"] == "array"
        assert options["items"] == {"type": "string"}
        assert "not a string" in options["description"]
        assert "options=[" not in (fn.description or "")
        assert "options=[" not in (tool.get_system_prompt() or "")

    def test_stringified_options_are_coerced_before_validate_call(self):
        """Models often emit options as a JSON string on the first call.

        pydantic validate_call then raised ``Input should be a valid list``
        and the user saw a failed tool before the retry got the type right.
        """
        import os

        os.environ.setdefault("OPENAI_API_KEY", "test-key")
        from agentica.tools.ask_user_question_tool import AskUserQuestionTool
        from agentica.tools.base import get_function_call

        captured = {}

        def cb(prompt, options=None):
            captured["options"] = options
            return "1"

        tool = AskUserQuestionTool(input_callback=cb)
        tool.functions["ask_user_question"].process_entrypoint()
        options = [
            "稳定版 a364f26 为主表（推荐）：LME-198 配对 38.38% vs mem0 28.28%（+10.10pp）",
            "最新 main dcf6c58 为主表",
            "两版都进正文",
        ]
        call = get_function_call(
            name="ask_user_question",
            arguments=json.dumps(
                {"prompt": "用哪一套？", "options": json.dumps(options, ensure_ascii=False)},
                ensure_ascii=False,
            ),
            functions=tool.functions,
        )
        assert call is not None
        assert call.error is None
        assert call.arguments["options"] == options
        assert asyncio.run(call.execute()) is True
        assert captured["options"] == options


class TestAskUserQuestionTimeout:
    def test_callback_timeout_becomes_timed_out_result_without_an_answer(self):
        from agentica.tools.ask_user_question_tool import (
            AskUserQuestionTimeout,
            AskUserQuestionTool,
        )

        def cb(prompt, options=None):
            raise AskUserQuestionTimeout(300)

        tool = AskUserQuestionTool(input_callback=cb)
        result = json.loads(asyncio.run(tool.ask_user_question(
            prompt="Which?", options=["A（推荐）", "B"],
        )))
        from agentica.tools.ask_user_question_tool import TIMED_OUT_NOTE

        assert result == {
            "prompt": "Which?",
            "response": "",
            "timed_out": True,
            "waited_seconds": 300,
            "note": TIMED_OUT_NOTE,
            "options": ["A（推荐）", "B"],
        }
        # Silence licenses a reversible choice, never an irreversible step.
        assert "Silence is not consent" in TIMED_OUT_NOTE
        assert "do not do that step" in TIMED_OUT_NOTE

    def test_system_prompt_keeps_permission_questions_out_of_the_tool(self):
        from agentica.tools.ask_user_question_tool import AskUserQuestionTool

        text = AskUserQuestionTool().get_system_prompt()
        assert '"timed_out": true' in text
        assert "permission system" in text


class TestSeveralQuestionsInOneCall:
    """Batching: settle every ambiguity before starting work.

    The point is not a nicer prompt. A model unsure about three things used to
    ask, wait, ask, wait, ask — three interruptions, and the answer to the third
    arrived after it had already started the work the first two shaped. One call
    collects them in a single sitting. Surfaces keep their existing one-question
    callback, so this needed no UI change anywhere.
    """

    def test_each_question_is_asked_and_answered_in_order(self):
        from agentica.tools.ask_user_question_tool import AskUserQuestionTool

        seen = []

        def cb(prompt, options=None):
            seen.append(prompt)
            return f"answer {len(seen)}"

        tool = AskUserQuestionTool(input_callback=cb)
        result = json.loads(asyncio.run(tool.ask_user_question(
            questions=[
                {"id": "db", "question": "Which database?"},
                {"id": "scope", "question": "Which module?"},
            ]
        )))

        assert seen == ["Which database?", "Which module?"], "asked in order"
        assert [a["id"] for a in result["answers"]] == ["db", "scope"]
        assert [a["response"] for a in result["answers"]] == ["answer 1", "answer 2"]

    def test_the_answer_echoes_each_id_because_labels_can_repeat(self):
        from agentica.tools.ask_user_question_tool import AskUserQuestionTool

        tool = AskUserQuestionTool(input_callback=lambda p, o=None: "same")
        result = json.loads(asyncio.run(tool.ask_user_question(
            questions=[
                {"id": "one", "question": "Q1", "options": ["same"]},
                {"id": "two", "question": "Q2", "options": ["same"]},
            ]
        )))

        assert [a["id"] for a in result["answers"]] == ["one", "two"]

    def test_a_header_prefixes_the_prompt_so_the_card_keeps_its_context(self):
        from agentica.tools.ask_user_question_tool import AskUserQuestionTool

        seen = []
        tool = AskUserQuestionTool(
            input_callback=lambda p, o=None: seen.append(p) or "ok"
        )
        asyncio.run(tool.ask_user_question(
            questions=[{"id": "x", "header": "Storage", "question": "Which?"}]
        ))

        assert "[Storage] Which?" == seen[0]

    def test_options_travel_with_their_own_question(self):
        from agentica.tools.ask_user_question_tool import AskUserQuestionTool

        seen = []
        tool = AskUserQuestionTool(
            input_callback=lambda p, o=None: seen.append((p, o)) or "ok"
        )
        asyncio.run(tool.ask_user_question(
            questions=[
                {"id": "a", "question": "Pick", "options": ["X", "Y"]},
                {"id": "b", "question": "Free form"},
            ]
        ))

        assert seen == [("Pick", ["X", "Y"]), ("Free form", None)]

    def test_a_timeout_stops_the_batch_but_keeps_what_was_answered(self):
        """Nobody at the keyboard: stop asking, keep the answers already given.

        Arming one prompt after another that nobody will answer keeps the tool
        blocked and the model waiting, so the remaining questions are reported
        as skipped instead.
        """
        from agentica.tools.ask_user_question_tool import (
            AskUserQuestionTool,
            AskUserQuestionTimeout,
            TIMED_OUT_NOTE,
        )

        calls = []

        def cb(prompt, options=None):
            calls.append(prompt)
            if len(calls) == 2:
                raise AskUserQuestionTimeout(120)
            return "kept"

        tool = AskUserQuestionTool(input_callback=cb)
        result = json.loads(asyncio.run(tool.ask_user_question(
            questions=[
                {"id": "a", "question": "Q1"},
                {"id": "b", "question": "Q2"},
                {"id": "c", "question": "Q3"},
            ]
        )))

        first, second, third = result["answers"]
        assert first["response"] == "kept"
        assert second["timed_out"] is True and second["note"] == TIMED_OUT_NOTE
        assert third["timed_out"] is True and third["skipped"] is True
        assert len(calls) == 2, "the unanswered questions are not armed"

    def test_a_batch_missing_an_id_is_still_answerable(self):
        """A question without an id gets one, rather than being refused.

        The id exists so the answer can be attributed; an index is a fine
        attribution, and refusing would cost the user the whole batch.
        """
        from agentica.tools.ask_user_question_tool import AskUserQuestionTool

        tool = AskUserQuestionTool(input_callback=lambda p, o=None: "ok")
        result = json.loads(asyncio.run(tool.ask_user_question(
            questions=[{"question": "no id here"}]
        )))

        assert result["answers"][0]["id"] == "q1"
        assert result["answers"][0]["response"] == "ok"

    def test_duplicate_ids_are_reported_not_asked(self):
        """Ids are the answer's only handle: two of them collide."""

        from agentica.tools.ask_user_question_tool import AskUserQuestionTool

        asked = []
        tool = AskUserQuestionTool(input_callback=lambda p, o=None: asked.append(p) or "x")
        result = json.loads(asyncio.run(tool.ask_user_question(
            questions=[{"id": "a", "question": "1"}, {"id": "a", "question": "2"}]
        )))

        assert "reuses id" in result["error"]
        assert asked == [], "refused before interrupting the user"

    def test_prompt_and_questions_together_are_refused(self):
        """Two shapes in one call means the model misread the schema; guessing
        which it meant would silently drop half the questions."""
        from agentica.tools.ask_user_question_tool import AskUserQuestionTool

        asked = []
        tool = AskUserQuestionTool(input_callback=lambda p, o=None: asked.append(p) or "x")
        result = json.loads(asyncio.run(tool.ask_user_question(
            prompt="one", questions=[{"id": "a", "question": "two"}]
        )))

        assert "not both" in result["error"]
        assert asked == []

    def test_an_empty_call_is_reported_rather_than_armed(self):
        from agentica.tools.ask_user_question_tool import AskUserQuestionTool

        asked = []
        tool = AskUserQuestionTool(input_callback=lambda p, o=None: asked.append(p) or "x")
        result = json.loads(asyncio.run(tool.ask_user_question()))

        assert "`prompt`" in result["error"] and "`questions`" in result["error"]
        assert asked == []

    def test_the_single_question_result_shape_did_not_change(self):
        """Existing surfaces parse the single shape; batching must not move it.

        The CLI renders ``prompt`` back as the transcript's record of the
        exchange, and the gateway reads ``response``; both predate batching.
        """
        from agentica.tools.ask_user_question_tool import AskUserQuestionTool

        tool = AskUserQuestionTool(input_callback=lambda p, o=None: "yes")
        result = json.loads(asyncio.run(tool.ask_user_question(
            prompt="Proceed?", options=["Yes", "No"]
        )))

        assert result == {
            "prompt": "Proceed?",
            "response": "yes",
            "options": ["Yes", "No"],
        }
