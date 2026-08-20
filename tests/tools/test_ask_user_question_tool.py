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
