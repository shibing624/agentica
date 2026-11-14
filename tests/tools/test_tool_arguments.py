# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for tool-call argument decoding (agentica/tools/base.py).

The decoder must hand the function exactly what the model wrote. Argument
values are payloads (message bodies, code, prose), not configuration to be
tidied up.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.tools.base import Function, get_function_call


def say(text: str, times: int = 1, loud: bool = False) -> str:
    """Say something.

    Args:
        text: What to say.
        times: How many times.
        loud: Whether to shout.
    """
    return text * times


def _functions():
    fn = Function.from_callable(say)
    fn.process_entrypoint(strict=False)
    return {"say": fn}


def _call(payload: dict):
    return get_function_call("say", json.dumps(payload), functions=_functions())


class TestArgumentsReachTheToolVerbatim:
    def test_python_literals_inside_a_string_are_not_rewritten(self):
        """The bug: a blind "True" -> "true" replace over the raw JSON turned
        `swapped = True` in a peer message into `swapped = true`, which is a
        NameError on the receiving side."""
        code = "swapped = False\nif x is None:\n    swapped = True\n"

        call = _call({"text": code})

        assert call.error is None
        assert call.arguments["text"] == code

    def test_surrounding_whitespace_is_kept(self):
        call = _call({"text": "  indented line\n\n"})

        assert call.arguments["text"] == "  indented line\n\n"

    def test_a_string_that_reads_none_stays_a_string(self):
        for word in ("None", "null", "true", "False"):
            call = _call({"text": word})

            assert call.arguments["text"] == word

    def test_schema_typed_values_are_still_coerced(self):
        call = _call({"text": "hi", "times": "3", "loud": "true"})

        assert call.arguments["times"] == 3
        assert call.arguments["loud"] is True


class TestMalformedArguments:
    def test_a_python_repr_dict_is_recovered(self):
        """Some models emit a Python repr instead of JSON."""
        call = get_function_call(
            "say",
            "{'text': 'hi', 'loud': True, 'times': None}",
            functions=_functions(),
        )

        assert call.error is None
        assert call.arguments == {"text": "hi", "loud": True, "times": None}

    def test_unparseable_arguments_report_an_error(self):
        call = get_function_call("say", "{text: hi", functions=_functions())

        assert call.error is not None
        assert call.arguments is None

    def test_a_non_object_payload_is_rejected(self):
        call = get_function_call("say", '["hi"]', functions=_functions())

        assert call.error is not None


class TestAnEvictedArgumentIsNeverExecuted:
    """Layer 1 dropped a payload, so the value is gone.

    The marker is a JSON object, not a string: a string in this slot is
    a shell command / file write / arbitrary string tool argument. Observed
    live in older sessions: three string-marker tool calls delivered the old
    marker as their entire argument. The tool result must not echo that marker
    back or the model copies it again.
    """

    def test_the_object_marker_is_refused(self):
        from agentica.compression.tool_call_args import omitted_tool_arg

        call = _call({"text": omitted_tool_arg(1885)})

        assert call.error is not None
        assert call.arguments is None
        assert "not executed" in call.error.lower()
        assert "$evicted" not in call.error
        assert "<evicted-tool-arg" not in call.error

    def test_a_json_encoded_object_marker_is_refused(self):
        from agentica.compression.tool_call_args import omitted_tool_arg

        call = _call({"text": json.dumps(omitted_tool_arg(1183))})

        assert call.error is not None
        assert call.arguments is None

    def test_the_old_string_marker_still_in_jsonl_is_refused(self):
        """Sessions already evicted with the string form. Do not send it."""
        call = _call({"text": "<evicted-tool-arg chars=1553>"})

        assert call.error is not None
        assert call.arguments is None

    def test_the_truncated_old_string_the_model_re_emitted_is_refused(self):
        call = _call({"text": "<evicted-tool-arg chars=1553"})

        assert call.error is not None
        assert call.arguments is None

    def test_a_nested_marker_is_refused_and_the_field_is_named(self):
        from agentica.compression.tool_call_args import omitted_tool_arg

        call = get_function_call(
            "say",
            json.dumps({"text": {"deep": [omitted_tool_arg(7)]}}),
            functions=_functions(),
        )

        assert call.error is not None
        assert "text.deep[0]" in call.error

    def test_a_real_payload_that_mentions_the_old_marker_still_runs(self):
        real = "grep -rn '<evicted-tool-arg chars=' agentica/"

        call = _call({"text": real})

        assert call.error is None
        assert call.arguments["text"] == real

    def test_a_marker_with_trailing_text_still_runs(self):
        text = "<evicted-tool-arg chars=5> and then real words"

        call = _call({"text": text})

        assert call.error is None
        assert call.arguments["text"] == text

    def test_normal_arguments_are_untouched(self):
        call = _call({"text": "hi", "times": 2})

        assert call.error is None
        assert call.arguments == {"text": "hi", "times": 2}

    def test_string_argument_call_is_refused_before_execution(self):
        from agentica.compression.tool_call_args import omitted_tool_arg

        def write_note(target: str, message: str) -> str:
            """Write a note."""
            raise AssertionError("the tool must not be reached")

        fn = Function.from_callable(write_note)
        fn.process_entrypoint(strict=False)
        call = get_function_call(
            "write_note",
            json.dumps({"target": "vpetmac-05", "message": omitted_tool_arg(1183)}),
            functions={"write_note": fn},
        )

        assert call.error is not None
        assert call.arguments is None
        assert "message" in call.error
