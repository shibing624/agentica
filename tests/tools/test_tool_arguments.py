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
