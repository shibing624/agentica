# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: What the model sees when it calls a tool wrong.

A model that hallucinates a tool name or a parameter name gets one tool result
back. That result is the only feedback it has, so it has to say what was
accepted — production logs show runs where the model repeated the same wrong
call (``Bash``, ``ToolSearch``, ``patch``) answered by the same useless string.
"""

import unittest

from pydantic import ValidationError, validate_call

from agentica.tools.base import Function, _format_tool_error, coerce_tool_args, unknown_tool_error


def _run_tool(command: str, timeout: int = 120, work_dir: str = None) -> str:
    return command


def _read_file(file_path: str, offset: int = 0) -> str:
    return file_path


def _validation_error(function: Function, **kwargs) -> ValidationError:
    try:
        validate_call(function.entrypoint)(**kwargs)
    except ValidationError as exc:
        return exc
    raise AssertionError(f"{kwargs} validated")


class TestUnknownToolResult(unittest.TestCase):
    def test_names_the_requested_tool_and_the_available_ones(self):
        functions = {
            "execute": Function.from_callable(_run_tool),
            "read_file": Function.from_callable(_read_file),
        }
        self.assertEqual(
            unknown_tool_error("Bash", functions),
            "No tool named 'Bash' exists; the call was not run. Available tools: execute, read_file.",
        )

    def test_list_is_capped_when_many_tools_exist(self):
        functions = {f"tool_{i:02d}": Function.from_callable(_run_tool) for i in range(40)}
        message = unknown_tool_error("Bash", functions)

        self.assertIn("tool_29", message)
        self.assertNotIn("tool_30", message)
        self.assertIn("(10 more)", message)

    def test_parse_tool_calls_answers_with_it(self):
        from agentica.model.openai import OpenAIChat
        from agentica.model.message import Message

        model = OpenAIChat(id="gpt-4o-mini", api_key="fake")
        model.functions = {"execute": Function.from_callable(_run_tool)}
        assistant = Message(role="assistant", tool_calls=[
            {"id": "c1", "type": "function", "function": {"name": "Bash", "arguments": "{}"}},
        ])
        messages = []
        model.parse_tool_calls(assistant, messages)

        self.assertEqual(messages[0].content, unknown_tool_error("Bash", model.functions))
        self.assertTrue(messages[0].tool_call_error)


    def test_ollama_refusal_answers_its_own_id_once(self):
        """Ollama used to send a bare user message; the id then got a second, padded 'interrupted' result."""
        from agentica.model.message import Message
        from agentica.model.ollama.chat import Ollama

        model = Ollama(id="qwen")
        model.functions = {"execute": Function.from_callable(_run_tool)}
        assistant = Message(role="assistant", tool_calls=[
            {"id": "c1", "type": "function", "function": {"name": "Bash", "arguments": "{}"}},
        ])
        messages = []
        _, metadata = model.parse_tool_calls(assistant, messages)
        model.format_tool_results([], messages, metadata)

        self.assertEqual(
            [(m.role, m.tool_call_id, m.content, m.tool_call_error) for m in messages],
            [("tool", "c1", unknown_tool_error("Bash", model.functions), True)],
        )

class TestWrongParameterName(unittest.TestCase):
    """``ArgsKwargs((), {'cmd': ...})`` is pydantic's repr of the call, not a hint."""

    def test_wrong_name_lists_accepted_parameters_without_the_call_repr(self):
        function = Function.from_callable(_run_tool)
        message = _format_tool_error(function, _validation_error(function, cmd="ls"))

        self.assertNotIn("ArgsKwargs", message)
        self.assertIn("command: Missing required argument", message)
        self.assertIn("cmd: Unexpected keyword argument", message)
        self.assertTrue(message.endswith("Accepted parameters: command, timeout, work_dir"), message)

    def test_camel_case_name_points_at_the_right_one(self):
        function = Function.from_callable(_read_file)
        message = _format_tool_error(function, _validation_error(function, filePath="/tmp/x"))

        self.assertTrue(message.endswith("Accepted parameters: file_path, offset"), message)

    def test_bad_value_reports_the_value_and_no_parameter_list(self):
        function = Function.from_callable(_read_file)
        message = _format_tool_error(function, _validation_error(function, file_path="/x", offset="abc"))

        self.assertIn("offset", message)
        self.assertIn("'abc'", message)
        self.assertNotIn("Accepted parameters", message)


class TestArgumentCoercion(unittest.TestCase):
    """A model writing "60" where an int is expected should not fail the call."""

    def test_numeric_string_becomes_int(self):
        function = Function.from_callable(_run_tool)
        coerced = coerce_tool_args({"command": "ls", "timeout": "60"}, function)
        self.assertEqual(coerced["timeout"], 60)


if __name__ == "__main__":
    unittest.main()
