# -*- coding: utf-8 -*-
"""Regression tests for the Anthropic ``tools[].input_schema`` payload.

These lock down four bugs in ``Claude.get_tools()`` /
``_get_structured_output_tool()``. The last one is what the API rejects with::

    tools.<i>.custom.input_schema.required: Input should be a valid list

1. ``required`` was recomputed as "every property whose type does not include
   null". ``agentica.utils.json_schema.get_json_schema`` deliberately encodes
   Optional by *omitting* the name from ``required`` (a ``["type", "null"]``
   array breaks other providers), so that rule marked every optional parameter
   mandatory: ``read_file`` demanded offset/limit/tail, ``execute`` demanded
   timeout/parallel_safe. Only the native anthropic path was affected; the
   OpenAI path sends ``Function.to_dict()`` and was always correct.
2. Each property was flattened to ``{"type": ..., "description": ...}``,
   dropping enum / items / nested properties.
3. ``deferred`` and unavailable functions leaked into the provider schema.
   ``get_tools_for_api`` applies both gates for OpenAI; this path iterates
   ``self.functions`` and skipped them.
4. A ``response_model`` whose fields all have defaults produces a pydantic
   schema with **no** ``required`` key, so the synthetic ``structured_output``
   tool serialised ``"required": null`` and the request 400'd.
"""

import unittest
from typing import Optional

from pydantic import BaseModel

from agentica.model.anthropic.claude import Claude
from agentica.tools.base import Function


def _claude(**kwargs) -> Claude:
    return Claude(id="claude-opus-5", api_key="test-key", **kwargs)


class TestRequiredMirrorsTheSignature(unittest.TestCase):
    def test_optional_parameters_are_not_marked_required(self):
        from agentica.tools.builtin.file_tool import BuiltinFileTool

        model = _claude()
        model.add_tool(BuiltinFileTool())
        by_name = {t["name"]: t for t in model.get_tools()}

        # file_path has no default; offset/limit/tail do.
        self.assertEqual(by_name["read_file"]["input_schema"]["required"], ["file_path"])
        self.assertIn("offset", by_name["read_file"]["input_schema"]["properties"])
        self.assertEqual(by_name["glob"]["input_schema"]["required"], ["pattern"])
        self.assertEqual(by_name["grep"]["input_schema"]["required"], ["pattern"])
        self.assertEqual(
            by_name["write_file"]["input_schema"]["required"], ["file_path", "content"]
        )

    def test_required_matches_what_the_openai_path_sends(self):
        """The two providers must not disagree about which args are mandatory."""
        from agentica.tools.builtin.execute_tool import BuiltinExecuteTool
        from agentica.tools.builtin.file_tool import BuiltinFileTool

        model = _claude()
        model.add_tool(BuiltinFileTool())
        model.add_tool(BuiltinExecuteTool())

        for tool in model.get_tools():
            fn = model.functions[tool["name"]]
            expected = fn.parameters.get("required") or []
            with self.subTest(tool=tool["name"]):
                self.assertEqual(tool["input_schema"]["required"], expected)

    def test_required_is_always_a_list_even_when_the_schema_lies(self):
        """An MCP server supplies inputSchema verbatim; it may be malformed."""
        for bogus in ("file_path", None, {"0": "a"}, 7):
            fn = Function(
                name="mcp_x",
                description="d",
                parameters={
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "required": bogus,
                },
                skip_entrypoint_processing=True,
            )
            model = _claude()
            model.functions = {"mcp_x": fn}
            got = model.get_tools()[0]["input_schema"]["required"]
            with self.subTest(bogus=bogus):
                self.assertIsInstance(got, list)
                self.assertEqual(got, [])

    def test_required_drops_names_absent_from_properties(self):
        fn = Function(
            name="mcp_y",
            description="d",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a", "ghost"],
            },
            skip_entrypoint_processing=True,
        )
        model = _claude()
        model.functions = {"mcp_y": fn}
        self.assertEqual(model.get_tools()[0]["input_schema"]["required"], ["a"])


class TestPropertySchemasSurvive(unittest.TestCase):
    def test_enum_items_and_nested_properties_are_kept(self):
        fn = Function(
            name="rich",
            description="d",
            parameters={
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["a", "b"], "description": "pick"},
                    "names": {"type": "array", "items": {"type": "string"}},
                    "opts": {"type": "object", "properties": {"deep": {"type": "number"}}},
                },
                "required": ["mode"],
            },
            skip_entrypoint_processing=True,
        )
        model = _claude()
        model.functions = {"rich": fn}
        props = model.get_tools()[0]["input_schema"]["properties"]

        self.assertEqual(props["mode"]["enum"], ["a", "b"])
        self.assertEqual(props["mode"]["description"], "pick")
        self.assertEqual(props["names"]["items"], {"type": "string"})
        self.assertEqual(props["opts"]["properties"]["deep"]["type"], "number")


class TestDeferredAndUnavailableAreHidden(unittest.TestCase):
    def _model_with_file_tools(self) -> Claude:
        from agentica.tools.builtin.file_tool import BuiltinFileTool

        model = _claude()
        model.add_tool(BuiltinFileTool())
        return model

    def test_deferred_function_is_not_advertised(self):
        model = self._model_with_file_tools()
        model.functions["grep"].deferred = True

        names = [t["name"] for t in model.get_tools()]
        self.assertNotIn("grep", names)
        self.assertIn("read_file", names)
        # Still executable — deferred hides the schema, not the capability.
        self.assertIn("grep", model.functions)

    def test_unavailable_function_is_not_advertised(self):
        model = self._model_with_file_tools()
        model.functions["grep"].available_when = lambda: False

        self.assertNotIn("grep", [t["name"] for t in model.get_tools()])


class _AllFieldsOptional(BaseModel):
    a: Optional[str] = None
    b: int = 3


class _HasMandatoryField(BaseModel):
    a: str
    b: Optional[int] = None


class TestStructuredOutputTool(unittest.TestCase):
    def _tool_for(self, model_cls):
        model = _claude()
        model.response_format = model_cls
        model.use_structured_outputs = True
        return model._get_structured_output_tool()

    def test_all_optional_model_still_sends_a_list(self):
        """pydantic omits "required" entirely here -> used to serialise null."""
        self.assertNotIn("required", _AllFieldsOptional.model_json_schema())

        schema = self._tool_for(_AllFieldsOptional)["input_schema"]
        self.assertEqual(schema["required"], [])
        self.assertIsInstance(schema["required"], list)

    def test_mandatory_field_is_preserved(self):
        schema = self._tool_for(_HasMandatoryField)["input_schema"]
        self.assertEqual(schema["required"], ["a"])

    def test_every_tool_in_the_request_has_a_list_required(self):
        from agentica.tools.builtin.file_tool import BuiltinFileTool

        model = _claude()
        model.add_tool(BuiltinFileTool())
        model.response_format = _AllFieldsOptional
        model.use_structured_outputs = True

        kwargs = model.prepare_request_kwargs("system")
        self.assertTrue(kwargs["tools"])
        for tool in kwargs["tools"]:
            with self.subTest(tool=tool["name"]):
                self.assertIsInstance(tool["input_schema"]["required"], list)
                self.assertIsInstance(tool["input_schema"]["properties"], dict)


if __name__ == "__main__":
    unittest.main()
