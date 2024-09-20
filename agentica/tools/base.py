# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This module provides classes for managing tools.
It includes an abstract base class for tools and a class for managing tool instances.

part of the code from https://github.com/phidatahq/phidata
"""
from __future__ import annotations

import json
from collections import OrderedDict
from typing import Callable, get_type_hints, Any, Dict, Union, get_args, get_origin, Optional

from pydantic import BaseModel, validate_call

from agentica.utils.log import logger


class Function(BaseModel):
    """Model for Function"""

    # The name of the function to be called.
    # Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.
    name: str
    # A description of what the function does, used by the model to choose when and how to call the function.
    description: Optional[str] = None
    # The parameters the functions accepts, described as a JSON Schema object.
    # To describe a function that accepts no parameters, provide the value {"type": "object", "properties": {}}.
    parameters: Dict[str, Any] = {"type": "object", "properties": {}}
    entrypoint: Optional[Callable] = None

    # If True, the arguments are sanitized before being passed to the function.
    sanitize_arguments: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True, include={"name", "description", "parameters"})

    @classmethod
    def from_callable(cls, c: Callable) -> "Function":
        from inspect import getdoc

        parameters = {"type": "object", "properties": {}}
        try:
            # logger.debug(f"Getting type hints for {c}")
            type_hints = get_type_hints(c)
            # logger.debug(f"Type hints for {c}: {type_hints}")
            # logger.debug(f"Getting JSON schema for {type_hints}")
            parameters = get_json_schema(type_hints)
            # logger.debug(f"JSON schema for {c}: {parameters}")
            # logger.debug(f"Type hints for {c.__name__}: {type_hints}")
        except Exception as e:
            logger.warning(f"Could not parse args for {c.__name__}: {e}")

        return cls(
            name=c.__name__,
            description=getdoc(c),
            parameters=parameters,
            entrypoint=validate_call(c),
        )

    def get_type_name(self, t):
        name = str(t)
        if "list" in name or "dict" in name:
            return name
        else:
            return t.__name__

    def get_definition_for_prompt(self) -> Optional[str]:
        """Returns a function definition that can be used in a prompt."""
        if self.entrypoint is None:
            return None

        type_hints = get_type_hints(self.entrypoint)
        return_type = type_hints.get("return", None)
        returns = None
        if return_type is not None:
            returns = self.get_type_name(return_type)

        function_info = {
            "name": self.name,
            "description": self.description,
            "arguments": self.parameters.get("properties", {}),
            "returns": returns,
        }
        return json.dumps(function_info, indent=2)

    def get_definition_for_prompt_dict(self) -> Optional[Dict[str, Any]]:
        """Returns a function definition that can be used in a prompt."""

        if self.entrypoint is None:
            return None

        type_hints = get_type_hints(self.entrypoint)
        return_type = type_hints.get("return", None)
        returns = None
        if return_type is not None:
            returns = self.get_type_name(return_type)

        function_info = {
            "name": self.name,
            "description": self.description,
            "arguments": self.parameters.get("properties", {}),
            "returns": returns,
        }
        return function_info


class FunctionCall(BaseModel):
    """Model for Function Call"""

    # The function to be called.
    function: Function
    # The arguments to call the function with.
    arguments: Optional[Dict[str, Any]] = None
    # The result of the function call.
    result: Optional[Any] = None
    # The ID of the function call.
    call_id: Optional[str] = None

    # Error while parsing arguments or running the function.
    error: Optional[str] = None

    def get_call_str(self) -> str:
        """Returns a string representation of the function call."""
        if self.arguments is None:
            return f"{self.function.name}()"

        trimmed_arguments = {}
        for k, v in self.arguments.items():
            if isinstance(v, str) and len(v) > 100:
                trimmed_arguments[k] = v[:100] + "..."
            else:
                trimmed_arguments[k] = v
        call_str = f"{self.function.name}({', '.join([f'{k}={v}' for k, v in trimmed_arguments.items()])})"
        return call_str

    def execute(self) -> bool:
        """Runs the function call.

        @return: True if the function call was successful, False otherwise.
        """
        if self.function.entrypoint is None:
            return False

        logger.debug(f"Running: {self.get_call_str()}")

        # Call the function with no arguments if none are provided.
        if self.arguments is None:
            try:
                self.result = self.function.entrypoint()
                return True
            except Exception as e:
                logger.warning(f"Could not run function {self.get_call_str()}")
                logger.exception(e)
                self.result = str(e)
                return False

        try:
            self.result = self.function.entrypoint(**self.arguments)
            return True
        except Exception as e:
            logger.warning(f"Could not run function {self.get_call_str()}")
            logger.exception(e)
            self.result = str(e)
            return False


class Toolkit:
    """Toolkit for managing functions."""

    def __init__(self, name: str = "tool"):
        self.name: str = name
        self.functions: Dict[str, Function] = OrderedDict()

    def register(self, function: Callable, sanitize_arguments: bool = True):
        try:
            f = Function.from_callable(function)
            f.sanitize_arguments = sanitize_arguments
            self.functions[f.name] = f
            logger.debug(f"Function: {f.name} registered with {self.name}")
            logger.debug(f"Json Schema: {f.to_dict()}")
        except Exception as e:
            logger.warning(f"Failed to create Function for: {function.__name__}")
            raise e

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name} functions={list(self.functions.keys())}>"

    def __str__(self):
        return self.__repr__()


class Tool(BaseModel):
    """Model for Tools"""

    # The type of tool
    type: str
    # The function to be called if type = "function"
    function: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)


def get_json_type_for_py_type(arg: str) -> str:
    """
    Get the JSON schema type for a given type.
    :param arg: The type to get the JSON schema type for.
    :return: The JSON schema type.

    See: https://json-schema.org/understanding-json-schema/reference/type.html#type-specific-keywords
    """
    # logger.info(f"Getting JSON type for: {arg}")
    if arg in ("int", "float"):
        return "number"
    elif arg == "str":
        return "string"
    elif arg == "bool":
        return "boolean"
    elif arg in ("NoneType", "None"):
        return "null"
    return arg


def get_json_schema_for_arg(t: Any) -> Optional[Any]:
    # logger.info(f"Getting JSON schema for arg: {t}")
    json_schema = None
    type_args = get_args(t)
    # logger.info(f"Type args: {type_args}")
    type_origin = get_origin(t)
    # logger.info(f"Type origin: {type_origin}")
    if type_origin is not None:
        if type_origin == list:
            json_schema_for_items = get_json_schema_for_arg(type_args[0])
            json_schema = {"type": "array", "items": json_schema_for_items}
        elif type_origin == dict:
            json_schema = {"type": "object", "properties": {}}
        elif type_origin == Union:
            json_schema = {"type": [get_json_type_for_py_type(arg.__name__) for arg in type_args]}
    else:
        json_schema = {"type": get_json_type_for_py_type(t.__name__)}
    return json_schema


def get_json_schema(type_hints: Dict[str, Any]) -> Dict[str, Any]:
    json_schema: Dict[str, Any] = {"type": "object", "properties": {}}
    for k, v in type_hints.items():
        # logger.info(f"Parsing arg: {k} | {v}")
        if k == "return":
            continue
        arg_json_schema = get_json_schema_for_arg(v)
        if arg_json_schema is not None:
            # logger.info(f"json_schema: {arg_json_schema}")
            json_schema["properties"][k] = arg_json_schema
        else:
            logger.warning(f"Could not parse argument {k} of type {v}")
    return json_schema


def get_function_call(
        name: str,
        arguments: Optional[str] = None,
        call_id: Optional[str] = None,
        functions: Optional[Dict[str, Function]] = None,
) -> Optional[FunctionCall]:
    logger.debug(f"Getting function {name}")
    logger.debug(f"Arguments: {arguments}, Call ID: {call_id}, name: {name}, functions: {functions}")
    if functions is None:
        return None

    function_to_call: Optional[Function] = None
    if name in functions:
        function_to_call = functions[name]
    if function_to_call is None:
        logger.error(f"Function {name} not found")
        return None

    function_call = FunctionCall(function=function_to_call)
    if call_id is not None:
        function_call.call_id = call_id
    if arguments is not None and arguments != "":
        try:
            if function_to_call.sanitize_arguments:
                if "None" in arguments:
                    arguments = arguments.replace("None", "null")
                if "True" in arguments:
                    arguments = arguments.replace("True", "true")
                if "False" in arguments:
                    arguments = arguments.replace("False", "false")
            _arguments = json.loads(arguments)
        except Exception as e:
            logger.error(f"Unable to decode function arguments:\n{arguments}\nError: {e}")
            function_call.error = f"Error while decoding function arguments:\n{arguments}\nError: {e}\n\n " \
                                  f"Please make sure we can json.loads() the arguments and retry."
            return function_call

        if not isinstance(_arguments, dict):
            logger.error(f"Function arguments are not a valid JSON object: {arguments}")
            function_call.error = "Function arguments are not a valid JSON object.\n\n Please fix and retry."
            return function_call

        try:
            clean_arguments: Dict[str, Any] = {}
            for k, v in _arguments.items():
                if isinstance(v, str):
                    _v = v.strip().lower()
                    if _v in ("none", "null"):
                        clean_arguments[k] = None
                    elif _v == "true":
                        clean_arguments[k] = True
                    elif _v == "false":
                        clean_arguments[k] = False
                    else:
                        clean_arguments[k] = v.strip()
                else:
                    clean_arguments[k] = v

            function_call.arguments = clean_arguments
        except Exception as e:
            logger.error(f"Unable to parsing function arguments:\n{arguments}\nError: {e}")
            function_call.error = f"Error while parsing function arguments: {e}\n\n Please fix and retry."
            return function_call
    return function_call


def get_function_call_for_tool_call(
        tool_call: Dict[str, Any], functions: Optional[Dict[str, Function]] = None
) -> Optional[FunctionCall]:
    if tool_call.get("type") == "function":
        _tool_call_id = tool_call.get("id")
        _tool_call_function = tool_call.get("function")
        if _tool_call_function is not None:
            _tool_call_function_name = _tool_call_function.get("name")
            _tool_call_function_arguments_str = _tool_call_function.get("arguments")
            if _tool_call_function_name is not None:
                return get_function_call(
                    name=_tool_call_function_name,
                    arguments=_tool_call_function_arguments_str,
                    call_id=_tool_call_id,
                    functions=functions,
                )
    return None


def extract_tool_call_from_string(text: str, start_tag: str = "<tool_call>", end_tag: str = "</tool_call>"):
    start_index = text.find(start_tag) + len(start_tag)
    end_index = text.find(end_tag)

    # Extracting the content between the tags
    return text[start_index:end_index].strip()


def remove_tool_calls_from_string(text: str, start_tag: str = "<tool_call>", end_tag: str = "</tool_call>"):
    """Remove multiple tool calls from a string."""
    while start_tag in text and end_tag in text:
        start_index = text.find(start_tag)
        end_index = text.find(end_tag) + len(end_tag)
        text = text[:start_index] + text[end_index:]
    return text


def extract_tool_from_xml(xml_str):
    # Find tool_name
    tool_name_start = xml_str.find("<tool_name>") + len("<tool_name>")
    tool_name_end = xml_str.find("</tool_name>")
    tool_name = xml_str[tool_name_start:tool_name_end].strip()

    # Find and process parameters block
    params_start = xml_str.find("<parameters>") + len("<parameters>")
    params_end = xml_str.find("</parameters>")
    parameters_block = xml_str[params_start:params_end].strip()

    # Extract individual parameters
    arguments = {}
    while parameters_block:
        # Find the next tag and its closing
        tag_start = parameters_block.find("<") + 1
        tag_end = parameters_block.find(">")
        tag_name = parameters_block[tag_start:tag_end]

        # Find the tag's closing counterpart
        value_start = tag_end + 1
        value_end = parameters_block.find(f"</{tag_name}>")
        value = parameters_block[value_start:value_end].strip()

        # Add to arguments
        arguments[tag_name] = value

        # Move past this tag
        parameters_block = parameters_block[value_end + len(f"</{tag_name}>"):].strip()

    return {"tool_name": tool_name, "parameters": arguments}


def remove_function_calls_from_string(
        text: str, start_tag: str = "<function_calls>", end_tag: str = "</function_calls>"
):
    """Remove multiple function calls from a string."""
    while start_tag in text and end_tag in text:
        start_index = text.find(start_tag)
        end_index = text.find(end_tag) + len(end_tag)
        text = text[:start_index] + text[end_index:]
    return text
