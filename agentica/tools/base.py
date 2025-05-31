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
from typing import Callable, get_type_hints, Any, Dict, Union, get_args, get_origin, Optional, Type, TypeVar, List
from pydantic import BaseModel, Field, validate_call
from agentica.model.message import Message
from agentica.utils.log import logger

T = TypeVar("T")


class ToolCallException(Exception):
    def __init__(
            self,
            exc,
            user_message: Optional[Union[str, Message]] = None,
            agent_message: Optional[Union[str, Message]] = None,
            messages: Optional[List[Union[dict, Message]]] = None,
            stop_execution: bool = False,
    ):
        super().__init__(exc)
        self.user_message = user_message
        self.agent_message = agent_message
        self.messages = messages
        self.stop_execution = stop_execution


class RetryAgentRun(ToolCallException):
    """Exception raised when a tool call should be retried."""


class StopAgentRun(ToolCallException):
    """Exception raised when an agent should stop executing entirely."""

    def __init__(
            self,
            exc,
            user_message: Optional[Union[str, Message]] = None,
            agent_message: Optional[Union[str, Message]] = None,
            messages: Optional[List[Union[dict, Message]]] = None,
    ):
        super().__init__(
            exc, user_message=user_message, agent_message=agent_message, messages=messages, stop_execution=True
        )


class Function(BaseModel):
    """Model for storing functions that can be called by an agent."""

    # The name of the function to be called.
    # Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.
    name: str
    # A description of what the function does, used by the model to choose when and how to call the function.
    description: Optional[str] = None
    # The parameters the functions accepts, described as a JSON Schema object.
    # To describe a function that accepts no parameters, provide the value {"type": "object", "properties": {}}.
    parameters: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        description="JSON Schema object describing function parameters",
    )
    strict: Optional[bool] = None

    # The function to be called.
    entrypoint: Optional[Callable] = None
    # If True, the entrypoint processing is skipped and the Function is used as is.
    skip_entrypoint_processing: bool = False
    # If True, the arguments are sanitized before being passed to the function.
    sanitize_arguments: bool = True
    # If True, the function call will show the result along with sending it to the model.
    show_result: bool = False
    # If True, the agent will stop after the function call.
    stop_after_tool_call: bool = False
    # Hook that runs before the function is executed.
    # If defined, can accept the FunctionCall instance as a parameter.
    pre_hook: Optional[Callable] = None
    # Hook that runs after the function is executed, regardless of success/failure.
    # If defined, can accept the FunctionCall instance as a parameter.
    post_hook: Optional[Callable] = None

    # --*-- FOR INTERNAL USE ONLY --*--
    # The agent that the function is associated with
    _agent: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True, include={"name", "description", "parameters", "strict"})

    @classmethod
    def from_callable(cls, c: Callable, strict: bool = False) -> "Function":
        from inspect import getdoc, signature
        from agentica.utils.json_util import get_json_schema

        function_name = c.__name__
        parameters = {"type": "object", "properties": {}, "required": []}
        try:
            sig = signature(c)
            type_hints = get_type_hints(c)

            # If function has an the agent argument, remove the agent parameter from the type hints
            if "agent" in sig.parameters:
                del type_hints["agent"]
            # logger.info(f"Type hints for {function_name}: {type_hints}")

            # Filter out return type and only process parameters
            param_type_hints = {
                name: type_hints[name]
                for name in sig.parameters
                if name in type_hints and name != "return" and name != "agent"
            }
            # logger.info(f"Arguments for {function_name}: {param_type_hints}")

            # Get JSON schema for parameters only
            parameters = get_json_schema(type_hints=param_type_hints, strict=strict)

            # If strict=True mark all fields as required
            # See: https://platform.openai.com/docs/guides/structured-outputs/supported-schemas#all-fields-must-be-required
            if strict:
                parameters["required"] = [name for name in parameters["properties"] if name != "agent"]
            else:
                # Mark a field as required if it has no default value
                parameters["required"] = [
                    name
                    for name, param in sig.parameters.items()
                    if param.default == param.empty and name != "self" and name != "agent"
                ]

            # logger.debug(f"JSON schema for {function_name}: {parameters}")
        except Exception as e:
            logger.warning(f"Could not parse args for {function_name}: {e}", exc_info=True)

        return cls(
            name=function_name,
            description=getdoc(c),
            parameters=parameters,
            entrypoint=validate_call(c),
        )

    def process_entrypoint(self, strict: bool = False):
        """Process the entrypoint and make it ready for use by an agent."""
        from inspect import getdoc, signature
        from agentica.utils.json_util import get_json_schema
        if self.skip_entrypoint_processing:
            return
        if self.entrypoint is None:
            return

        parameters = {"type": "object", "properties": {}, "required": []}
        try:
            sig = signature(self.entrypoint)
            type_hints = get_type_hints(self.entrypoint)

            # If function has an the agent argument, remove the agent parameter from the type hints
            if "agent" in sig.parameters:
                del type_hints["agent"]
            # logger.info(f"Type hints for {self.name}: {type_hints}")

            # Filter out return type and only process parameters
            param_type_hints = {
                name: type_hints[name]
                for name in sig.parameters
                if name in type_hints and name != "return" and name != "agent"
            }
            # logger.info(f"Arguments for {self.name}: {param_type_hints}")

            # Get JSON schema for parameters only
            parameters = get_json_schema(type_hints=param_type_hints, strict=strict)
            # If strict=True mark all fields as required
            # See: https://platform.openai.com/docs/guides/structured-outputs/supported-schemas#all-fields-must-be-required
            if strict:
                parameters["required"] = [name for name in parameters["properties"] if name != "agent"]
            else:
                # Mark a field as required if it has no default value
                parameters["required"] = [
                    name
                    for name, param in sig.parameters.items()
                    if param.default == param.empty and name != "self" and name != "agent"
                ]

            # logger.debug(f"JSON schema for {self.name}: {parameters}")
        except Exception as e:
            logger.warning(f"Could not parse args for {self.name}: {e}", exc_info=True)

        self.description = getdoc(self.entrypoint) or self.description
        self.parameters = parameters
        self.entrypoint = validate_call(self.entrypoint)

    def get_type_name(self, t: Type[T]):
        name = str(t)
        if "list" in name or "dict" in name:
            return name
        else:
            return t.__name__

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

    def get_definition_for_prompt(self) -> Optional[str]:
        """Returns a function definition that can be used in a prompt."""
        import json

        function_info = self.get_definition_for_prompt_dict()
        if function_info is not None:
            return json.dumps(function_info, indent=2)
        return None


class FunctionCall(BaseModel):
    """Model for Function Calls"""

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
                trimmed_arguments[k] = "..."
            else:
                trimmed_arguments[k] = v
        call_str = f"{self.function.name}({', '.join([f'{k}={v}' for k, v in trimmed_arguments.items()])})"
        return call_str

    def execute(self) -> bool:
        """Execute the function.

        Returns:
            bool: True if the function call was successful, False otherwise.
        """
        from inspect import signature

        if self.function.entrypoint is None:
            self.error = f"No entrypoint found for function: {self.function.name}"
            logger.warning(self.error)
            return False

        logger.debug(f"Running: {self.get_call_str()}")
        function_call_success = False

        # Execute pre-hook if it exists
        if self.function.pre_hook is not None:
            try:
                pre_hook_args = {}
                # Check if the pre-hook has and agent argument
                if "agent" in signature(self.function.pre_hook).parameters:
                    pre_hook_args["agent"] = self.function._agent
                # Check if the pre-hook has an fc argument
                if "fc" in signature(self.function.pre_hook).parameters:
                    pre_hook_args["fc"] = self
                self.function.pre_hook(**pre_hook_args)
            except ToolCallException as e:
                logger.debug(f"{e.__class__.__name__}: {e}")
                self.error = str(e)
                raise
            except Exception as e:
                logger.warning(f"Error in pre-hook callback: {e}")
                logger.exception(e)

        # Call the function with no arguments if none are provided.
        if self.arguments is None:
            try:
                entrypoint_args = {}
                # Check if the entrypoint has and agent argument
                if "agent" in signature(self.function.entrypoint).parameters:
                    entrypoint_args["agent"] = self.function._agent
                # Check if the entrypoint has an fc argument
                if "fc" in signature(self.function.entrypoint).parameters:
                    entrypoint_args["fc"] = self

                self.result = self.function.entrypoint(**entrypoint_args)
                function_call_success = True
            except ToolCallException as e:
                logger.debug(f"{e.__class__.__name__}: {e}")
                self.error = str(e)
                raise
            except Exception as e:
                logger.warning(f"Could not run function {self.get_call_str()}")
                logger.exception(e)
                self.error = str(e)
                return function_call_success
        else:
            try:
                entrypoint_args = {}
                # Check if the entrypoint has and agent argument
                if "agent" in signature(self.function.entrypoint).parameters:
                    entrypoint_args["agent"] = self.function._agent
                # Check if the entrypoint has an fc argument
                if "fc" in signature(self.function.entrypoint).parameters:
                    entrypoint_args["fc"] = self

                self.result = self.function.entrypoint(**entrypoint_args, **self.arguments)
                function_call_success = True
            except ToolCallException as e:
                logger.debug(f"{e.__class__.__name__}: {e}")
                self.error = str(e)
                raise
            except Exception as e:
                logger.warning(f"Could not run function {self.get_call_str()}")
                logger.exception(e)
                self.error = str(e)
                return function_call_success

        # Execute post-hook if it exists
        if self.function.post_hook is not None:
            try:
                post_hook_args = {}
                # Check if the post-hook has and agent argument
                if "agent" in signature(self.function.post_hook).parameters:
                    post_hook_args["agent"] = self.function._agent
                # Check if the post-hook has an fc argument
                if "fc" in signature(self.function.post_hook).parameters:
                    post_hook_args["fc"] = self
                self.function.post_hook(**post_hook_args)
            except ToolCallException as e:
                logger.debug(f"{e.__class__.__name__}: {e}")
                self.error = str(e)
                raise
            except Exception as e:
                logger.warning(f"Error in post-hook callback: {e}")
                logger.exception(e)

        return function_call_success


class ModelTool(BaseModel):
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


class Tool:
    """Tool for managing functions."""

    def __init__(self, name: str = "tool"):
        self.name: str = name
        self.functions: Dict[str, Function] = OrderedDict()

    def register(self, function: Callable[..., Any], sanitize_arguments: bool = True):
        """Register a function with the toolkit.

        Args:
            function: The callable to register
            sanitize_arguments: If True, the arguments will be sanitized before being passed to the function.

        Returns:
            The registered function
        """
        try:
            f = Function(
                name=function.__name__,
                description=function.__doc__ or "",
                entrypoint=function,
                sanitize_arguments=sanitize_arguments,
            )
            self.functions[f.name] = f
            logger.debug(f"Function: {f.name} registered with {self.name}")
        except Exception as e:
            logger.warning(f"Failed to create Function for: {function.__name__}")
            raise e

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name} functions={list(self.functions.keys())}>"

    def __str__(self):
        return self.__repr__()
