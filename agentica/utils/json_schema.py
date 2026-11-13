# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Convert Python type hints into OpenAI tool-calling JSON Schema.

Used by ``Function.from_callable()`` (tools/base.py) to auto-generate the
``parameters`` block sent to the LLM as a tool definition. Not a JSON parser
-- for parsing JSON OUT of LLM text responses see ``json_parse.py``.
"""
import re
from typing import Any, Dict, Tuple, Union, get_args, get_origin, Optional, Literal

from agentica.utils.log import logger

_SECTION_HEADER_RE = re.compile(
    r"^(Args|Arguments|Parameters|Returns|Return|Raises|Yields):\s*$"
)
_ARG_ENTRY_RE = re.compile(
    r"^\*{0,2}([A-Za-z_]\w*)(?:\s*\([^)]*\))?\s*:\s*(.*)$"
)


def parse_docstring_sections(doc: Optional[str]) -> Tuple[str, Dict[str, str]]:
    """Split a cleaned (``inspect.getdoc``) docstring into body prose and per-arg descriptions.

    Extracts Google-style ``Args:`` / ``Arguments:`` / ``Parameters:`` entries
    into ``{param_name: description}`` and strips ``Args:`` / ``Returns:`` /
    ``Raises:`` / ``Yields:`` blocks from the top-level tool description so
    argument docs live on ``parameters.properties.<name>.description`` rather
    than being duplicated as raw docstring boilerplate.
    """
    if not doc:
        return "", {}

    body_lines = []
    arg_parts: Dict[str, list] = {}
    section: Optional[str] = None
    arg_indent: Optional[int] = None
    current_arg: Optional[str] = None

    for raw_line in doc.splitlines():
        stripped = raw_line.strip()
        indent = len(raw_line) - len(raw_line.lstrip(" "))

        if indent == 0 and stripped:
            header = _SECTION_HEADER_RE.match(stripped)
            if header:
                kind = header.group(1)
                section = "args" if kind in ("Args", "Arguments", "Parameters") else "skip"
                arg_indent = None
                current_arg = None
                continue
            section = None
            arg_indent = None
            current_arg = None

        if section is None:
            body_lines.append(raw_line)
            continue

        if section == "skip":
            continue

        # Inside Args:
        if not stripped:
            continue
        if arg_indent is None:
            arg_indent = indent
        if indent <= arg_indent:
            entry = _ARG_ENTRY_RE.match(stripped)
            if entry:
                arg_indent = indent
                current_arg = entry.group(1)
                first = entry.group(2).strip()
                arg_parts[current_arg] = [first] if first else []
                continue
        if current_arg is not None:
            arg_parts[current_arg].append(stripped)

    body = "\n".join(body_lines).strip()
    body = re.sub(r"\n{3,}", "\n\n", body)
    arg_docs = {
        name: " ".join(part for part in parts if part)
        for name, parts in arg_parts.items()
        if any(parts)
    }
    return body, arg_docs


def get_json_type_for_py_type(arg: str) -> str:
    """
    Get the JSON schema type for a given type.
    :param arg: The type to get the JSON schema type for.
    :return: The JSON schema type.
    """
    # logger.info(f"Getting JSON type for: {arg}")
    if arg in ("int", "float", "complex", "Decimal"):
        return "number"
    elif arg in ("str", "string"):
        return "string"
    elif arg in ("bool", "boolean"):
        return "boolean"
    elif arg in ("NoneType", "None"):
        return "null"
    elif arg in ("list", "tuple", "set", "frozenset"):
        return "array"
    elif arg in ("dict", "mapping"):
        return "object"

    # If the type is not recognized, return "object"
    return "object"


def get_json_schema_for_arg(t: Any) -> Optional[Dict[str, Any]]:
    # log_info(f"Getting JSON schema for arg: {t}")
    type_args = get_args(t)
    # log_info(f"Type args: {type_args}")
    type_origin = get_origin(t)
    # log_info(f"Type origin: {type_origin}")

    if type_origin is not None:
        if type_origin is Literal:
            # Literal["a", "b", "c"] → {"type": "string", "enum": ["a", "b", "c"]}
            enum_values = list(type_args)
            py_types = {type(v).__name__ for v in enum_values}
            json_type = "string"
            if py_types == {"int"}:
                json_type = "number"
            elif py_types == {"bool"}:
                json_type = "boolean"
            return {"type": json_type, "enum": enum_values}
        if type_origin in (list, tuple, set, frozenset):
            json_schema_for_items = get_json_schema_for_arg(type_args[0]) if type_args else {"type": "string"}
            return {"type": "array", "items": json_schema_for_items}
        elif type_origin is dict:
            # Handle dict types - use simple object schema for maximum API compatibility
            # Note: additionalProperties may not be supported by some APIs
            return {"type": "object"}
        elif type_origin is Union or type(None) in type_args:
            # Filter out NoneType from Union
            non_none_types = [arg for arg in type_args if arg is not type(None)]
            
            # If only one non-None type, return its schema directly
            if len(non_none_types) == 1:
                return get_json_schema_for_arg(non_none_types[0])
            
            # For multiple types, try to use anyOf (may not be supported by all APIs)
            # Fall back to string type for maximum compatibility
            types = []
            for arg in non_none_types:
                try:
                    schema = get_json_schema_for_arg(arg)
                    if schema:
                        types.append(schema)
                except Exception:
                    continue
            
            # If we have multiple types, just use string for compatibility
            # Many APIs don't support anyOf
            if len(types) > 1:
                return {"type": "string"}
            elif len(types) == 1:
                return types[0]
            return None

    return {"type": get_json_type_for_py_type(t.__name__)}


def get_json_schema(type_hints: Dict[str, Any], strict: bool = False) -> Dict[str, Any]:
    json_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {},
    }
    if strict:
        json_schema["additionalProperties"] = False

    for k, v in type_hints.items():
        # logger.info(f"Parsing arg: {k} | {v}")
        if k == "return":
            continue

        try:
            # Check if type is Optional (Union with NoneType)
            type_origin = get_origin(v)
            type_args = get_args(v)
            is_optional = (type_origin is Union or type(None) in type_args) and len(type_args) == 2 and any(arg is type(None) for arg in type_args)

            # Get the actual type if it's Optional
            if is_optional:
                v = next(arg for arg in type_args if arg is not type(None))

            arg_json_schema = get_json_schema_for_arg(v)
            if arg_json_schema is not None:
                # Note: Many APIs (Google Gemini, etc.) don't support type as array
                # So we don't convert Optional to ["type", "null"] format
                # Optional fields are handled by not including them in "required"
                json_schema["properties"][k] = arg_json_schema

            else:
                logger.warning(f"Could not parse argument {k} of type {v}")
        except Exception as e:
            logger.error(f"Error processing argument {k}: {str(e)}")
            continue

    return json_schema
