import hashlib
import json
import re
from typing import Optional, Type

from pydantic import BaseModel, ValidationError
from agentica.utils.log import logger

TOOL_RESULT_TOKEN_LIMIT = 16000  # Same threshold as eviction
TRUNCATION_GUIDANCE = "... [results truncated, try being more specific with your parameters]"


def hash_string_sha256(input_string):
    # Encode the input string to bytes
    encoded_string = input_string.encode("utf-8")

    # Create a SHA-256 hash object
    sha256_hash = hashlib.sha256()

    # Update the hash object with the encoded string
    sha256_hash.update(encoded_string)

    # Get the hexadecimal digest of the hash
    hex_digest = sha256_hash.hexdigest()

    return hex_digest


def parse_structured_output(content: str, response_model: Type[BaseModel]) -> Optional[BaseModel]:
    """Parse LLM text output into a Pydantic model.

    Attempts multiple parsing strategies:
    1. Direct JSON validation (model_validate_json + json.loads fallback)
    2. Extract JSON from markdown code blocks
    3. Find the outermost JSON object via brace matching
    4. Last resort: json.loads on original content
    """

    def _try_parse(text: str) -> Optional[BaseModel]:
        """Try parsing text as JSON into the response model.

        Uses model_validate_json first (strict), then json.loads + model_validate
        which supports populate_by_name / alias.
        """
        # Strict JSON parsing
        try:
            return response_model.model_validate_json(text)
        except (ValidationError, json.JSONDecodeError):
            pass
        # Relaxed: json.loads + model_validate (supports alias / populate_by_name)
        try:
            data = json.loads(text)
            return response_model.model_validate(data)
        except (ValidationError, json.JSONDecodeError):
            pass
        return None

    # ---- Attempt 1: direct parse ----
    result = _try_parse(content)
    if result is not None:
        return result

    # ---- Attempt 2: extract JSON from markdown code blocks ----
    cleaned = content
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[-1].split("```")[0].strip()
    elif "```" in cleaned:
        parts = cleaned.split("```")
        if len(parts) >= 3:
            cleaned = parts[1].strip()

    if cleaned != content:
        result = _try_parse(cleaned)
        if result is not None:
            return result

    # ---- Attempt 3: find the outermost { ... } via brace matching ----
    json_str = _extract_outermost_json(cleaned)
    if json_str is None:
        json_str = _extract_outermost_json(content)

    if json_str is not None:
        result = _try_parse(json_str)
        if result is not None:
            return result
        logger.warning(f"Failed to parse extracted JSON into {response_model.__name__}")

    # ---- Attempt 4: last resort ----
    try:
        data = json.loads(content)
        return response_model.model_validate(data)
    except (ValidationError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to parse structured output: {e}")

    return None


def _extract_outermost_json(text: str) -> Optional[str]:
    """Extract the outermost JSON object from text using brace matching.

    This avoids destructive regex cleaning that can corrupt JSON values
    containing markdown characters like #, *, _, etc.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    end = start

    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            if in_string:
                escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                return text[start:end + 1]

    # If braces are unbalanced, return from start to last '}'
    last_brace = text.rfind("}")
    if last_brace > start:
        return text[start:last_brace + 1]

    return None


def truncate_if_too_long(result: list[str] | str) -> list[str] | str:
    """Truncate list or string result if it exceeds the token limit."""
    if isinstance(result, list):
        total_chars = sum(len(item) for item in result)
        if total_chars > TOOL_RESULT_TOKEN_LIMIT:
            return result[: len(result) * TOOL_RESULT_TOKEN_LIMIT // total_chars] + [TRUNCATION_GUIDANCE]
        return result
    # string
    if len(result) > TOOL_RESULT_TOKEN_LIMIT:
        return result[: TOOL_RESULT_TOKEN_LIMIT] + "\n" + TRUNCATION_GUIDANCE
    return result