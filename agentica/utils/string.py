import hashlib
import json
import re
from typing import Optional, Type

from pydantic import BaseModel, ValidationError
from agentica.utils.log import logger

TOOL_RESULT_TOKEN_LIMIT = 20000  # Same threshold as eviction
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
    structured_output = None
    try:
        # First attempt: direct JSON validation
        structured_output = response_model.model_validate_json(content)
    except (ValidationError, json.JSONDecodeError):
        # Second attempt: Extract JSON from markdown code blocks and clean
        content = content

        # Handle code blocks
        if "```json" in content:
            content = content.split("```json")[-1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()

        # Clean the JSON string
        # Remove markdown formatting
        content = re.sub(r"[*_`#]", "", content)

        # Handle newlines and control characters
        content = content.replace("\n", " ").replace("\r", "")
        content = re.sub(r"[\x00-\x1F\x7F]", "", content)

        # Escape quotes only in values, not keys
        def escape_quotes_in_values(match):
            key = match.group(1)
            value = match.group(2)
            # Escape quotes in the value portion only
            escaped_value = value.replace('"', '\\"')
            return f'"{key}": "{escaped_value}'

        # Find and escape quotes in field values
        content = re.sub(r'"(?P<key>[^"]+)"\s*:\s*"(?P<value>.*?)(?="\s*(?:,|\}))', escape_quotes_in_values, content)

        try:
            # Try parsing the cleaned JSON
            structured_output = response_model.model_validate_json(content)
        except (ValidationError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to parse cleaned JSON: {e}")

            try:
                # Final attempt: Try parsing as Python dict
                data = json.loads(content)
                structured_output = response_model.model_validate(data)
            except (ValidationError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to parse as Python dict: {e}")

    return structured_output


def truncate_if_too_long(result: list[str] | str) -> list[str] | str:
    """Truncate list or string result if it exceeds token limit (rough estimate: 4 chars/token)."""
    if isinstance(result, list):
        total_chars = sum(len(item) for item in result)
        if total_chars > TOOL_RESULT_TOKEN_LIMIT * 4:
            return result[: len(result) * TOOL_RESULT_TOKEN_LIMIT * 4 // total_chars] + [TRUNCATION_GUIDANCE]
        return result
    # string
    if len(result) > TOOL_RESULT_TOKEN_LIMIT * 4:
        return result[: TOOL_RESULT_TOKEN_LIMIT * 4] + "\n" + TRUNCATION_GUIDANCE
    return result