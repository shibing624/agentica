# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Token counting utilities for messages, tools, and multi-modal content.

Supports:
- Text token counting with tiktoken (OpenAI), with CJK-aware character fallback
- Image token counting based on OpenAI's vision model formula
- Audio/Video token estimation
- Tool/Function definition token counting
"""
import json
import math
import unicodedata
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import tiktoken
from pydantic import BaseModel
from agentica.media import Audio, Image, Video
from agentica.model.message import Message
from agentica.tools.base import Function, ModelTool
from agentica.utils.log import logger

# Default image dimensions used as fallback when actual dimensions cannot be determined
DEFAULT_IMAGE_WIDTH = 1024
DEFAULT_IMAGE_HEIGHT = 1024


@lru_cache(maxsize=16)
def _get_tiktoken_encoding(model_id: str):
    """Get tiktoken encoding for a model, with caching.

    OpenAI model ids map directly. Unknown ids (e.g. ``deepseek-v4-flash``)
    fall back to a general encoding. When tiktoken cannot load an encoding
    (an offline host with no network to fetch the ``.tiktoken`` blob), return
    ``None`` so callers fall back to the char-based estimator instead of
    hanging on a slow fetch or crashing the run.
    """
    try:
        return tiktoken.encoding_for_model(model_id.lower())
    except KeyError:
        pass
    try:
        return tiktoken.get_encoding("o200k_base")
    except Exception:
        logger.warning(
            "tiktoken encoding unavailable (offline host?); "
            "falling back to char-based token estimate for %r",
            model_id,
        )
        return None


def _estimate_tokens_by_chars(text: str) -> int:
    """Estimate token count using character-based heuristics.

    - CJK characters (Chinese/Japanese/Korean): ~1 token per character
    - Other characters (English, numbers, punctuation, etc.): ~1 token per 4 characters
    """
    cjk_chars = 0
    other_chars = 0
    for ch in text:
        if unicodedata.category(ch).startswith('L') and ord(ch) > 0x2E7F:
            # CJK Unified Ideographs and extensions, Hangul, Kana, etc.
            cjk_chars += 1
        else:
            other_chars += 1
    return cjk_chars + (other_chars + 3) // 4


# =============================================================================
# Tool Token Counting
# =============================================================================
# OpenAI counts tool/function tokens by converting them to a TypeScript-like
# namespace format.
# =============================================================================


def _format_function_definitions(tools: List[Dict[str, Any]]) -> str:
    """
    Formats tool definitions as a TypeScript namespace for token counting.
    
    Returns:
        A TypeScript namespace string representation of all tools.
    """
    lines = []
    lines.append("namespace functions {")
    lines.append("")

    for tool in tools:
        # Handle both {"function": {...}} and direct function dict formats
        function = tool.get("function", tool)
        if function_description := function.get("description"):
            lines.append(f"// {function_description}")

        function_name = function.get("name", "")
        parameters = function.get("parameters", {})
        properties = parameters.get("properties", {})

        if properties:
            lines.append(f"type {function_name} = (_: {{")
            lines.append(_format_object_parameters(parameters, 0))
            lines.append("}) => any;")
        else:
            # Functions with no parameters
            lines.append(f"type {function_name} = () => any;")
        lines.append("")

    lines.append("} // namespace functions")
    return "\n".join(lines)


def _format_object_parameters(parameters: Dict[str, Any], indent: int) -> str:
    """Format JSON Schema object properties as TypeScript object properties."""
    properties = parameters.get("properties", {})
    if not properties:
        return ""

    required_params = parameters.get("required", [])
    lines = []

    for key, props in properties.items():
        # Add property description as a comment
        description = props.get("description")
        if description:
            lines.append(f"// {description}")

        # Required params have no "?", optional params have "?"
        question = "" if required_params and key in required_params else "?"
        lines.append(f"{key}{question}: {_format_type(props, indent)},")

    return "\n".join([" " * max(0, indent) + line for line in lines])


def _format_type(props: Dict[str, Any], indent: int) -> str:
    """Convert a JSON Schema type to its TypeScript equivalent."""
    type_name = props.get("type", "any")

    if type_name == "string":
        if "enum" in props:
            return " | ".join([f'"{item}"' for item in props["enum"]])
        return "string"
    elif type_name == "array":
        items = props.get("items", {})
        return f"{_format_type(items, indent)}[]"
    elif type_name == "object":
        return f"{{\n{_format_object_parameters(props, indent + 2)}\n}}"
    elif type_name in ["integer", "number"]:
        if "enum" in props:
            return " | ".join([f'"{item}"' for item in props["enum"]])
        return "number"
    elif type_name == "boolean":
        return "boolean"
    elif type_name == "null":
        return "null"
    else:
        return "any"


def _normalize_visual_dimensions(width: int, height: int) -> Tuple[int, int]:
    """Normalize visual dimensions to OpenAI vision counting bounds."""
    if width <= 0 or height <= 0:
        return 0, 0

    normalized_width = width
    normalized_height = height

    if max(normalized_width, normalized_height) > 2000:
        scale = 2000 / max(normalized_width, normalized_height)
        normalized_width = int(normalized_width * scale)
        normalized_height = int(normalized_height * scale)

    if min(normalized_width, normalized_height) > 768:
        scale = 768 / min(normalized_width, normalized_height)
        normalized_width = int(normalized_width * scale)
        normalized_height = int(normalized_height * scale)

    return normalized_width, normalized_height


def _count_tiled_visual_tokens(width: int, height: int) -> int:
    """Count tokens for a normalized high-detail visual input."""
    normalized_width, normalized_height = _normalize_visual_dimensions(width, height)
    if normalized_width <= 0 or normalized_height <= 0:
        return 0

    tiles = math.ceil(normalized_width / 512) * math.ceil(normalized_height / 512)
    return 85 + (170 * tiles)


# =============================================================================
# Image Dimension Parsing
# =============================================================================


def _get_image_type(data: bytes) -> Optional[str]:
    """Returns the image format from magic bytes in the file header."""
    if len(data) < 12:
        return None
    # PNG: 8-byte signature
    if data[0:8] == b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a":
        return "png"
    # GIF: "GIF8" followed by "9a" or "7a"
    if data[0:4] == b"GIF8" and data[5:6] == b"a":
        return "gif"
    # JPEG: SOI marker
    if data[0:3] == b"\xff\xd8\xff":
        return "jpeg"
    # HEIC/HEIF: ftyp box at offset 4
    if data[4:8] == b"ftyp":
        return "heic"
    # WebP: RIFF container with WEBP identifier
    if data[0:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def _decode_data_url_bytes(url: str) -> Optional[bytes]:
    """Decode a base64 data URL into bytes."""
    if not url.startswith("data:") or ";base64," not in url:
        return None

    import base64

    _, encoded = url.split(";base64,", 1)
    try:
        return base64.b64decode(encoded)
    except Exception:
        return None


def _parse_image_dimensions_from_bytes(data: bytes, img_type: Optional[str] = None) -> Tuple[int, int]:
    """Returns the image dimensions (width, height) from raw image bytes."""
    import io
    import struct

    if img_type is None:
        img_type = _get_image_type(data)

    if img_type == "png":
        # PNG IHDR chunk: width at offset 16, height at offset 20 (big-endian)
        return struct.unpack(">LL", data[16:24])
    elif img_type == "gif":
        # GIF logical screen descriptor: width/height at offset 6 (little-endian)
        return struct.unpack("<HH", data[6:10])
    elif img_type == "jpeg":
        # JPEG requires scanning for SOF markers
        with io.BytesIO(data) as f:
            f.seek(0)
            size = 2
            ftype = 0
            while not 0xC0 <= ftype <= 0xCF or ftype in (0xC4, 0xC8, 0xCC):
                f.seek(size, 1)
                byte = f.read(1)
                while ord(byte) == 0xFF:
                    byte = f.read(1)
                ftype = ord(byte)
                size = struct.unpack(">H", f.read(2))[0] - 2
            f.seek(1, 1)
            h, w = struct.unpack(">HH", f.read(4))
        return w, h
    elif img_type == "webp":
        if data[12:16] == b"VP8X":
            w = struct.unpack("<I", data[24:27] + b"\x00")[0] + 1
            h = struct.unpack("<I", data[27:30] + b"\x00")[0] + 1
            return w, h
        elif data[12:16] == b"VP8 ":
            w = struct.unpack("<H", data[26:28])[0] & 0x3FFF
            h = struct.unpack("<H", data[28:30])[0] & 0x3FFF
            return w, h
        elif data[12:16] == b"VP8L":
            bits = struct.unpack("<I", data[21:25])[0]
            w = (bits & 0x3FFF) + 1
            h = ((bits >> 14) & 0x3FFF) + 1
            return w, h

    return DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT


def _load_image_header_bytes(image: Image) -> Optional[bytes]:
    """Load local image bytes needed for dimension parsing.

    Token estimation must stay side-effect light. We only inspect bytes we
    already have locally: direct content, local file paths, or inline data URLs.
    Remote URLs deliberately skip network fetches and fall back to defaults.
    """
    if image.content:
        return image.content if isinstance(image.content, bytes) else image.content.encode()

    if image.filepath:
        with open(image.filepath, "rb") as f:
            return f.read(65536)

    if image.url:
        return _decode_data_url_bytes(image.url)

    return None


def _get_image_dimensions(image: Image) -> Tuple[int, int]:
    """Returns the image dimensions (width, height) from an Image object."""
    try:
        img_format = image.format
        data = _load_image_header_bytes(image)
        if data is None:
            return DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT

        return _parse_image_dimensions_from_bytes(data, img_format)
    except Exception:
        return DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT


def _get_compatible_attr(obj: Any, name: str, default: Any) -> Any:
    """Read optional attributes from heterogeneous media objects.

    Audio/Video token estimation is used across provider-specific objects that
    do not all share one strict runtime class, so this remains a narrow
    compatibility boundary.
    """
    return getattr(obj, name, default) or default


def _count_message_content_item_tokens(item: Any, model_id: str) -> Tuple[List[str], int]:
    """Count tokens for one message content item, returning text and media totals."""
    text_parts: List[str] = []
    media_tokens = 0

    if isinstance(item, str):
        text_parts.append(item)
        return text_parts, media_tokens

    if not isinstance(item, dict):
        text_parts.append(str(item))
        return text_parts, media_tokens

    item_type = item.get("type", "")
    if item_type == "text":
        text = item.get("text", "")
        if isinstance(text, str) and text:
            text_parts.append(text)
        return text_parts, media_tokens

    if item_type == "image_url":
        image_url_data = item.get("image_url", {})
        if not isinstance(image_url_data, dict):
            text_parts.append(json.dumps(item, ensure_ascii=False))
            return text_parts, media_tokens

        url = image_url_data.get("url")
        detail = image_url_data.get("detail", "auto")
        if isinstance(url, str) and url:
            media_tokens += count_image_tokens(Image(url=url, detail=detail))
        else:
            text_parts.append(json.dumps(item, ensure_ascii=False))
        return text_parts, media_tokens

    text_parts.append(json.dumps(item, ensure_ascii=False))
    return text_parts, media_tokens


def _to_tool_dict(tool: Union[Function, ModelTool, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Normalize supported tool representations into dict form."""
    if isinstance(tool, Function):
        return tool.to_dict()
    if isinstance(tool, ModelTool):
        return tool.to_dict()
    if isinstance(tool, dict):
        return tool
    return None


# =============================================================================
# Token Counting Functions
# =============================================================================


def count_text_tokens(text: str, model_id: str = "gpt-4o") -> int:
    """Count tokens in a text string using tiktoken (char-based fallback)."""
    if not text:
        return 0
    encoding = _get_tiktoken_encoding(model_id)
    if encoding is None:
        return _estimate_tokens_by_chars(text)
    return len(encoding.encode(text, disallowed_special=()))


def count_image_tokens(image: Union[Image, str, bytes]) -> int:
    """
    Count tokens for an image based on OpenAI's vision model formula.

    Formula:
    1. If max(width, height) > 2000: scale to fit in 2000px on longest side
    2. If min(width, height) > 768: scale so shortest side is 768px
    3. tiles = ceil(width/512) * ceil(height/512)
    4. tokens = 85 + (170 * tiles)
    """
    # String URLs / base64 strings don't carry dimension info — use low-detail default
    if isinstance(image, (str, bytes)):
        return 85

    width, height = _get_image_dimensions(image)
    detail = image.detail or "auto"

    if width <= 0 or height <= 0:
        return 0

    # Low detail: fixed 85 tokens
    if detail == "low":
        return 85

    return _count_tiled_visual_tokens(width, height)


def count_audio_tokens(audio: Audio, duration: Optional[float] = None) -> int:
    """
    Estimate tokens for audio based on duration.
    Uses ~25 tokens per second (conservative estimate).
    """
    if duration is None:
        duration = _get_compatible_attr(audio, "duration", 0)
    if duration <= 0:
        return 0
    return int(duration * 25)


def count_video_tokens(video: Video, duration: Optional[float] = None, fps: float = 1.0) -> int:
    """
    Estimate tokens for video by treating it as a sequence of images.
    """
    if duration is None:
        duration = _get_compatible_attr(video, "duration", 0)
    if duration <= 0:
        return 0

    width = _get_compatible_attr(video, "width", 512)
    height = _get_compatible_attr(video, "height", 512)
    fps = _get_compatible_attr(video, "fps", fps)
    tokens_per_frame = _count_tiled_visual_tokens(width, height)

    num_frames = max(int(duration * fps), 1)
    return num_frames * tokens_per_frame


def count_tool_tokens(
    tools: Sequence[Union[Function, ModelTool, Dict[str, Any]]],
    model_id: str = "gpt-4o",
) -> int:
    """Count tokens consumed by tool/function definitions."""
    if not tools:
        return 0

    # Convert Function objects to dict format
    tool_dicts = []
    for tool in tools:
        tool_dict = _to_tool_dict(tool)
        if tool_dict is not None:
            tool_dicts.append(tool_dict)

    formatted = _format_function_definitions(tool_dicts)
    return count_text_tokens(formatted, model_id)


def count_schema_tokens(
    output_schema: Optional[Union[Dict, Type[BaseModel]]],
    model_id: str = "gpt-4o",
) -> int:
    """Estimate tokens for output schema."""
    if output_schema is None:
        return 0

    try:
        if isinstance(output_schema, type) and issubclass(output_schema, BaseModel):
            schema = output_schema.model_json_schema()
        elif isinstance(output_schema, dict):
            schema = output_schema
        else:
            return 0

        schema_json = json.dumps(schema)
        return count_text_tokens(schema_json, model_id)
    except Exception:
        return 0


def count_message_tokens(message: Message, model_id: str = "gpt-4o") -> int:
    """Count tokens in a single message, including text and media."""
    tokens = 0
    text_parts: List[str] = []

    content = message.compressed_content if message.compressed_content is not None else message.content
    if content:
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for item in content:
                item_text_parts, item_tokens = _count_message_content_item_tokens(item, model_id)
                text_parts.extend(item_text_parts)
                tokens += item_tokens
        else:
            text_parts.append(str(content))

    # Collect tool call arguments
    if message.tool_calls:
        for tool_call in message.tool_calls:
            if isinstance(tool_call, dict) and "function" in tool_call:
                args = tool_call["function"].get("arguments", "")
                text_parts.append(str(args))

    # Collect tool response id
    if message.tool_call_id:
        text_parts.append(message.tool_call_id)

    # Collect reasoning content
    if message.reasoning_content:
        text_parts.append(message.reasoning_content)

    # Collect name field
    if message.name:
        text_parts.append(message.name)

    # Count all text tokens
    if text_parts:
        tokens += count_text_tokens(" ".join(text_parts), model_id)

    # Count media tokens
    if message.images:
        for image in message.images:
            tokens += count_image_tokens(image)

    if message.audio:
        if isinstance(message.audio, (list, tuple)):
            for audio in message.audio:
                tokens += count_audio_tokens(audio)
        else:
            tokens += count_audio_tokens(message.audio)

    if message.videos:
        for video in message.videos:
            tokens += count_video_tokens(video)

    return tokens


def count_tokens(
    messages: List["Message"],
    tools: Optional[List[Union["Function", Dict[str, Any]]]] = None,
    model_id: str = "gpt-4o",
    output_schema: Optional[Union[Dict, Type[BaseModel]]] = None,
) -> int:
    """
    Count total tokens for messages, tools, and output schema.
    
    Args:
        messages: List of Message objects
        tools: Optional list of tools/functions
        model_id: Model identifier for tokenizer selection
        output_schema: Optional output schema (Pydantic model or dict)
    
    Returns:
        Total token count
    """
    total = 0
    model_id = model_id.lower()

    # Count message tokens
    if messages:
        for msg in messages:
            total += count_message_tokens(msg, model_id)

    # Add tool tokens
    if tools:
        total += count_tool_tokens(tools, model_id)

    # Add output schema tokens
    if output_schema is not None:
        total += count_schema_tokens(output_schema, model_id)

    return total
