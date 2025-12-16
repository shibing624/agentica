# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Token counting utilities for messages, tools, and multi-modal content.

Supports:
- Text token counting with tiktoken (OpenAI) and HuggingFace tokenizers
- Image token counting based on OpenAI's vision model formula
- Audio/Video token estimation
- Tool/Function definition token counting
"""
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

from pydantic import BaseModel

from agentica.utils.log import logger

# Default image dimensions used as fallback when actual dimensions cannot be determined
DEFAULT_IMAGE_WIDTH = 1024
DEFAULT_IMAGE_HEIGHT = 1024


@lru_cache(maxsize=16)
def _get_tiktoken_encoding(model_id: str):
    """Get tiktoken encoding for a model, with caching."""
    model_id = model_id.lower()
    try:
        import tiktoken

        try:
            # Use model-specific encoding
            return tiktoken.encoding_for_model(model_id)
        except KeyError:
            return tiktoken.get_encoding("o200k_base")
    except ImportError:
        logger.warning("tiktoken not installed. Please install it using `pip install tiktoken`.")
        return None


@lru_cache(maxsize=16)
def _get_hf_tokenizer(model_id: str):
    """Get HuggingFace tokenizer for specific models."""
    try:
        from tokenizers import Tokenizer

        model_id = model_id.lower()

        # Llama-3 models
        if "llama-3" in model_id or "llama3" in model_id:
            return Tokenizer.from_pretrained("Xenova/llama-3-tokenizer")

        # Llama-2 models
        if "llama-2" in model_id or "llama2" in model_id or "replicate" in model_id:
            return Tokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

        # Cohere command-r models
        if "command-r" in model_id:
            return Tokenizer.from_pretrained("Xenova/c4ai-command-r-v01-tokenizer")

        return None
    except ImportError:
        logger.warning("tokenizers not installed. Please install it using `pip install tokenizers`.")
        return None
    except Exception:
        return None


def _select_tokenizer(model_id: str) -> Tuple[str, Any]:
    """Select the best available tokenizer for a model."""
    # Priority 1: HuggingFace tokenizers for models with specific tokenizers
    hf_tokenizer = _get_hf_tokenizer(model_id)
    if hf_tokenizer is not None:
        return ("huggingface", hf_tokenizer)

    # Priority 2: tiktoken for OpenAI models
    tiktoken_enc = _get_tiktoken_encoding(model_id)
    if tiktoken_enc is not None:
        return ("tiktoken", tiktoken_enc)

    # Fallback: No tokenizer available, will use character-based estimation
    return ("none", None)


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


def _get_image_dimensions(image: "Image") -> Tuple[int, int]:
    """Returns the image dimensions (width, height) from an Image object."""
    try:
        # Try to get format hint from metadata
        img_format = getattr(image, 'format', None)

        # Get raw bytes from the appropriate source
        if hasattr(image, 'content') and image.content:
            data = image.content if isinstance(image.content, bytes) else image.content.encode()
        elif hasattr(image, 'filepath') and image.filepath:
            with open(image.filepath, "rb") as f:
                data = f.read(100)  # Only need header bytes
        elif hasattr(image, 'url') and image.url:
            import httpx
            response = httpx.get(image.url, timeout=5)
            data = response.content
        else:
            return DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT

        return _parse_image_dimensions_from_bytes(data, img_format)
    except Exception:
        return DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT


# =============================================================================
# Token Counting Functions
# =============================================================================


def count_text_tokens(text: str, model_id: str = "gpt-4o") -> int:
    """Count tokens in a text string."""
    if not text:
        return 0
    tokenizer_type, tokenizer = _select_tokenizer(model_id)
    if tokenizer_type == "huggingface":
        return len(tokenizer.encode(text).ids)
    elif tokenizer_type == "tiktoken":
        return len(tokenizer.encode(text, disallowed_special=()))
    else:
        # Fallback: ~4 characters per token
        return len(text) // 4


def count_image_tokens(image: "Image") -> int:
    """
    Count tokens for an image based on OpenAI's vision model formula.
    
    Formula:
    1. If max(width, height) > 2000: scale to fit in 2000px on longest side
    2. If min(width, height) > 768: scale so shortest side is 768px
    3. tiles = ceil(width/512) * ceil(height/512)
    4. tokens = 85 + (170 * tiles)
    """
    width, height = _get_image_dimensions(image)
    detail = getattr(image, 'detail', None) or "auto"

    if width <= 0 or height <= 0:
        return 0

    # Low detail: fixed 85 tokens
    if detail == "low":
        return 85

    # For auto/high detail, calculate based on dimensions
    if max(width, height) > 2000:
        scale = 2000 / max(width, height)
        width, height = int(width * scale), int(height * scale)

    if min(width, height) > 768:
        scale = 768 / min(width, height)
        width, height = int(width * scale), int(height * scale)

    tiles = math.ceil(width / 512) * math.ceil(height / 512)
    return 85 + (170 * tiles)


def count_audio_tokens(audio: "Audio", duration: Optional[float] = None) -> int:
    """
    Estimate tokens for audio based on duration.
    Uses ~25 tokens per second (conservative estimate).
    """
    if duration is None:
        duration = getattr(audio, 'duration', 0) or 0
    if duration <= 0:
        return 0
    return int(duration * 25)


def count_video_tokens(video: "Video", duration: Optional[float] = None, fps: float = 1.0) -> int:
    """
    Estimate tokens for video by treating it as a sequence of images.
    """
    if duration is None:
        duration = getattr(video, 'duration', 0) or 0
    if duration <= 0:
        return 0

    width = getattr(video, 'width', 512) or 512
    height = getattr(video, 'height', 512) or 512
    fps = getattr(video, 'fps', fps) or fps

    # Calculate tokens per frame
    w, h = width, height
    if max(w, h) > 2000:
        scale = 2000 / max(w, h)
        w, h = int(w * scale), int(h * scale)
    if min(w, h) > 768:
        scale = 768 / min(w, h)
        w, h = int(w * scale), int(h * scale)
    tiles = math.ceil(w / 512) * math.ceil(h / 512)
    tokens_per_frame = 85 + (170 * tiles)

    num_frames = max(int(duration * fps), 1)
    return num_frames * tokens_per_frame


def count_tool_tokens(
    tools: Sequence[Union["Function", Dict[str, Any]]],
    model_id: str = "gpt-4o",
) -> int:
    """Count tokens consumed by tool/function definitions."""
    if not tools:
        return 0

    # Convert Function objects to dict format
    tool_dicts = []
    for tool in tools:
        if hasattr(tool, 'to_dict'):
            tool_dicts.append(tool.to_dict())
        elif isinstance(tool, dict):
            tool_dicts.append(tool)

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


def count_message_tokens(message: "Message", model_id: str = "gpt-4o") -> int:
    """Count tokens in a single message, including text and media."""
    tokens = 0
    text_parts: List[str] = []

    # Collect content text
    content = message.content
    if hasattr(message, 'compressed_content') and message.compressed_content:
        content = message.compressed_content
    
    if content:
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict):
                    item_type = item.get("type", "")
                    if item_type == "text":
                        text_parts.append(item.get("text", ""))
                    elif item_type == "image_url":
                        # Handle OpenAI-style content lists
                        image_url_data = item.get("image_url", {})
                        url = image_url_data.get("url") if isinstance(image_url_data, dict) else None
                        detail = image_url_data.get("detail", "auto") if isinstance(image_url_data, dict) else "auto"
                        # Create a simple object to pass to count_image_tokens
                        class TempImage:
                            pass
                        temp_image = TempImage()
                        temp_image.url = url
                        temp_image.detail = detail
                        temp_image.content = None
                        temp_image.filepath = None
                        tokens += count_image_tokens(temp_image)
                    else:
                        text_parts.append(json.dumps(item))
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
    if hasattr(message, 'reasoning_content') and message.reasoning_content:
        text_parts.append(message.reasoning_content)

    # Collect name field
    if message.name:
        text_parts.append(message.name)

    # Count all text tokens
    if text_parts:
        tokens += count_text_tokens(" ".join(text_parts), model_id)

    # Count media tokens
    if hasattr(message, 'images') and message.images:
        for image in message.images:
            tokens += count_image_tokens(image)

    if hasattr(message, 'audio') and message.audio:
        if isinstance(message.audio, (list, tuple)):
            for audio in message.audio:
                tokens += count_audio_tokens(audio)
        else:
            tokens += count_audio_tokens(message.audio)

    if hasattr(message, 'videos') and message.videos:
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
