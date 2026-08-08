# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
Shared async file I/O and frontmatter parsing utilities.

Used by workspace.py and agentica/experience/compiled_store.py.
Single source of truth — do not duplicate these functions.
"""
import asyncio
import functools
import json
import re
from pathlib import Path
from typing import List, Optional


async def async_read_text(path: Path, encoding: str = "utf-8") -> str:
    """Read text file in executor to avoid blocking event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(path.read_text, encoding=encoding))


async def async_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write text file in executor to avoid blocking event loop."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, functools.partial(path.write_text, content, encoding=encoding))


def extract_frontmatter_value(content: str, key: str) -> Optional[str]:
    """Extract a value from YAML frontmatter."""
    m = re.search(rf"^{re.escape(key)}:\s*(.+)$", content, re.MULTILINE)
    return m.group(1).strip() if m else None


def extract_frontmatter_int(content: str, key: str, default: int = 0) -> int:
    """Extract an integer value from YAML frontmatter."""
    m = re.search(rf"^{re.escape(key)}:\s*(\d+)", content, re.MULTILINE)
    return int(m.group(1)) if m else default


def strip_frontmatter(content: str) -> str:
    """Remove a leading YAML frontmatter block from ``content``.

    The block must open on the very first line and close on a line that is
    exactly ``---``. Scanning for the next ``---`` anywhere instead cuts inside
    a value: ``source_tasks`` stores raw user queries, which routinely contain
    ``---``, and the tail of the frontmatter then gets spliced into the body.
    Because the body is carried forward on every rewrite, that tail accumulates
    and eventually lands in the system prompt.
    """
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return content.strip()
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[i + 1:]).strip()
    # Unterminated block: treat the whole file as body rather than dropping it.
    return content.strip()


def extract_frontmatter_list(content: str, key: str) -> List[str]:
    """Extract a list value from frontmatter, written as a JSON array.

    We deliberately use JSON-array syntax (e.g. ``source_tasks: ["a", "b"]``)
    instead of YAML's block-style list to keep parsing pure-stdlib and to
    avoid multi-line frontmatter changes that would complicate the existing
    ``extract_frontmatter_value`` regex callers. JSON arrays are valid YAML,
    so any standard YAML reader still parses these values.
    """
    raw = extract_frontmatter_value(content, key)
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return []
    if isinstance(parsed, list):
        return [str(x) for x in parsed if isinstance(x, (str, int, float))]
    return []


def format_frontmatter_list(values: List[str]) -> str:
    """Render a list of strings as a one-line JSON array for frontmatter."""
    return json.dumps(values, ensure_ascii=False)
