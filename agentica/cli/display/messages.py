# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: User-message, markdown, token-stat, and diff rendering
"""

import difflib
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

from rich.markdown import Markdown
from rich.padding import Padding
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from agentica.cli.runtime import get_console
from agentica.peers import PeerMessage

from .console import (
    COLORS,
    _PASTE_PATH_RE,
    clear_truncated_blocks,
)

def parse_file_mentions(text: str) -> Tuple[str, List[Path]]:
    """Parse @file mentions and return text with mentioned files.
    
    Uses lookbehind to avoid matching email addresses.
    """
    pattern = r"(?:^|(?<=\s))@([\w./-]+)"
    mentioned_files = []
    
    for match in re.finditer(pattern, text):
        file_path_str = match.group(1)
        file_path = Path(file_path_str).expanduser()
        if not file_path.is_absolute():
            file_path = Path.cwd() / file_path
        if file_path.exists() and file_path.is_file():
            mentioned_files.append(file_path)
    
    # Remove @ mentions from text for cleaner display
    processed_text = re.sub(pattern, r'\1', text)
    return processed_text, mentioned_files


def inject_file_contents(prompt_text: str, mentioned_files: List[Path]) -> str:
    """Inject file contents into the prompt."""
    if not mentioned_files:
        return prompt_text
    
    context_parts = [prompt_text, "\n\n## Referenced Files\n"]
    for file_path in mentioned_files:
        try:
            content = file_path.read_text(encoding="utf-8")
            # Limit file content to reasonable size
            if len(content) > 20000:
                content = content[:20000] + "\n... (file truncated)"
            context_parts.append(
                f"\n### {file_path.name}\nPath: `{file_path}`\n```\n{content}\n```"
            )
        except Exception as e:
            context_parts.append(f"\n### {file_path.name}\n[Error reading file: {e}]")
    
    return "\n".join(context_parts)

def _format_attachment_size(path: Path) -> str:
    """Return a compact file size for an attachment label."""
    size = path.stat().st_size
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{round(size / 1024)}KB"
    return f"{size / (1024 * 1024):.1f}MB"


def display_user_message(
    text: str,
    *,
    pasted_blocks: int = 0,
    pasted_lines: int = 0,
    images: Optional[List[Path]] = None,
) -> None:
    """Display the full user message and image attachments in one input panel.

    Long messages are no longer folded behind Ctrl+O — the complete text is
    always rendered inline so the user sees exactly what they sent.
    """
    # A new user turn starts here: drop the previous turn's folded blocks so
    # Ctrl+O expands the CURRENT turn's tool results, not stale history.
    clear_truncated_blocks()
    cleaned = _PASTE_PATH_RE.sub("", text).strip()
    if not cleaned and pasted_blocks:
        cleaned = f"[Pasted text: {pasted_lines} lines]"

    # Always render the complete message (no Ctrl+O fold); only colorize
    # @file mentions.
    pattern = r"(@[\w./-]+)"
    parts = re.split(pattern, cleaned)
    rich_text = Text()
    for part in parts:
        if part.startswith("@"):
            rich_text.append(part, style="bold magenta")
        else:
            rich_text.append(part, style=f"bold {COLORS['user']}")

    if pasted_blocks:
        suffix = "s" if pasted_blocks > 1 else ""
        rich_text.append(
            f" ({pasted_blocks} pasted block{suffix}, {pasted_lines} lines total)",
            style="dim",
        )

    for index, image in enumerate(images or [], start=1):
        if rich_text.plain:
            rich_text.append("\n")
        rich_text.append(
            f"📎 Image #{index} attached: {image.name} ({_format_attachment_size(image)})",
            style="dim",
        )

    _echo_panel(rich_text)


def _echo_panel(
    body: Text,
    *,
    marker: str = "❯",
    marker_style: str = "bold bright_yellow",
) -> None:
    """Render one incoming request in the transcript's history panel.

    Echoed on a subtle full-width background so it is easy to find while
    scanning a long conversation. No trailing blank line here: the response
    section (start_tool_section / _start_response) adds its spacing. A separate
    content column keeps wrapped and explicit continuation lines aligned.
    ``overflow="fold"`` is required: Rich Table defaults to ellipsis and
    silently truncates long (pasted or relayed) turns with "…".
    """
    history = Table.grid(padding=(0, 1), expand=True)
    history.add_column(no_wrap=True)
    history.add_column(ratio=1, overflow="fold")
    history.add_row(Text(marker, style=marker_style), body)
    console = get_console()
    console.print()
    console.print(Padding(history, (0, 1), style="on rgb(35,35,35)"))


def display_peer_messages(messages: List[PeerMessage]) -> None:
    """Show messages that just arrived from another session.

    Both user and agent messages are independent incoming requests, so both use
    the same history-panel shape. Their markers preserve the authority boundary:
    ``❯`` means the human spoke, while ``↳ 🖥️ <name>`` identifies another
    session's agent. The model-facing header separately enforces that an agent
    message does not carry user authority.
    """
    for message in messages:
        if message.from_user:
            body = Text()
            body.append(f"via {message.from_name}  ", style="dim")
            body.append(message.text, style=f"bold {COLORS['user']}")
            _echo_panel(body)
        else:
            body = Text()
            body.append(f"🖥️ {message.from_name}\n", style=f"bold {COLORS['tool']}")
            body.append(message.text, style=f"bold {COLORS['user']}")
            _echo_panel(body, marker="↳", marker_style=f"bold {COLORS['tool']}")


def get_file_completions(document_text: str) -> List[str]:
    """Get file completions for @ mentions."""
    import glob as glob_module

    # Find the @ mention being typed
    match = re.search(r"@([\w./-]*)$", document_text)
    if not match:
        return []
    
    partial = match.group(1)
    
    if partial:
        # Search for files matching the partial path (current dir only, not recursive)
        search_pattern = f"{partial}*"
        matches = glob_module.glob(search_pattern, recursive=False)
        # Also search one level of subdirectories (limited depth)
        if os.sep not in partial and "/" not in partial:
            for d in os.listdir("."):
                if os.path.isdir(d) and not d.startswith("."):
                    sub_matches = glob_module.glob(os.path.join(d, f"{partial}*"))
                    matches.extend(sub_matches[:5])
    else:
        # Show files in current directory
        matches = glob_module.glob("*")
    
    # Filter to only files (not directories) and limit results
    completions = []
    seen = set()
    for m in matches[:20]:
        if m in seen:
            continue
        seen.add(m)
        if os.path.isfile(m):
            completions.append(m)
        elif os.path.isdir(m):
            completions.append(m + "/")
    
    return completions

def _has_markdown(text: str) -> bool:
    """Detect if text contains Markdown formatting worth rendering."""
    if not text:
        return False
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            return True
        if stripped.startswith(("- ", "* ", "+ ", "> ")):
            return True
        if any(stripped.startswith(f"{i}. ") for i in range(1, 10)):
            return True
        if "```" in line or "**" in line or "`" in line:
            return True
    return "\n|" in text and "\n|-" in text


def render_markdown_response(console_instance, text: str) -> None:
    """Render a complete response as rich Markdown if it contains formatting."""
    if _has_markdown(text):
        console_instance.print(Markdown(text))
    else:
        console_instance.print(text, style=COLORS["agent"])


def display_diff(console_instance, file_path: str, old_content: str, new_content: str) -> None:
    """Display unified diff between old and new file content."""
    diff_lines = list(difflib.unified_diff(
        old_content.splitlines(),
        new_content.splitlines(),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        n=3,
        lineterm="",
    ))
    if diff_lines:
        diff_text = "\n".join(diff_lines)
        console_instance.print(Syntax(diff_text, "diff", theme="monokai", line_numbers=False))
