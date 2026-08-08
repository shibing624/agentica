# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: CLI display package entry — common rendering helpers for the TUI
"""

from agentica.cli.display.console import (
    COLORS,
    clear_truncated_blocks,
    display_agent_execution_error,
    get_last_truncated,
    get_truncated_blocks,
    remember_truncated,
)
from agentica.cli.display.help_header import (
    format_session_summary,
    print_header,
    resumable_session_id,
    show_help,
)
from agentica.cli.display.messages import (
    display_diff,
    display_user_message,
    get_file_completions,
    inject_file_contents,
    parse_file_mentions,
    render_markdown_response,
)
from agentica.cli.display.status_bar import (
    build_context_bar,
    build_status_bar_fragments,
    context_pct_style,
    display_token_stats,
    format_duration_compact,
)
from agentica.cli.display.stream import StreamDisplayManager
from agentica.cli.display.tool_format import display_tool_call, format_tool_display

__all__ = [
    "COLORS",
    "StreamDisplayManager",
    "build_context_bar",
    "build_status_bar_fragments",
    "clear_truncated_blocks",
    "context_pct_style",
    "display_agent_execution_error",
    "display_diff",
    "display_token_stats",
    "display_tool_call",
    "display_user_message",
    "format_duration_compact",
    "format_session_summary",
    "format_tool_display",
    "get_file_completions",
    "get_last_truncated",
    "get_truncated_blocks",
    "inject_file_contents",
    "parse_file_mentions",
    "print_header",
    "remember_truncated",
    "render_markdown_response",
    "resumable_session_id",
    "show_help",
]
