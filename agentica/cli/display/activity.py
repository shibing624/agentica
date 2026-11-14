# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tool activity categories shared with Web's workActivity.ts vocabulary.
"""

CATEGORY_BY_TOOL = {
    "read_file": "read",
    "read_image": "read",
    "ls": "read",
    "list_dir": "read",
    "glob": "search",
    "grep": "search",
    "search_session": "search",
    "search_files": "search",
    "apply_patch": "edit",
    "edit_file": "edit",
    "multi_edit": "edit",
    "write_file": "write",
    "execute": "commands",
    "shell": "commands",
    "bash": "commands",
    "run_python": "commands",
    "python": "commands",
    "web_search": "webSearch",
    "fetch_url": "webFetch",
    "web_fetch": "webFetch",
    "task": "subagents",
    "write_todos": "plan",
    "goal": "plan",
    "ask_user_question": "questions",
}


def tool_category(tool_name: str) -> str:
    """Return the activity category used by Web group headers for one tool."""
    return CATEGORY_BY_TOOL.get(tool_name, "tools")
