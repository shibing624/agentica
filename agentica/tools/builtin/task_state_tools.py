# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Canonical built-in task-state tools.
"""

import json
import re
from collections import OrderedDict
from datetime import date, timedelta
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from agentica.tools.base import Tool
from agentica.utils.log import logger

if TYPE_CHECKING:
    from agentica.agent import Agent


class BuiltinTodoTool(Tool):
    """
    Built-in task management tool providing write_todos function.
    Used for tracking progress of complex tasks.
    Todos are stored on the Agent instance when available, making them
    visible to the agent via tool_result and periodic reminders.

    Design (mirrors CC TodoWriteTool):
    - write_todos tool_result contains full todo state + guidance text
    - All-completed auto-clear: when every item is completed, list is cleared
    - Verification nudge: when 3+ tasks all completed and none is a verification
      step, tool_result appends a reminder to verify before reporting done
    - No system prompt injection of todos (avoids token waste / cache busting)
    - Periodic reminder injected by Runner when LLM hasn't called write_todos
      for N turns (see Runner._inject_todo_reminder)
    """

    WRITE_TODOS_SYSTEM_PROMPT = dedent("""## `write_todos`

    Use this tool for complex objectives to track each necessary step and give the user visibility into your progress.
    Writing todos takes time and tokens — only use it for complex many-step problems (3+ distinct steps), not for simple few-step requests.

    Critical rules:
    - Mark todos as completed as soon as each step is done. Do not batch completions.
    - The `write_todos` tool should NEVER be called multiple times in parallel.
    - Revise the todo list as you go — new information may reveal new tasks or make old tasks irrelevant.
    - The todo list will be shown in your tool results when you update it.
    - If you haven't updated it in a while, you may receive a reminder with the current state.""")

    _VERIFICATION_NUDGE = (
        "\n\nNOTE: You just closed out 3+ tasks and none of them was a verification step. "
        "Before writing your final summary, verify your work by running tests, linting, "
        "or checking the actual output. Do not self-declare completion without evidence -- "
        "review the results to confirm correctness."
    )

    def __init__(self):
        """Initialize BuiltinTodoTool."""
        super().__init__(name="builtin_todo_tool")
        self._agent: Optional["Agent"] = None
        self._todos: List[Dict[str, Any]] = []
        self.register(self.write_todos, is_destructive=True)

    def set_agent(self, agent: "Agent") -> None:
        """Receive agent reference so todos are stored on the agent."""
        self._agent = agent

    def clone(self) -> "BuiltinTodoTool":
        """Fresh instance so each agent owns its ``_agent`` slot and todos.

        Preserves the source's exposed ``functions`` keys so an upstream
        registry filter (e.g. ``SubagentRegistry._select_child_tools``) is not
        silently undone when the agent re-clones during ``_post_init``.
        """
        new = BuiltinTodoTool()
        if set(new.functions) != set(self.functions):
            new.functions = OrderedDict(
                (name, new.functions[name])
                for name in self.functions
                if name in new.functions
            )
        return new

    @property
    def todos(self) -> List[Dict[str, Any]]:
        if self._agent is not None:
            return self._agent.todos
        return self._todos

    @todos.setter
    def todos(self, value: List[Dict[str, Any]]) -> None:
        if self._agent is not None:
            self._agent.todos = value
        else:
            self._todos = value

    def get_system_prompt(self) -> Optional[str]:
        """Get the system prompt for todo tool usage guidance."""
        return self.WRITE_TODOS_SYSTEM_PROMPT

    @staticmethod
    def _needs_verification_nudge(todos: List[Dict[str, str]]) -> bool:
        """Check if verification nudge should be appended to tool_result."""
        if len(todos) < 3:
            return False
        if not all(t.get("status") == "completed" for t in todos):
            return False
        verification_pattern = re.compile(r"verif|test|lint|check|review|validate", re.IGNORECASE)
        if any(verification_pattern.search(t.get("content", "")) for t in todos):
            return False
        return True

    def write_todos(self, todos: Optional[List[Dict[str, str]]] = None) -> str:
        """Create and manage a structured task list.

        Use this tool to create and manage a structured task list for your current work session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.

        Only use this tool if you think it will be helpful in staying organized. If the user's request is trivial and takes less than 3 steps, it is better to NOT use this tool and just do the task directly.

        ## When to Use This Tool
        Use this tool in these scenarios:
        1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
        2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
        3. User explicitly requests todo list - When the user directly asks you to use the todo list
        4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
        5. The plan may need future revisions or updates based on results from the first few steps

        ## How to Use This Tool
        1. When you start working on a task - Mark it as in_progress BEFORE beginning work.
        2. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation.
        3. You can also update future tasks, such as deleting them if they are no longer necessary, or adding new tasks that are necessary. Don't change previously completed tasks.
        4. You can make several updates to the todo list at once. For example, when you complete a task, you can mark the next task you need to start as in_progress.

        ## When NOT to Use This Tool
        It is important to skip using this tool when:
        1. There is only a single, straightforward task
        2. The task is trivial and tracking it provides no benefit
        3. The task can be completed in less than 3 trivial steps
        4. The task is purely conversational or informational

        ## Task States and Management

        1. **Task States**: Use these states to track progress:
        - pending: Task not yet started
        - in_progress: Currently working on (you can have multiple tasks in_progress at a time if they are not related to each other and can be run in parallel)
        - completed: Task finished successfully

        2. **Task Management**:
        - Update task status in real-time as you work
        - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
        - Complete current tasks before starting new ones
        - Remove tasks that are no longer relevant from the list entirely
        - IMPORTANT: When you write this todo list, you should mark your first task (or tasks) as in_progress immediately!.
        - IMPORTANT: Unless all tasks are completed, you should always have at least one task in_progress to show the user that you are working on something.

        3. **Task Completion Requirements**:
        - ONLY mark a task as completed when you have FULLY accomplished it
        - If you encounter errors, blockers, or cannot finish, keep the task as in_progress
        - When blocked, create a new task describing what needs to be resolved
        - Never mark a task as completed if:
            - There are unresolved issues or errors
            - Work is partial or incomplete
            - You encountered blockers that prevent completion
            - You couldn't find necessary resources or dependencies
            - Quality standards haven't been met

        4. **Task Breakdown**:
        - Create specific, actionable items
        - Break complex tasks into smaller, manageable steps
        - Use clear, descriptive task names

        Being proactive with task management demonstrates attentiveness and ensures you complete all requirements successfully
        Remember: If you only need to make a few tool calls to complete a task, and it is clear what you need to do, it is better to just do the task directly and NOT call this tool at all.

        Each task item should contain:
        - content: Task description
        - status: Task status ("pending", "in_progress", "completed")
        """
        if todos is None:
            raise ValueError(
                "'todos' parameter is required. Please provide a list of tasks "
                "with 'content' and 'status' fields."
            )
        if len(todos) == 0:
            raise ValueError("'todos' list cannot be empty. Please provide at least one task.")
        valid_statuses = {"pending", "in_progress", "completed"}
        validated_todos = []

        for i, todo in enumerate(todos):
            if not isinstance(todo, dict):
                raise ValueError(f"Todo item {i} must be a dictionary")

            content = todo.get("content", "")
            status = todo.get("status", "pending")

            if not content:
                raise ValueError(f"Todo item {i} must have 'content' field")
            if status not in valid_statuses:
                raise ValueError(
                    f"Invalid status '{status}' for todo item {i}. "
                    f"Must be one of: {valid_statuses}"
                )

            validated_todos.append({
                "id": str(i + 1),
                "content": content,
                "status": status,
            })

        nudge_needed = self._needs_verification_nudge(validated_todos)
        all_done = all(t["status"] == "completed" for t in validated_todos)
        if all_done:
            self.todos = []
        else:
            self.todos = validated_todos

        logger.debug(f"Updated todo list: {len(validated_todos)} items, all_done={all_done}")

        result_message = (
            f"Todos have been modified successfully ({len(validated_todos)} items). "
            "Ensure that you continue to use the todo list to track your progress. "
            "Please proceed with the current tasks if applicable."
        )
        if nudge_needed:
            result_message += self._VERIFICATION_NUDGE

        return json.dumps({
            "message": result_message,
            "todos": validated_todos,
            "all_completed": all_done,
            "verification_nudge": nudge_needed,
        }, ensure_ascii=False, indent=2)


class BuiltinMemoryTool(Tool):
    """
    Built-in memory tool for LLM to autonomously save and search long-term memories.
    """

    MEMORY_SYSTEM_PROMPT: str = ""
    DEFAULT_SEARCH_LIMIT: int = 10
    DEFAULT_MAX_SEARCH_CHARS: int = 6000
    DEFAULT_CONVERSATION_DAYS: int = 7
    MIN_SEARCH_SCORE: float = 0.1

    def __init__(self):
        super().__init__(name="builtin_memory_tool")
        self._workspace = None
        self._sync_memories_to_global_agent_md = False

        from agentica.hooks import MEMORY_EXCLUSION_SPEC, MEMORY_TYPE_SPEC

        self.MEMORY_SYSTEM_PROMPT = dedent("""\
        ## Long-term Memory

        You have access to `save_memory` and `search_memory` tools for persistent memory across sessions.
        `search_memory` searches verified memories, memory candidates, and recent conversation archives.
        Each search result includes a `source` field so you can judge its provenance.

        Memories capture context NOT derivable from the current project state.
        Code patterns, architecture, git history, and file structure are derivable
        (via grep/git/AGENTS.md) and must NOT be saved as memories.

        If the user explicitly asks you to remember something, save it immediately
        as whichever type fits best. If they ask you to forget, tell them to delete
        the relevant memory file.

        ### Memory types

        """) + MEMORY_TYPE_SPEC + dedent("""
        **feedback** — Guidance on how to approach work: what to avoid AND what
          to keep doing.
          When to save: any time the user corrects an approach ('don't do X') OR
          confirms a non-obvious approach worked ('yes exactly', 'perfect').
          Body structure: lead with the rule, then Why, then How to apply.

        ### How to save
        Call `save_memory` with:
        - `title`: short, searchable name (e.g. "user_role", "prefer_pytest")
        - `content`: what to remember and how to apply it
        - `memory_type`: one of "user", "feedback", "project", "reference"

        ### What NOT to save

        """) + MEMORY_EXCLUSION_SPEC + (
            "\n- Duplicate of existing memory (search first before saving)."
        )

        self.register(self.save_memory, is_destructive=True)
        self.register(self.search_memory, concurrency_safe=True, is_read_only=True)

    def set_workspace(self, workspace) -> None:
        """Set the workspace reference for memory persistence."""
        self._workspace = workspace

    def set_sync_global_agent_md(self, enabled: bool) -> None:
        """Enable syncing user/feedback memories into ~/.agentica/AGENTS.md."""
        self._sync_memories_to_global_agent_md = enabled

    def clone(self) -> "BuiltinMemoryTool":
        """Fresh instance so each agent owns its ``_workspace`` slot."""
        new = BuiltinMemoryTool()
        if set(new.functions) != set(self.functions):
            new.functions = OrderedDict(
                (name, new.functions[name])
                for name in self.functions
                if name in new.functions
            )
        return new

    def get_system_prompt(self) -> Optional[str]:
        return self.MEMORY_SYSTEM_PROMPT

    async def save_memory(
        self,
        title: str,
        content: str,
        memory_type: str = "project",
    ) -> str:
        """Save important information to long-term memory for future sessions."""
        if self._workspace is None:
            raise RuntimeError("No workspace configured. Memory cannot be saved.")

        valid_types = {"user", "feedback", "project", "reference"}
        if memory_type not in valid_types:
            raise ValueError(
                f"Invalid memory_type '{memory_type}'. Must be one of: {valid_types}"
            )

        if not title.strip():
            raise ValueError("title cannot be empty.")
        if not content.strip():
            raise ValueError("content cannot be empty.")

        filepath = await self._workspace.write_memory_entry(
            title=title.strip(),
            content=content.strip(),
            memory_type=memory_type,
            description=title.strip(),
            sync_to_global_agent_md=(
                self._sync_memories_to_global_agent_md and memory_type in {"user", "feedback"}
            ),
        )

        logger.debug(f"Memory saved: {title} -> {filepath}")
        return f"Memory saved: '{title}' (type: {memory_type}) -> {filepath}"

    def search_memory(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        max_chars: int = DEFAULT_MAX_SEARCH_CHARS,
        conversation_days: int = DEFAULT_CONVERSATION_DAYS,
    ) -> str:
        """Search verified memories, memory candidates, and recent conversation archives.

        All three sources are scored against the query in one shared pool. Each
        result carries a `source` field ("memory" | "memory_candidate" |
        "conversation") so the agent can judge provenance. Returns JSON of the
        top matches, capped by `limit` and total `max_chars`. The aggregated
        ``MEMORY.md`` is excluded because it is already injected into the
        system prompt.
        """
        if self._workspace is None:
            raise RuntimeError("No workspace configured.")
        if not query.strip():
            return "No memories found matching '': empty query."

        limit = max(1, int(limit))
        max_chars = max(1, int(max_chars))
        conversation_days = max(1, int(conversation_days))

        query_lower = query.lower().strip()
        scored: List[Tuple[float, float, Dict]] = []
        for mtime, entry in self._collect_searchable_memory_entries(conversation_days):
            score = self._workspace.compute_relevance_score(query_lower, entry["content"].lower())
            if score >= self.MIN_SEARCH_SCORE:
                entry = {**entry, "score": round(score, 4)}
                scored.append((score, mtime, entry))

        if not scored:
            return f"No memories found matching '{query}'"

        scored.sort(key=lambda item: (-item[0], -item[1]))
        return json.dumps(
            self._limit_search_results([entry for _, _, entry in scored], limit=limit, max_chars=max_chars),
            ensure_ascii=False,
            indent=2,
        )

    def _collect_searchable_memory_entries(
        self, conversation_days: int
    ) -> List[Tuple[float, Dict]]:
        """Collect (mtime, entry) pairs across all three memory sources."""
        if self._workspace is None:
            raise RuntimeError("No workspace configured.")

        entries: List[Tuple[float, Dict]] = []

        for file_path in self._workspace.get_all_memory_files():
            if file_path.name == self._workspace.config.memory_md:
                continue
            entry = self._build_file_search_entry(file_path, "memory")
            if entry:
                entries.append(entry)

        for candidate in self._workspace.list_memory_candidates():
            entry = self._build_file_search_entry(Path(candidate["path"]), "memory_candidate")
            if entry:
                entries.append(entry)

        for file_path in self._get_recent_conversation_files(conversation_days):
            entries.extend(self._build_conversation_search_entries(file_path))

        return entries

    def _build_file_search_entry(
        self, file_path: Path, source: str
    ) -> Optional[Tuple[float, Dict]]:
        if self._workspace is None:
            raise RuntimeError("No workspace configured.")

        if not file_path.exists():
            return None
        try:
            content = file_path.read_text(encoding="utf-8").strip()
        except OSError as e:
            logger.warning(f"unable to read memory search file {file_path}: {e}")
            return None
        content = self._strip_frontmatter(content)
        if not content:
            return None
        return file_path.stat().st_mtime, {
            "content": content,
            "file_path": str(file_path.relative_to(self._workspace.path)),
            "source": source,
        }

    def _get_recent_conversation_files(self, conversation_days: int) -> List[Path]:
        if self._workspace is None:
            raise RuntimeError("No workspace configured.")

        cutoff = date.today() - timedelta(days=conversation_days - 1)
        recent_files: List[Path] = []
        for file_path in self._workspace.get_conversation_files():
            try:
                file_date = date.fromisoformat(file_path.stem)
            except ValueError:
                continue
            if file_date >= cutoff:
                recent_files.append(file_path)
        return recent_files

    def _build_conversation_search_entries(
        self, file_path: Path
    ) -> List[Tuple[float, Dict]]:
        if self._workspace is None:
            raise RuntimeError("No workspace configured.")

        try:
            content = file_path.read_text(encoding="utf-8").strip()
        except OSError as e:
            logger.warning(f"unable to read conversation archive {file_path}: {e}")
            return []
        if not content:
            return []

        mtime = file_path.stat().st_mtime
        rel_path = str(file_path.relative_to(self._workspace.path))
        entries: List[Tuple[float, Dict]] = []
        for block in reversed(re.split(r"\n\n---\n\n", content)):
            block = block.strip()
            if not block:
                continue
            entries.append((mtime, {
                "content": block,
                "file_path": rel_path,
                "source": "conversation",
            }))
        return entries

    @staticmethod
    def _strip_frontmatter(content: str) -> str:
        return re.sub(r"^---\n[\s\S]*?\n---\n", "", content, count=1).strip()

    @staticmethod
    def _limit_search_results(results: List[Dict], limit: int, max_chars: int) -> List[Dict]:
        limited = []
        used_chars = 0
        for result in results[:limit]:
            remaining = max_chars - used_chars
            if remaining <= 0:
                break
            content = result["content"]
            if len(content) > remaining:
                result = {**result, "content": content[:remaining], "truncated": True}
            used_chars += len(result["content"])
            limited.append(result)
        return limited
