# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Canonical built-in task-state tools.
"""

import json
from collections import Counter, OrderedDict
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from agentica.tools.base import Tool
from agentica.utils.async_file import extract_frontmatter_list, extract_frontmatter_value, strip_frontmatter
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
    - write_todos tool_result is a one-line ack, not the list. The model just
      sent the list; echoing it back adds no information and every step advance
      would then cost a full list round-trip that stays in context forever.
      The CLI and the web UI both render the list from ``tool_args``, and the
      Runner's periodic reminder re-injects the state when it goes stale.
    - All-completed auto-clear: when every item is completed, list is cleared
    - No system prompt injection (usage guidance lives in the docstring only)
    - Periodic reminder injected by Runner when LLM hasn't called write_todos
      for N turns (see Runner._inject_todo_reminder)
    """

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

    def write_todos(self, todos: Optional[List[Dict[str, str]]] = None) -> str:
        """Create and update a structured task list for the current session.

        Use it when the work has 3+ distinct steps — distinct means separate
        pieces of work, not 3 tool calls serving one step. Reading four files to
        answer one question is one step, not four.

        Skip it for a question you can just answer, a single edit, or one command.

        - Exactly one item is `in_progress` while work remains.
        - Update as you go; do not batch completions. Add tasks you discover,
          drop ones that turned out to be irrelevant.
        - Mark `completed` only once the work is actually done and verified,
          never on intent. If blocked, keep it `in_progress` and add a
          follow-up item describing the blocker.
        - Never call it multiple times in parallel.

        Each item: {"content": task description,
                    "status": "pending" | "in_progress" | "completed"}.
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

        all_done = all(t["status"] == "completed" for t in validated_todos)
        if all_done:
            self.todos = []
        else:
            self.todos = validated_todos

        logger.debug(f"Updated todo list: {len(validated_todos)} items, all_done={all_done}")

        total = len(validated_todos)
        if all_done:
            result_message = f"All {total} todos completed; list cleared."
        else:
            counts = Counter(t["status"] for t in validated_todos)
            breakdown = ", ".join(
                f"{counts[status]} {label}"
                for status, label in (
                    ("completed", "done"),
                    ("in_progress", "in progress"),
                    ("pending", "pending"),
                )
                if counts[status]
            )
            result_message = f"Todos updated ({total} items: {breakdown})."

        return result_message


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

        from agentica.prompts.memory import MEMORY_SYSTEM_PROMPT

        self.MEMORY_SYSTEM_PROMPT = MEMORY_SYSTEM_PROMPT

        self.register(self.save_memory, is_destructive=True)
        self.register(self.search_memory, concurrency_safe=True, is_read_only=True)

    def set_workspace(self, workspace) -> None:
        """Set the workspace reference for memory persistence."""
        self._workspace = workspace

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
        # The user-level AGENTS.md path is resolved from the workspace, never
        # written into the prompt text: AGENTICA_HOME can be moved, and a
        # multi-user workspace keeps a per-user copy. The default CLI user may
        # expose ~/.agentica/AGENTS.md as a symlink, but the canonical path is
        # still the per-user workspace file.
        if self._workspace is None:
            return self.MEMORY_SYSTEM_PROMPT
        return self.MEMORY_SYSTEM_PROMPT.replace(
            "<user-agents-md>", str(self._workspace.user_agent_md_path())
        )

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
        "conversation") so the agent can judge provenance, plus `title`,
        `memory_type`, `memory_source`, and `evidence_refs` from the file
        frontmatter (conversation blocks point at `{file}#block=N`). Returns
        JSON of the top matches, capped by `limit` and total `max_chars`. The
        aggregated ``MEMORY.md`` is excluded because the index is already
        injected into the system prompt (title + hook + path).
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
            raw_content = file_path.read_text(encoding="utf-8").strip()
        except OSError as e:
            logger.warning(f"unable to read memory search file {file_path}: {e}")
            return None
        content = strip_frontmatter(raw_content)
        if not content:
            return None
        return file_path.stat().st_mtime, {
            "content": content,
            "file_path": str(file_path.relative_to(self._workspace.path)),
            "source": source,
            "title": extract_frontmatter_value(raw_content, "name") or file_path.stem,
            "memory_type": extract_frontmatter_value(raw_content, "type") or "",
            "memory_source": extract_frontmatter_value(raw_content, "source") or "",
            "evidence_refs": extract_frontmatter_list(raw_content, "evidence_refs"),
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
        blocks = [block.strip() for block in re.split(r"\n\n---\n\n", content) if block.strip()]
        for block_index, block in reversed(list(enumerate(blocks))):
            entries.append((mtime, {
                "content": block,
                "file_path": rel_path,
                "source": "conversation",
                "title": file_path.stem,
                "memory_type": "conversation",
                "memory_source": "conversation_archive",
                "evidence_refs": [f"{rel_path}#block={block_index}"],
            }))
        return entries

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
