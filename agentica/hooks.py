# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Lifecycle hooks for Agent runs.

Two levels of hooks:
- AgentHooks: per-agent hooks (on_start, on_end), set on Agent instance
- RunHooks: global run-level hooks (on_agent_start, on_agent_end, on_llm_start,
  on_llm_end, on_tool_start, on_tool_end, on_agent_transfer), passed to run()
- ConversationArchiveHooks: auto-archives conversations to workspace after each run
"""
import asyncio
import json
import re
from typing import Any, Optional, List, Dict, Tuple

from agentica.experience import extract_state
from agentica.experience.skill_upgrade import SkillEvolutionManager
from agentica.learning_report import (
    LearningReport,
    LearningStatus,
    write_learning_report,
)
from agentica.model.message import Message
from agentica.tools.skill_tool import SkillTool
from agentica.utils.log import logger

# ─── Shared memory type specification ───────────────────────────────────────
# Used by both BuiltinMemoryTool (LLM-facing system prompt) and
# MemoryExtractHooks (extraction sub-call). Keep in sync by importing
# from this single source.

MEMORY_TYPE_SPEC = (
    "**user** — User's role, goals, preferences, and knowledge.\n"
    "  When to save: when you learn details about the user's role, preferences, "
    "responsibilities, or knowledge.\n"
    "  Example: 'User is a data scientist focused on observability/logging.'\n\n"
    "**project** — Information about ongoing work, goals, bugs, or incidents "
    "NOT derivable from code or git history.\n"
    "  When to save: when you learn who is doing what, why, or by when. "
    "Convert relative dates to absolute dates.\n\n"
    "**reference** — Pointers to external resources: issue trackers, dashboards, "
    "wikis, documentation sites, or internal tools.\n"
    "  When to save: when you learn about an external system the team uses.\n"
)

MEMORY_EXCLUSION_SPEC = (
    "- Code patterns, conventions, architecture, file paths, or project structure "
    "— derivable by reading the codebase.\n"
    "- Git history, recent changes, or who-changed-what — `git log`/`git blame` "
    "are authoritative.\n"
    "- Debugging solutions or fix recipes — the fix is in the code.\n"
    "- Anything already documented in AGENTS.md files.\n"
    "- Ephemeral task details: in-progress work, temporary state, current "
    "conversation context.\n"
    "- Activity logs, PR lists, or task summaries — only the *surprising* or "
    "*non-obvious* part is worth keeping.\n"
)


class AgentHooks:
    """Per-agent lifecycle hooks.

    Subclass and override the methods you need. Attach to an Agent via
    ``Agent(hooks=MyHooks())``.

    Example::

        class LoggingHooks(AgentHooks):
            async def on_start(self, agent, **kwargs):
                print(f"{agent.name} starting")

            async def on_end(self, agent, output, **kwargs):
                print(f"{agent.name} produced: {output}")
    """

    async def on_start(self, agent: Any, **kwargs) -> None:
        """Called when this agent begins a run."""
        pass

    async def on_end(self, agent: Any, output: Any, **kwargs) -> None:
        """Called when this agent finishes a run."""
        pass


class RunHooks:
    """Global run-level lifecycle hooks.

    These hooks observe the entire run, including LLM calls, tool calls,
    and agent transfers. Pass to ``agent.run(hooks=MyRunHooks())``.

    Example::

        class MetricsHooks(RunHooks):
            def __init__(self):
                self.event_counter = 0

            async def on_agent_start(self, agent, **kwargs):
                self.event_counter += 1
                print(f"#{self.event_counter}: Agent {agent.name} started")

            async def on_llm_start(self, agent, messages, **kwargs):
                self.event_counter += 1
                print(f"#{self.event_counter}: LLM call started")

            async def on_llm_end(self, agent, response, **kwargs):
                self.event_counter += 1
                print(f"#{self.event_counter}: LLM call ended")

            async def on_tool_start(self, agent, tool_name, tool_call_id, tool_args, **kwargs):
                self.event_counter += 1
                print(f"#{self.event_counter}: Tool {tool_name} started")

            async def on_tool_end(self, agent, tool_name, tool_call_id, tool_args, result, **kwargs):
                self.event_counter += 1
                print(f"#{self.event_counter}: Tool {tool_name} ended")

            async def on_agent_transfer(self, from_agent, to_agent, **kwargs):
                self.event_counter += 1
                print(f"#{self.event_counter}: Transfer from {from_agent.name} to {to_agent.name}")

            async def on_agent_end(self, agent, output, **kwargs):
                self.event_counter += 1
                print(f"#{self.event_counter}: Agent {agent.name} ended")
    """

    # Hooks with parallelizable=True can run concurrently via asyncio.gather
    # in _CompositeRunHooks.on_agent_end. Set to False for hooks that must
    # run after all parallel hooks complete (e.g. skill spawning).
    parallelizable: bool = True

    async def on_agent_start(self, agent: Any, **kwargs) -> None:
        """Called when any agent begins execution within this run."""
        pass

    async def on_agent_end(self, agent: Any, output: Any, **kwargs) -> None:
        """Called when any agent finishes execution within this run."""
        pass

    async def on_llm_start(
        self,
        agent: Any,
        messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> None:
        """Called before each LLM API call."""
        pass

    async def on_llm_end(
        self,
        agent: Any,
        response: Any = None,
        **kwargs,
    ) -> None:
        """Called after each LLM API call returns."""
        pass

    async def on_tool_start(
        self,
        agent: Any,
        tool_name: str = "",
        tool_call_id: str = "",
        tool_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Called before a tool begins execution."""
        pass

    async def on_tool_end(
        self,
        agent: Any,
        tool_name: str = "",
        tool_call_id: str = "",
        tool_args: Optional[Dict[str, Any]] = None,
        result: Any = None,
        is_error: bool = False,
        elapsed: float = 0.0,
        **kwargs,
    ) -> None:
        """Called after a tool finishes execution."""
        pass

    async def on_agent_transfer(
        self,
        from_agent: Any,
        to_agent: Any,
        **kwargs,
    ) -> None:
        """Called when a task is transferred from one agent to another."""
        pass

    async def on_user_prompt(
        self,
        agent: Any,
        message: str,
        **kwargs,
    ) -> Optional[str]:
        """Called before a user prompt is processed.

        Return a modified message string to replace the original, or None to
        keep it unchanged.  Mirrors CC's UserPromptSubmit hook.
        """
        return None

    async def on_pre_compact(
        self,
        agent: Any,
        messages: Optional[List] = None,
        **kwargs,
    ) -> None:
        """Called just before context compression is triggered.

        Use for: saving state, logging, custom archival before messages are
        compressed/dropped.  Mirrors CC's PreCompact hook.
        """
        pass

    async def on_post_compact(
        self,
        agent: Any,
        messages: Optional[List] = None,
        **kwargs,
    ) -> None:
        """Called right after context compression completes.

        ``messages`` is the compressed result (may be much shorter than before).
        Use for: post-compression analytics, re-injecting critical context.
        Mirrors CC's PostCompact hook.
        """
        pass


class ConversationArchiveHooks(RunHooks):
    """RunHooks that auto-archives conversations to workspace after each agent run.

    Captures user input and agent output from each run and appends them to
    the daily conversation archive in the workspace.

    Usage::

        from agentica.hooks import ConversationArchiveHooks

        hooks = ConversationArchiveHooks()
        response = await agent.run("Hello", config=RunConfig(hooks=hooks))
    """

    def __init__(self):
        pass

    async def on_agent_end(self, agent: Any, output: Any, **kwargs) -> None:
        """Archive conversation after agent completes.

        Reads agent.run_input directly (set by Runner before on_agent_end).
        """
        workspace = agent.workspace
        if workspace is None:
            return

        messages_to_archive = []

        # Read run_input directly — Runner sets it before calling on_agent_end
        run_input = agent.run_input
        if run_input and isinstance(run_input, str):
            messages_to_archive.append({"role": "user", "content": run_input})

        # Collect agent output
        if output and isinstance(output, str):
            messages_to_archive.append({"role": "assistant", "content": output})

        if not messages_to_archive:
            return

        try:
            session_id = agent.run_id
            filepath = await workspace.archive_conversation(messages_to_archive, session_id=session_id)
            logger.debug(f"Conversation saved to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to archive conversation: {e}")


class MemoryExtractHooks(RunHooks):
    """Boundary-triggered memory extraction.

    Per-turn behavior is zero-cost: completed turns are appended to a
    per-session in-memory buffer. The LLM extraction runs only when the
    buffer is flushed, which happens on two boundaries:

      1. Every ``every_n_turns`` turns (default 10).
      2. ``on_pre_compact`` — context is about to be summarized/dropped,
         so we extract whatever is buffered before it's lost.

    Plus a cross-process frequency cap (``min_seconds_between``) so two
    parallel agents in the same ``~/.agentica/`` don't both extract at the
    same wall-clock second.

    If the LLM already called ``save_memory`` during the buffered window,
    the extraction is skipped — the model has already self-curated.

    Usage::

        hooks = MemoryExtractHooks(every_n_turns=10, min_seconds_between=60)
        response = await agent.run("Hello", config=RunConfig(hooks=hooks))
    """

    # Prompt for the memory extraction sub-call.
    # Uses shared MEMORY_TYPE_SPEC / MEMORY_EXCLUSION_SPEC constants
    # (same source as BuiltinMemoryTool.MEMORY_SYSTEM_PROMPT).
    _EXTRACT_PROMPT = (
        "You are a memory extraction assistant. Review the conversation below and "
        "extract key information worth remembering for future sessions.\n\n"
        "Memories capture context NOT derivable from the current project state. "
        "Code patterns, architecture, git history, and file structure are derivable "
        "(via grep/git/AGENTS.md) and must NOT be saved as memories.\n\n"
        "## Memory types\n\n"
        + MEMORY_TYPE_SPEC +
        "\n## What NOT to save\n\n"
        + MEMORY_EXCLUSION_SPEC +
        "\n## Output format\n\n"
        "For each memory, output a JSON object with fields:\n"
        '  {"title": "short_name", "content": "what to remember", '
        '"type": "user|project|reference"}\n\n'
        "Do NOT extract feedback/corrections — those are handled separately.\n\n"
        "Output a JSON array of memories. If nothing worth remembering, output: []\n\n"
        "Conversation:\n"
    )

    _STATE_SLOT = "memory_extract"

    def __init__(
        self,
        sync_memories_to_global_agent_md: bool = False,
        every_n_turns: int = 10,
        min_seconds_between: int = 60,
    ):
        """
        Args:
            sync_memories_to_global_agent_md: Mirror user-type memories into global AGENT.md.
            every_n_turns: Flush the buffer through the LLM every N turns.
                0 disables periodic extraction (on_pre_compact still flushes).
            min_seconds_between: Cross-process frequency cap; skip when the
                previous extraction was less than N seconds ago. 0 disables.
        """
        self._tool_calls: Dict[str, List[str]] = {}  # agent_id -> list of tool names called
        self._sync_memories_to_global_agent_md = sync_memories_to_global_agent_md
        self._every_n_turns = max(0, int(every_n_turns))
        self._min_seconds_between = max(0, int(min_seconds_between))
        # Per-session buffer of (user, assistant, save_memory_called) tuples.
        # Drained when a boundary fires; survives across runs in the same
        # session_id so multi-turn batching works.
        self._buffers: Dict[str, List[Tuple[str, str, bool]]] = {}

    async def on_agent_start(self, agent: Any, **kwargs) -> None:
        self._tool_calls[agent.agent_id] = []

    async def on_tool_end(self, agent: Any, tool_name: str = "", **kwargs) -> None:
        """Track tool calls to detect if save_memory was already used."""
        agent_id = agent.agent_id
        if agent_id not in self._tool_calls:
            self._tool_calls[agent_id] = []
        self._tool_calls[agent_id].append(tool_name)

    async def on_agent_end(self, agent: Any, output: Any, **kwargs) -> None:
        """Buffer the turn; flush through the LLM when boundary policy says so."""
        agent_id = agent.agent_id
        tool_calls = self._tool_calls.pop(agent_id, [])

        workspace = agent.workspace
        if workspace is None:
            return

        run_input = agent.run_input
        user_msg = run_input if isinstance(run_input, str) else ""
        assistant_msg = output if isinstance(output, str) else ""
        if not user_msg and not assistant_msg:
            return

        session_key = agent.session_id or agent.agent_id
        save_memory_called = "save_memory" in tool_calls
        buf = self._buffers.setdefault(session_key, [])
        buf.append((user_msg, assistant_msg, save_memory_called))

        # Periodic boundary: only flush if we've accumulated N turns AND
        # the global frequency cap allows it. Both gates are intentional —
        # every_n_turns bounds work-per-session; min_seconds_between bounds
        # work-per-wallclock across concurrent agents.
        if self._every_n_turns == 0 or len(buf) < self._every_n_turns:
            return
        if extract_state.should_skip(self._STATE_SLOT, self._min_seconds_between):
            logger.debug(
                "Memory extraction skipped: last run within %ds window",
                self._min_seconds_between,
            )
            return

        await self._flush(agent, workspace, session_key)

    async def on_pre_compact(self, agent: Any, messages=None, **kwargs) -> None:
        """Context is about to be compressed — extract anything we have buffered first."""
        workspace = agent.workspace
        if workspace is None:
            return
        session_key = agent.session_id or agent.agent_id
        if not self._buffers.get(session_key):
            return
        # No idle-gate on pre_compact: this is a "last chance before data loss"
        # path, so we always flush. min_seconds_between is for periodic noise
        # reduction; data-loss prevention overrides it.
        await self._flush(agent, workspace, session_key)

    async def _flush(self, agent: Any, workspace: Any, session_key: str) -> None:
        """Drain buffer for ``session_key`` and run one LLM extraction over it."""
        buf = self._buffers.pop(session_key, [])
        if not buf:
            return

        # If any turn in the window already called save_memory, the model
        # self-curated — skip extraction. Mirrors CC's hasMemoryWritesSince.
        if any(save_called for _, _, save_called in buf):
            logger.debug(
                "Skipping memory extraction: save_memory called within batch (session=%s, turns=%d)",
                session_key, len(buf),
            )
            return

        parts: List[str] = []
        for user_msg, assistant_msg, _ in buf:
            if user_msg:
                parts.append(f"User: {user_msg}")
            if assistant_msg:
                parts.append(f"Assistant: {assistant_msg}")
        conversation_text = "\n\n".join(parts)

        # Cheap floor: a single trivial turn ("hi" / "ok") still rolls up to
        # a tiny conversation_text; not worth a sub-LLM round trip.
        if len(conversation_text) < 200:
            return

        model = agent.auxiliary_model or agent.model
        if model is None:
            return

        extract_state.stamp(self._STATE_SLOT)
        await self._extract_and_save(model, workspace, conversation_text)

    async def _extract_and_save(self, model: Any, workspace: Any, conversation_text: str) -> None:
        """Run the LLM extraction call and persist results."""
        extract_messages = [
            Message(role="user", content=self._EXTRACT_PROMPT + conversation_text),
        ]

        try:
            model_response = await model.response(extract_messages)
            if not model_response or not model_response.content:
                return

            # Parse JSON array from response
            text = model_response.content.strip()
            # Handle markdown code blocks
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            memories = json.loads(text)
            if not isinstance(memories, list) or not memories:
                return

            for mem in memories:
                if not isinstance(mem, dict):
                    continue
                title = mem.get("title", "").strip()
                content = mem.get("content", "").strip()
                mem_type = mem.get("type", "project").strip()
                if not title or not content:
                    continue
                if mem_type not in ("user", "project", "reference"):
                    mem_type = "project"

                await workspace.write_memory_entry(
                    title=title,
                    content=content,
                    memory_type=mem_type,
                    description=title,
                    sync_to_global_agent_md=(
                        self._sync_memories_to_global_agent_md and mem_type == "user"
                    ),
                    source="auto_extract",
                )
                logger.debug(f"Auto-extracted memory: {title} (type: {mem_type})")

        except json.JSONDecodeError as e:
            logger.debug(f"Memory extraction: LLM returned invalid JSON: {e}")
        except Exception as e:
            logger.warning(f"Memory extraction failed: {e}")


class ExperienceCaptureHooks(RunHooks):
    """Capture tool failures, user corrections, and success patterns.

    Tool errors and success patterns are captured deterministically (zero LLM cost).
    User corrections are classified by an auxiliary LLM model for accuracy —
    keyword matching is too fragile for nuanced human feedback.

    Persists experiences to workspace at on_agent_end for cross-session learning.

    Delegates to three experience-layer classes:
    - ExperienceEventStore: append-only raw event persistence
    - ExperienceCompiler: pure/stateless compilation (errors/successes -> cards)
    - CompiledExperienceStore: card CRUD, lifecycle, sync

    Usage::

        from agentica.hooks import ExperienceCaptureHooks
        from agentica.agent.config import ExperienceConfig

        hooks = ExperienceCaptureHooks(ExperienceConfig())
        response = await agent.run("Hello", config=RunConfig(hooks=hooks))
    """

    parallelizable = False

    # LLM prompt for feedback classification
    _FEEDBACK_CLASSIFY_PROMPT = (
        "You are judging whether the user's latest message is a correction or "
        "behavioral feedback to the assistant.\n\n"
        "Inputs:\n"
        "- Previous assistant message\n"
        "- Current user message\n\n"
        "Decide:\n"
        "1. Is the user correcting the assistant, rejecting its approach, or "
        "imposing a behavioral constraint?\n"
        "2. Is this a reusable rule worth remembering for future sessions?\n"
        "3. If yes, normalize it into a reusable rule.\n\n"
        "Important:\n"
        "- Do not rely on literal phrases. Indirect corrections still count.\n"
        "- Code snippets, log lines, or hypothetical examples in quotes are NOT corrections.\n"
        "- A pure retry request (e.g. 'try again', 'read another file') is NOT "
        "a correction.\n"
        "- When the user explicitly states a workflow, procedure, or rule (e.g. "
        "'always do X before Y', 'the rule is ...', '下次请先 ...'), set "
        "should_persist=true and persist_target=\"experience\".\n"
        "- Set persist_target=\"experience\" for any cross-session reusable "
        "rule; use \"none\" only for turn-specific feedback or non-corrections.\n\n"
        "Rule field requirements (CRITICAL — this becomes the dedup key):\n"
        "- If the user gives an explicit quoted rule string (\"the rule is: '<X>'\", "
        "'apply this rule: \"<X>\"', etc.), copy <X> verbatim into the rule field. "
        "Do not paraphrase, do not add steps, do not prepend 'Always'.\n"
        "- Otherwise, condense to a single short verb-object phrase, "
        "<= 8 words, no leading 'Always/Never/Please', no trailing period.\n"
        "- The rule must be the same string every time the same intent recurs — "
        "it is hashed to a filename.\n\n"
        "Return JSON only with these fields:\n"
        '{"is_correction": bool, "confidence": float (0-1), '
        '"category": "factual|preference|workflow|tool_usage|rejection|not_correction", '
        '"scope": "turn_only|session|cross_session", '
        '"should_persist": bool, '
        '"persist_target": "none|experience", '
        '"rule": "verb-object phrase, <= 8 words", '
        '"why": "reason this matters", '
        '"how_to_apply": "when and where to apply this rule"}\n\n'
    )

    _RULE_PREFIX_PATTERNS = (
        re.compile(
            r"(?is)(?:^|[。.!?\n]\s*)(?:the\s+rule\s+is|remember(?:\s+this)?|remember|记住|规则(?:是)?)[：:\s]+"
            r"[\"'“”‘’]?(?P<rule>.+?)[\"'“”‘’]?\s*$"
        ),
        re.compile(r"(?is)(?P<rule>(?:下次|以后)请先.+)$"),
        re.compile(r"(?is)(?P<rule>不要再.+)$"),
        re.compile(r"(?is)(?P<rule>必须先.+再.+)$"),
        re.compile(r"(?is)(?P<rule>always\s+.+\s+before\s+.+)$"),
        re.compile(r"(?is)(?P<rule>never\s+.+)$"),
    )
    _STRONG_NEGATIVE_PATTERNS = (
        re.compile(r"(?i)\byou made a mistake\b"),
        re.compile(r"(?i)\byou are wrong\b"),
        re.compile(r"(?i)\bthat(?:'s| is) incorrect\b"),
        re.compile(r"(?i)\bwrong\b"),
        re.compile(r"你犯错了"),
        re.compile(r"你又错了"),
        re.compile(r"这不对"),
        re.compile(r"不对"),
        re.compile(r"太蠢了"),
    )

    def __init__(self, config: Any):
        self._config = config
        # Per-agent state (keyed by agent_id)
        self._tool_errors: Dict[str, List[Dict]] = {}
        self._tool_successes: Dict[str, List[Dict]] = {}
        self._last_assistant_output: Dict[str, Optional[str]] = {}
        self._skills_used: Dict[str, set] = {}  # agent_id -> set of skill names loaded via get_skill_info
        self._correction_detected: Dict[str, bool] = {}  # agent_id -> True if correction persisted this run
        # Cache for the tool_recovery gate. Keyed by workspace events.jsonl
        # path; value is (last_seen_size, set_of_failed_tool_names). Avoids
        # rescanning the entire jsonl on every on_agent_end. Refreshed when
        # the file has grown since last lookup.
        self._failed_tools_cache: Dict[str, Tuple[int, set]] = {}
        # Per-session buffer of (user_msg, previous_assistant, original_task)
        # tuples for the batched judge LLM call. The cheap prefilter (explicit
        # rule prefix + strong negation) still fires per-turn so explicit
        # corrections aren't delayed; only the LLM fall-through is batched.
        self._judge_buffers: Dict[str, List[Tuple[str, str, str]]] = {}

    async def on_agent_start(self, agent: Any, **kwargs) -> None:
        """Initialize per-agent capture state."""
        aid = agent.agent_id
        self._tool_errors[aid] = []
        self._tool_successes[aid] = []
        self._last_assistant_output[aid] = None
        self._skills_used[aid] = set()
        self._correction_detected[aid] = False

    @staticmethod
    def _extract_original_task(agent: Any) -> str:
        """Best-effort extraction of the user-facing task for this run.

        Read at ``on_agent_end`` time because the Runner sets
        ``agent.run_input`` only just before invoking ``on_agent_end`` —
        it is None during ``on_agent_start``. Multi-task aggregation
        (one card touched by N runs) happens in CompiledExperienceStore,
        so per-run capture here is enough.

        Order of preference:
            1. ``agent.run_input`` if it is a non-empty string
            2. ``agent.run_input["content"]`` if it is a Message-dict
            3. First user-role message in ``agent.working_memory.messages``
            4. Empty string (caller treats it as "unknown")
        """
        run_input = getattr(agent, "run_input", None)
        if isinstance(run_input, str) and run_input.strip():
            return run_input.strip()[:500]
        if isinstance(run_input, dict):
            content = run_input.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()[:500]
        wm = getattr(agent, "working_memory", None)
        messages = getattr(wm, "messages", None) or []
        for msg in messages:
            role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
            content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
            if role == "user" and isinstance(content, str) and content.strip():
                return content.strip()[:500]
        return ""

    async def on_user_prompt(self, agent: Any, message: str, **kwargs) -> Optional[str]:
        """No-op: classification uses agent.run_input at on_agent_end time."""
        return None  # Never modify the message

    async def on_tool_end(
        self,
        agent: Any,
        tool_name: str = "",
        tool_call_id: str = "",
        tool_args: Optional[Dict[str, Any]] = None,
        result: Any = None,
        is_error: bool = False,
        elapsed: float = 0.0,
        **kwargs,
    ) -> None:
        """Record tool errors and successes."""
        aid = agent.agent_id

        if is_error and self._config.capture_tool_errors:
            result_str = str(result)[:500] if result else ""
            self._tool_errors.setdefault(aid, []).append({
                "tool": tool_name,
                "args": tool_args or {},
                "error": result_str,
                "elapsed": elapsed,
            })
        elif not is_error and self._config.capture_success_patterns:
            self._tool_successes.setdefault(aid, []).append({
                "tool": tool_name,
                "elapsed": elapsed,
            })

        # Track generated skills only when get_skill_info returned real content.
        # After方案A: skill_tool raises on missing/disabled skills, so is_error=True
        # filters out those cases. No need to inspect the result text.
        if tool_name == "get_skill_info" and tool_args and not is_error:
            skill_name = tool_args.get("skill_name") or tool_args.get("name", "")
            if skill_name:
                self._skills_used.setdefault(aid, set()).add(skill_name)

    async def _get_past_failed_tools(self, event_store: Any) -> set:
        """Set of tool names with at least one prior ``tool_error`` event.

        Cheap path: if events.jsonl size is unchanged since the last call,
        return the cached set. Otherwise re-read the file and refresh the
        cache (size + set). Detects peer writers because any append from
        another process / hook instance grows the file.

        Note: ``tool_error`` events written earlier in the current
        ``on_agent_end`` (step 1, before step 1b runs) are visible here
        because they were already appended via ``event_store.append`` —
        which is why a tool that fails AND succeeds in the same turn
        correctly counts as a recovery.
        """
        path = event_store.events_path
        cache_key = str(path)
        try:
            current_size = path.stat().st_size
        except FileNotFoundError:
            return self._failed_tools_cache.setdefault(cache_key, (0, set()))[1]

        cached = self._failed_tools_cache.get(cache_key)
        if cached is not None and cached[0] == current_size:
            return cached[1]

        failed = {
            e.get("tool")
            for e in await event_store.read_all()
            if e.get("event_type") == "tool_error" and e.get("tool")
        }
        self._failed_tools_cache[cache_key] = (current_size, failed)
        return failed

    async def on_agent_end(self, agent: Any, output: Any, **kwargs) -> None:
        """Persist captured experiences to workspace.

        Flow: write raw events -> compile cards -> persist -> lifecycle -> sync.
        Delegates compilation to ExperienceCompiler (pure, no I/O).
        Delegates persistence to ExperienceEventStore / CompiledExperienceStore.

        Also emits a structured LearningReport (arch_v5.md Phase 2) so the
        operator can answer "did this run actually learn anything?" without
        grepping logs.
        """
        from agentica.experience.compiler import ExperienceCompiler

        # Always drain accumulated state first — even when workspace is None.
        # Without this, state leaks into the next run that DOES have a workspace.
        aid = agent.agent_id
        errors = self._tool_errors.pop(aid, [])
        successes = self._tool_successes.pop(aid, [])
        self._last_assistant_output.pop(aid, None)
        skills_used = self._skills_used.pop(aid, set())
        correction_this_run = self._correction_detected.pop(aid, False)

        workspace = agent.workspace
        if workspace is None:
            return

        # Build the LearningReport scaffold up-front so each branch below can
        # mutate counters / status as it goes. Persistence happens at the end.
        report = LearningReport(
            run_id=agent.run_id or aid,
            agent_id=aid,
            session_id=agent.session_id,
            tool_errors_captured=len(errors),
        )

        # Read run_input directly — Runner sets it before calling on_agent_end
        run_input = agent.run_input
        user_msg = run_input if isinstance(run_input, str) else None

        # Extract previous assistant message from working_memory for correction context.
        previous_assistant_text = self._get_previous_assistant_text(agent)

        # Get stores from workspace
        event_store = workspace.get_experience_event_store()
        compiled_store = workspace.get_compiled_experience_store()

        # Stamp the run's original_task on every error / success dict so the
        # downstream pure compilers (compile_tool_errors / _success_pattern)
        # can read it and embed it into CompiledCard.source_task without
        # needing extra arguments. Captured lazily here because Runner sets
        # agent.run_input only just before on_agent_end.
        original_task = self._extract_original_task(agent)
        if original_task:
            for err in errors:
                err.setdefault("original_task", original_task)
            for s in successes:
                s.setdefault("original_task", original_task)

        # ── 1. Write raw events (pure builder + store) ──
        raw_events = ExperienceCompiler.build_raw_events(
            errors=errors,
            user_msg=user_msg,
            previous_assistant=previous_assistant_text,
            successes=successes,
            capture_corrections=self._config.capture_user_corrections,
        )
        for event in raw_events:
            if original_task:
                event["original_task"] = original_task
            await event_store.append(event)
        if raw_events:
            logger.debug(
                f"[experience] appended {len(raw_events)} raw event(s): "
                f"types={[e.get('event_type') for e in raw_events]}"
            )

        # ── 1b. Emit tool_recovery events (decoupled from skill installation) ──
        # Semantic: emit one ``tool_recovery`` per (run, tool) when a tool
        # invocation succeeds in this run AND the same tool has any prior
        # ``tool_error`` event in the workspace's events.jsonl (including
        # earlier in the very same run — failures from raw_events written
        # in step 1 above are visible here because they were already
        # appended). This is the bootstrap signal that drives
        # ``min_success_applications``; it is independent of whether any
        # skill is installed (solves the chicken-and-egg problem).
        #
        # Concurrent writers: each agent run is independent evidence of
        # recovery — two parallel agents that both succeed at write_file
        # legitimately contribute two recovery events. No lock is needed;
        # events.jsonl is append-only and the in-process cache only gates
        # the cheap path (a stat+size delta forces a refresh whenever a
        # peer has written).
        if successes and self._config.capture_success_patterns:
            failed_tools = await self._get_past_failed_tools(event_store)
            emitted_tools: set = set()
            for s in successes:
                tool = s.get("tool")
                if not tool or tool not in failed_tools or tool in emitted_tools:
                    continue
                emitted_tools.add(tool)
                recovery_event: Dict[str, Any] = {
                    "event_type": "tool_recovery",
                    "tool": tool,
                    "elapsed": s.get("elapsed", 0.0),
                }
                if original_task:
                    recovery_event["original_task"] = original_task
                await event_store.append(recovery_event)

        # ── 2. Compile and persist experience cards ──

        # 2a. Tool errors (deterministic, zero LLM cost)
        # Dedup by title: same run may produce duplicate error titles (e.g. two
        # PermissionErrors from the same tool), which would inflate repeat_count.
        error_cards = ExperienceCompiler.compile_tool_errors(errors)
        seen_titles: set = set()
        for card in error_cards:
            if card.title in seen_titles:
                continue
            seen_titles.add(card.title)
            try:
                await compiled_store.write(card)
                report.cards_written += 1
            except Exception as e:
                logger.warning(f"Failed to write tool error experience: {e}")

        # 2b. LLM-based correction classification.
        # Structural pre-filter (no keyword heuristics): a correction requires
        # a previous assistant turn to correct. First-turn messages cannot be
        # corrections by definition, so skip entirely.
        if (
            self._config.capture_user_corrections
            and user_msg
            and previous_assistant_text
        ):
            # Cheap deterministic prefilter still runs per-turn — explicit
            # rules ("下次请先…", "the rule is …") and strong negations
            # ("你错了") shouldn't wait for a batch boundary to take effect.
            prefiltered = self._prefilter_feedback_classification(user_msg)
            if prefiltered is not None:
                prefiltered["user_message"] = user_msg
                was_correction = await self._persist_feedback_classification(
                    event_store=event_store,
                    compiled_store=compiled_store,
                    result=prefiltered,
                    threshold=self._config.feedback_confidence_threshold,
                    original_task=original_task,
                )
                if was_correction:
                    correction_this_run = True
                    report.corrections_persisted += 1
                    report.cards_written += 1
            else:
                # Fall-through goes to the LLM judge — batch it. Buffer this
                # turn for a future flush (boundary policy: every_n_turns or
                # on_pre_compact). One LLM call later scans the whole window
                # for corrections, replacing N per-turn calls with 1.
                session_key = agent.session_id or agent.agent_id
                self._judge_buffers.setdefault(session_key, []).append(
                    (user_msg, previous_assistant_text, original_task)
                )
                model = self._get_classification_model(agent)
                if model is not None and self._should_flush_judge(session_key):
                    await self._flush_judge(
                        model, event_store, compiled_store, session_key,
                    )

        # 2c. Success pattern
        success_card = ExperienceCompiler.compile_success_pattern(successes)
        if success_card and not errors:
            try:
                await compiled_store.write(success_card)
                report.cards_written += 1
            except Exception as e:
                logger.warning(f"Failed to write success pattern experience: {e}")

        # ── 3. Lifecycle sweep ──
        try:
            await compiled_store.run_lifecycle(
                promotion_count=self._config.promotion_count,
                promotion_window_days=self._config.promotion_window_days,
                demotion_days=self._config.demotion_days,
                archive_days=self._config.archive_days,
            )
        except Exception as e:
            logger.debug(f"Experience lifecycle sweep failed: {e}")

        # ── 3.5 Skill upgrade (after lifecycle, before sync) ──
        skill_cfg = self._config.skill_upgrade
        if skill_cfg is not None and skill_cfg.mode != "off":
            try:
                manager = SkillEvolutionManager()
                upgrade_model = self._get_classification_model(agent)
                if upgrade_model is not None:
                    gen_dir = workspace._get_user_generated_skills_dir()
                    exp_dir = workspace._get_user_experience_dir()
                    skill_tool = None
                    for tool in agent.tools or []:
                        if isinstance(tool, SkillTool):
                            skill_tool = tool
                            break
                    should_reload_generated_skills = False

                    # Phase A: try to spawn new skill from experience
                    # (draft mode only generates, shadow mode generates + installs)
                    candidates = manager.get_candidate_cards(
                        exp_dir=exp_dir,
                        min_repeat_count=skill_cfg.min_repeat_count,
                        min_tier=skill_cfg.min_tier,
                    )
                    logger.debug(
                        f"[skill-upgrade] scanned {exp_dir.name}/: "
                        f"{len(candidates)} candidate(s) "
                        f"(min_repeat={skill_cfg.min_repeat_count}, "
                        f"min_tier={skill_cfg.min_tier!r})"
                    )
                    if candidates:
                        existing = set(
                            [
                                d.name for d in gen_dir.iterdir()
                                if d.is_dir() and (d / "SKILL.md").exists()
                            ]
                            if gen_dir.exists() else []
                        )
                        if skill_tool is not None:
                            existing.update(skill.name for skill in skill_tool.registry.list_all())
                        spawned = await manager.maybe_spawn_skill(
                            model=upgrade_model,
                            candidates=candidates,
                            existing_skills=sorted(existing),
                            generated_skills_dir=gen_dir,
                            event_store=event_store,
                            min_success_applications=skill_cfg.min_success_applications,
                            hooks=skill_cfg.lifecycle_hooks,
                        )
                        logger.debug(
                            f"[skill-upgrade] maybe_spawn_skill → {spawned!r} "
                            f"(mode={skill_cfg.mode}, "
                            f"min_success_applications={skill_cfg.min_success_applications})"
                        )
                        # In draft mode, mark as draft instead of shadow
                        if spawned and skill_cfg.mode == "draft":
                            meta_path = gen_dir / spawned / "meta.json"
                            meta = manager.read_meta(meta_path)
                            if meta:
                                meta["status"] = "draft"
                                manager.write_meta(meta_path, meta)

                        if spawned and skill_cfg.mode == "shadow":
                            should_reload_generated_skills = True
                        if spawned:
                            report.skill_candidate = spawned
                            report.skill_state_change = (
                                "spawned_draft" if skill_cfg.mode == "draft"
                                else "spawned_shadow"
                            )
                            report.upgrade_decision = "spawned"

                    # Phase B: record episode only for skills actually used this run
                    if skill_cfg.mode == "shadow" and skills_used and gen_dir.exists():
                        outcome = "failure" if errors or correction_this_run else "success"
                        query_text = user_msg or ""
                        for skill_dir in gen_dir.iterdir():
                            if not skill_dir.is_dir():
                                continue
                            meta = manager.read_meta(skill_dir / "meta.json")
                            skill_name = meta.get("skill_name", "")
                            if not skill_name or skill_name not in skills_used:
                                continue
                            if meta.get("status") not in ("shadow", "auto"):
                                continue
                            if skill_tool is not None:
                                loaded_skill = skill_tool.registry.get(skill_name)
                                if loaded_skill is None or loaded_skill.location != "generated":
                                    continue
                            episodes_path = skill_dir / "episodes.jsonl"
                            manager.record_episode(
                                episodes_path=episodes_path,
                                outcome=outcome,
                                query=query_text,
                                tool_errors=len(errors),
                                user_corrected=correction_this_run,
                            )
                            manager.update_meta_after_episode(
                                skill_dir / "meta.json", outcome,
                            )
                            # Phase C: checkpoint judgment
                            decision = await manager.maybe_update_skill_state(
                                model=upgrade_model,
                                skill_dir=skill_dir,
                                checkpoint_interval=skill_cfg.checkpoint_interval,
                                rollback_consecutive_failures=skill_cfg.rollback_consecutive_failures,
                                hooks=skill_cfg.lifecycle_hooks,
                            )
                            if decision is not None:
                                should_reload_generated_skills = True
                                report.skill_state_change = decision
                                report.upgrade_decision = decision

                    if should_reload_generated_skills and skill_tool is not None:
                        skill_tool.reload_generated_skills()
                        agent.refresh_tool_system_prompts()
            except Exception as e:
                logger.debug(f"Skill upgrade check failed: {e}")
                report.mark_error(f"skill_upgrade_failed: {e}")

        # ── 4. Sync to global AGENTS.md ──
        if self._config.sync_to_global_agent_md:
            try:
                global_md = workspace._get_global_agent_md_path()
                await compiled_store.sync_to_global_agent_md(global_md)
            except Exception as e:
                logger.debug(f"Experience sync to global AGENTS.md failed: {e}")

        # ── 5. Emit LearningReport (arch_v5.md Phase 2) ──
        if (
            report.cards_written
            or report.corrections_persisted
            or report.skill_state_change
        ):
            report.mark_learned()
        elif errors:
            report.mark_skipped("errors_observed_but_no_card_persisted")
        else:
            report.skip_reason = report.skip_reason or "nothing_actionable"

        # write_learning_report owns its own I/O fault tolerance (logs warnings,
        # returns None on failure). No outer try here -- it would only mask
        # programming bugs as silent debug noise.
        write_learning_report(workspace, report)

    @staticmethod
    def _get_previous_assistant_text(agent: Any) -> Optional[str]:
        """Get the last assistant message text from working_memory.

        At on_agent_end time, the current run's messages haven't been added
        to working_memory yet, so the last assistant message reflects previous
        runs — which is exactly what a user correction refers to.
        """
        messages = agent.working_memory.messages
        for msg in reversed(messages):
            if msg.role == "assistant" and msg.content:
                return msg.content if isinstance(msg.content, str) else str(msg.content)
        return None

    @staticmethod
    def _get_classification_model(agent: Any) -> Any:
        """Get the model for feedback classification.

        Prefers auxiliary_model (cheaper), falls back to main model.
        Returns None if no model is available.
        """
        model = agent.auxiliary_model
        if model is not None:
            return model
        return agent.model

    @staticmethod
    def _clean_prefilter_rule(text: str) -> str:
        """Trim wrapper punctuation while preserving the user's rule wording."""
        return text.strip().strip(" \t\r\n\"'“”‘’").rstrip("。.!?")

    @classmethod
    def _prefilter_feedback_classification(cls, user_message: str) -> Optional[Dict[str, Any]]:
        """Cheap deterministic prefilter for explicit rules and strong rejection signals."""
        text = user_message.strip()
        if not text:
            return None

        for pattern in cls._RULE_PREFIX_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            rule = cls._clean_prefilter_rule(match.group("rule"))
            if not rule:
                continue
            return {
                "is_correction": True,
                "confidence": 1.0,
                "category": "workflow",
                "scope": "cross_session",
                "should_persist": True,
                "persist_target": "experience",
                "rule": rule,
                "why": "User stated an explicit reusable rule.",
                "how_to_apply": "Apply this rule on future runs when the same behavior is relevant.",
            }

        if any(pattern.search(text) for pattern in cls._STRONG_NEGATIVE_PATTERNS):
            return {
                "is_correction": True,
                "confidence": 1.0,
                "category": "rejection",
                "scope": "turn_only",
                "should_persist": False,
                "persist_target": "none",
                "rule": "",
                "why": "User explicitly rejected the assistant's behavior.",
                "how_to_apply": "",
            }

        return None

    _JUDGE_STATE_SLOT = "correction_judge"

    # Prompt for the batched judge: one call scans an entire turn window.
    _BATCH_JUDGE_PROMPT = (
        "You are auditing a recent conversation window to find user corrections "
        "or behavioral feedback addressed to the assistant.\n\n"
        "For each turn, decide whether the user's message was a correction, "
        "rejection, or a reusable rule the assistant should remember.\n\n"
        "Important:\n"
        "- A pure retry request ('try again') or a question is NOT a correction.\n"
        "- Code snippets / log lines in quotes are NOT corrections.\n"
        "- An explicit workflow or rule ('always do X before Y', '下次请先 ...', "
        "'the rule is …') IS a correction worth persisting.\n\n"
        "Output a JSON array; one object per correction you found:\n"
        '{"turn_index": int, "rule": "verb-object phrase <=8 words", '
        '"confidence": float, "category": "factual|preference|workflow|tool_usage|rejection", '
        '"scope": "turn_only|session|cross_session", '
        '"should_persist": bool, "persist_target": "experience|none", '
        '"why": "...", "how_to_apply": "..."}\n\n'
        "If nothing in the window is a correction, output []. Do not invent corrections.\n\n"
        "Window (oldest first):\n"
    )

    def _should_flush_judge(self, session_key: str) -> bool:
        n = self._config.judge_every_n_turns
        if n <= 0:
            return False
        if len(self._judge_buffers.get(session_key, [])) < n:
            return False
        if extract_state.should_skip(self._JUDGE_STATE_SLOT, self._config.judge_min_seconds_between):
            logger.debug(
                "Correction judge skipped: last run within %ds window",
                self._config.judge_min_seconds_between,
            )
            return False
        return True

    async def _flush_judge(
        self,
        model: Any,
        event_store: Any,
        compiled_store: Any,
        session_key: str,
    ) -> None:
        """Drain the judge buffer for ``session_key`` and run one batched LLM scan."""
        buf = self._judge_buffers.pop(session_key, [])
        if not buf:
            return

        # Build the window as numbered turns so the LLM can refer back by index.
        turn_lines: List[str] = []
        for idx, (user_msg, prev_asst, _) in enumerate(buf):
            turn_lines.append(
                f"--- Turn {idx} ---\n"
                f"Previous assistant: {prev_asst[:800]}\n"
                f"User: {user_msg[:800]}"
            )
        prompt = self._BATCH_JUDGE_PROMPT + "\n\n".join(turn_lines)

        extract_state.stamp(self._JUDGE_STATE_SLOT)
        try:
            response = await model.response([Message(role="user", content=prompt)])
            if not response or not response.content:
                return
            text = response.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            findings = json.loads(text)
            if not isinstance(findings, list):
                return
        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"Batched correction judge failed: {e}")
            return

        threshold = self._config.feedback_confidence_threshold
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            turn_idx = finding.get("turn_index")
            if not isinstance(turn_idx, int) or not (0 <= turn_idx < len(buf)):
                continue
            user_msg, _, original_task = buf[turn_idx]
            # Promote a few canonical fields so the persistence helper
            # treats this like a single-turn classification.
            finding["is_correction"] = True
            finding["user_message"] = user_msg
            await self._persist_feedback_classification(
                event_store=event_store,
                compiled_store=compiled_store,
                result=finding,
                threshold=threshold,
                original_task=original_task or "",
            )

    async def on_pre_compact(self, agent: Any, messages=None, **kwargs) -> None:
        """Context is about to be compressed — flush the judge buffer first."""
        workspace = agent.workspace
        if workspace is None:
            return
        session_key = agent.session_id or agent.agent_id
        if not self._judge_buffers.get(session_key):
            return
        model = self._get_classification_model(agent)
        if model is None:
            return
        event_store = workspace.get_experience_event_store()
        compiled_store = workspace.get_compiled_experience_store()
        await self._flush_judge(model, event_store, compiled_store, session_key)

    @staticmethod
    async def _persist_feedback_classification(
        event_store: Any,
        compiled_store: Any,
        result: Dict[str, Any],
        threshold: float,
        original_task: str = "",
    ) -> bool:
        """Write the classification event and persist a correction card when warranted."""
        from agentica.experience.compiler import ExperienceCompiler

        is_correction = result.get("is_correction", False)
        confidence = result.get("confidence", 0.0)
        rule = (result.get("rule") or "").strip()
        correction_key = (
            ExperienceCompiler.correction_key_from_rule(rule) if rule else ""
        )

        classify_event: Dict[str, Any] = {
            "event_type": "correction_classification",
            "is_correction": is_correction,
            "confidence": confidence,
            "should_persist": result.get("should_persist", False),
            "persist_target": result.get("persist_target", "none"),
            "user_message": result.get("user_message", "")[:300],
            "rule": rule,
            "correction_key": correction_key,
        }
        if original_task:
            classify_event["original_task"] = original_task
        await event_store.append(classify_event)

        if not is_correction or confidence < threshold:
            return False
        if not result.get("should_persist", False) or result.get("persist_target") == "none":
            return False

        if original_task and "original_task" not in result:
            result["original_task"] = original_task
        card = ExperienceCompiler.compile_correction(result)
        if card:
            await compiled_store.write(card)
            return True
        return False

    async def _classify_and_persist_feedback(
        self,
        model: Any,
        event_store: Any,
        compiled_store: Any,
        user_message: str,
        previous_assistant_text: str,
        original_task: str = "",
    ) -> bool:
        """Classify user feedback with LLM and persist if appropriate.

        Uses ExperienceCompiler for card building (pure logic).
        Delegates I/O to event_store and compiled_store.

        Returns:
            True if a correction was persisted to the experience store.
        """
        threshold = self._config.feedback_confidence_threshold
        prefetched = self._prefilter_feedback_classification(user_message)
        if prefetched is not None:
            prefetched["user_message"] = user_message
            return await self._persist_feedback_classification(
                event_store=event_store,
                compiled_store=compiled_store,
                result=prefetched,
                threshold=threshold,
                original_task=original_task,
            )

        # Title is derived deterministically from `rule` in compile_correction,
        # so we no longer need to feed existing titles back to the LLM.
        prompt = (
            self._FEEDBACK_CLASSIFY_PROMPT
            + f"Previous assistant message:\n{previous_assistant_text[:1000]}\n\n"
            + f"Current user message:\n{user_message[:1000]}\n"
        )

        try:
            response = await model.response([
                Message(role="user", content=prompt),
            ])
            if not response or not response.content:
                return False

            text = response.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            result = json.loads(text)
            if not isinstance(result, dict):
                return False
            result["user_message"] = user_message
            return await self._persist_feedback_classification(
                event_store=event_store,
                compiled_store=compiled_store,
                result=result,
                threshold=threshold,
                original_task=original_task,
            )

        except json.JSONDecodeError:
            logger.debug("Feedback classification: LLM returned invalid JSON")
            return False
        except Exception as e:
            logger.warning(f"Feedback classification failed ({type(e).__name__}): {e}")
            return False


class _CompositeRunHooks(RunHooks):
    """Internal wrapper that dispatches to multiple RunHooks instances.

    Used to combine auto-injected hooks (e.g. ConversationArchiveHooks)
    with user-provided hooks without requiring users to manage composition.
    """

    def __init__(self, hooks_list: List[RunHooks]):
        self._hooks_list = hooks_list

    async def on_agent_start(self, agent: Any, **kwargs) -> None:
        for h in self._hooks_list:
            await h.on_agent_start(agent=agent, **kwargs)

    async def on_agent_end(self, agent: Any, output: Any, **kwargs) -> None:
        parallel = [h for h in self._hooks_list if h.parallelizable]
        serial = [h for h in self._hooks_list if not h.parallelizable]
        if parallel:
            await asyncio.gather(*(
                h.on_agent_end(agent=agent, output=output, **kwargs)
                for h in parallel
            ))
        for h in serial:
            await h.on_agent_end(agent=agent, output=output, **kwargs)

    async def on_llm_start(self, agent: Any, messages: Optional[List[Dict[str, Any]]] = None, **kwargs) -> None:
        for h in self._hooks_list:
            await h.on_llm_start(agent=agent, messages=messages, **kwargs)

    async def on_llm_end(self, agent: Any, response: Any = None, **kwargs) -> None:
        for h in self._hooks_list:
            await h.on_llm_end(agent=agent, response=response, **kwargs)

    async def on_tool_start(self, agent: Any, tool_name: str = "", tool_call_id: str = "",
                            tool_args: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        for h in self._hooks_list:
            await h.on_tool_start(agent=agent, tool_name=tool_name, tool_call_id=tool_call_id,
                                  tool_args=tool_args, **kwargs)

    async def on_tool_end(self, agent: Any, tool_name: str = "", tool_call_id: str = "",
                          tool_args: Optional[Dict[str, Any]] = None, result: Any = None,
                          is_error: bool = False, elapsed: float = 0.0, **kwargs) -> None:
        for h in self._hooks_list:
            await h.on_tool_end(agent=agent, tool_name=tool_name, tool_call_id=tool_call_id,
                                tool_args=tool_args, result=result, is_error=is_error,
                                elapsed=elapsed, **kwargs)

    async def on_agent_transfer(self, from_agent: Any, to_agent: Any, **kwargs) -> None:
        for h in self._hooks_list:
            await h.on_agent_transfer(from_agent=from_agent, to_agent=to_agent, **kwargs)

    async def on_user_prompt(self, agent: Any, message: str, **kwargs) -> Optional[str]:
        result = None
        for h in self._hooks_list:
            r = await h.on_user_prompt(agent=agent, message=message, **kwargs)
            if r is not None:
                result = r
                message = r  # chain: next hook sees the modified message
        return result

    async def on_pre_compact(self, agent: Any, messages=None, **kwargs) -> None:
        for h in self._hooks_list:
            await h.on_pre_compact(agent=agent, messages=messages, **kwargs)

    async def on_post_compact(self, agent: Any, messages=None, **kwargs) -> None:
        for h in self._hooks_list:
            await h.on_post_compact(agent=agent, messages=messages, **kwargs)
