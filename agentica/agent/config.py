# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Config dataclasses for Agent V2 architecture.

Provides layered configuration:
- PromptConfig: Prompt engineering details
- ToolConfig: Tool calling behavior
- WorkspaceMemoryConfig: Workspace memory settings
"""

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    Union,
)

# Imported at runtime (not TYPE_CHECKING) so the field annotation
# ``Optional[SkillLifecycleHooks]`` below resolves at class-build time.
# Pydantic 2.12+ eagerly resolves forward refs when a subclass of
# ``Workflow`` exposes ``Agent`` typed fields; a deferred TYPE_CHECKING
# import leaves the name undefined and raises ``PydanticUserError:
# `<Workflow subclass>` is not fully defined``.
from agentica.experience.skill_lifecycle_hooks import SkillLifecycleHooks


@dataclass
class PromptConfig:
    """Prompt construction configuration.

    Most users only need Agent.instructions. These parameters are for advanced customization.
    """
    # Custom system prompt (overrides default build logic)
    system_prompt: Optional[Union[str, Callable]] = None
    system_prompt_template: Optional[Any] = None  # PromptTemplate
    system_message_role: str = "system"
    user_message_role: str = "user"
    user_prompt_template: Optional[Any] = None  # PromptTemplate
    use_default_user_message: bool = True

    # System message building details
    task: Optional[str] = None
    role: Optional[str] = None
    guidelines: Optional[List[str]] = None
    expected_output: Optional[str] = None
    additional_context: Optional[str] = None
    introduction: Optional[str] = None
    references_format: Literal["json", "yaml"] = "json"

    # Prompt behavior switches
    add_name_to_instructions: bool = False
    add_datetime_to_instructions: bool = True
    prevent_hallucinations: bool = False
    prevent_prompt_leakage: bool = False
    limit_tool_access: bool = False
    enable_agentic_prompt: bool = False
    # Minimal mode: skip all section assembly, use a one-line system prompt.
    # Mirrors CC's CLAUDE_CODE_SIMPLE for minimum token consumption.
    # Useful for testing, lightweight tasks, or cost-sensitive scenarios.
    minimal: bool = False

    # Output formatting
    output_language: Optional[str] = None
    markdown: bool = False

    # Todo reminder: inject a gentle user-role reminder when the LLM hasn't
    # called write_todos for this many assistant turns. 0 = disabled.
    # Mirrors CC's TODO_REMINDER_CONFIG.TURNS_SINCE_WRITE (default 10).
    todo_reminder_interval: int = 10


@dataclass
class ToolConfig:
    """Tool calling configuration."""
    support_tool_calls: bool = True
    tool_call_limit: Optional[int] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    auto_load_mcp: bool = False
    # Knowledge tools
    search_knowledge: bool = True
    update_knowledge: bool = False
    # References
    add_references: bool = False
    # Compression
    compress_tool_results: bool = False
    compression_manager: Optional[Any] = None

    # ---- Deep / Agentic capabilities (Model-layer hooks) ----

    # Context overflow handling: when token usage exceeds the threshold (0–1 fraction
    # of context_window), truncate old non-system messages before the next LLM call.
    # 0.0 = disabled. Recommended: 0.8 (warn at 80%, hard-truncate at 90%).
    context_overflow_threshold: float = 0.0


@dataclass
class WorkspaceMemoryConfig:
    """Workspace memory loading configuration."""
    load_workspace_context: bool = True
    load_workspace_memory: bool = True
    # Maximum number of memory entries to inject per run (relevance-ranked).
    # Maps to CC's "select up to N memories" in sidequery.
    max_memory_entries: int = 5
    # Auto-archive conversation to workspace after each run() (zero LLM cost,
    # just appends raw messages to conversations/ directory).
    auto_archive: bool = False
    # Auto-extract memories from conversation using an LLM call.
    # Only fires when the LLM didn't call save_memory during the run.
    # Batched at boundaries (not per-turn) so token cost is bounded.
    auto_extract_memory: bool = False
    # Boundary policy: flush the pending turn buffer through the extraction
    # LLM every N turns. 0 disables the periodic trigger; on_pre_compact
    # still flushes whatever has accumulated. Default 10 turns matches the
    # typical CLI session length where users want roughly one extraction
    # per "topic".
    extract_every_n_turns: int = 10
    # Frequency cap (seconds) across processes. Even if every_n_turns says
    # "flush", we skip when the previous extraction ran less than this many
    # seconds ago. Persisted to ~/.agentica/extract_state.json. 0 disables.
    extract_min_seconds_between: int = 60
    # Recompile confirmed user/feedback memories into ~/.agentica/AGENTS.md so
    # future sessions automatically inherit long-lived preferences.
    sync_memories_to_global_agent_md: bool = False


@dataclass
class HistoryConfig:
    """Filter rules applied to multi-turn history before it's injected into the prompt.

    Active only when ``Agent.add_history_to_context=True``. The filter runs on a
    *copy* of the historical messages — the underlying ``working_memory.runs``
    is never mutated, so different filters can be tried across runs without
    losing data.

    Pipeline (in this order):
        working_memory.get_messages_from_last_n_runs(...)   # built-in tool-result truncation
              ↓
        excluded_tools  → drop matching tool messages + paired assistant.tool_calls
              ↓
        assistant_max_chars  → truncate long assistant replies
              ↓
        Agent.history_filter(history)  → user-defined Callable (final say)
              ↓
        consistency fix  → strip orphan assistant.tool_calls (no matching tool result)

    Attributes:
        excluded_tools: Tool name patterns whose results are dropped from history.
            Glob style via ``fnmatch`` (e.g. ``"search_*"``, ``"web_search"``).
            Matched tool messages are removed AND the corresponding ``tool_calls``
            entries on the preceding assistant message are stripped, so the OpenAI
            API contract ("each tool_call must be followed by its tool result")
            is preserved.
        assistant_max_chars: If set, truncate ``assistant`` message content longer
            than this length. ``None`` = no truncation. Tool calls are not affected.
            Only applies when ``content`` is a string — multimodal assistant turns
            (where ``content`` is a list of content blocks) are left untouched to
            avoid accidentally dropping image/audio parts.
    """
    excluded_tools: List[str] = field(default_factory=list)
    assistant_max_chars: Optional[int] = None


@dataclass
class ToolRuntimeConfig:
    """Runtime configuration for a single tool.

    Controls whether a tool is enabled at Agent level.
    Query-level override via run(enabled_tools=[...]).
    """
    name: str
    enabled: bool = True


@dataclass
class SkillRuntimeConfig:
    """Runtime configuration for a single skill.

    Controls whether a skill is enabled at Agent level.
    Query-level override via run(enabled_skills=[...]).
    """
    name: str
    enabled: bool = True


@dataclass
class ExperienceConfig:
    """Self-evolution experience capture configuration.

    Tool error and success pattern capture is deterministic (zero LLM cost).
    User correction classification uses auxiliary_model.

    Lifecycle:
        1. Capture: tool errors, correction classification, success sequences
        2. Promote: repeat_count >= promotion_count within promotion_window_days → tier=hot
        3. Demote: unused > demotion_days → tier=warm
        4. Archive: unused > archive_days → tier=cold
    """
    # Capture switches. Default to False so libraries do not silently persist
    # behavioral data to disk; opt in explicitly when self-evolution is desired.
    capture_tool_errors: bool = False
    capture_user_corrections: bool = False
    capture_success_patterns: bool = False

    # LLM classification confidence threshold for persisting corrections
    feedback_confidence_threshold: float = 0.8

    # Boundary policy for the LLM correction judge. Cheap prefilter (explicit
    # rule prefixes + strong negations) still fires per-turn — only the
    # fall-through LLM scan is batched. Every N turns, the buffered
    # (user, assistant) pairs are passed to one LLM call that scans for
    # corrections across the window. 0 disables the periodic trigger;
    # on_pre_compact still flushes accumulated buffer.
    judge_every_n_turns: int = 10
    # Frequency cap (seconds) across processes. Same idea as
    # extract_min_seconds_between. Persisted to ~/.agentica/extract_state.json.
    judge_min_seconds_between: int = 60

    # Promotion lifecycle
    promotion_count: int = 3
    # Used in lifecycle sweep: repeats must occur within this window to promote
    promotion_window_days: int = 7
    demotion_days: int = 30
    archive_days: int = 90

    # Injection
    max_experiences_in_prompt: int = 5

    # Sync confirmed experiences to ~/.agentica/AGENTS.md
    sync_to_global_agent_md: bool = False

    # Skill upgrade pipeline (None = disabled)
    skill_upgrade: Optional["SkillUpgradeConfig"] = None


@dataclass
class SkillUpgradeConfig:
    """Experience → Skill automatic upgrade pipeline configuration.

    When enabled, high-value experiences that cross the threshold are
    automatically compiled into SKILL.md files and installed as shadow skills
    in the workspace. Runtime episodes are recorded, and at checkpoint
    intervals an LLM judges whether to keep, promote, revise, or rollback.

    Modes:
        off: No skill upgrade (same as setting skill_upgrade=None on ExperienceConfig)
        draft: Generate SKILL.md drafts only, do not install
        shadow: Generate + auto-install to workspace-local generated_skills (default)
    """
    mode: str = "shadow"
    min_repeat_count: int = 3
    min_tier: str = "hot"
    checkpoint_interval: int = 5
    rollback_consecutive_failures: int = 2
    notify_user: bool = True
    # Require at least N ``tool_recovery`` events in the workspace before
    # any candidate is allowed to spawn a skill.
    #
    # A ``tool_recovery`` is emitted by ExperienceCaptureHooks whenever a
    # tool succeeds in a run AND that same tool has previously failed
    # somewhere in this workspace's events.jsonl. It signals "the agent
    # has demonstrated it CAN solve this problem on its own" — without it
    # we'd be capturing skills for things the agent perpetually fails at.
    # The semantic is decoupled from skill installation (no chicken-egg).
    #
    # Defaults to 1 (gate on). Set to 0 for cold-start / demo workspaces
    # where no successful recovery has happened yet.
    min_success_applications: int = 1
    # Optional lifecycle hooks for extensions like multi-critic admission
    # gates, append-only provenance audit logs or LLM-driven repair-or-discard
    # maintenance. ``None`` falls back to a no-op at call sites; research
    # extensions (e.g. ``evaluation/vag/lifecycle/``) implement
    # ``SkillLifecycleHooks`` and pass an instance here.
    lifecycle_hooks: Optional[SkillLifecycleHooks] = None


@dataclass
class SandboxConfig:
    """Sandbox execution isolation configuration (best-effort).

    Controls file system and command execution boundaries for security.
    NOTE: This is a best-effort safety net, NOT a true security sandbox.
    Determined attackers can bypass these checks (e.g. via encoding, symlinks,
    or indirect execution). Use OS-level sandboxing (Docker, seccomp, etc.)
    for untrusted code.

    Attributes:
        enabled: Whether sandbox restrictions are active
        writable_dirs: List of directory paths the agent is allowed to write to.
        blocked_paths: Path components that are always blocked for read/write.
            Access to any path containing these path components is denied.
            Uses path component matching (not substring) to avoid false positives.
        blocked_commands: Shell command patterns that are blocked from execution.
            Uses regex boundary matching to reduce false positives.
        allowed_commands: Optional whitelist of allowed command prefixes.
            If set (non-None), ONLY commands whose first token matches one of
            these prefixes are permitted. None means no whitelist restriction
            (all commands allowed, subject to blocked_commands).
            Example: ["python", "pip", "git", "pytest"] restricts the agent to
            only run Python/pip/git/pytest commands.
            NOTE: This is prefix-matched against the first token of the command,
            so "python" allows both "python script.py" and "python3 -c '...'".
        max_execution_time: Maximum seconds for a single command execution
    """
    enabled: bool = False
    writable_dirs: List[str] = field(default_factory=list)
    blocked_paths: List[str] = field(default_factory=lambda: [
        ".ssh", ".gnupg", ".aws", ".azure", ".config/gcloud",
        ".env", ".netrc", "id_rsa", "id_ed25519",
    ])
    blocked_commands: List[str] = field(default_factory=lambda: [
        "rm -rf /", "rm -rf /*", "mkfs", "dd if=",
        ":(){ :|:& };:", "chmod -R 777 /",
        "> /dev/sda", "curl|sh", "curl |sh", "wget|sh", "wget |sh",
    ])
    allowed_commands: Optional[List[str]] = None
    max_execution_time: int = 300


@dataclass
class AgentDefinition:
    """Identity and capability definition for an Agent.

    This groups the "what this agent is" fields into one object so callers do
    not need to pass a long flat constructor for common structured setups.
    """

    model: Optional[Any] = None
    auxiliary_model: Optional[Any] = None
    name: Optional[str] = None
    agent_id: Optional[str] = None
    description: Optional[str] = None
    when_to_use: Optional[str] = None
    instructions: Optional[Union[str, List[str], Callable]] = None
    tools: Optional[List[Any]] = None
    knowledge: Optional[Any] = None
    workspace: Optional[Any] = None
    work_dir: Optional[str] = None
    response_model: Optional[Type[Any]] = None


@dataclass
class AgentExecutionConfig:
    """Execution behavior for an Agent."""

    add_history_to_context: bool = False
    num_history_turns: int = 3
    use_structured_outputs: bool = False
    debug: bool = False
    enable_tracing: bool = False
    session_id: Optional[str] = None
    hooks: Optional[Any] = None


@dataclass
class AgentMemoryConfig:
    """Memory and experience behavior for an Agent."""

    enable_long_term_memory: bool = False
    enable_experience_capture: bool = False
    long_term_memory_config: WorkspaceMemoryConfig = field(default_factory=WorkspaceMemoryConfig)
    experience_config: ExperienceConfig = field(default_factory=ExperienceConfig)
    working_memory: Optional[Any] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class AgentSafetyConfig:
    """Guardrail and sandbox configuration for an Agent."""

    sandbox_config: Optional[SandboxConfig] = None
    tool_input_guardrails: List[Any] = field(default_factory=list)
    tool_output_guardrails: List[Any] = field(default_factory=list)
    input_guardrails: List[Any] = field(default_factory=list)
    output_guardrails: List[Any] = field(default_factory=list)
