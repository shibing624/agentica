# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Agent base class - V2 architecture with layered configuration.

Architecture:
- Agent defines identity and capabilities ("who I am, what I can do")
- Runner handles execution (LLM calls, tool calls, streaming, memory updates)
- Mixins: PromptsMixin, AsToolMixin, ToolsMixin, PrinterMixin
- Session state: in-memory WorkingMemory (serializable via to_dict/from_dict)
- Multi-modal: images/videos/audio passed as run() parameters, not stored on Agent

Parameters organized in three layers:
1. Core definition (~10): model, name, instructions, tools, knowledge, etc.
2. Common config (~5): add_history_to_context, debug, enable_tracing, etc.
3. Packed config (3): PromptConfig, ToolConfig, WorkspaceMemoryConfig
"""
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)

if TYPE_CHECKING:
    from agentica.goals import GoalRunResult
import copy
import os
import time
import weakref
from inspect import signature
from uuid import uuid4
from pathlib import Path
from dataclasses import dataclass, field
from agentica.utils.log import logger, set_log_level_to_debug, set_log_level_to_info
from agentica.model.message import Message
from agentica.tools.base import ModelTool, Tool, Function
from agentica.tools.skill_tool import SkillTool
from agentica.model.base import Model
from agentica.run_response import RunResponse, AgentCancelledError
from agentica.run_config import RunConfig
from agentica.run_context import RunContext, TaskAnchor
from agentica.memory import WorkingMemory
from agentica.memory.session_log import SessionLog
from agentica.compression import CompressionManager
from agentica.config import LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY
from agentica.agent.config import (
    PromptConfig, ToolConfig, WorkspaceMemoryConfig, HistoryConfig, SandboxConfig,
    ToolRuntimeConfig, SkillRuntimeConfig, ExperienceConfig,
    AgentDefinition, AgentExecutionConfig, AgentMemoryConfig, AgentSafetyConfig,
)
from agentica.agent.history_filter import HistoryFilter
from agentica.hooks import (
    AgentHooks, RunHooks, ConversationArchiveHooks, MemoryExtractHooks,
    ExperienceCaptureHooks, _CompositeRunHooks,
)
from agentica.runner import Runner

# Import mixin classes — pure method containers, no state, no __init__
from agentica.agent.prompts import PromptsMixin
from agentica.agent.as_tool import AsToolMixin
from agentica.agent.tools import ToolsMixin
from agentica.agent.printer import PrinterMixin


@dataclass(init=False)
class Agent(PromptsMixin, AsToolMixin, ToolsMixin, PrinterMixin):
    """AI Agent — defines identity and capabilities.

    Agent only describes "who I am, what I can do".
    Session persistence is handled by external SessionManager.

    Parameters are organized in three layers:
    1. Core definition (~10): model, name, instructions, tools, etc.
    2. Common config (~5): add_history_to_context, debug, etc.
    3. Packed config (3): prompt_config, tool_config, long_term_memory_config

    For output_language, markdown, search_knowledge etc., set them via
    prompt_config=PromptConfig(...) or tool_config=ToolConfig(...).

    Example - Minimal:
        >>> agent = Agent(instructions="You are a helpful assistant.")
        >>> response = await agent.run("Hello!")

    Example - Full:
        >>> agent = Agent(
        ...     name="analyst",
        ...     model=OpenAIChat(id="gpt-4o"),
        ...     instructions="You are a data analyst.",
        ...     tools=[web_search, calculator],
        ...     knowledge=my_knowledge,
        ...     response_model=AnalysisReport,
        ...     prompt_config=PromptConfig(markdown=True, output_language="Chinese"),
        ... )
    """

    # ============================
    # Layer 1: Core definition
    # ============================
    model: Optional[Model] = None
    # Auxiliary model for low-cost side tasks (compression, memory extraction, evaluation).
    # When set, CompressionManager and other subsystems use this instead of the main model.
    auxiliary_model: Optional[Model] = None
    # Cross-provider fallback model chain. Activated PER-CALL (not per-run):
    # each LLM call starts from `model`; fallbacks are tried in order only when
    # the primary fails (content_filter / exhausted-retry timeout / 5xx).
    # The next call again starts from the primary. Use cross-provider models —
    # same provider often shares the moderation layer, defeating the purpose.
    # RunConfig.fallback_models, if provided, overrides this default for one run.
    fallback_models: List[Model] = field(default_factory=list)
    name: Optional[str] = None
    agent_id: str = ""
    description: Optional[str] = None
    when_to_use: Optional[str] = None  # Hint for LLM: when to delegate tasks to this agent
    instructions: Optional[Union[str, List[str], Callable]] = None
    tools: Optional[List[Union[ModelTool, Tool, Callable, Dict, Function]]] = None
    knowledge: Optional[Any] = None  # Knowledge type
    workspace: Optional[Any] = None  # Workspace type
    user_id: Optional[str] = None
    work_dir: Optional[str] = None  # Working directory for file operations (used by builtin tools)
    enable_long_term_memory: bool = False  # Whether to enable long-term memory tools and hooks
    enable_experience_capture: bool = False  # Whether to enable experience capture (self-evolution)
    response_model: Optional[Type[Any]] = None

    # ============================
    # Layer 2: Common config
    # ============================
    add_history_to_context: bool = False
    num_history_turns: int = 3
    use_structured_outputs: bool = False
    debug: bool = False
    enable_tracing: bool = False

    # Session persistence (CC-style append-only JSONL):
    # Set session_id to enable. Stored at .sessions/{session_id}.jsonl
    # Supports compact boundaries for resume from last compaction point.
    session_id: Optional[str] = None

    # Lifecycle hooks (per-agent)
    hooks: Optional[AgentHooks] = None

    # ============================
    # Layer 3: Packed config
    # ============================
    prompt_config: PromptConfig = field(default_factory=PromptConfig)
    tool_config: ToolConfig = field(default_factory=ToolConfig)
    long_term_memory_config: WorkspaceMemoryConfig = field(default_factory=WorkspaceMemoryConfig)
    experience_config: ExperienceConfig = field(default_factory=ExperienceConfig)
    sandbox_config: Optional[SandboxConfig] = None
    history_config: HistoryConfig = field(default_factory=HistoryConfig)
    history_filter: Optional[HistoryFilter] = None

    # Tool-level guardrails (run before/after each tool call)
    tool_input_guardrails: List[Any] = field(default_factory=list)
    tool_output_guardrails: List[Any] = field(default_factory=list)

    # ============================
    # Runtime
    # ============================
    working_memory: WorkingMemory = field(default_factory=WorkingMemory)
    run_id: Optional[str] = field(default=None, init=False, repr=False)
    run_input: Optional[Any] = field(default=None, init=False, repr=False)
    # SDK-first internal run lifecycle (arch_v5.md Phase 0/1).
    # Created by Runner at run start, holds the current run's TaskAnchor and status.
    # External code should treat this as read-only.
    run_context: Optional[RunContext] = field(default=None, init=False, repr=False)
    # Session-scoped TaskAnchor: pinned ONCE on the first run of a session and
    # reused for every subsequent run. Reset when session_id changes so a new
    # conversation can establish its own anchor. Read by prompts.py + Runner.
    task_anchor: Optional[TaskAnchor] = field(default=None, init=False, repr=False)
    _anchor_session_id: Optional[str] = field(default=None, init=False, repr=False)
    # Append-only JSONL session log. Lazily created when session_id is set
    # (see _init_execution) or by the first call to get_goal_manager().
    # Declared here so attribute access is never speculative (no getattr).
    _session_log: Optional[Any] = field(default=None, init=False, repr=False)
    # Persistent standing-goal state machine. Lazily created by
    # ``get_goal_manager()``; bound to ``_session_log`` for persistence.
    goal_manager: Optional[Any] = field(default=None, init=False, repr=False)
    # Set by subagent.spawn() before child.run() so the child Runner can record
    # parent_run_id + RunSource.subagent in its RunContext. None for top-level runs.
    _parent_run_id: Optional[str] = field(default=None, init=False, repr=False)
    run_response: RunResponse = field(default_factory=RunResponse, init=False, repr=False)
    stream: Optional[bool] = field(default=None, init=False, repr=False)
    stream_intermediate_steps: bool = field(default=False, init=False, repr=False)
    _cancelled: bool = field(default=False, init=False, repr=False)
    _running: bool = field(default=False, init=False, repr=False)
    # Asyncio handles for hard cancellation (set by Runner._run_impl).
    # Allows cancel() to inject CancelledError into blocking awaits
    # (LLM API call, tool execution, subagent run) from another thread.
    _run_loop: Optional[Any] = field(default=None, init=False, repr=False)
    _run_task: Optional[Any] = field(default=None, init=False, repr=False)
    # Live event callback. Fired synchronously from the asyncio thread by
    # the Runner (compression progress) and BuiltinTaskTool (subagent
    # progress). The CLI registers this to render real-time sub-process
    # visibility. Signature: callback(event: dict) -> None.
    _event_callback: Optional[Any] = field(default=None, init=False, repr=False)

    # Run-level hooks (set per-run via run(hooks=...))
    _run_hooks: Optional[RunHooks] = field(default=None, init=False, repr=False)
    # Default run hooks (auto-injected, e.g. ConversationArchiveHooks when auto_archive=True)
    _default_run_hooks: Optional[RunHooks] = field(default=None, init=False, repr=False)
    # Per-run cost budget (USD). Set by Runner before _run_impl, read by Model.
    _run_max_cost_usd: Optional[float] = field(default=None, init=False, repr=False)
    # Per-run cross-provider fallback model chain. Set by Runner from RunConfig.
    # Triggered per-call by content_filter / exhausted-retry timeout / 5xx.
    _run_fallback_models: List[Any] = field(default_factory=list, init=False, repr=False)
    # Max LLM loop turns. None = unlimited (main agent default).
    # Subagents set this via SubagentConfig.max_turns as a safety net.
    _max_turns: Optional[int] = field(default=None, init=False, repr=False)

    # Tool/Skill runtime configs (Agent-level enable/disable)
    _tool_runtime_configs: Dict[str, ToolRuntimeConfig] = field(default_factory=dict, init=False, repr=False)
    _skill_runtime_configs: Dict[str, SkillRuntimeConfig] = field(default_factory=dict, init=False, repr=False)

    # Query-level enabled_tools/enabled_skills (set per-run, cleared after run)
    _enabled_tools: Optional[List[str]] = field(default=None, init=False, repr=False)
    _enabled_skills: Optional[List[str]] = field(default=None, init=False, repr=False)

    # Task list (populated by BuiltinTodoTool.write_todos)
    todos: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    # Context for tools and prompt functions (runtime input)
    context: Optional[Dict[str, Any]] = None

    def __init__(
            self,
            *,
            # ---- Core definition ----
            model: Optional[Model] = None,
            auxiliary_model: Optional[Model] = None,
            fallback_models: Optional[List[Model]] = None,
            name: Optional[str] = None,
            agent_id: Optional[str] = None,
            description: Optional[str] = None,
            when_to_use: Optional[str] = None,
            instructions: Optional[Union[str, List[str], Callable]] = None,
            tools: Optional[List[Union[ModelTool, Tool, Callable, Dict, Function]]] = None,
            knowledge: Optional[Any] = None,
            workspace: Optional[Union[Any, str]] = None,  # Workspace or str path
            user_id: Optional[str] = None,  # User ID for multi-user workspace isolation
            work_dir: Optional[str] = None,  # Working directory for file operations
            enable_long_term_memory: bool = False,  # Enable long-term memory tools and hooks
            enable_experience_capture: bool = False,  # Enable experience capture (self-evolution)
            response_model: Optional[Type[Any]] = None,
            # ---- Common config ----
            add_history_to_context: bool = False,
            num_history_turns: int = 8,
            use_structured_outputs: bool = False,
            debug: bool = False,
            enable_tracing: bool = False,
            hooks: Optional[AgentHooks] = None,
            # ---- Session persistence ----
            session_id: Optional[str] = None,
            # ---- Packed config ----
            prompt_config: Optional[PromptConfig] = None,
            tool_config: Optional[ToolConfig] = None,
            long_term_memory_config: Optional[WorkspaceMemoryConfig] = None,
            experience_config: Optional[ExperienceConfig] = None,
            sandbox_config: Optional[SandboxConfig] = None,
            history_config: Optional[HistoryConfig] = None,
            history_filter: Optional[HistoryFilter] = None,
            tool_input_guardrails: Optional[List[Any]] = None,
            tool_output_guardrails: Optional[List[Any]] = None,
            # ---- Agent-level guardrails (run on user input / agent output) ----
            input_guardrails: Optional[List[Any]] = None,
            output_guardrails: Optional[List[Any]] = None,
            # ---- Runtime ----
            working_memory: Optional[WorkingMemory] = None,
            context: Optional[Dict[str, Any]] = None,
    ):
        self._init_definition(
            model=model,
            auxiliary_model=auxiliary_model,
            fallback_models=fallback_models,
            name=name,
            agent_id=agent_id,
            description=description,
            when_to_use=when_to_use,
            instructions=instructions,
            tools=tools,
            knowledge=knowledge,
            workspace=workspace,
            user_id=user_id,
            work_dir=work_dir,
            enable_long_term_memory=enable_long_term_memory,
            enable_experience_capture=enable_experience_capture,
            response_model=response_model,
        )
        self._init_execution(
            add_history_to_context=add_history_to_context,
            num_history_turns=num_history_turns,
            use_structured_outputs=use_structured_outputs,
            debug=debug,
            enable_tracing=enable_tracing,
            hooks=hooks,
            session_id=session_id,
        )
        self._init_packed_config(
            prompt_config=prompt_config,
            tool_config=tool_config,
            long_term_memory_config=long_term_memory_config,
            experience_config=experience_config,
            sandbox_config=sandbox_config,
            history_config=history_config,
            history_filter=history_filter,
            tool_input_guardrails=tool_input_guardrails,
            tool_output_guardrails=tool_output_guardrails,
            input_guardrails=input_guardrails,
            output_guardrails=output_guardrails,
        )
        self._init_runtime(
            working_memory=working_memory,
            context=context,
        )

        # Create Runner instance
        self._runner = Runner(self)

        # Post-init setup
        self._post_init()

    def _init_definition(
        self,
        *,
        model: Optional[Model],
        auxiliary_model: Optional[Model],
        fallback_models: Optional[List[Model]],
        name: Optional[str],
        agent_id: Optional[str],
        description: Optional[str],
        when_to_use: Optional[str],
        instructions: Optional[Union[str, List[str], Callable]],
        tools: Optional[List[Union[ModelTool, Tool, Callable, Dict, Function]]],
        knowledge: Optional[Any],
        workspace: Optional[Union[Any, str]],
        user_id: Optional[str],
        work_dir: Optional[str],
        enable_long_term_memory: bool,
        enable_experience_capture: bool,
        response_model: Optional[Type[Any]],
    ) -> None:
        """Initialize identity and capability definition."""
        self.model = model
        self.auxiliary_model = auxiliary_model
        self.fallback_models = list(fallback_models) if fallback_models else []
        self.name = name
        self.agent_id = agent_id or str(uuid4())
        self.description = description
        self.when_to_use = when_to_use
        self.instructions = instructions
        self.tools = tools
        self.knowledge = knowledge
        self.response_model = response_model
        self.work_dir = work_dir
        self.enable_long_term_memory = enable_long_term_memory
        self.enable_experience_capture = enable_experience_capture
        self.user_id = user_id

        if isinstance(workspace, str):
            from agentica.workspace import Workspace
            self.workspace = Workspace(workspace, user_id=user_id)
            self.user_id = self.workspace.user_id
        else:
            self.workspace = workspace
            if user_id is not None and workspace is not None:
                existing = workspace.user_id
                if existing not in (None, "default") and existing != user_id:
                    logger.warning(
                        f"Agent user_id={user_id!r} overrides Workspace user_id={existing!r}"
                    )
                workspace.set_user(user_id)
            if self.workspace is not None:
                self.user_id = self.workspace.user_id

    def _init_execution(
        self,
        *,
        add_history_to_context: bool,
        num_history_turns: int,
        use_structured_outputs: bool,
        debug: bool,
        enable_tracing: bool,
        hooks: Optional[AgentHooks],
        session_id: Optional[str],
    ) -> None:
        """Initialize execution behavior and session state."""
        self.add_history_to_context = add_history_to_context
        self.num_history_turns = num_history_turns
        self.use_structured_outputs = use_structured_outputs
        self.debug = debug
        self.enable_tracing = enable_tracing
        self.hooks = hooks

        self.session_id = session_id
        self._session_log = None
        if session_id is not None:
            self._session_log = SessionLog(session_id=session_id)

    def _init_packed_config(
        self,
        *,
        prompt_config: Optional[PromptConfig],
        tool_config: Optional[ToolConfig],
        long_term_memory_config: Optional[WorkspaceMemoryConfig],
        experience_config: Optional[ExperienceConfig],
        sandbox_config: Optional[SandboxConfig],
        history_config: Optional[HistoryConfig],
        history_filter: Optional[HistoryFilter],
        tool_input_guardrails: Optional[List[Any]],
        tool_output_guardrails: Optional[List[Any]],
        input_guardrails: Optional[List[Any]],
        output_guardrails: Optional[List[Any]],
    ) -> None:
        """Initialize structured config objects and safety controls."""
        self.prompt_config = prompt_config or PromptConfig()
        self.tool_config = tool_config or ToolConfig()
        self.long_term_memory_config = long_term_memory_config or WorkspaceMemoryConfig()
        self.experience_config = experience_config or ExperienceConfig()
        self.sandbox_config = sandbox_config
        self.history_config = history_config or HistoryConfig()
        self.history_filter = history_filter
        self.tool_input_guardrails = tool_input_guardrails or []
        self.tool_output_guardrails = tool_output_guardrails or []
        self.input_guardrails = input_guardrails or []
        self.output_guardrails = output_guardrails or []

    def _init_runtime(
        self,
        *,
        working_memory: Optional[WorkingMemory],
        context: Optional[Dict[str, Any]],
    ) -> None:
        """Initialize mutable per-agent runtime state."""
        self.working_memory = working_memory or WorkingMemory()
        self.context = context
        self.run_id = None
        self.run_input = None
        self.run_context = None
        self.task_anchor = None
        self._anchor_session_id = None
        self._parent_run_id = None
        self.run_response = RunResponse()
        self.stream = None
        self.stream_intermediate_steps = False
        self._cancelled = False
        self._running = False
        self._run_loop = None
        self._run_task = None
        self._event_callback = None
        self._run_hooks = None
        self._default_run_hooks = None
        self._tool_runtime_configs: Dict[str, ToolRuntimeConfig] = {}
        self._skill_runtime_configs: Dict[str, SkillRuntimeConfig] = {}
        self._enabled_tools = None
        self._enabled_skills = None
        self._run_max_cost_usd = None
        self._run_fallback_models = []
        self.todos = []
        self._tool_policy_prompts: List[str] = []
        self._session_guidance_prompts: List[str] = []

        # Session-level set of memory filenames already surfaced (dedup across turns).
        # Prevents the same memory entry from occupying system prompt slots every turn.
        self._surfaced_memories: set = set()

        # Context-overflow protection dedup state. Guards prevent the same
        # overflow warning from being emitted on every loop iteration.
        self._overflow_warning_emitted: bool = False

    def _post_init(self):
        """Post-initialization setup."""
        if self.debug:
            set_log_level_to_debug()
            logger.debug("Set Log level: debug")
        else:
            set_log_level_to_info()

        self._warn_misconfigured_long_term_memory()

        # Auto-load MCP tools
        if self.tool_config.auto_load_mcp:
            self._load_mcp_tools()

        # Isolate stateful tools per-agent. Tools that bind agent/workspace
        # references (BuiltinTodoTool, BuiltinTaskTool, BuiltinMemoryTool,
        # SkillTool) override Tool.clone() to return a fresh instance, so the
        # caller's original tool object is never overwritten when the same
        # logical tool config is shared across multiple agents (Swarm clones,
        # subagents, manual reuse). Stateless tools clone to ``self`` (no-op).
        # Tool list may also contain raw callables / ModelTool / Function — left
        # as-is since they hold no agent state. Must happen BEFORE any mutation
        # on the tool list (skill dir injection, agent wiring, prompt merge).
        if self.tools:
            self.tools = [t.clone() if isinstance(t, Tool) else t for t in self.tools]

        # Register BuiltinMemoryTool when long-term memory is enabled and workspace exists
        if self.enable_long_term_memory and self.workspace is not None:
            from agentica.tools.buildin_tools import BuiltinMemoryTool
            has_memory_tool = any(isinstance(t, BuiltinMemoryTool) for t in (self.tools or []))
            if not has_memory_tool:
                memory_tool = BuiltinMemoryTool()
                if self.tools is None:
                    self.tools = [memory_tool]
                else:
                    self.tools = list(self.tools) + [memory_tool]

        # Inject generated-skill dirs into any SkillTool BEFORE first use.
        self._inject_generated_skill_dirs()

        # Bind agent-aware tools to this agent (todos, parent_agent, workspace,
        # skill registry filtering). Centralized here so Agent.clone() can call
        # the same helper without duplicating the wiring logic.
        self._wire_tools_to_self()

        # Merge tool system prompts into instructions (read-only over tools).
        self._merge_tool_system_prompts()

        # Load runtime config from workspace YAML
        self._load_runtime_config()

        # Initialize compression manager
        if self.tool_config.compress_tool_results and self.tool_config.compression_manager is None:
            self.tool_config.compression_manager = CompressionManager(
                model=self.auxiliary_model or self.model,
                compress_tool_results=True,
                workspace=self.workspace,
            )
        # Wire auxiliary model into existing compression manager if not already set
        elif (
            self.auxiliary_model is not None
            and self.tool_config.compression_manager is not None
            and self.tool_config.compression_manager.model is None
        ):
            self.tool_config.compression_manager.model = self.auxiliary_model

        # Tracing: check Langfuse config when enabled
        if self.enable_tracing:
            if not LANGFUSE_SECRET_KEY or not LANGFUSE_PUBLIC_KEY:
                logger.warning(
                    "enable_tracing=True but Langfuse is not configured. "
                    "Set environment variables to enable:\n"
                    "  LANGFUSE_SECRET_KEY=sk-lf-xxx\n"
                    "  LANGFUSE_PUBLIC_KEY=pk-lf-xxx\n"
                    "  LANGFUSE_BASE_URL=https://cloud.langfuse.com  # or self-hosted\n"
                    "Install: pip install langfuse"
                )

        # Auto-archive: inject ConversationArchiveHooks when long-term memory is enabled and auto_archive=True (zero cost)
        # Auto-extract: inject MemoryExtractHooks when long-term memory is enabled and auto_extract_memory=True (LLM cost)
        # Auto-experience: inject ExperienceCaptureHooks when experience capture is enabled (zero LLM cost)
        auto_hooks: list = []
        if self.enable_long_term_memory and self.workspace is not None:
            if self.long_term_memory_config.auto_archive:
                auto_hooks.append(ConversationArchiveHooks())
            if self.long_term_memory_config.auto_extract_memory:
                auto_hooks.append(
                    MemoryExtractHooks(
                        sync_memories_to_global_agent_md=(
                            self.long_term_memory_config.sync_memories_to_global_agent_md
                        ),
                        every_n_turns=self.long_term_memory_config.extract_every_n_turns,
                        min_seconds_between=self.long_term_memory_config.extract_min_seconds_between,
                    )
                )
        if self.enable_experience_capture and self.workspace is not None:
            auto_hooks.append(ExperienceCaptureHooks(self.experience_config))
        if auto_hooks:
            self._default_run_hooks = _CompositeRunHooks(auto_hooks)

    def _warn_misconfigured_long_term_memory(self) -> None:
        """Warn when long-term memory config is set but the master switch is off.

        Common pitfall: user passes ``long_term_memory_config=WorkspaceMemoryConfig(auto_archive=True)``
        and a Workspace, but forgets ``enable_long_term_memory=True``. The hooks
        in ``_post_init`` are gated on the boolean flag, so the config is silently
        ignored. Fail loud at construction time instead.
        """
        # Only check opt-in fields (auto_archive / auto_extract_memory default to False).
        # load_workspace_* default to True, so they'd trigger on every plain Agent(workspace=...).
        cfg = self.long_term_memory_config
        opted_in = bool(cfg and (cfg.auto_archive or cfg.auto_extract_memory))
        if opted_in and not self.enable_long_term_memory:
            logger.warning(
                "long_term_memory_config has auto_archive / auto_extract_memory enabled, "
                "but enable_long_term_memory=False. These settings will be IGNORED. "
                "Pass enable_long_term_memory=True to activate."
            )
        if self.enable_long_term_memory and self.workspace is None:
            logger.warning(
                "enable_long_term_memory=True but workspace=None. Long-term memory, "
                "auto-archive and memory extraction will all be disabled. "
                "Pass workspace=Workspace('~/.agentica/workspace', user_id=...) to activate."
            )

    def _wire_tools_to_self(self) -> None:
        """Bind agent-aware tools to this agent.

        Tools are expected to have already been cloned (see ``_post_init`` /
        ``clone``) so that mutations here only affect this agent's private
        instances and never bleed into other agents that share the same
        logical tool config.
        """
        if not self.tools:
            return
        from agentica.tools.buildin_tools import BuiltinTodoTool, BuiltinMemoryTool
        from agentica.tools.builtin_task_tool import BuiltinTaskTool
        from agentica.tools.skill_tool import SkillTool

        for tool in self.tools:
            if isinstance(tool, Tool):
                tool.set_agent_model(self.model)
            if isinstance(tool, BuiltinTodoTool):
                tool.set_agent(self)
            elif isinstance(tool, BuiltinTaskTool):
                tool.set_parent_agent(self)
            elif isinstance(tool, BuiltinMemoryTool):
                tool.set_workspace(self.workspace)
                tool.set_sync_global_agent_md(
                    self.long_term_memory_config.sync_memories_to_global_agent_md
                )
            elif isinstance(tool, SkillTool):
                tool._agent = self

    def _inject_generated_skill_dirs(self) -> None:
        """Attach workspace generated skill dirs to any SkillTool before prompt merge."""
        if not self.enable_experience_capture or self.workspace is None:
            return
        if self.experience_config.skill_upgrade is None:
            return

        gen_dir = str(self.workspace._get_user_generated_skills_dir())
        for tool in self.tools or []:
            if isinstance(tool, SkillTool) and gen_dir not in tool._custom_skill_dirs:
                tool._custom_skill_dirs.append(gen_dir)

    def refresh_tool_system_prompts(self) -> None:
        """Rebuild cached tool/session guidance prompts after tool state changes."""
        self._merge_tool_system_prompts()

    async def get_workspace_context_prompt(self) -> Optional[str]:
        """Dynamically load workspace context for system prompt.

        Prefers frozen snapshot (if freeze_snapshots() was called) to keep
        system prompt prefix stable across turns for prompt cache hits.
        Falls back to live read if no snapshot exists.
        """
        if not self.workspace or not self.long_term_memory_config.load_workspace_context:
            return None
        if not self.workspace.exists():
            return None
        # Prefer frozen snapshot for prompt cache stability
        frozen = self.workspace.get_frozen_context()
        if frozen is not None:
            return frozen
        context = await self.workspace.get_context_prompt()
        return context if context else None

    async def get_workspace_memory_prompt(self, query: str = "") -> Optional[str]:
        """Dynamically load relevant workspace memory for system prompt.

        Prefers frozen snapshot (if freeze_snapshots() was called) to keep
        system prompt prefix stable across turns for prompt cache hits.
        Falls back to live relevance-based recall if no snapshot exists.

        Args:
            query: Current user query string for relevance scoring.

        Returns:
            Formatted memory string, or None if workspace/memory not configured.
        """
        if not self.enable_long_term_memory:
            return None
        if not self.workspace or not self.long_term_memory_config.load_workspace_memory:
            return None
        # Prefer frozen snapshot for prompt cache stability
        frozen = self.workspace.get_frozen_memory()
        if frozen is not None:
            return frozen
        memory = await self.workspace.get_relevant_memories(
            query=query,
            limit=self.long_term_memory_config.max_memory_entries,
            already_surfaced=self._surfaced_memories,
        )
        return memory if memory else None

    async def get_experience_prompt(self, query: str = "") -> Optional[str]:
        """Load relevant experiences for system prompt injection.

        Args:
            query: Current user query for relevance scoring.

        Returns:
            Formatted experience string, or None if experience not enabled.
        """
        if not self.enable_experience_capture or not self.workspace:
            return None
        experiences = await self.workspace.get_relevant_experiences(
            query=query,
            limit=self.experience_config.max_experiences_in_prompt,
        )
        return experiences if experiences else None

    def _load_mcp_tools(self):
        """Auto-load MCP tools from mcp_config.json/yaml if available."""
        try:
            import asyncio
            from agentica.mcp.config import MCPConfig
            from agentica.tools.mcp_tool import McpTool, CompositeMultiMcpTool

            config = MCPConfig()
            if not config.servers:
                return

            mcp_tool = McpTool.from_config(config_path=config.config_path)

            async def init_mcp():
                await mcp_tool.__aenter__()
                await mcp_tool.__aexit__(None, None, None)

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, init_mcp())
                    future.result(timeout=30)
            else:
                asyncio.run(init_mcp())

            if self.tools is None:
                self.tools = [mcp_tool]
            else:
                self.tools = list(self.tools) + [mcp_tool]

            logger.info(f"Auto-loaded MCP tools from: {config.config_path}")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to auto-load MCP tools: {e}")

    def add_tool_policy_prompt(self, prompt: Optional[str]) -> None:
        """Add a static tool-usage policy section to the system prompt."""
        if prompt and prompt not in self._tool_policy_prompts:
            self._tool_policy_prompts.append(prompt)

    def add_session_guidance(self, prompt: Optional[str]) -> None:
        """Add a dynamic per-session guidance section to the system prompt."""
        if prompt and prompt not in self._session_guidance_prompts:
            self._session_guidance_prompts.append(prompt)

    def _merge_tool_system_prompts(self) -> None:
        """Collect tool prompts and split them into static vs dynamic sections."""
        if not self.tools:
            return

        from agentica.tools.skill_tool import SkillTool

        self._tool_policy_prompts = []
        self._session_guidance_prompts = []

        for tool in self.tools:
            if isinstance(tool, Tool) and hasattr(tool, 'get_system_prompt'):
                prompt = tool.get_system_prompt()
                if not prompt:
                    continue
                if isinstance(tool, SkillTool):
                    self.add_session_guidance(prompt)
                else:
                    self.add_tool_policy_prompt(prompt)

        logger.debug(
            "Collected %d tool prompts and %d session guidance prompts",
            len(self._tool_policy_prompts),
            len(self._session_guidance_prompts),
        )

    def cancel(self):
        """Cancel the current run.

        Hard-cancellation: sets the soft flag AND injects ``CancelledError``
        into the running asyncio task via ``loop.call_soon_threadsafe``.
        This unblocks any pending ``await`` (LLM API call, tool execution,
        subagent run) immediately, instead of waiting for the next safe
        boundary check.

        Safe to call from another thread.
        """
        self._cancelled = True
        loop = self._run_loop
        task = self._run_task
        if loop is not None and task is not None and not task.done():
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass

    def _check_cancelled(self):
        """Check if cancelled and raise AgentCancelledError if so."""
        if self._cancelled:
            self._cancelled = False
            raise AgentCancelledError("Agent run cancelled by user")

    @property
    def is_streamable(self) -> bool:
        """For structured outputs we disable streaming."""
        return self.response_model is None

    @property
    def identifier(self) -> Optional[str]:
        return self.name or self.agent_id

    @classmethod
    def from_parts(
        cls,
        definition: Optional[AgentDefinition] = None,
        execution: Optional[AgentExecutionConfig] = None,
        memory: Optional[AgentMemoryConfig] = None,
        safety: Optional[AgentSafetyConfig] = None,
    ) -> "Agent":
        """Create an Agent from grouped config parts.

        This provides a compact alternative to the flat 40+ parameter
        constructor without breaking the existing `Agent(...)` call style.
        """
        definition = definition or AgentDefinition()
        execution = execution or AgentExecutionConfig()
        memory = memory or AgentMemoryConfig()
        safety = safety or AgentSafetyConfig()
        return cls(
            model=definition.model,
            auxiliary_model=definition.auxiliary_model,
            name=definition.name,
            agent_id=definition.agent_id,
            description=definition.description,
            when_to_use=definition.when_to_use,
            instructions=definition.instructions,
            tools=definition.tools,
            knowledge=definition.knowledge,
            workspace=definition.workspace,
            work_dir=definition.work_dir,
            response_model=definition.response_model,
            add_history_to_context=execution.add_history_to_context,
            num_history_turns=execution.num_history_turns,
            use_structured_outputs=execution.use_structured_outputs,
            debug=execution.debug,
            enable_tracing=execution.enable_tracing,
            hooks=execution.hooks,
            session_id=execution.session_id,
            enable_long_term_memory=memory.enable_long_term_memory,
            enable_experience_capture=memory.enable_experience_capture,
            long_term_memory_config=memory.long_term_memory_config,
            experience_config=memory.experience_config,
            working_memory=memory.working_memory,
            context=memory.context,
            sandbox_config=safety.sandbox_config,
            tool_input_guardrails=safety.tool_input_guardrails,
            tool_output_guardrails=safety.tool_output_guardrails,
            input_guardrails=safety.input_guardrails,
            output_guardrails=safety.output_guardrails,
        )

    @classmethod
    def from_workspace(
        cls,
        workspace_path: str,
        model: Optional["Model"] = None,
        initialize: bool = True,
        **kwargs
    ) -> "Agent":
        """Create Agent from workspace path."""
        from agentica.workspace import Workspace

        workspace = Workspace(workspace_path)
        if initialize and not workspace.exists():
            workspace.initialize()

        return cls(workspace=workspace, model=model, **kwargs)

    def add_instruction(self, instruction: str):
        """Dynamically add instruction to Agent."""
        if not instruction:
            return
        if self.instructions is None:
            self.instructions = [instruction]
        elif isinstance(self.instructions, str):
            self.instructions = [self.instructions, instruction]
        elif isinstance(self.instructions, list):
            self.instructions = list(self.instructions) + [instruction]
        else:
            logger.warning(f"Cannot add instruction: instructions is {type(self.instructions)}")
            return
        logger.debug(f"Added instruction to agent: {instruction[:50]}...")

    # =========================================================================
    # Tool/Skill runtime control
    # =========================================================================

    def enable_tool(self, name: str) -> None:
        """Enable a tool by name (function name or tool class name)."""
        self._tool_runtime_configs[name] = ToolRuntimeConfig(name=name, enabled=True)

    def disable_tool(self, name: str) -> None:
        """Disable a tool by name (function name or tool class name)."""
        self._tool_runtime_configs[name] = ToolRuntimeConfig(name=name, enabled=False)

    def enable_skill(self, name: str) -> None:
        """Enable a skill by name."""
        self._skill_runtime_configs[name] = SkillRuntimeConfig(name=name, enabled=True)

    def disable_skill(self, name: str) -> None:
        """Disable a skill by name."""
        self._skill_runtime_configs[name] = SkillRuntimeConfig(name=name, enabled=False)

    def _is_tool_enabled(self, func_name: str) -> bool:
        """Check if a tool function is enabled.

        Priority: query-level (enabled_tools) > agent-level (runtime_configs) > default (True).
        """
        # Query-level whitelist: if set, only listed tools are allowed
        if self._enabled_tools is not None:
            return func_name in self._enabled_tools
        # Agent-level config
        cfg = self._tool_runtime_configs.get(func_name)
        if cfg is not None:
            return cfg.enabled
        return True

    def _is_skill_enabled(self, skill_name: str) -> bool:
        """Check if a skill is enabled.

        Priority: query-level (enabled_skills) > agent-level (runtime_configs) > default (True).
        """
        if self._enabled_skills is not None:
            return skill_name in self._enabled_skills
        cfg = self._skill_runtime_configs.get(skill_name)
        if cfg is not None:
            return cfg.enabled
        return True

    def _load_runtime_config(self) -> None:
        """Load tool/skill runtime configs from workspace YAML.

        Searches for `.agentica/runtime_config.yaml` in:
        1. workspace path (if workspace is set)
        2. current working directory

        YAML format:
            tools:
              execute:
                enabled: false
              write_file:
                enabled: true
            skills:
              iwiki-doc:
                enabled: false
        """
        config_name = ".agentica/runtime_config.yaml"
        config_path = None

        # Try workspace path first
        if self.workspace is not None:
            candidate = self.workspace.path / config_name
            if candidate.exists():
                config_path = candidate

        # Fallback: current working directory
        if config_path is None:
            candidate = Path(os.getcwd()) / config_name
            if candidate.exists():
                config_path = candidate

        if config_path is None:
            return

        try:
            import yaml
        except ImportError:
            logger.debug("PyYAML not installed, skipping runtime config loading")
            return

        try:
            data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return

            # Load tool configs
            tools_data = data.get("tools")
            if isinstance(tools_data, dict):
                for name, cfg in tools_data.items():
                    if isinstance(cfg, dict):
                        enabled = cfg.get("enabled", True)
                        self._tool_runtime_configs[name] = ToolRuntimeConfig(name=name, enabled=enabled)

            # Load skill configs
            skills_data = data.get("skills")
            if isinstance(skills_data, dict):
                for name, cfg in skills_data.items():
                    if isinstance(cfg, dict):
                        enabled = cfg.get("enabled", True)
                        self._skill_runtime_configs[name] = SkillRuntimeConfig(name=name, enabled=enabled)

            logger.debug(f"Loaded runtime config from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load runtime config from {config_path}: {e}")

    def clone(self) -> "Agent":
        """Create a lightweight clone of this Agent for concurrent execution.

        Shares heavy config (tools, instructions, knowledge) but creates a
        fresh Model instance and resets all mutable runtime state.
        Safe for parallel asyncio.gather() calls.
        """
        clone = copy.copy(self)
        # Model isolation: route through SubagentRegistry._clone_parent_model so
        # Agent.clone() / SubagentRegistry.spawn() / Swarm._clone_agent_for_task
        # share ONE concurrency-safe handoff strategy: shallow copy + reset
        # tools/functions/tool_choice/metrics + fresh ``Usage`` + drop HTTP
        # client (the original belongs to the source agent's loop).
        # Previously this used ``copy.deepcopy`` which (a) attempts to deepcopy
        # the bound HTTP client / sessions and (b) silently diverged from the
        # subagent path's reset semantics.
        if self.model is not None:
            from agentica.subagent import SubagentRegistry
            clone.model = SubagentRegistry._clone_parent_model(self.model)
        clone.agent_id = str(uuid4())
        clone._init_runtime(
            working_memory=WorkingMemory(),
            context=dict(self.context) if isinstance(self.context, dict) else self.context,
        )
        clone._session_log = None
        # Per-agent enable/disable runtime configs are mutable dicts. Without
        # explicit re-init, ``copy.copy`` aliases them, so ``clone.enable_tool``
        # / ``clone.disable_tool`` would silently mutate the source agent's
        # tool gating. Same for skill runtime configs.
        clone._tool_runtime_configs = dict(self._tool_runtime_configs)
        clone._skill_runtime_configs = dict(self._skill_runtime_configs)
        # Inherit hook dedup state from source so cloned sub-agents do not
        # re-emit the same overflow warning that the parent already handled.
        clone._overflow_warning_emitted = self._overflow_warning_emitted
        # Fresh Runner bound to the clone
        clone._runner = Runner(clone)
        # Tool isolation: stateful tools (todos, parent_agent, workspace, skill
        # filtering) are cloned per-agent and rewired so the clone does not
        # share or overwrite the source agent's tool state slots.
        if self.tools:
            clone.tools = [t.clone() if isinstance(t, Tool) else t for t in self.tools]
            clone._wire_tools_to_self()
        return clone

    def add_introduction(self, introduction: str) -> None:
        """Add an introduction message to memory."""
        if introduction is None:
            return
        for message in self.working_memory.messages:
            if message.role == "assistant" and message.content == introduction:
                return
        self.working_memory.add_message(Message(role="assistant", content=introduction))

    def _resolve_context(self) -> None:
        logger.debug("Resolving context")
        if self.context is None:
            return
        # context may be a dict / string / callable / any resolved object.
        # Only dict-shaped context has per-key callables to resolve; everything
        # else is treated as an already-resolved value and passed through.
        if not isinstance(self.context, dict):
            if callable(self.context):
                sig = signature(self.context)
                resolved = self.context(agent=self) if "agent" in sig.parameters else self.context()
                if resolved is not None:
                    self.context = resolved
            return
        for ctx_key, ctx_value in self.context.items():
            if callable(ctx_value):
                try:
                    sig = signature(ctx_value)
                    resolved_ctx_value = None
                    if "agent" in sig.parameters:
                        resolved_ctx_value = ctx_value(agent=self)
                    else:
                        resolved_ctx_value = ctx_value()
                    if resolved_ctx_value is not None:
                        self.context[ctx_key] = resolved_ctx_value
                except Exception as e:
                    logger.warning(f"Failed to resolve context for {ctx_key}: {e}")
            else:
                self.context[ctx_key] = ctx_value

    def update_model(self) -> None:
        if self.model is None:
            from agentica.model.defaults import create_default_model
            logger.debug("Model not set, resolving default model from configured provider credentials")
            self.model = create_default_model()
            self._wire_tools_to_self()
        model_cls = self.model.name or self.model.__class__.__name__
        logger.debug(
            f"Agent '{self.name}' using {model_cls}("
            f"id={self.model.id}, provider={self.model.provider}, "
            f"thinking={self.model.describe_thinking_mode()})"
        )

        # Clear previously registered tools/functions and metrics to prevent accumulation
        # across multiple run() calls on the same Agent instance.
        if self.model.functions:
            self.model.functions.clear()
        if self.model.tools:
            self.model.tools.clear()
        self.model.metrics.clear()
        # Reset tool-call state so each agent run starts clean.
        # Prevents function_call_stack / tool_choice leaking between runs
        # (or between agents that share the same Model instance).
        self.model.function_call_stack = None
        self.model.tool_choice = None

        # Set agent reference on model (legacy, for backward compatibility with
        # direct model.run_function_calls() calls in tests/examples).
        # In normal Runner-driven execution, run_tools=False and Runner owns
        # tool execution, so _agent_ref is not needed.
        self.model._agent_ref = weakref.ref(self)

        # Set response_format
        if self.response_model is not None and self.model.response_format is None:
            if self.use_structured_outputs and self.model.supports_structured_outputs:
                logger.debug("Setting Model.response_format to Agent.response_model")
                self.model.response_format = self.response_model
                self.model.use_structured_outputs = True
            else:
                self.model.response_format = {"type": "json_object"}

        # Add tools to the Model (with runtime filtering)
        agent_tools = self.get_tools()
        if agent_tools is not None and self.tool_config.support_tool_calls:
            for tool in agent_tools:
                if (
                        self.response_model is not None
                        and self.use_structured_outputs
                        and self.model.supports_structured_outputs
                ):
                    self.model.add_tool(tool=tool, strict=True, agent=self)
                else:
                    self.model.add_tool(tool=tool, agent=self)

            # Filter out disabled functions from model after add_tool
            self._filter_model_functions()

        # Set tool_choice
        if self.model.tool_choice is None and self.tool_config.tool_choice is not None:
            self.model.tool_choice = self.tool_config.tool_choice

        # Set tool_call_limit
        if self.tool_config.tool_call_limit is not None:
            self.model.tool_call_limit = self.tool_config.tool_call_limit

        # Add trace identity to Models for Langfuse/OpenAI integration.
        for model in [self.model, *self.fallback_models]:
            model.user_id = self.user_id
            model.session_id = self.session_id
            model.agent_name = self.name

    def _build_pre_tool_hook(self):
        """Build the pre-tool hook function based on ToolConfig settings.

        Returns an async callable (messages, function_calls) -> bool, or None
        if no hook is active. Returning False tells the Runner to proceed with
        tool execution normally.

        Capability (opt-in via ToolConfig):

        Context overflow handling
           Triggered when: tool_config.context_overflow_threshold > 0 and
           estimated token usage / context_window >= threshold.
           Two-stage action (compress-then-evict):
             1. If a compression_manager is wired, try reversible compression
                first (summarize/truncate tool results). Compression preserves
                information — prefer it over destructive eviction.
             2. Only if usage still exceeds the hard limit (threshold + 5pp)
                after compression, FIFO-evict oldest non-system messages.
           Estimation uses ~4 chars/token heuristic for speed; accurate token
           counting requires the tokenizer.

        Returning None means no hook is registered (fast path, no overhead).
        """
        overflow_threshold = self.tool_config.context_overflow_threshold

        # Fast path: feature is disabled
        if overflow_threshold <= 0.0:
            return None

        agent_ref = self  # captured in closure

        def _estimate_usage_ratio(msgs: list, ctx_window: int) -> float:
            total_chars = sum(
                len(str(m.content)) if m.content else 0
                for m in msgs
            )
            return (total_chars / 4.0) / ctx_window

        async def _pre_tool_hook(messages: list, function_calls: list) -> bool:
            model = agent_ref.model
            if model is None:
                return False

            if overflow_threshold <= 0.0:
                return False

            context_window = model.context_window or 128000
            usage_ratio = _estimate_usage_ratio(messages, context_window)

            if usage_ratio < overflow_threshold:
                return False

            hard_limit = min(overflow_threshold + 0.05, 0.95)

            # ---- Stage 1: reversible compression (prefer over eviction) ----
            compression_manager = agent_ref.tool_config.compression_manager
            compressed = False
            if compression_manager is not None:
                _uid_compress = (
                    agent_ref.workspace.user_id
                    if agent_ref.workspace is not None
                    else None
                )
                await compression_manager.compress(messages, user_id=_uid_compress)
                compressed = True
                usage_ratio = _estimate_usage_ratio(messages, context_window)

            # ---- Stage 2: FIFO evict oldest non-system messages if still over ----
            evicted = 0
            while usage_ratio >= hard_limit and len(messages) > 2:
                for idx, m in enumerate(messages):
                    if m.role != "system":
                        messages.pop(idx)
                        evicted += 1
                        break
                else:
                    break  # Only system messages left
                usage_ratio = _estimate_usage_ratio(messages, context_window)

            # Demote to debug + only once per Agent instance lifetime to
            # avoid flooding the CLI across multiple user turns.
            if not agent_ref._overflow_warning_emitted:
                logger.debug(
                    f"Agent '{agent_ref.identifier}': context overflow handled "
                    f"(estimated {usage_ratio:.0%} of {context_window} tokens). "
                    f"Compressed={compressed}, evicted {evicted} old messages. "
                    "Set tool_config=ToolConfig(context_overflow_threshold=0.0) to disable."
                )
                agent_ref._overflow_warning_emitted = True

            return False  # proceed with tool execution

        return _pre_tool_hook

    def _build_post_tool_hook(self):
        """Build the post-tool hook function for todo reminder injection.

        Returns an async callable (messages, function_call_results) -> None, or None
        if no todo tool is registered.

        Mirrors CC's getTodoReminderAttachments: after each tool batch, count how many
        assistant turns have passed since the last write_todos call. If the count exceeds
        todo_reminder_interval and there are active todos, inject a gentle user-role
        reminder message containing the current todo list state.

        This is ephemeral -- the reminder appears in the in-flight messages only and
        does not persist to memory, avoiding permanent context pollution.
        """
        from agentica.tools.buildin_tools import BuiltinTodoTool

        # Check if agent has a BuiltinTodoTool registered
        has_todo_tool = False
        if self.tools:
            for tool in self.tools:
                if isinstance(tool, BuiltinTodoTool):
                    has_todo_tool = True
                    break

        if not has_todo_tool:
            return None

        reminder_interval = self.prompt_config.todo_reminder_interval
        if reminder_interval <= 0:
            return None

        agent_ref = self  # captured in closure

        async def _post_tool_hook(messages: list, function_call_results: list) -> None:
            # Count assistant turns since last write_todos call
            turns_since_write = 0
            turns_since_reminder = 0
            found_write = False
            found_reminder = False

            for m in reversed(messages):
                if m.role == "assistant":
                    if not found_write:
                        turns_since_write += 1
                    if not found_reminder:
                        turns_since_reminder += 1
                elif m.role == "tool" and m.tool_name == "write_todos" and not found_write:
                    found_write = True
                elif (
                    m.role == "user"
                    and isinstance(m.content, str)
                    and "[Todo Reminder]" in m.content
                    and not found_reminder
                ):
                    found_reminder = True

                if found_write and found_reminder:
                    break

            # Only inject if enough turns have passed since both last write and last reminder
            if turns_since_write < reminder_interval:
                return
            if turns_since_reminder < reminder_interval:
                return

            # Only inject if there are active (non-empty) todos
            todos = agent_ref.todos
            if not todos:
                return

            # Build reminder message (mirrors CC's todo_reminder attachment content)
            todo_items = "\n".join(
                f"  {i + 1}. [{t.get('status', 'pending')}] {t.get('content', '')}"
                for i, t in enumerate(todos)
            )
            reminder_content = (
                "[Todo Reminder] The write_todos tool hasn't been used recently. "
                "If you're working on tasks that would benefit from tracking progress, "
                "consider using the write_todos tool to update your progress. "
                "Also consider cleaning up the todo list if it has become stale. "
                "Only use it if relevant to the current work.\n\n"
                f"Current todo list:\n{todo_items}"
            )
            messages.append(Message(role="user", content=reminder_content))
            logger.debug(f"Injected todo reminder ({len(todos)} items, {turns_since_write} turns since write)")

        return _post_tool_hook

    def _filter_model_functions(self) -> None:
        """Filter disabled functions from the model.

        Removes functions that are disabled via agent-level config or query-level whitelist.
        This is called after update_model() adds all tools, so we filter at the function level.
        """
        if self.model is None or self.model.functions is None:
            return

        # If no filtering configured, skip
        if self._enabled_tools is None and not self._tool_runtime_configs:
            return

        disabled_funcs = []
        for func_name in list(self.model.functions.keys()):
            if not self._is_tool_enabled(func_name):
                disabled_funcs.append(func_name)

        if not disabled_funcs:
            return

        for func_name in disabled_funcs:
            del self.model.functions[func_name]

        # Rebuild model.tools list to match remaining functions
        if self.model.tools is not None:
            self.model.tools = [
                t for t in self.model.tools
                if not (isinstance(t, dict) and t.get("type") == "function"
                        and t.get("function", {}).get("name") in disabled_funcs)
            ]

        logger.debug(f"Filtered {len(disabled_funcs)} disabled tools: {disabled_funcs}")

    # =========================================================================
    # Run API — delegates to self._runner (public API unchanged)
    # =========================================================================

    async def run(
        self,
        message: Optional[Union[str, List, Dict, Message]] = None,
        *,
        audio: Optional[Any] = None,
        images: Optional[Sequence[Any]] = None,
        videos: Optional[Sequence[Any]] = None,
        messages: Optional[Sequence[Union[Dict, Message]]] = None,
        add_messages: Optional[List[Union[Dict, Message]]] = None,
        config: Optional[RunConfig] = None,
        **kwargs: Any,
    ) -> RunResponse:
        """Run the Agent and return the final response (non-streaming)."""
        return await self._runner.run(
            message=message,
            audio=audio,
            images=images,
            videos=videos,
            messages=messages,
            add_messages=add_messages,
            config=config,
            **kwargs,
        )

    async def run_stream(
        self,
        message: Optional[Union[str, List, Dict, Message]] = None,
        *,
        audio: Optional[Any] = None,
        images: Optional[Sequence[Any]] = None,
        videos: Optional[Sequence[Any]] = None,
        messages: Optional[Sequence[Union[Dict, Message]]] = None,
        add_messages: Optional[List[Union[Dict, Message]]] = None,
        config: Optional[RunConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[RunResponse]:
        """Run the Agent and stream incremental responses."""
        async for chunk in self._runner.run_stream(
            message=message,
            audio=audio,
            images=images,
            videos=videos,
            messages=messages,
            add_messages=add_messages,
            config=config,
            **kwargs,
        ):
            yield chunk

    # ============================================================
    # Standing goal loop (ergonomic SDK surface, see agentica/goals.py)
    # ============================================================
    def get_goal_manager(
        self,
        *,
        default_turn_budget: Optional[int] = None,
        event_callback: Optional[Callable[..., None]] = None,
    ) -> Any:
        """Return (and lazily create) this agent's ``GoalManager``.

        Creates a ``SessionLog`` on the fly if the agent was built without
        a ``session_id``. Idempotent: subsequent calls return the same
        manager so any persisted GoalState stays consistent.

        Args:
            default_turn_budget: Overrides the safety-net turn cap (see
                ``agentica.goals.DEFAULT_TURN_BUDGET``; ``token_budget`` /
                ``wall_clock_budget_sec`` are the real hard caps). Ignored
                if a manager already exists.
            event_callback: ``(RunEventType, dict) -> None`` hook for
                ``goal.set / continuing / completed / paused`` events.

        Returns:
            ``agentica.goals.GoalManager``.
        """
        # Local import keeps module import graph cheap for users that
        # never touch the goal loop.
        from agentica.goals import GoalManager, DEFAULT_TURN_BUDGET

        if self._session_log is None:
            if self.session_id is None:
                self.session_id = str(uuid4())
            self._session_log = SessionLog(session_id=self.session_id)

        if self.goal_manager is None:
            self.goal_manager = GoalManager(
                self._session_log,
                default_turn_budget=(
                    default_turn_budget if default_turn_budget is not None
                    else DEFAULT_TURN_BUDGET
                ),
                judge_model=self.auxiliary_model or self.model,
                event_callback=event_callback,
            )
            # Load any persisted state from a previous session.
            self.goal_manager.load()
        elif event_callback is not None:
            # Allow re-binding the callback (cheap, no-mutation otherwise).
            self.goal_manager.event_callback = event_callback

        return self.goal_manager

    def enable_goal_tool(self) -> None:
        """Attach ``GoalTool.update_goal`` so the model can self-mark the
        goal ``complete`` or ``paused``, short-circuiting the external
        judge. Idempotent.
        """
        from agentica.tools.goal_tool import GoalTool

        mgr = self.get_goal_manager()
        if self.tools is None:
            self.tools = []
        for t in self.tools:
            if isinstance(t, GoalTool):
                return
        self.tools.append(GoalTool(mgr.session_log))

    async def run_goal(
        self,
        objective: str,
        *,
        turn_budget: Optional[int] = None,
        token_budget: Optional[int] = None,
        wall_clock_budget_sec: Optional[float] = None,
        attach_goal_tool: bool = True,
        event_callback: Optional[Callable[..., None]] = None,
    ) -> "GoalRunResult":
        """Drive the standing-goal loop until completion / pause / budget.

        Ergonomic entry point: callers do NOT touch ``SessionLog``,
        ``GoalManager``, or ``GoalTool`` directly. The loop:

            1. Sets the objective on the manager (resets turns_used etc).
            2. Binds ``TaskAnchor`` to the objective so retrieval / prompt
               anchoring use it for every turn.
            3. Optionally attaches ``GoalTool`` so the model can short
               circuit the judge.
            4. Runs ``self.run()`` repeatedly, feeding each turn's
               ``token_delta`` and wall-clock seconds into the manager.
            5. Stops when the manager says the goal is complete /
               paused / budget_limited.

        Args:
            objective: The standing goal text. Used as the first prompt.
            turn_budget: Max LLM turns. ``None`` falls back to
                ``DEFAULT_TURN_BUDGET = 100`` (runaway safety net — cannot be
                fully disabled; pass a large number like ``10_000`` instead).
            token_budget: Max cumulative input+output tokens. ``None`` =
                unlimited. Recommended for production: ``50_000``–``200_000``
                for coding tasks.
            wall_clock_budget_sec: Max agent wall-clock seconds. ``None`` =
                unlimited. Recommended ``1800``–``3600`` for long tasks.

                The three budgets are **independent hard caps — whichever
                hits first stops the loop** (AND/intersection semantics).
                Priority each turn: ``budget > tool short-circuit > judge``.
            attach_goal_tool: Register ``GoalTool`` on this agent. Set
                False if you want the external judge to be authoritative.
            event_callback: ``goal.*`` event hook.

        Returns:
            ``agentica.goals.GoalRunResult`` with final status / reason /
            ``RunResponse`` / GoalState snapshot / turns_used.
        """
        from agentica.goals import GoalRunResult

        mgr = self.get_goal_manager(event_callback=event_callback)
        state = mgr.set(
            objective,
            turn_budget=turn_budget,
            token_budget=token_budget,
            wall_clock_budget_sec=wall_clock_budget_sec,
        )

        # Pin the anchor up front so the first turn already uses it.
        self.task_anchor = TaskAnchor(
            goal=state.objective, source_query=state.objective,
        )
        self._anchor_session_id = self.session_id

        if attach_goal_tool:
            self.enable_goal_tool()

        prompt = state.objective
        last_run_response: Optional[RunResponse] = None
        tokens_baseline = 0

        while True:
            t0 = time.monotonic()
            response = await self.run(prompt)
            elapsed = time.monotonic() - t0
            last_run_response = response

            token_delta = 0
            if response.cost_tracker is not None:
                total_now = (
                    response.cost_tracker.total_input_tokens
                    + response.cost_tracker.total_output_tokens
                )
                token_delta = max(0, total_now - tokens_baseline)
                tokens_baseline = total_now

            # Pluck (tool_name, is_error) pairs from the just-finished
            # turn so the judge sees what work actually happened and the
            # manager can track consecutive tool failures. No LLM call —
            # names + flags only.
            tool_pairs: List[Tuple[str, bool]] = []
            for tc in response.tool_calls:
                if tc.tool_name:
                    tool_pairs.append((tc.tool_name, bool(tc.is_error)))

            decision = await mgr.evaluate_after_turn(
                response.content or "",
                token_delta=token_delta,
                elapsed_sec=elapsed,
                tool_calls=tool_pairs or None,
            )

            if not decision.should_continue:
                final_state = mgr.load()
                return GoalRunResult(
                    status=decision.status,
                    reason=decision.reason,
                    run_response=last_run_response,
                    goal=final_state,
                    turns_used=final_state.turns_used if final_state else 0,
                )
            prompt = decision.continuation_prompt

    def run_sync(
        self,
        message: Optional[Union[str, List, Dict, Message]] = None,
        *,
        audio: Optional[Any] = None,
        images: Optional[Sequence[Any]] = None,
        videos: Optional[Sequence[Any]] = None,
        messages: Optional[Sequence[Union[Dict, Message]]] = None,
        add_messages: Optional[List[Union[Dict, Message]]] = None,
        config: Optional[RunConfig] = None,
        **kwargs: Any,
    ) -> RunResponse:
        """Synchronous wrapper for `run()` (non-streaming only)."""
        return self._runner.run_sync(
            message=message,
            audio=audio,
            images=images,
            videos=videos,
            messages=messages,
            add_messages=add_messages,
            config=config,
            **kwargs,
        )

    def run_stream_sync(
        self,
        message: Optional[Union[str, List, Dict, Message]] = None,
        *,
        audio: Optional[Any] = None,
        images: Optional[Sequence[Any]] = None,
        videos: Optional[Sequence[Any]] = None,
        messages: Optional[Sequence[Union[Dict, Message]]] = None,
        add_messages: Optional[List[Union[Dict, Message]]] = None,
        config: Optional[RunConfig] = None,
        **kwargs: Any,
    ) -> Iterator[RunResponse]:
        """Synchronous wrapper for `run_stream()`."""
        return self._runner.run_stream_sync(
            message=message,
            audio=audio,
            images=images,
            videos=videos,
            messages=messages,
            add_messages=add_messages,
            config=config,
            **kwargs,
        )

    # =========================================================================
    # SDK shutdown helper — drain boundary-triggered work before process exit.
    # =========================================================================

    async def flush_pending(self) -> None:
        """Drain any in-memory work that hooks have buffered for later flushing.

        Memory extraction (``MemoryExtractHooks``) and the batched correction
        judge (``ExperienceCaptureHooks``) intentionally batch turns instead
        of calling the LLM every turn. They flush on natural boundaries
        (``every_n_turns``, ``on_pre_compact``). For SDK callers that own
        the process lifecycle — e.g. a CLI exiting after the last turn, a
        worker about to be recycled — anything still in the buffer is lost
        because the buffers are in-process by design (multi-tenant safety:
        no shared disk state, no cross-user collisions).

        ``flush_pending`` gives integrators an explicit hook to drain those
        buffers for THIS agent (this ``user_id`` + ``session_id``) before
        shutdown:

        .. code-block:: python

            try:
                response = await agent.run("…")
            finally:
                await agent.flush_pending()  # call from your shutdown hook

        - Force-flushes the buffers tied to this agent's ``(user_id,
          session_id)`` key, bypassing ``every_n_turns`` and the idle gate.
        - Safe to call even with no hooks attached (no-op).
        - Other tenants' buffers are untouched: this method scopes only
          to the agent it's called on.
        - Idempotent: a second call with nothing buffered is a no-op.
        """
        for hooks in (self._default_run_hooks, self._run_hooks):
            if hooks is None:
                continue
            try:
                await hooks.flush_pending(agent=self)
            except Exception as e:
                logger.warning(f"flush_pending failed for {type(hooks).__name__}: {e}")

    def flush_pending_sync(self) -> None:
        """Synchronous wrapper for :meth:`flush_pending`.

        Convenient for ``atexit``-style shutdown paths that aren't async.
        Runs the async drain in a private event loop, mirroring how
        :meth:`run_sync` wraps :meth:`run`.
        """
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running():
            raise RuntimeError(
                "flush_pending_sync() cannot be called from a running event loop; "
                "use `await agent.flush_pending()` instead."
            )
        asyncio.run(self.flush_pending())
