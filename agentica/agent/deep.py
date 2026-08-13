# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: DeepAgent — batteries-included product preset.

A pre-configured Agent preset for CLI, Gateway, and daily dogfood workflows.
Use plain Agent for SDK integrations that need the smallest stable surface.

DeepAgent enables the product defaults users expect from an unattended assistant:
- 40+ built-in tools (file ops, web search, execute, subagent task, todos)
- Runner agentic loop: LLM ↔ tool-call auto-loop with multi-turn reasoning
- Two-layer compression (tool-result budget → Layer 1 evict → Layer 2
  native/LLM summarise; reactive compact on prompt_too_long)
- Death spiral detection + cost tracking + cost budget
- Repeated tool-call detection (inject "change strategy" at 3 repeats)
- Workspace memory (AGENTS.md, MEMORY.md, daily memory, relevance recall)
- Conversation archive (auto_archive for search_conversations)
- Agentic prompt (heartbeat, tools guide, self-verification)
- Sandbox isolation (optional, off by default)
- Multi-turn history

Usage:
    from agentica import DeepAgent

    # One-liner: full-featured agent
    agent = DeepAgent()
    response = agent.run_sync("Research the latest advances in RAG")
    print(response.content)

    # Enable memory tool (LLM can save/search memories)
    agent = DeepAgent(enable_long_term_memory=False)  # explicitly disable long-term memory

    # Enable human-in-the-loop
    agent = DeepAgent(include_ask_user_question=True)

    # Disable web search (file-only agent)
    agent = DeepAgent(include_web_search=False, include_fetch_url=False)

    # Custom task subagent model
    from agentica import OpenAIChat
    agent = DeepAgent(task_model=OpenAIChat(id="gpt-4o-mini"))

    # With cost budget
    from agentica import RunConfig
    response = await agent.run("Analyze X", config=RunConfig(max_cost_usd=1.0))
    print(response.cost_tracker.total_cost_usd)

    # Unified 3-tier tool permission (see agentica.agent.permissions):
    # "ask" (read-only tools only), "auto" (writes restricted to work_dir),
    # "allow-all" (no restriction — default, matches historical behavior).
    agent = DeepAgent(permission_mode="auto")
    agent.set_permission_mode("allow-all")  # switch at runtime, no rebuild

    # Custom sandbox override (advanced — bypasses permission_mode's default)
    from agentica import SandboxConfig
    agent = DeepAgent(sandbox_config=SandboxConfig(enabled=True, writable_dirs=["./output"]))

    # Every Agent parameter is declared explicitly — a typo raises TypeError here
    agent = DeepAgent(debug=True, enable_tracing=True, response_model=MyModel)
"""
import os
from typing import Any, Callable, Dict, List, Optional, Type, Union

from agentica.agent.base import Agent
from agentica.config import AGENTICA_NUM_HISTORY_TURNS
from agentica.agent.config import (
    ExperienceConfig,
    HistoryConfig,
    PromptConfig,
    SandboxConfig,
    ToolConfig,
    WorkspaceMemoryConfig,
)
from agentica.agent.history_filter import HistoryFilter
from agentica.hooks import AgentHooks
from agentica.memory import WorkingMemory
from agentica.model.base import Model
from agentica.tools.base import Tool, ModelTool, Function
from agentica.workspace import Workspace


class DeepAgent(Agent):
    """Batteries-included product preset.

    DeepAgent = Agent + builtin tools + workspace memory + compression +
    self-evolution defaults. It is intended for CLI/Gateway/product surfaces,
    not as the minimal SDK core contract.

    Enabled by default:
    - Two-layer compression pipeline (eviction + LLM summarisation)
    - MCP auto-loading from local mcp_config.json/yaml when available
    - Workspace memory with relevance recall (max_memory_entries=10)
    - Conversation auto-archive (auto_archive=True)
    - Memory auto-extract after each run (auto_extract_memory=True) —
      falls back to auxiliary_model to extract memories when the LLM did
      not call save_memory during the run.
    - Workspace memory stays per-workspace by default; syncing memories into
      the user-global AGENTS.md remains opt-in
    - auxiliary_model: defaults to the main model (same instance), so the
      whole stack runs on one API key without DeepAgent picking a hardcoded
      OpenAI sibling. Pass an explicit auxiliary_model (any provider, any
      size) to override — e.g. a cheaper same-provider variant for side
      tasks like compression / memory extraction / correction classification
      / experience lifecycle.
    - Agentic prompt with datetime and agent name
    - Self-evolution: enable_experience_capture=True + ExperienceConfig with all capture_*
      switches on (tool errors, user corrections, success patterns), while
      global AGENTS sync and skill auto-upgrade stay opt-in

    All parameters are optional — sensible defaults are applied.

    Every Agent parameter is declared explicitly in ``__init__``; there is no
    ``**kwargs``. An unknown or misspelled name therefore raises a TypeError
    naming DeepAgent at construction, instead of being forwarded into
    ``Agent.__init__`` and failing there without mentioning this class. The
    flip side is that the ``include_*`` toggles are not a stable contract to
    build kwargs against — a whitelist surface that disables all of them wants
    plain ``Agent`` plus the presets it actually needs.
    """

    def __init__(
        self,
        *,
        model: Optional[Model] = None,
        auxiliary_model: Optional[Model] = None,
        name: str = "DeepAgent",
        tools: Optional[List[Union[ModelTool, Tool, Callable, Dict, Function]]] = None,
        workspace: Optional[Union[Any, str]] = None,
        user_id: Optional[str] = None,
        work_dir: Optional[str] = None,
        session_id: Optional[str] = None,
        add_history_to_context: bool = True,
        num_history_turns: int = AGENTICA_NUM_HISTORY_TURNS,
        prompt_config: Optional[PromptConfig] = None,
        tool_config: Optional[ToolConfig] = None,
        long_term_memory_config: Optional[WorkspaceMemoryConfig] = None,
        experience_config: Optional[ExperienceConfig] = None,
        sandbox_config: Optional[SandboxConfig] = None,
        permission_mode: str = "allow-all",
        # Builtin tool toggles — mirror get_builtin_tools() params
        include_file_tools: bool = True,
        include_execute: bool = True,
        include_web_search: bool = True,
        include_fetch_url: bool = True,
        include_todos: bool = True,
        include_task: bool = True,
        include_skills: bool = True,
        include_ask_user_question: bool = False,
        web_search_provider: Optional[str] = None,
        enable_long_term_memory: bool = True,
        enable_diagnostics: bool = False,
        diagnostics_servers: Optional[List[str]] = None,
        diagnostics_errors_only: bool = True,
        background_process_registry: Optional[Any] = None,
        # Warns (never blocks) when another live session has the same file
        # uncommitted; see agentica/peer_conflicts.py. Supplied by the CLI,
        # which is where a session's presence identity lives.
        peer_conflict_checker: Optional[Any] = None,
        task_model: Optional[Model] = None,
        custom_skill_dirs: Optional[List[str]] = None,
        ask_user_question_callback: Optional[Callable] = None,
        # ---- Plain Agent parameters, forwarded unchanged ----
        # Declared one by one instead of collected in **kwargs. Agent.__init__
        # is keyword-only with no **kwargs of its own, so a forwarded typo (or
        # a parameter this class renames) used to die inside Agent with a
        # message that never mentioned DeepAgent.
        agent_id: Optional[str] = None,
        description: Optional[str] = None,
        instructions: Optional[Union[str, List[str], Callable]] = None,
        knowledge: Optional[Any] = None,
        auxiliary_task_models: Optional[Dict[str, Model]] = None,
        fallback_models: Optional[List[Model]] = None,
        fallback_on_break: bool = False,
        max_api_retry: int = 1,
        response_model: Optional[Type[Any]] = None,
        use_structured_outputs: bool = False,
        # DeepAgent is the self-evolving preset: on by default, unlike Agent.
        enable_experience_capture: bool = True,
        debug: bool = False,
        enable_tracing: bool = False,
        hooks: Optional[Union[AgentHooks, List[AgentHooks]]] = None,
        session_base_dir: Optional[str] = None,
        enable_session_log: bool = True,
        history_config: Optional[HistoryConfig] = None,
        history_filter: Optional[HistoryFilter] = None,
        tool_input_guardrails: Optional[List[Any]] = None,
        tool_output_guardrails: Optional[List[Any]] = None,
        input_guardrails: Optional[List[Any]] = None,
        output_guardrails: Optional[List[Any]] = None,
        working_memory: Optional[WorkingMemory] = None,
        context: Optional[Dict[str, Any]] = None,
        environment_context: Optional[str] = None,
    ):
        if model is None:
            from agentica.model.defaults import create_default_model
            model = create_default_model()

        # Default auxiliary_model — reuse the main model so the whole stack
        # runs on a single API key. Pass a different model explicitly to
        # offload side tasks (compression, memory extraction, correction
        # classification, experience lifecycle) onto a cheaper/faster sibling.
        if auxiliary_model is None:
            auxiliary_model = model

        # Default workspace
        if workspace is None:
            workspace = Workspace(os.path.expanduser("~/.agentica/workspace"), user_id=user_id)

        # Default work_dir
        if work_dir is None:
            work_dir = os.getcwd()

        # Unified 3-tier tool permission (see agentica.agent.permissions).
        # Resolve a single concrete SandboxConfig instance up front — this
        # exact object is shared by every builtin tool AND self.sandbox_config
        # (set below via super().__init__()), so a later
        # ``agent.set_permission_mode(...)`` mutation (flips .enabled in
        # place) is immediately observed by already-constructed tools without
        # rebuilding the Agent. An explicit sandbox_config always wins as-is
        # (advanced override); permission_mode only supplies the default.
        from agentica.agent.permissions import validate_permission_mode, sandbox_should_be_enabled
        validate_permission_mode(permission_mode)
        if sandbox_config is None:
            # writable_dirs must be seeded with work_dir: SandboxConfig only
            # enforces work_dir as a fallback when writable_dirs is non-empty
            # (see BuiltinFileTool._validate_write_path) — an empty list is a
            # silent no-op even with enabled=True.
            sandbox_config = SandboxConfig(
                enabled=sandbox_should_be_enabled(permission_mode), writable_dirs=[work_dir]
            )

        # Builtin tools + user-provided tools
        from agentica.tools.builtin import get_builtin_tools
        all_tools: List[Union[ModelTool, Tool, Callable, Dict, Function]] = list(
            get_builtin_tools(
                work_dir=work_dir,
                include_file_tools=include_file_tools,
                include_execute=include_execute,
                include_web_search=include_web_search,
                include_fetch_url=include_fetch_url,
                include_todos=include_todos,
                include_task=include_task,
                include_skills=include_skills,
                include_ask_user_question=include_ask_user_question,
                web_search_provider=web_search_provider,
                task_model=task_model,
                custom_skill_dirs=custom_skill_dirs,
                ask_user_question_callback=ask_user_question_callback,
                sandbox_config=sandbox_config,
                background_process_registry=background_process_registry,
                enable_diagnostics=enable_diagnostics,
                diagnostics_servers=diagnostics_servers,
                diagnostics_errors_only=diagnostics_errors_only,
                peer_conflict_checker=peer_conflict_checker,
            )
        )
        if tools:
            all_tools.extend(tools)

        # Opinionated config defaults (user can override by passing their own)
        if prompt_config is None:
            prompt_config = PromptConfig(
                markdown=True,
                enable_agentic_prompt=True,
                add_datetime_to_instructions=True,
                add_name_to_instructions=True,
            )

        if tool_config is None:
            tool_config = ToolConfig(
                auto_load_mcp=True,
                permission_mode=permission_mode,
            )
        else:
            # permission_mode kwarg always wins, even over a caller-supplied
            # tool_config, so it never silently diverges from the
            # sandbox_config resolved above (both must agree on the mode).
            tool_config.permission_mode = permission_mode

        if long_term_memory_config is None:
            long_term_memory_config = WorkspaceMemoryConfig(
                auto_archive=True,
                auto_extract_memory=True,
                # Boundary-triggered (every 10 turns or on_pre_compact) instead
                # of per-turn — keeps token cost bounded and removes the post-
                # response blocking that used to make the user wait several
                # seconds before the next prompt.
                extract_every_n_turns=10,
                extract_min_seconds_between=60,
                load_workspace_context=True,
                load_workspace_memory=True,
                max_memory_entries=10,
            )

        # DeepAgent is the product preset: capture errors + corrections only.
        # Capturing a tool error is not the same as injecting it: the cards and
        # events are what the skill-upgrade pipeline grounds gotchas in, while
        # `CompiledExperienceStore.get_relevant` keeps them out of the prompt.
        # Pure success sequences are intentionally dropped — they don't carry
        # actionable lessons and just inflate the telemetry.
        # Users can pass their own experience_config to override.
        if experience_config is None:
            experience_config = ExperienceConfig(
                capture_tool_errors=True,
                capture_user_corrections=True,
                capture_success_patterns=False,
                # Cheap prefilter still per-turn; LLM fall-through judge
                # batches every 10 turns (or on_pre_compact). Same idea as
                # extract_every_n_turns above.
                judge_every_n_turns=10,
                judge_min_seconds_between=60,
                skill_upgrade=None,
            )

        super().__init__(
            model=model,
            auxiliary_model=auxiliary_model,
            name=name,
            tools=all_tools,
            workspace=workspace,
            user_id=user_id,
            work_dir=work_dir,
            enable_long_term_memory=enable_long_term_memory,
            session_id=session_id,
            add_history_to_context=add_history_to_context,
            num_history_turns=num_history_turns,
            prompt_config=prompt_config,
            tool_config=tool_config,
            long_term_memory_config=long_term_memory_config,
            experience_config=experience_config,
            sandbox_config=sandbox_config,
            agent_id=agent_id,
            description=description,
            instructions=instructions,
            knowledge=knowledge,
            auxiliary_task_models=auxiliary_task_models,
            fallback_models=fallback_models,
            fallback_on_break=fallback_on_break,
            max_api_retry=max_api_retry,
            response_model=response_model,
            use_structured_outputs=use_structured_outputs,
            enable_experience_capture=enable_experience_capture,
            debug=debug,
            enable_tracing=enable_tracing,
            hooks=hooks,
            session_base_dir=session_base_dir,
            enable_session_log=enable_session_log,
            history_config=history_config,
            history_filter=history_filter,
            tool_input_guardrails=tool_input_guardrails,
            tool_output_guardrails=tool_output_guardrails,
            input_guardrails=input_guardrails,
            output_guardrails=output_guardrails,
            working_memory=working_memory,
            context=context,
            environment_context=environment_context,
        )
