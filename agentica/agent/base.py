# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Agent base class - V2 architecture with layered configuration.

V2 changes from V1:
- 57 params → ~15 params (core + common + Config objects)
- Removed: SessionMixin, MediaMixin
- Removed: backward-compat aliases (llm, knowledge_base, output_model, etc.)
- Removed: session fields (session_id, db, user_id, etc.) → SessionManager
- Removed: convenience params (search_knowledge, output_language, markdown) → use Config objects
- Added: PromptConfig, ToolConfig, WorkspaceMemoryConfig, TeamConfig
- Added: RunConfig support in run()/run_stream()
"""
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
)
from uuid import uuid4
from dataclasses import dataclass, field
from agentica.utils.log import logger, set_log_level_to_debug, set_log_level_to_info
from agentica.model.openai import OpenAIChat
from agentica.tools.base import ModelTool, Tool, Function
from agentica.model.base import Model
from agentica.run_response import RunResponse, AgentCancelledError
from agentica.memory import WorkingMemory
from agentica.agent.config import PromptConfig, ToolConfig, WorkspaceMemoryConfig, TeamConfig

# Import mixin classes — pure method containers, no state, no __init__
from agentica.agent.prompts import PromptsMixin
from agentica.agent.runner import RunnerMixin
from agentica.agent.team import TeamMixin
from agentica.agent.tools import ToolsMixin
from agentica.agent.printer import PrinterMixin


@dataclass(init=False)
class Agent(PromptsMixin, RunnerMixin, TeamMixin, ToolsMixin, PrinterMixin):
    """AI Agent — defines identity and capabilities.

    Agent only describes "who I am, what I can do".
    Session persistence is handled by external SessionManager.

    Parameters are organized in three layers:
    1. Core definition (~10): model, name, instructions, tools, etc.
    2. Common config (~5): add_history_to_messages, debug, etc.
    3. Packed config (4): prompt_config, tool_config, long_term_memory_config, team_config

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
    name: Optional[str] = None
    agent_id: str = ""
    description: Optional[str] = None
    instructions: Optional[Union[str, List[str], Callable]] = None
    tools: Optional[List[Union[ModelTool, Tool, Callable, Dict, Function]]] = None
    knowledge: Optional[Any] = None  # Knowledge type
    team: Optional[List["Agent"]] = None
    workspace: Optional[Any] = None  # Workspace type
    response_model: Optional[Type[Any]] = None

    # ============================
    # Layer 2: Common config
    # ============================
    add_history_to_messages: bool = False
    history_window: int = 3
    structured_outputs: bool = False
    debug: bool = False
    tracing: bool = False

    # ============================
    # Layer 3: Packed config
    # ============================
    prompt_config: PromptConfig = field(default_factory=PromptConfig)
    tool_config: ToolConfig = field(default_factory=ToolConfig)
    long_term_memory_config: WorkspaceMemoryConfig = field(default_factory=WorkspaceMemoryConfig)
    team_config: TeamConfig = field(default_factory=TeamConfig)

    # ============================
    # Runtime
    # ============================
    working_memory: WorkingMemory = field(default_factory=WorkingMemory)
    run_id: Optional[str] = field(default=None, init=False, repr=False)
    run_input: Optional[Any] = field(default=None, init=False, repr=False)
    run_response: RunResponse = field(default_factory=RunResponse, init=False, repr=False)
    stream: Optional[bool] = field(default=None, init=False, repr=False)
    stream_intermediate_steps: bool = field(default=False, init=False, repr=False)
    _cancelled: bool = field(default=False, init=False, repr=False)

    # Context for tools and prompt functions (runtime input)
    context: Optional[Dict[str, Any]] = None

    def __init__(
            self,
            *,
            # ---- Core definition ----
            model: Optional[Model] = None,
            name: Optional[str] = None,
            agent_id: Optional[str] = None,
            description: Optional[str] = None,
            instructions: Optional[Union[str, List[str], Callable]] = None,
            tools: Optional[List[Union[ModelTool, Tool, Callable, Dict, Function]]] = None,
            knowledge: Optional[Any] = None,
            team: Optional[List["Agent"]] = None,
            workspace: Optional[Union[Any, str]] = None,  # Workspace or str path
            response_model: Optional[Type[Any]] = None,
            # ---- Common config ----
            add_history_to_messages: bool = False,
            history_window: int = 3,
            structured_outputs: bool = False,
            debug: bool = False,
            tracing: bool = False,
            # ---- Packed config ----
            prompt_config: Optional[PromptConfig] = None,
            tool_config: Optional[ToolConfig] = None,
            long_term_memory_config: Optional[WorkspaceMemoryConfig] = None,
            team_config: Optional[TeamConfig] = None,
            # ---- Runtime ----
            working_memory: Optional[WorkingMemory] = None,
            context: Optional[Dict[str, Any]] = None,
    ):
        # Core
        self.model = model
        self.name = name
        self.agent_id = agent_id or str(uuid4())
        self.description = description
        self.instructions = instructions
        self.tools = tools
        self.knowledge = knowledge
        self.team = team
        self.response_model = response_model

        # Handle workspace: str → Workspace(path=str)
        if isinstance(workspace, str):
            from agentica.workspace import Workspace
            self.workspace = Workspace(workspace)
        else:
            self.workspace = workspace

        # Common
        self.add_history_to_messages = add_history_to_messages
        self.history_window = history_window
        self.structured_outputs = structured_outputs
        self.debug = debug
        self.tracing = tracing

        # Packed config (use defaults if not provided)
        self.prompt_config = prompt_config or PromptConfig()
        self.tool_config = tool_config or ToolConfig()
        self.long_term_memory_config = long_term_memory_config or WorkspaceMemoryConfig()
        self.team_config = team_config or TeamConfig()

        # Runtime
        self.working_memory = working_memory or WorkingMemory()
        self.context = context
        self.run_id = None
        self.run_input = None
        self.run_response = RunResponse()
        self.stream = None
        self.stream_intermediate_steps = False
        self._cancelled = False

        # Post-init setup
        self._post_init()

    def _post_init(self):
        """Post-initialization setup."""
        if self.debug:
            set_log_level_to_debug()
            logger.debug("Set Log level: debug")
        else:
            set_log_level_to_info()

        # Auto-load MCP tools
        if self.tool_config.auto_load_mcp:
            self._load_mcp_tools()

        # Merge tool system prompts into instructions
        self._merge_tool_system_prompts()

        # Initialize compression manager
        if self.tool_config.compress_tool_results and self.tool_config.compression_manager is None:
            from agentica.compression import CompressionManager
            self.tool_config.compression_manager = CompressionManager(
                model=self.model,
                compress_tool_results=True,
            )

        # Tracing: check Langfuse config when enabled
        if self.tracing:
            from agentica.config import LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY
            if not LANGFUSE_SECRET_KEY or not LANGFUSE_PUBLIC_KEY:
                logger.warning(
                    "tracing=True but Langfuse is not configured. "
                    "Set environment variables to enable:\n"
                    "  LANGFUSE_SECRET_KEY=sk-lf-xxx\n"
                    "  LANGFUSE_PUBLIC_KEY=pk-lf-xxx\n"
                    "  LANGFUSE_BASE_URL=https://cloud.langfuse.com  # or self-hosted\n"
                    "Install: pip install langfuse"
                )

    async def get_workspace_context_prompt(self) -> Optional[str]:
        """Dynamically load workspace context for system prompt."""
        if not self.workspace or not self.long_term_memory_config.load_workspace_context:
            return None
        if not self.workspace.exists():
            return None
        context = await self.workspace.get_context_prompt()
        return context if context else None

    async def get_workspace_memory_prompt(self) -> Optional[str]:
        """Dynamically load workspace memory for system prompt."""
        if not self.workspace or not self.long_term_memory_config.load_workspace_memory:
            return None
        memory = await self.workspace.get_memory_prompt(days=self.long_term_memory_config.memory_days)
        return memory if memory else None

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
                if isinstance(mcp_tool, CompositeMultiMcpTool):
                    await mcp_tool.__aenter__()
                    await mcp_tool.__aexit__(None, None, None)
                else:
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

    def _merge_tool_system_prompts(self) -> None:
        """Collect system prompts from all tools and merge into instructions."""
        if not self.tools:
            return

        tool_prompts_map: Dict[str, str] = {}
        for tool in self.tools:
            if isinstance(tool, Tool) and hasattr(tool, 'get_system_prompt'):
                prompt = tool.get_system_prompt()
                if prompt:
                    tool_class_name = tool.__class__.__name__
                    if tool_class_name not in tool_prompts_map or len(prompt) > len(tool_prompts_map[tool_class_name]):
                        tool_prompts_map[tool_class_name] = prompt

        if not tool_prompts_map:
            return

        tool_sections = []
        for tool_name, prompt in tool_prompts_map.items():
            tool_sections.append(f"<tool_instructions name=\"{tool_name}\">\n{prompt}\n</tool_instructions>")

        merged_prompt = "<tool_system_prompts>\n" + "\n\n".join(tool_sections) + "\n</tool_system_prompts>"

        if self.instructions is None:
            self.instructions = [merged_prompt]
        elif isinstance(self.instructions, str):
            self.instructions = [self.instructions, merged_prompt]
        elif isinstance(self.instructions, list):
            self.instructions = list(self.instructions) + [merged_prompt]

        logger.debug(f"Merged {len(tool_prompts_map)} tool system prompts into instructions: {list(tool_prompts_map.keys())}")

    def cancel(self):
        """Cancel the current run. Can be called from another thread/task."""
        self._cancelled = True

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

    async def save_memory(self, content: str, long_term: bool = False):
        """Save memory to workspace."""
        if self.workspace:
            await self.workspace.write_memory(content, to_daily=not long_term)
            logger.debug(f"Saved memory to workspace: long_term={long_term}")
        else:
            logger.warning("No workspace configured, memory not saved")

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

    def has_team(self) -> bool:
        return self.team is not None and len(self.team) > 0

    def add_introduction(self, introduction: str) -> None:
        """Add an introduction message to memory."""
        from agentica.model.message import Message
        if introduction is None:
            return
        for message in self.working_memory.messages:
            if message.role == "assistant" and message.content == introduction:
                return
        self.working_memory.add_message(Message(role="assistant", content=introduction))

    def _resolve_context(self) -> None:
        from inspect import signature

        logger.debug("Resolving context")
        if self.context is not None:
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
            logger.debug("Model not set, Using OpenAIChat as default")
            self.model = OpenAIChat()
        logger.debug(f"Agent, using model: {self.model}")

        # Set response_format
        if self.response_model is not None and self.model.response_format is None:
            if self.structured_outputs and self.model.supports_structured_outputs:
                logger.debug("Setting Model.response_format to Agent.response_model")
                self.model.response_format = self.response_model
                self.model.structured_outputs = True
            else:
                self.model.response_format = {"type": "json_object"}

        # Add tools to the Model
        agent_tools = self.get_tools()
        if agent_tools is not None and self.tool_config.support_tool_calls:
            for tool in agent_tools:
                if (
                        self.response_model is not None
                        and self.structured_outputs
                        and self.model.supports_structured_outputs
                ):
                    self.model.add_tool(tool=tool, strict=True, agent=self)
                else:
                    self.model.add_tool(tool=tool, agent=self)

        # Set tool_choice
        if self.model.tool_choice is None and self.tool_config.tool_choice is not None:
            self.model.tool_choice = self.tool_config.tool_choice

        # Set tool_call_limit
        if self.tool_config.tool_call_limit is not None:
            self.model.tool_call_limit = self.tool_config.tool_call_limit

        # Add agent name to the Model for Langfuse tracing
        if self.name is not None:
            self.model.agent_name = self.name
