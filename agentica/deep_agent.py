# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: DeepAgent - Enhanced Agent with built-in tools and agentic capabilities

DeepAgent is an enhanced version of Agent that automatically includes built-in tools:
- ls: List directory contents
- read_file: Read file content
- write_file: Write file content
- edit_file: Edit file (string replacement)
- glob: File pattern matching
- grep: Search file content
- execute: Execute Python code
- web_search: Web search
- fetch_url: Fetch URL content
- write_todos: Create and manage task list
- read_todos: Read current task list
- task: Launch subagent to handle complex tasks
- list_skills: List available skills
- get_skill_info: Get skill information
- user_input: Human-in-the-loop input (optional)

Key Features:
1. ReAct Loop Control: Step-by-step execution with reflection
2. Smart Context Management: Two-threshold hysteresis mechanism
3. Forced Termination: Auto-stop when context limit reached
4. Reflection & Strategy Adjustment: Detect repetitive behavior
5. Forced Iteration Control: Checkpoint reminders
"""

from collections import deque
from typing import (
    Any,
    List,
    Optional,
)
from dataclasses import dataclass, field

from agentica.agent import Agent
from agentica.deep_tools import get_builtin_tools, BuiltinTaskTool
from agentica.model.message import Message
from agentica.utils.log import logger
from agentica.utils.tokens import count_message_tokens

from agentica.prompts.base.deep_agent import (
    get_step_reflection_prompt,
    get_force_answer_prompt,
    get_repetitive_behavior_prompt,
    get_force_strategy_change_prompt,
    get_iteration_checkpoint_prompt,
)


@dataclass(init=False)
class DeepAgent(Agent):
    """
    DeepAgent - Enhanced Agent with built-in tools and agentic capabilities.

    DeepAgent inherits from Agent and automatically adds the following built-in tools:
    - File system tools: ls, read_file, write_file, edit_file, glob, grep
    - Code execution tool: execute
    - Web tools: web_search, fetch_url
    - Task management tools: write_todos, read_todos
    - Subagent tool: task
    - Skill tools: list_skills, get_skill_info
    - Human-in-the-loop: user_input (optional)

    Key Features:
    1. ReAct Loop Control: Multi-round execution with configurable max_rounds
    2. Smart Context Management: Two-threshold mechanism for context overflow
    3. Forced Termination: Auto-stop when context limit reached
    4. Reflection & Strategy Adjustment: Detect and handle repetitive behavior
    5. Forced Iteration Control: Checkpoint reminders

    Example:
        ```python
        from agentica import DeepAgent, OpenAIChat

        agent = DeepAgent(
            model=OpenAIChat(id="gpt-4o"),
            description="A powerful research assistant",
        )

        # Run the agent
        response = agent.run("Research the latest developments in AI agents")
        print(response.content)
        ```
    """

    # DeepAgent specific configuration
    work_dir: Optional[str] = None
    include_file_tools: bool = True
    include_execute: bool = True
    include_web_search: bool = True
    include_fetch_url: bool = True
    include_todos: bool = True
    include_task: bool = True
    include_skills: bool = True
    include_user_input: bool = False
    custom_skill_dirs: Optional[List[str]] = None

    # These hooks are injected into the Model layer's recursive tool-calling loop
    # via _pre_tool_hook / _tool_call_hook / _post_tool_hook on the Model instance.
    # They only affect DeepAgent — plain Agent never sets these hooks.

    # ReAct Loop Control
    enable_step_reflection: bool = False  # May interfere with model reasoning
    reflection_frequency: int = 3  # Inject reflection prompt every N steps

    # Context Management (dual-threshold hysteresis)
    context_soft_limit: Optional[int] = None  # Soft threshold: start compression (default: model.context_window * 0.6)
    context_hard_limit: Optional[int] = None  # Hard threshold: force termination (default: model.context_window * 0.8)
    enable_context_overflow_handling: bool = True  # Enabled by default

    # Repetitive Behavior Detection
    enable_repetition_detection: bool = True  # Enabled by default
    max_same_tool_calls: int = 3  # Max consecutive calls to same tool

    # Forced Iteration Control (checkpoint + must-continue reminders)
    enable_forced_iteration: bool = True  # Enabled by default
    iteration_checkpoint_frequency: int = 5  # Inject iteration checkpoint every N steps

    # Internal state
    _tool_call_history: deque = field(default_factory=lambda: deque(maxlen=10))
    _hook_step_counter: int = field(default=0, init=False, repr=False)

    def __init__(
            self,
            *,
            # ---- DeepAgent specific parameters ----
            work_dir: Optional[str] = None,
            include_file_tools: bool = True,
            include_execute: bool = True,
            include_web_search: bool = True,
            include_fetch_url: bool = True,
            include_todos: bool = True,
            include_task: bool = True,
            include_skills: bool = True,
            include_user_input: bool = False,
            custom_skill_dirs: Optional[List[str]] = None,
            enable_step_reflection: bool = False,
            reflection_frequency: int = 3,
            context_soft_limit: Optional[int] = None,
            context_hard_limit: Optional[int] = None,
            enable_context_overflow_handling: bool = True,
            enable_repetition_detection: bool = True,
            max_same_tool_calls: int = 5,
            enable_forced_iteration: bool = True,
            iteration_checkpoint_frequency: int = 5,
            # ---- All other Agent parameters via kwargs ----
            **kwargs,
    ):
        """
        Initialize DeepAgent.

        DeepAgent-specific args:
            work_dir: Working directory for file operations
            include_file_tools: Include file tools
            include_execute: Include code execution tool
            include_web_search: Include web search tool
            include_fetch_url: Include URL fetching tool
            include_todos: Include task management tools
            include_task: Include subagent task tool
            include_skills: Include skill tools
            include_user_input: Include human-in-the-loop tool
            custom_skill_dirs: Custom skill directories to load
            enable_step_reflection: Enable reflection after steps
            reflection_frequency: How often to inject reflection prompts
            context_soft_limit: Token count to start compression
            context_hard_limit: Token count to force termination
            enable_context_overflow_handling: Enable context overflow handling
            enable_repetition_detection: Enable repetitive behavior detection
            max_same_tool_calls: Max consecutive calls to same tool
            enable_forced_iteration: Enable forced iteration control (checkpoint + must-continue)
            iteration_checkpoint_frequency: How often to inject iteration checkpoints
            **kwargs: All Agent.__init__ parameters (model, name, instructions, tools, etc.)
        """
        from agentica.agent.config import PromptConfig

        # Store DeepAgent specific configuration
        self.work_dir = work_dir
        self.include_file_tools = include_file_tools
        self.include_execute = include_execute
        self.include_web_search = include_web_search
        self.include_fetch_url = include_fetch_url
        self.include_todos = include_todos
        self.include_task = include_task
        self.include_skills = include_skills
        self.include_user_input = include_user_input
        self.custom_skill_dirs = custom_skill_dirs

        # ReAct Loop Control
        self.enable_step_reflection = enable_step_reflection
        self.reflection_frequency = reflection_frequency

        # Context Management
        model = kwargs.get('model')
        cw = getattr(model, 'context_window', None) if model else None
        mot = getattr(model, 'max_output_tokens', None) if model else None
        effective_context = (cw - (mot or 0)) if cw else None
        self.context_soft_limit = context_soft_limit if context_soft_limit is not None else (int(effective_context * 0.6) if effective_context else 80000)
        self.context_hard_limit = context_hard_limit if context_hard_limit is not None else (int(effective_context * 0.8) if effective_context else 120000)
        self.enable_context_overflow_handling = enable_context_overflow_handling

        # Repetitive Behavior Detection
        self.enable_repetition_detection = enable_repetition_detection
        self.max_same_tool_calls = max_same_tool_calls
        self._tool_call_history = deque(maxlen=10)

        # Forced Iteration Control
        self.enable_forced_iteration = enable_forced_iteration
        self.iteration_checkpoint_frequency = iteration_checkpoint_frequency
        self._hook_step_counter = 0

        # Get built-in tools
        builtin_tools = get_builtin_tools(
            work_dir=work_dir,
            include_file_tools=include_file_tools,
            include_execute=include_execute,
            include_web_search=include_web_search,
            include_fetch_url=include_fetch_url,
            include_todos=include_todos,
            include_task=include_task,
            include_skills=include_skills,
            custom_skill_dirs=custom_skill_dirs,
        )

        # Add user input tool if enabled
        if include_user_input:
            from agentica.tools.user_input_tool import UserInputTool
            builtin_tools.append(UserInputTool())

        # Merge user-provided tools with built-in tools
        user_tools = kwargs.pop('tools', None)
        all_tools = list(builtin_tools)
        if user_tools:
            all_tools = self._merge_tools_with_dedup(all_tools, user_tools)

        custom_tool_count = len(user_tools) if user_tools else 0
        logger.debug(f"DeepAgent initialized with {len(builtin_tools)} builtin tools and {custom_tool_count} custom tools")

        # Build prompt_config with enable_agentic_prompt
        prompt_config = kwargs.pop('prompt_config', None)
        if prompt_config is None:
            prompt_config = PromptConfig()
        prompt_config.enable_agentic_prompt = True

        super().__init__(
            tools=all_tools,
            prompt_config=prompt_config,
            **kwargs,
        )

        # Set parent agent reference for task tool
        self._setup_task_tool()

        # Set workspace for memory tool (workspace is initialized in parent class)
        self._setup_memory_tool()

        # Bind DeepAgent hooks into the Model layer's recursive tool-calling loop
        self._bind_hooks_to_model()

    def _merge_tools_with_dedup(
            self,
            builtin_tools: List[Any],
            user_tools: List[Any],
    ) -> List[Any]:
        """Merge user tools with builtin tools."""
        result = list(builtin_tools)
        result.extend(user_tools)
        return result

    def _setup_task_tool(self) -> None:
        """Set up the task tool with parent agent reference."""
        if not self.include_task:
            return

        for tool in self.tools or []:
            if isinstance(tool, BuiltinTaskTool):
                tool.set_parent_agent(self)
                break

    def _setup_memory_tool(self) -> None:
        """Set up the memory tool with workspace reference."""
        from agentica.deep_tools import BuiltinMemoryTool
        
        # Find and configure the memory tool with workspace
        for tool in self.tools or []:
            if isinstance(tool, BuiltinMemoryTool):
                if self.workspace:
                    tool.set_workspace(self.workspace)
                    logger.debug(f"Memory tool configured with workspace: {self.workspace.path}")
                break

    def _bind_hooks_to_model(self) -> None:
        """Bind DeepAgent hooks into the Model layer's recursive tool-calling loop.

        The Model layer handles multi-round tool calling via recursive response() calls.
        By setting _pre_tool_hook / _tool_call_hook / _post_tool_hook on the Model,
        DeepAgent's context management, repetition detection, and iteration checkpoint
        features become active within that recursive loop.

        These hooks only affect DeepAgent instances — plain Agent never sets them.
        """
        if self.model is None:
            return

        agent = self  # capture for closures

        def pre_tool_hook(function_call_results: list) -> tuple:
            """Called before tool execution. Returns (should_force_answer, optional_msg)."""
            agent._hook_step_counter += 1
            if not agent.enable_context_overflow_handling:
                return False, None
            model_messages = getattr(agent.model, '_current_messages', None)
            if model_messages is None:
                return False, None
            current_tokens = agent._estimate_context_tokens(model_messages)
            return agent._handle_context_overflow(model_messages, current_tokens)

        def tool_call_hook(tool_name: str):
            """Called per tool call. Returns optional warning message."""
            return agent._check_repetitive_behavior(tool_name)

        def post_tool_hook(function_call_results: list) -> None:
            """Called after tool execution. Injects reflection/checkpoint/must-continue prompts."""
            step = agent._hook_step_counter
            # Inject reflection prompt
            if agent.enable_step_reflection and step > 0 and step % agent.reflection_frequency == 0:
                function_call_results.append(Message(
                    role="user",
                    content=get_step_reflection_prompt()
                ))
                logger.debug(f"Injected reflection prompt at step {step}")
            # Iteration checkpoint
            if agent.enable_forced_iteration and step > 0 and step % agent.iteration_checkpoint_frequency == 0:
                function_call_results.append(Message(
                    role="user",
                    content=get_iteration_checkpoint_prompt(step)
                ))
                logger.debug(f"Injected iteration checkpoint at step {step}")

        self.model._pre_tool_hook = pre_tool_hook
        self.model._tool_call_hook = tool_call_hook
        self.model._post_tool_hook = post_tool_hook

    def _estimate_context_tokens(self, messages: List[Message]) -> int:
        """Estimate the current context token count using tokens.py utilities."""
        model_id = getattr(self.model, 'id', 'gpt-4o') if self.model else 'gpt-4o'
        total = 0
        for msg in messages:
            total += count_message_tokens(msg, model_id)
        return total

    def _check_repetitive_behavior(self, tool_name: str) -> Optional[str]:
        """Check for repetitive tool call patterns.
        
        Returns a warning message if repetitive behavior is detected.
        Returns a stronger force-change message if repetition persists beyond 2x threshold.
        """
        if not self.enable_repetition_detection:
            return None

        self._tool_call_history.append(tool_name)

        if len(self._tool_call_history) < self.max_same_tool_calls:
            return None

        # Check if the last N calls are all the same tool
        recent_calls = list(self._tool_call_history)[-self.max_same_tool_calls:]
        if len(set(recent_calls)) == 1:
            # Check for persistent repetition (2x threshold) -> force strategy change
            double_threshold = self.max_same_tool_calls * 2
            if len(self._tool_call_history) >= double_threshold:
                extended = list(self._tool_call_history)[-double_threshold:]
                if len(set(extended)) == 1:
                    logger.warning(f"Persistent repetition detected: {tool_name} called {double_threshold} times")
                    return get_force_strategy_change_prompt(tool_name, double_threshold)
            return get_repetitive_behavior_prompt(tool_name, self.max_same_tool_calls)

        return None

    def _handle_context_overflow(
            self,
            messages: List[Message],
            current_tokens: int
    ) -> tuple[bool, Optional[str]]:
        """
        Handle context overflow using two-threshold hysteresis mechanism.
        
        Args:
            messages: Current message list
            current_tokens: Current token count
            
        Returns:
            (should_force_answer, optional_warning_message)
        """
        if not self.enable_context_overflow_handling:
            return False, None

        if current_tokens < self.context_soft_limit:
            return False, None

        if current_tokens >= self.context_hard_limit:
            # Hard threshold: force termination
            logger.warning(f"Context hard limit reached: {current_tokens} >= {self.context_hard_limit}")
            return True, get_force_answer_prompt()

        # Soft threshold: trigger compression
        logger.info(f"Context soft limit reached: {current_tokens} >= {self.context_soft_limit}")

        # Compress tool results if compression manager is available
        if self.tool_config.compression_manager is not None:
            self.tool_config.compression_manager.compress(messages)
            logger.debug(f"Compressed tool results, stats: {self.tool_config.compression_manager.get_stats()}")

        return False, None

    def get_builtin_tool_names(self) -> List[str]:
        """Get list of currently enabled built-in tool names."""
        tool_names = []

        if self.include_file_tools:
            tool_names.extend(["ls", "read_file", "write_file", "edit_file", "glob", "grep"])

        if self.include_execute:
            tool_names.append("execute")

        if self.include_web_search:
            tool_names.append("web_search")

        if self.include_fetch_url:
            tool_names.append("fetch_url")

        if self.include_todos:
            tool_names.extend(["write_todos", "read_todos"])

        if self.include_task:
            tool_names.append("task")

        if self.include_skills:
            tool_names.extend(["list_skills", "get_skill_info"])

        if self.include_user_input:
            tool_names.extend(["user_input", "confirm"])

        return tool_names

    def get_builtin_tool_descriptions(self) -> dict:
        """Get descriptions for built-in tools from the actual registered tool instances.

        Extracts descriptions dynamically from tool.functions (Dict[str, Function])
        rather than relying on a hardcoded static mapping.

        Returns:
            Dict mapping tool_name -> description string
        """
        descriptions = {}
        if not self.tools:
            return descriptions

        for tool in self.tools:
            if not hasattr(tool, 'functions'):
                continue
            for func_name, func_obj in tool.functions.items():
                desc = getattr(func_obj, 'description', None)
                if desc:
                    # Extract first non-empty line as short description
                    first_line = ""
                    for line in desc.split("\n"):
                        stripped = line.strip()
                        if stripped:
                            first_line = stripped
                            break
                    if len(first_line) > 80:
                        first_line = first_line[:77] + "..."
                    if first_line:
                        descriptions[func_name] = first_line

        return descriptions

    def reset_tool_history(self) -> None:
        """Reset the tool call history for repetition detection."""
        self._tool_call_history.clear()
        self._hook_step_counter = 0

    def __repr__(self) -> str:
        """Return string representation of DeepAgent."""
        builtin_tools = self.get_builtin_tool_names()
        mot = getattr(self.model, 'max_output_tokens', None) if self.model else None
        return (
            f"DeepAgent(name={self.name}, "
            f"max_output_tokens={mot}, "
            f"builtin_tools={len(builtin_tools)})"
        )


if __name__ == '__main__':
    agent = DeepAgent(
        name="TestDeepAgent",
        description="A test deep agent",
        debug=True,
    )

    print(f"Created: {agent}")
    print(f"Builtin tools: {agent.get_builtin_tool_names()}")
    print(f"enable_agentic_prompt: {agent.prompt_config.enable_agentic_prompt}")
