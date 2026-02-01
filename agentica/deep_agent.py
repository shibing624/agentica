# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: DeepAgent - Enhanced Agent with built-in tools and deep research capabilities

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
5. Deep Research System Prompt: Optimized for thorough investigation
6. Enhanced Iteration Control: HEARTBEAT-style forced iteration (Phase 3)
"""
from __future__ import annotations

from datetime import datetime
from collections import deque
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)
from dataclasses import dataclass, field

from agentica.agent import Agent
from agentica.tools.base import ModelTool, Tool, Function
from agentica.deep_tools import get_builtin_tools, BuiltinTaskTool
from agentica.model.message import Message
from agentica.utils.log import logger

# Import prompts from the centralized prompt module
from agentica.prompts.base.deep_agent import (
    get_deep_research_prompt,
    get_step_reflection_prompt,
    get_force_answer_prompt,
    get_repetitive_behavior_prompt,
    get_iteration_checkpoint_prompt,
    get_must_continue_prompt,
)


@dataclass(init=False)
class DeepAgent(Agent):
    """
    DeepAgent - Enhanced Agent with built-in tools and deep research capabilities.

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
    5. Deep Research System Prompt: Optimized for thorough investigation (optional)

    Example:
        ```python
        from agentica import DeepAgent, OpenAIChat

        # Create DeepAgent with deep research mode
        agent = DeepAgent(
            model=OpenAIChat(id="gpt-4o"),
            description="A powerful research assistant",
            enable_deep_research=True,
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

    # Deep Research Mode (only affects system prompt, not multi-round capability)
    enable_deep_research: bool = False

    # ReAct Loop Control (uses Agent's max_rounds)
    enable_step_reflection: bool = True  # Enable reflection after each step
    reflection_frequency: int = 3  # Inject reflection prompt every N steps

    # Context Management (Two-threshold hysteresis mechanism)
    context_soft_limit: int = 80000  # Soft threshold: start compression
    context_hard_limit: int = 120000  # Hard threshold: force termination
    enable_context_overflow_handling: bool = True

    # Repetitive Behavior Detection
    enable_repetition_detection: bool = True
    max_same_tool_calls: int = 3  # Max consecutive calls to same tool

    # HEARTBEAT-style Forced Iteration Control (Phase 3)
    enable_forced_iteration: bool = True  # Enable HEARTBEAT-style iteration control
    iteration_checkpoint_frequency: int = 5  # Inject iteration checkpoint every N steps

    # Tool call history for repetition detection
    _tool_call_history: deque = field(default_factory=lambda: deque(maxlen=10))

    def __init__(
            self,
            # DeepAgent specific parameters
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
            # Deep Research Mode (only affects system prompt)
            enable_deep_research: bool = False,
            # ReAct Loop Control
            enable_step_reflection: bool = True,
            reflection_frequency: int = 3,
            # Context Management
            context_soft_limit: int = 80000,
            context_hard_limit: int = 120000,
            enable_context_overflow_handling: bool = True,
            # Repetitive Behavior Detection
            enable_repetition_detection: bool = True,
            max_same_tool_calls: int = 3,
            # HEARTBEAT-style Forced Iteration Control (Phase 3)
            enable_forced_iteration: bool = True,
            iteration_checkpoint_frequency: int = 5,
            # User-provided custom tools
            tools: Optional[List[Union[ModelTool, Tool, Callable, Dict, Function]]] = None,
            # Instructions from user
            instructions: Optional[Union[str, List[str]]] = None,
            # System prompt override
            system_prompt: Optional[str] = None,
            # Other parameters passed to Agent via kwargs
            **kwargs,
    ):
        """
        Initialize DeepAgent.

        Args:
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
            enable_deep_research: Enable deep research system prompt (does not affect multi-round)
            enable_step_reflection: Enable reflection after steps
            reflection_frequency: How often to inject reflection prompts
            context_soft_limit: Token count to start compression
            context_hard_limit: Token count to force termination
            enable_context_overflow_handling: Enable context overflow handling
            enable_repetition_detection: Enable repetitive behavior detection
            max_same_tool_calls: Max consecutive calls to same tool
            enable_forced_iteration: Enable HEARTBEAT-style forced iteration control
            iteration_checkpoint_frequency: How often to inject iteration checkpoints
            tools: User-provided custom tools list
            instructions: User-provided instructions
            system_prompt: Override system prompt (if not using deep research mode)
            **kwargs: Other parameters passed to Agent (including max_rounds, enable_multi_round)
        """
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

        # Deep Research Mode (only affects system prompt)
        self.enable_deep_research = enable_deep_research

        # ReAct Loop Control
        self.enable_step_reflection = enable_step_reflection
        self.reflection_frequency = reflection_frequency

        # Context Management
        self.context_soft_limit = context_soft_limit
        self.context_hard_limit = context_hard_limit
        self.enable_context_overflow_handling = enable_context_overflow_handling

        # Repetitive Behavior Detection
        self.enable_repetition_detection = enable_repetition_detection
        self.max_same_tool_calls = max_same_tool_calls
        self._tool_call_history = deque(maxlen=10)

        # HEARTBEAT-style Forced Iteration Control (Phase 3)
        self.enable_forced_iteration = enable_forced_iteration
        self.iteration_checkpoint_frequency = iteration_checkpoint_frequency

        # Get built-in tools
        builtin_tools = get_builtin_tools(
            base_dir=work_dir,
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
        all_tools = list(builtin_tools)
        if tools:
            all_tools = self._merge_tools_with_dedup(all_tools, tools)

        custom_tool_count = len(tools) if tools else 0
        logger.debug(f"DeepAgent initialized with {len(builtin_tools)} builtin tools and {custom_tool_count} custom tools")

        # Determine system prompt
        final_system_prompt = system_prompt
        if enable_deep_research and system_prompt is None:
            # Use deep research prompt with current date
            current_date = datetime.now().strftime("%Y-%m-%d")
            final_system_prompt = get_deep_research_prompt(current_date)

        # Multi-round and agentic prompt configuration:
        # - When enable_deep_research=True, multi-round and agentic_prompt are required
        # - When enable_deep_research=False, user can choose (defaults to False for DeepAgent)
        if enable_deep_research:
            # Deep research mode requires multi-round for iterative search and validation
            if kwargs.get('enable_multi_round') is False:
                logger.warning("enable_deep_research=True requires enable_multi_round=True, forcing enable_multi_round=True")
            kwargs['enable_multi_round'] = True
            # Deep research mode also requires agentic prompt enhancement
            if kwargs.get('enable_agentic_prompt') is False:
                logger.warning("enable_deep_research=True requires enable_agentic_prompt=True, forcing enable_agentic_prompt=True")
            kwargs['enable_agentic_prompt'] = True
        # Otherwise, respect user's setting or default to False
        
        # Default max_rounds to 15 if not specified
        if 'max_rounds' not in kwargs:
            kwargs['max_rounds'] = 15

        # Call parent class init with merged tools
        super().__init__(
            tools=all_tools,
            instructions=instructions,
            system_prompt=final_system_prompt,
            **kwargs
        )

        # Set parent agent reference for task tool
        self._setup_task_tool()

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

    def _estimate_context_tokens(self, messages: List[Message]) -> int:
        """
        Estimate the current context token count.
        
        Uses character count / 4 as a rough estimate if tiktoken is not available.
        """
        try:
            from agentica.utils.tokens import count_tokens
            model_id = getattr(self.model, 'id', 'gpt-4o') if self.model else 'gpt-4o'
            return count_tokens(messages, self.model.functions if hasattr(self.model, 'functions') else None, model_id)
        except Exception:
            # Fallback: estimate tokens as characters / 4
            total_chars = sum(len(str(m.content or "")) for m in messages)
            return total_chars // 4

    def _check_repetitive_behavior(self, tool_name: str) -> Optional[str]:
        """
        Check for repetitive tool call patterns.
        
        Returns a warning message if repetitive behavior is detected.
        """
        if not self.enable_repetition_detection:
            return None

        self._tool_call_history.append(tool_name)

        if len(self._tool_call_history) < self.max_same_tool_calls:
            return None

        # Check if the last N calls are all the same tool
        recent_calls = list(self._tool_call_history)[-self.max_same_tool_calls:]
        if len(set(recent_calls)) == 1:
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
        if self.compression_manager is not None:
            self.compression_manager.compress(messages)
            logger.debug(f"Compressed tool results, stats: {self.compression_manager.get_stats()}")

        return False, None

    def _inject_reflection_prompt(self, messages: List[Message], step: int) -> None:
        """Inject reflection prompt at appropriate intervals."""
        if not self.enable_step_reflection:
            return

        if step > 0 and step % self.reflection_frequency == 0:
            messages.append(Message(
                role="system",
                content=get_step_reflection_prompt()
            ))
            logger.debug(f"Injected reflection prompt at step {step}")

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

    def reset_tool_history(self) -> None:
        """Reset the tool call history for repetition detection."""
        self._tool_call_history.clear()

    # =============================================================================
    # Override Hook Methods from Agent
    # =============================================================================

    def _on_pre_step(
            self,
            step: int,
            messages: List[Message]
    ) -> tuple[bool, Optional[str]]:
        """
        Pre-step hook: Handle context overflow using two-threshold mechanism.
        
        Args:
            step: Current step number
            messages: Current message list
            
        Returns:
            (should_force_answer, optional_warning_message)
        """
        if not self.enable_context_overflow_handling:
            return False, None

        current_tokens = self._estimate_context_tokens(messages)
        return self._handle_context_overflow(messages, current_tokens)

    def _on_tool_call(self, tool_name: str, step: int) -> Optional[str]:
        """
        Tool call hook: Check for repetitive behavior patterns.
        
        Args:
            tool_name: Name of the tool being called
            step: Current step number
            
        Returns:
            Optional warning message if repetitive behavior detected
        """
        return self._check_repetitive_behavior(tool_name)

    def _on_post_step(self, step: int, messages: List[Message]) -> None:
        """
        Post-step hook: Inject reflection and iteration checkpoint prompts.

        Enhanced in Phase 3 with HEARTBEAT-style iteration control:
        - Inject reflection prompts at regular intervals
        - Inject iteration checkpoint prompts to ensure task completion

        Args:
            step: Current step number
            messages: Current message list (modified in place)
        """
        # Original reflection prompt injection
        self._inject_reflection_prompt(messages, step)

        # HEARTBEAT-style iteration checkpoint (Phase 3 Enhancement)
        if self.enable_forced_iteration and step > 0:
            if step % self.iteration_checkpoint_frequency == 0:
                checkpoint_prompt = get_iteration_checkpoint_prompt(step)
                messages.append(Message(
                    role="system",
                    content=checkpoint_prompt
                ))
                logger.debug(f"Injected iteration checkpoint at step {step}")

    def _get_iteration_reminder(self, step: int) -> str:
        """Generate an iteration reminder message for the current step.

        This is used by the PromptBuilder integration to provide
        HEARTBEAT-style reminders during multi-round execution.

        Args:
            step: Current step number

        Returns:
            Iteration reminder message
        """
        return f"""
Step {step} checkpoint:
- Have you fully solved the problem?
- Are there any remaining tasks in the task list?
- Did you verify your changes?

If not complete, continue working. Do NOT end your turn prematurely.
"""

    def __repr__(self) -> str:
        """Return string representation of DeepAgent."""
        builtin_tools = self.get_builtin_tool_names()
        return (
            f"DeepAgent(name={self.name}, "
            f"deep_research={self.enable_deep_research}, "
            f"max_rounds={self.max_rounds}, "
            f"builtin_tools={len(builtin_tools)})"
        )


# =============================================================================
# DeepResearchAgent - Specialized for deep research tasks
# =============================================================================

class DeepResearchAgent(DeepAgent):
    """
    DeepResearchAgent - Specialized agent for deep research tasks.
    
    This is a convenience class that pre-configures DeepAgent for deep research:
    - Enables deep research mode by default
    - Increases max_rounds for thorough investigation
    - Enables all relevant tools
    
    Example:
        ```python
        from agentica import DeepResearchAgent, OpenAIChat
        
        agent = DeepResearchAgent(
            model=OpenAIChat(id="gpt-4o"),
        )
        
        response = agent.run("Research the impact of AI on healthcare in 2024")
        print(response.content)
        ```
    """

    def __init__(
            self,
            max_rounds: int = 20,
            **kwargs,
    ):
        """
        Initialize DeepResearchAgent.
        
        Args:
            max_rounds: Maximum research rounds (default: 20)
            **kwargs: Other parameters passed to DeepAgent
        """
        # Set defaults for deep research
        kwargs.setdefault('enable_deep_research', True)
        kwargs.setdefault('include_todos', True)
        kwargs.setdefault('include_web_search', True)
        kwargs.setdefault('include_fetch_url', True)
        kwargs.setdefault('include_execute', True)
        kwargs.setdefault('enable_step_reflection', True)
        kwargs.setdefault('enable_context_overflow_handling', True)
        kwargs.setdefault('enable_repetition_detection', True)
        kwargs.setdefault('compress_tool_results', True)

        super().__init__(max_rounds=max_rounds, **kwargs)


if __name__ == '__main__':
    # Simple test
    from agentica import OpenAIChat

    # Create DeepAgent
    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o-mini"),
        name="TestDeepAgent",
        description="A test deep agent",
        enable_deep_research=True,
        max_rounds=15,
        debug_mode=True,
    )

    print(f"Created: {agent}")
    print(f"Builtin tools: {agent.get_builtin_tool_names()}")
    print(f"Max rounds: {agent.max_rounds}")

    # Test DeepResearchAgent
    research_agent = DeepResearchAgent(
        model=OpenAIChat(id="gpt-4o-mini"),
        name="TestResearchAgent",
        max_rounds=20,
        debug_mode=True,
        include_todos=False,
    )
    print(f"\nCreated: {research_agent}")
    print(f"Max rounds: {research_agent.max_rounds}")
    print(f"Tools: {research_agent.get_builtin_tool_names()}, {research_agent.tools}")
