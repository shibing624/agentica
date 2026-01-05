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


# =============================================================================
# Deep Research System Prompt
# =============================================================================

DEEP_RESEARCH_SYSTEM_PROMPT = """**任务目标:** 针对以下问题，请进行深入、详尽的调查与分析，并提供一个经过充分验证的、全面的答案。

**核心要求:** 在整个过程中，你需要**最大化地、策略性地使用你可用的工具** (例如：搜索引擎、代码执行器等)，并**清晰地展示你的思考、决策和验证过程**。不仅仅是给出最终答案，更要展现获得答案的严谨路径。

**行为指令:**

1. **启动调查 (Initiate Investigation):** 首先分析问题，识别关键信息点和潜在的约束条件。初步规划你需要哪些信息（使用todo工具制定你的调查计划），并使用工具（如搜索）开始收集。

2. **迭代式信息收集与反思 (Iterative Information Gathering & Reflection):**
   * **处理搜索失败:** 如果首次搜索（或后续搜索）未能找到相关结果或结果不佳，**必须**明确说明（例如："初步搜索未能找到关于'XXX'的直接信息，尝试调整关键词为'YYY'再次搜索。"），并调整策略（修改关键词、尝试不同搜索引擎或数据库、扩大搜索范围如增加top K结果数量并说明"之前的Top K结果不足，现在尝试查看更多页面获取信息"）。
   * **评估信息充分性:** 在获取部分信息后，**必须**停下来评估这些信息是否足以回答原始问题的所有方面（例如："已找到关于'AAA'的信息，但问题中提到的'BBB'方面尚未覆盖，需要继续搜索'BBB'相关内容。"）。
   * **追求信息深度:** 即使已有一些信息，如果觉得不够深入或全面，**必须**说明需要更多信息来源并继续搜索（例如："现有信息提供了基础，但为确保全面性，需要查找更多权威来源或不同角度的报道来深化理解。"）。
   * **信源考量:** 在引用信息时，**主动思考并简述**信息来源的可靠性或背景（例如："这个信息来自'XYZ网站'，该网站通常被认为是[领域]的权威来源/是一个用户生成内容平台，信息需要进一步核实。"）。

3. **多源/多工具交叉验证 (Multi-Source/Multi-Tool Cross-Validation):**
   * **主动验证:** **不要**满足于单一来源的信息。**必须**尝试使用不同工具或搜索不同来源来交叉验证关键信息点（例如："为确认'CCC'数据的准确性，让我们尝试用另一个搜索引擎或查询官方数据库进行验证。" 或 "让我们用代码计算器/Python工具来验证刚才推理中得到的数值/字符串处理结果。"）。
   * **工具切换:** 如果一个工具不适用或效果不佳，**明确说明**并尝试使用其他可用工具（例如："搜索引擎未能提供结构化数据，尝试使用代码执行器分析或提取网页内容。"）。

4. **约束条件检查 (Constraint Checklist):** 在整合信息和形成答案之前，**必须**明确回顾原始问题的所有约束条件，并逐一确认现有信息是否完全满足这些条件（例如："让我们检查一下：问题要求时间在'2023年后'，地点为'欧洲'，并且涉及'特定技术'。目前收集到的信息 A 满足时间，信息 B 满足地点，信息 C 涉及该技术... 所有约束均已覆盖。"）。

5. **计算与操作验证 (Calculation & Operation Verification):** 如果在你的思考链（Chain of Thought）中进行了任何计算、数据提取、字符串操作或其他逻辑推导，**必须**在最终确定前使用工具（如代码执行器）进行验证，并展示验证步骤（例如："推理得出总和为 X，现在使用代码验证：`print(a+b)` ... 结果确认是 X。"）。

6. **清晰的叙述:** 在每一步工具调用前后，用简短的语句**清晰说明你为什么要调用这个工具、期望获得什么信息、以及调用后的结果和下一步计划**。这包括上述所有反思和验证的插入语。

**制定计划:** 在开始收集信息之前，请先分析问题，并使用todo工具制定你的行动计划。

**格式要求:** 每次执行工具调用后，分析返回的信息，如果已收集到足够的信息，可以直接回答用户请求，否则继续执行工具调用。在整个过程中，请始终明确你的目标是回答用户请求。当通过充分的工具调用获取并验证了所有必要信息后，在 <answer>...</answer> 中输出一个详尽全面的报告。

**引用要求:** 报告中引用搜索信息时，必须使用Markdown链接格式标注来源，格式为：`[来源名称](URL)`。例如：`根据[OpenAI官方博客](https://openai.com/blog/xxx)的报道...`。禁止使用LaTeX的cite格式或占位符（如\\cite{{}}），必须使用实际的URL链接。

**报告要求:** 请确保深度、全面地回答任务中的所有子问题，采用符合用户提问的语言风格和结构，使用逻辑清晰、论证充分的长段落，禁止碎片化罗列。论证需要基于具体的数字和最新的权威引用，进行必要的关联对比分析、利弊权衡、风险讨论，并确保事实准确、术语清晰，避免模糊和绝对化措辞。

**当前日期:** {current_date}"""


# =============================================================================
# Reflection Prompts
# =============================================================================

STEP_REFLECTION_PROMPT = """
## 步骤反思

请在继续之前进行反思：

1. **信息评估**: 我获得了什么新信息？这些信息是否足够回答用户问题？
2. **策略检查**: 当前搜索策略是否有效？是否需要更换关键词或搜索角度？
3. **进度判断**: 是否应该继续搜索，还是信息已经足够可以给出答案？

重要：避免重复相同的搜索。如果连续两次搜索结果相似，应调整策略或停止。
"""

FORCE_ANSWER_PROMPT = """
⚠️ **上下文长度已达到限制**

你现在必须：
1. **停止所有工具调用** - 不要再进行任何搜索或工具操作
2. **基于已收集的信息给出最终答案** - 整合目前已有的所有信息
3. **如果信息不足，说明已知内容并指出缺失部分**

请立即在 <answer>...</answer> 中给出你的最终回答。
"""

REPETITIVE_BEHAVIOR_PROMPT = """
⚠️ **检测到重复行为模式**

你已经连续 {count} 次调用相同的工具 `{tool_name}`。这可能意味着：
1. 搜索策略需要调整 - 尝试不同的关键词或工具
2. 已有信息可能已经足够 - 考虑是否可以给出答案
3. 问题可能需要分解 - 尝试将问题拆分为更小的子问题

请调整你的策略，或者如果信息已经足够，直接给出最终答案。
"""


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
            final_system_prompt = DEEP_RESEARCH_SYSTEM_PROMPT.format(current_date=current_date)

        # DeepAgent always enables multi-round by default
        # When enable_deep_research=True, multi-round is required for iterative research
        if 'enable_multi_round' not in kwargs:
            kwargs['enable_multi_round'] = True
        elif enable_deep_research and kwargs.get('enable_multi_round') is False:
            # Warn user if they try to disable multi-round with deep research enabled
            logger.warning("enable_deep_research=True requires enable_multi_round=True, forcing enable_multi_round=True")
            kwargs['enable_multi_round'] = True
        
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
            return REPETITIVE_BEHAVIOR_PROMPT.format(
                count=self.max_same_tool_calls,
                tool_name=tool_name
            )

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
            return True, FORCE_ANSWER_PROMPT

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
                content=STEP_REFLECTION_PROMPT
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
        Post-step hook: Inject reflection prompts at appropriate intervals.
        
        Args:
            step: Current step number
            messages: Current message list (modified in place)
        """
        self._inject_reflection_prompt(messages, step)

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
