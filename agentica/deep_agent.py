# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: DeepAgent - Agent subclass with built-in tools

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
"""
from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)
from dataclasses import dataclass

from agentica.agent import Agent
from agentica.tools.base import ModelTool, Tool, Function
from agentica.deep_tools import get_builtin_tools, BuiltinTaskTool
from agentica.utils.log import logger


@dataclass(init=False)
class DeepAgent(Agent):
    """
    DeepAgent - Enhanced Agent with built-in tools.

    DeepAgent inherits from Agent and automatically adds the following built-in tools:
    - File system tools: ls, read_file, write_file, edit_file, glob, grep
    - Code execution tool: execute
    - Web tools: web_search, fetch_url
    - Task management tools: write_todos, read_todos
    - Subagent tool: task

    Users can control which tools to include via parameters and add custom tools.

    Example:
        ```python
        from agentica import DeepAgent, OpenAIChat

        # Create DeepAgent with all built-in tools
        agent = DeepAgent(
            model=OpenAIChat(id="gpt-4o"),
            description="A powerful coding assistant",
        )

        # Run the agent
        response = agent.run("List all Python files in the current directory")
        print(response.content)
        ```

    Example with custom configuration:
        ```python
        from agentica import DeepAgent, OpenAIChat
        from agentica.tools.calculator_tool import CalculatorTool

        # Create DeepAgent with some tools disabled and custom tools added
        agent = DeepAgent(
            model=OpenAIChat(id="gpt-4o"),
            include_execute=False,  # Disable code execution
            include_web_search=False,  # Disable web search
            tools=[CalculatorTool()],  # Add calculator tool
        )
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
            # User-provided custom tools
            tools: Optional[List[Union[ModelTool, Tool, Callable, Dict, Function]]] = None,
            # Instructions from user
            instructions: Optional[Union[str, List[str]]] = None,
            # Other parameters passed to Agent via kwargs
            **kwargs,
    ):
        """
        Initialize DeepAgent.

        Args:
            work_dir: Working directory for file operations, defaults to current directory
            include_file_tools: Include file tools (ls, read_file, write_file, edit_file, glob, grep)
            include_execute: Include code execution tool (execute)
            include_web_search: Include web search tool (web_search)
            include_fetch_url: Include URL fetching tool (fetch_url)
            include_todos: Include task management tools (write_todos, read_todos)
            include_task: Include subagent task tool (task)
            tools: User-provided custom tools list
            instructions: User-provided instructions (tool system prompts will be auto-appended)
            **kwargs: Other parameters passed to Agent
        """
        # Store DeepAgent specific configuration
        self.work_dir = work_dir
        self.include_file_tools = include_file_tools
        self.include_execute = include_execute
        self.include_web_search = include_web_search
        self.include_fetch_url = include_fetch_url
        self.include_todos = include_todos
        self.include_task = include_task

        # Get built-in tools
        builtin_tools = get_builtin_tools(
            base_dir=work_dir,
            include_file_tools=include_file_tools,
            include_execute=include_execute,
            include_web_search=include_web_search,
            include_fetch_url=include_fetch_url,
            include_todos=include_todos,
            include_task=include_task,
        )

        # Merge user-provided tools with built-in tools, handling duplicates
        all_tools = list(builtin_tools)
        if tools:
            all_tools = self._merge_tools_with_dedup(all_tools, tools)

        # Collect system prompts from all tools
        tool_system_prompts = self._collect_tool_system_prompts(all_tools)

        # Merge user instructions with tool system prompts
        merged_instructions = self._merge_instructions(instructions, tool_system_prompts)

        custom_tool_count = len(tools) if tools else 0
        logger.debug(f"DeepAgent initialized with {len(builtin_tools)} builtin tools and {custom_tool_count} custom tools")
        if tool_system_prompts:
            logger.debug(f"Injected {len(tool_system_prompts)} tool system prompts into instructions")

        # Call parent class init with merged tools and instructions
        super().__init__(tools=all_tools, instructions=merged_instructions, **kwargs)

        # Set parent agent reference for task tool (so it can access model)
        self._setup_task_tool()

    def _collect_tool_system_prompts(self, tools: List[Any]) -> List[str]:
        """
        Collect system prompts from all tools that provide them.

        Args:
            tools: List of tools to collect prompts from

        Returns:
            List of system prompt strings from tools
        """
        prompts = []
        for tool in tools:
            if isinstance(tool, Tool) and hasattr(tool, 'get_system_prompt'):
                prompt = tool.get_system_prompt()
                if prompt:
                    prompts.append(prompt)
        return prompts

    def _merge_tools_with_dedup(
            self,
            builtin_tools: List[Any],
            user_tools: List[Any],
    ) -> List[Any]:
        """
        Merge user tools with builtin tools, removing duplicate functions.
        
        When both builtin and user tools have similar functions, prefer builtin:
        - ls (builtin) over list_skill_files (SkillTool)
        - read_file (builtin) over read_skill_file (SkillTool)
        
        Args:
            builtin_tools: List of builtin tools
            user_tools: List of user-provided tools
            
        Returns:
            Merged list of tools with duplicates removed from user tools
        """
        # Get function names from builtin tools
        builtin_func_names = set()
        for tool in builtin_tools:
            if isinstance(tool, Tool) and hasattr(tool, 'functions'):
                for func_name in tool.functions.keys():
                    builtin_func_names.add(func_name)
        
        # Define function name mappings (user_func -> builtin_func)
        # If user tool has these functions and builtin has the mapped function, remove from user
        func_mappings = {
            'list_skill_files': 'ls',
            'read_skill_file': 'read_file',
        }
        
        # Check which functions should be removed from user tools
        funcs_to_remove = set()
        for user_func, builtin_func in func_mappings.items():
            if builtin_func in builtin_func_names:
                funcs_to_remove.add(user_func)
        
        # Process user tools and remove duplicate functions
        result = list(builtin_tools)
        for tool in user_tools:
            if isinstance(tool, Tool) and hasattr(tool, 'functions') and funcs_to_remove:
                # Remove duplicate functions from this tool
                for func_name in list(tool.functions.keys()):
                    if func_name in funcs_to_remove:
                        del tool.functions[func_name]
                        logger.debug(f"Removed duplicate function '{func_name}' from {tool.name}, using builtin instead")
            result.append(tool)
        
        return result

    def _merge_instructions(
            self,
            user_instructions: Optional[Union[str, List[str]]],
            tool_prompts: List[str],
    ) -> Optional[List[str]]:
        """
        Merge user instructions with tool system prompts.

        Args:
            user_instructions: User-provided instructions
            tool_prompts: System prompts from tools

        Returns:
            Merged list of instructions, or None if both are empty
        """
        result = []

        # Add user instructions first
        if user_instructions:
            if isinstance(user_instructions, str):
                result.append(user_instructions)
            else:
                result.extend(user_instructions)

        # Add tool system prompts
        result.extend(tool_prompts)

        return result if result else None

    def _setup_task_tool(self) -> None:
        """Set up the task tool with parent agent reference."""
        if not self.include_task:
            return

        # Find and configure the task tool
        for tool in self.tools or []:
            if isinstance(tool, BuiltinTaskTool):
                tool.set_parent_agent(self)
                break

    def get_builtin_tool_names(self) -> List[str]:
        """
        Get list of currently enabled built-in tool names.

        Returns:
            List of built-in tool names
        """
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

        return tool_names

    def __repr__(self) -> str:
        """Return string representation of DeepAgent."""
        builtin_tools = self.get_builtin_tool_names()
        return f"DeepAgent(name={self.name}, builtin_tools={builtin_tools})"


if __name__ == '__main__':
    # Simple test
    from agentica import OpenAIChat

    # Create DeepAgent
    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o-mini"),
        name="TestDeepAgent",
        description="A test deep agent",
        debug_mode=True,
    )

    print(f"Created: {agent}")
    print(f"Builtin tools: {agent.get_builtin_tool_names()}")
