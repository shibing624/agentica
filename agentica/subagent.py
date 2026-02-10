# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Subagent system for managing ephemeral agent tasks

This module implements a subagent system that allows main agents to:
- Spawn isolated subagents for complex tasks
- Track subagent lifecycle and results
- Support different subagent types with varying tool permissions
- Enable parallel execution of multiple subagents

Based on the subagent design pattern from modern AI coding assistants.
"""

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)
from datetime import datetime

from agentica.utils.log import logger

if TYPE_CHECKING:
    pass  # Reserved for future type imports


class SubagentType(str, Enum):
    """Types of subagents with different capabilities."""
    
    # General-purpose agent with full tool access
    GENERAL = "general"
    
    # Explore agent: read-only, specialized for codebase exploration
    EXPLORE = "explore"
    
    # Research agent: web search and document analysis
    RESEARCH = "research"
    
    # Code agent: code generation and execution
    CODE = "code"


@dataclass
class SubagentConfig:
    """Configuration for a subagent type."""
    
    # Subagent type identifier
    type: SubagentType
    
    # Human-readable name
    name: str
    
    # Description of the subagent's capabilities
    description: str
    
    # System prompt for this subagent type
    system_prompt: str
    
    # Allowed tools (None means all tools from parent, empty list means no tools)
    allowed_tools: Optional[List[str]] = None
    
    # Denied tools (takes precedence over allowed_tools)
    denied_tools: Optional[List[str]] = None
    
    # Maximum iterations for this subagent
    max_iterations: int = 10
    
    # Whether this subagent can spawn its own subagents
    can_spawn_subagents: bool = False


@dataclass
class SubagentRun:
    """Represents a single subagent execution."""
    
    # Unique run identifier
    run_id: str
    
    # Subagent type
    subagent_type: SubagentType
    
    # Session key for the subagent (isolated from parent)
    session_key: str
    
    # Parent session key (who spawned this subagent)
    parent_session_key: str
    
    # Task label/description
    task_label: str
    
    # Full task description
    task_description: str
    
    # Timestamp when started
    started_at: datetime
    
    # Current status
    status: Literal["pending", "running", "completed", "error", "cancelled"] = "pending"
    
    # Timestamp when ended (if finished)
    ended_at: Optional[datetime] = None
    
    # Result from the subagent
    result: Optional[str] = None
    
    # Error message if failed
    error: Optional[str] = None
    
    # Token usage statistics
    token_usage: Optional[Dict[str, int]] = None


class SubagentRegistry:
    """
    Registry for tracking and managing subagent runs.
    
    This is a singleton-like class that tracks all subagent executions
    across the application lifetime.
    """
    
    _instance: Optional["SubagentRegistry"] = None
    
    def __new__(cls) -> "SubagentRegistry":
        """Ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._runs: Dict[str, SubagentRun] = {}
            cls._instance._listeners: List[Callable[[SubagentRun], None]] = []
        return cls._instance
    
    def register(self, run: SubagentRun) -> None:
        """Register a new subagent run."""
        self._runs[run.run_id] = run
        logger.debug(f"Registered subagent run: {run.run_id} ({run.subagent_type.value})")
    
    def get(self, run_id: str) -> Optional[SubagentRun]:
        """Get a subagent run by ID."""
        return self._runs.get(run_id)
    
    def get_for_session(self, session_key: str) -> List[SubagentRun]:
        """Get all subagent runs spawned by a session."""
        return [
            run for run in self._runs.values()
            if run.parent_session_key == session_key
        ]
    
    def get_active(self) -> List[SubagentRun]:
        """Get all currently running subagents."""
        return [
            run for run in self._runs.values()
            if run.status in ("pending", "running")
        ]
    
    def update_status(
        self,
        run_id: str,
        status: Literal["running", "completed", "error", "cancelled"],
        result: Optional[str] = None,
        error: Optional[str] = None,
        token_usage: Optional[Dict[str, int]] = None,
    ) -> None:
        """Update the status of a subagent run."""
        run = self._runs.get(run_id)
        if run is None:
            logger.warning(f"Cannot update status: subagent run {run_id} not found")
            return
        
        run.status = status
        if status in ("completed", "error", "cancelled"):
            run.ended_at = datetime.now()
        if result is not None:
            run.result = result
        if error is not None:
            run.error = error
        if token_usage is not None:
            run.token_usage = token_usage
        
        # Notify listeners
        for listener in self._listeners:
            try:
                listener(run)
            except Exception as e:
                logger.error(f"Subagent listener error: {e}")
    
    def on_complete(self, callback: Callable[[SubagentRun], None]) -> None:
        """Register a callback for when a subagent completes."""
        self._listeners.append(callback)
    
    def cleanup_completed(self, max_age_seconds: int = 3600) -> int:
        """Remove completed/cancelled runs older than max_age_seconds."""
        now = datetime.now()
        to_remove = []
        
        for run_id, run in self._runs.items():
            if run.status in ("completed", "error", "cancelled") and run.ended_at:
                age = (now - run.ended_at).total_seconds()
                if age > max_age_seconds:
                    to_remove.append(run_id)
        
        for run_id in to_remove:
            del self._runs[run_id]
        
        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} old subagent runs")
        
        return len(to_remove)


def generate_subagent_session_key(parent_session_id: str, subagent_type: SubagentType) -> str:
    """
    Generate a unique session key for a subagent.
    
    Format: subagent:{parent_session_id}:{type}:{uuid}
    
    This format allows:
    - Identifying the parent session
    - Knowing the subagent type
    - Having a unique identifier
    """
    unique_id = str(uuid.uuid4())[:8]
    return f"subagent:{parent_session_id}:{subagent_type.value}:{unique_id}"


def is_subagent_session(session_id: Optional[str]) -> bool:
    """Check if a session ID belongs to a subagent."""
    if not session_id:
        return False
    return session_id.startswith("subagent:")


def parse_subagent_session(session_id: str) -> Optional[Dict[str, str]]:
    """
    Parse a subagent session ID.
    
    Returns:
        Dict with keys: parent_session_id, subagent_type, unique_id
        Or None if not a valid subagent session ID
    """
    if not is_subagent_session(session_id):
        return None
    
    parts = session_id.split(":")
    if len(parts) < 4:
        return None
    
    return {
        "parent_session_id": parts[1],
        "subagent_type": parts[2],
        "unique_id": parts[3],
    }


# ============== Default Subagent Configurations ==============

# Explore agent: specialized for codebase exploration (read-only)
EXPLORE_SUBAGENT_CONFIG = SubagentConfig(
    type=SubagentType.EXPLORE,
    name="Explore Agent",
    description="""Fast agent specialized for exploring codebases and searching for information.
Use this agent when you need to:
- Search for files using glob patterns
- Search file contents with regex
- Read and analyze source code
- Understand project structure""",
    system_prompt="""You are a file search specialist. You excel at thoroughly navigating and exploring codebases.

Your strengths:
- Rapidly finding files using glob patterns
- Searching code and text with powerful regex patterns
- Reading and analyzing file contents

Guidelines:
- Use glob for broad file pattern matching
- Use grep for searching file contents with regex
- Use read_file when you know the specific file path you need to read
- Use ls to list directory contents and understand project structure
- Adapt your search approach based on the thoroughness level specified by the caller
- Return file paths as absolute paths in your final response
- For clear communication, avoid using emojis
- Do NOT create or modify any files - you are read-only
- Do NOT run commands that modify the user's system state

Complete the user's search request efficiently and report your findings clearly.""",
    allowed_tools=["ls", "read_file", "glob", "grep"],  # Read-only tools
    denied_tools=["write_file", "edit_file", "execute", "task"],  # No write/execute/spawn
    max_iterations=15,
    can_spawn_subagents=False,
)


# General-purpose agent: full capabilities
GENERAL_SUBAGENT_CONFIG = SubagentConfig(
    type=SubagentType.GENERAL,
    name="General-Purpose Agent",
    description="""General-purpose agent for handling complex, multi-step tasks.
Use this agent for:
- Research and analysis tasks
- Multi-step file operations
- Complex reasoning tasks
- Any task that benefits from isolated context""",
    system_prompt="""You are a helpful assistant that completes tasks autonomously.

Guidelines:
1. **Stay focused** - Complete your assigned task, nothing else
2. **Be thorough** - Your final message is your deliverable
3. **Complete the task** - Don't ask for clarification, make reasonable assumptions
4. **Report clearly** - Provide a clear summary of what you accomplished

Focus on the specific task given to you and provide a clear, concise result.
Use the available tools to accomplish your task efficiently.""",
    allowed_tools=None,  # All tools from parent
    denied_tools=["task"],  # Cannot spawn nested subagents by default
    max_iterations=20,
    can_spawn_subagents=False,
)


# Research agent: web search and analysis
RESEARCH_SUBAGENT_CONFIG = SubagentConfig(
    type=SubagentType.RESEARCH,
    name="Research Agent",
    description="""Research agent specialized for web search and document analysis.
Use this agent for:
- Searching the web for information
- Fetching and analyzing web pages
- Synthesizing research findings""",
    system_prompt="""You are a research specialist that excels at finding and analyzing information.

Guidelines:
1. Use web_search to find relevant information on the web
2. Use fetch_url to read web page contents
3. Synthesize your findings into a clear, well-organized summary
4. Cite your sources when providing information
5. Be objective and fact-based in your analysis

Complete your research task and provide a comprehensive summary of your findings.""",
    allowed_tools=["web_search", "fetch_url", "read_file", "ls", "glob", "grep"],
    denied_tools=["write_file", "edit_file", "execute", "task"],
    max_iterations=15,
    can_spawn_subagents=False,
)


# Code agent: code generation and execution
CODE_SUBAGENT_CONFIG = SubagentConfig(
    type=SubagentType.CODE,
    name="Code Agent",
    description="""Code agent specialized for code generation and execution.
Use this agent for:
- Writing and executing code
- Running tests and commands
- Code analysis and debugging""",
    system_prompt="""You are a code specialist that excels at writing and executing code.

Guidelines:
1. Write clean, well-documented code
2. Test your code before reporting results
3. Handle errors gracefully and report them clearly
4. Follow best practices for the programming language being used
5. Provide clear explanations of what your code does

Complete your coding task and provide a summary of the results.""",
    allowed_tools=["read_file", "write_file", "edit_file", "execute", "ls", "glob", "grep"],
    denied_tools=["task"],  # Cannot spawn nested subagents
    max_iterations=20,
    can_spawn_subagents=False,
)


# Registry of all default subagent configurations
DEFAULT_SUBAGENT_CONFIGS: Dict[SubagentType, SubagentConfig] = {
    SubagentType.EXPLORE: EXPLORE_SUBAGENT_CONFIG,
    SubagentType.GENERAL: GENERAL_SUBAGENT_CONFIG,
    SubagentType.RESEARCH: RESEARCH_SUBAGENT_CONFIG,
    SubagentType.CODE: CODE_SUBAGENT_CONFIG,
}

# Custom subagent configurations (user-defined, keyed by string name)
_CUSTOM_SUBAGENT_CONFIGS: Dict[str, SubagentConfig] = {}


def register_custom_subagent(
    name: str,
    description: str,
    system_prompt: str,
    allowed_tools: Optional[List[str]] = None,
    denied_tools: Optional[List[str]] = None,
    max_iterations: int = 15,
) -> SubagentConfig:
    """
    Register a custom subagent type.
    
    This allows users to define their own subagent types without modifying code.
    Custom subagents are accessible by their name string (case-insensitive).
    
    Args:
        name: Unique name for the subagent (e.g., "code-reviewer", "data-analyst")
        description: Description of what this subagent does
        system_prompt: System prompt for the subagent
        allowed_tools: List of allowed tool names (None = inherit from parent)
        denied_tools: List of denied tool names
        max_iterations: Maximum iterations for this subagent
        
    Returns:
        The created SubagentConfig
        
    Example:
        >>> register_custom_subagent(
        ...     name="code-reviewer",
        ...     description="Reviews code for quality and bugs",
        ...     system_prompt="You are a code review expert...",
        ...     allowed_tools=["read_file", "ls", "glob", "grep"],
        ...     max_iterations=10,
        ... )
    """
    config = SubagentConfig(
        type=SubagentType.GENERAL,  # Custom subagents use GENERAL as base type
        name=name,
        description=description,
        system_prompt=system_prompt,
        allowed_tools=allowed_tools,
        denied_tools=denied_tools or ["task"],  # Prevent nesting by default
        max_iterations=max_iterations,
        can_spawn_subagents=False,
    )
    _CUSTOM_SUBAGENT_CONFIGS[name.lower()] = config
    logger.info(f"Registered custom subagent: {name}")
    return config


def unregister_custom_subagent(name: str) -> bool:
    """
    Unregister a custom subagent type.
    
    Args:
        name: Name of the subagent to unregister
        
    Returns:
        True if found and removed, False otherwise
    """
    key = name.lower()
    if key in _CUSTOM_SUBAGENT_CONFIGS:
        del _CUSTOM_SUBAGENT_CONFIGS[key]
        logger.info(f"Unregistered custom subagent: {name}")
        return True
    return False


def get_subagent_config(subagent_type: Union[str, SubagentType]) -> Optional[SubagentConfig]:
    """
    Get the configuration for a subagent type.
    
    Lookup order:
    1. Custom subagent configs (by name string)
    2. Default subagent configs (by SubagentType enum)
    3. Aliases (e.g., "general-purpose" -> GENERAL)
    """
    if isinstance(subagent_type, str):
        # First check custom configs (case-insensitive)
        custom_config = _CUSTOM_SUBAGENT_CONFIGS.get(subagent_type.lower())
        if custom_config is not None:
            return custom_config
        
        # Then try to parse as SubagentType enum
        try:
            subagent_type = SubagentType(subagent_type)
        except ValueError:
            # Try mapping common aliases
            aliases = {
                "general-purpose": SubagentType.GENERAL,
                "explorer": SubagentType.EXPLORE,
                "researcher": SubagentType.RESEARCH,
                "coder": SubagentType.CODE,
            }
            subagent_type = aliases.get(subagent_type.lower())
            if subagent_type is None:
                return None
    
    return DEFAULT_SUBAGENT_CONFIGS.get(subagent_type)


def get_available_subagent_types() -> List[Dict[str, str]]:
    """
    Get a list of available subagent types with their descriptions.
    
    Returns both default and custom subagent types.
    """
    result = []
    
    # Add default configs
    for config in DEFAULT_SUBAGENT_CONFIGS.values():
        result.append({
            "type": config.type.value,
            "name": config.name,
            "description": config.description,
            "is_custom": False,
        })
    
    # Add custom configs
    for name, config in _CUSTOM_SUBAGENT_CONFIGS.items():
        result.append({
            "type": name,  # Custom types use their name as type
            "name": config.name,
            "description": config.description,
            "is_custom": True,
        })
    
    return result


def get_custom_subagent_configs() -> Dict[str, SubagentConfig]:
    """Get all registered custom subagent configurations."""
    return _CUSTOM_SUBAGENT_CONFIGS.copy()
