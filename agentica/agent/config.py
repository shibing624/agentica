# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Config dataclasses for Agent V2 architecture.

Provides layered configuration:
- PromptConfig: Prompt engineering details
- ToolConfig: Tool calling behavior
- WorkspaceMemoryConfig: Workspace memory settings
- TeamConfig: Team collaboration settings
"""

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)


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

    # Output formatting
    output_language: Optional[str] = None
    markdown: bool = False


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
    # History tools
    read_chat_history: bool = False
    read_tool_call_history: bool = False
    # References
    add_references: bool = False
    # Compression
    compress_tool_results: bool = False
    compression_manager: Optional[Any] = None


@dataclass
class WorkspaceMemoryConfig:
    """Workspace memory loading configuration."""
    load_workspace_context: bool = True
    load_workspace_memory: bool = True
    memory_days: int = 2


@dataclass
class TeamConfig:
    """Team collaboration configuration."""
    respond_directly: bool = False
    add_transfer_instructions: bool = True
    team_response_separator: str = "\n"
