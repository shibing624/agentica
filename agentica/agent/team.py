# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Team collaboration and transfer methods for Agent

This module contains methods for team management, task transfer,
and tool retrieval.
"""

import json
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    TYPE_CHECKING,
)

from agentica.utils.log import logger
from agentica.tools.base import ModelTool, Tool, Function

if TYPE_CHECKING:
    from agentica.agent.base import Agent


class TeamMixin:
    """Mixin class containing team and tool methods for Agent."""

    def as_tool(
        self: "Agent",
        tool_name: Optional[str] = None,
        tool_description: Optional[str] = None,
        custom_output_extractor: Optional[Callable] = None,
    ) -> Function:
        """Convert this Agent to a Function that can be used by other agents.
        
        Args:
            tool_name: The name of the tool. Defaults to snake_case of agent name or 'agent_{id}'.
            tool_description: The tool description. Defaults to agent description or role.
            custom_output_extractor: Optional function to extract output from RunResponse.
            
        Returns:
            A Function instance that wraps this agent.
        """
        # Generate tool name: prefer agent name (snake_case), fallback to agent_id prefix
        if tool_name:
            name = tool_name
        elif self.name:
            # Convert to snake_case
            name = self.name.lower().replace(' ', '_').replace('-', '_')
        else:
            # Fallback to agent_id prefix
            name = f"agent_{self.agent_id[:8]}"
        
        # Generate description
        description = tool_description or self.description or self.role or f"Run the {name} agent."

        def agent_entrypoint(message: str) -> str:
            """Run the agent with the given message and return the response."""
            response = self.run(message, stream=False)
            
            # Use custom output extractor if provided
            if custom_output_extractor:
                return custom_output_extractor(response)
            
            # Default extraction
            if response and response.content:
                if isinstance(response.content, str):
                    return response.content
                return json.dumps(response.content, ensure_ascii=False)
            return "No response from agent."

        return Function(
            name=name,
            description=description,
            entrypoint=agent_entrypoint,
        )

    def get_transfer_function(self: "Agent") -> Function:
        """Get a function to transfer tasks to this agent.
        
        Returns:
            A Function instance that can transfer tasks to this agent.
        """
        agent_name = self.name or "agent"
        agent_description = self.description or self.role or f"Transfer task to {agent_name}"

        def transfer_to_agent(task: str) -> str:
            """Transfer a task to this agent.
            
            Args:
                task: The task description to transfer.
                
            Returns:
                The response from the agent.
            """
            logger.info(f"Transferring task to {agent_name}: {task}")
            response = self.run(message=task, stream=False)
            if response and response.content:
                if isinstance(response.content, str):
                    return response.content
                return json.dumps(response.content, ensure_ascii=False)
            return "No response from agent"

        return Function(
            name=f"transfer_to_{agent_name.lower().replace(' ', '_')}",
            description=agent_description,
            func=transfer_to_agent,
        )

    def get_transfer_prompt(self: "Agent") -> str:
        """Get prompt for transferring tasks to team members.
        
        Returns:
            A string with instructions for task transfer.
        """
        if not self.has_team():
            return ""

        transfer_prompt = "\n## Task Transfer\n"
        transfer_prompt += "You can transfer tasks to the following team members:\n"
        
        for member in self.team:
            member_name = member.name or "unnamed_agent"
            member_role = member.role or member.description or "No description"
            transfer_prompt += f"- **{member_name}**: {member_role}\n"

        transfer_prompt += "\nUse the appropriate transfer function to delegate tasks to team members.\n"
        return transfer_prompt

    def get_tools(self: "Agent") -> Optional[List[Union[ModelTool, Tool, Callable, Dict, Function]]]:
        """Get all tools available to this agent.
        
        This includes:
        - User-provided tools
        - Default tools (chat history, knowledge base search, etc.)
        - Team transfer functions (if enabled)
        
        Returns:
            A list of tools, or None if no tools are available.
        """
        tools: List[Union[ModelTool, Tool, Callable, Dict, Function]] = []

        # Add user-provided tools
        if self.tools is not None:
            tools.extend(self.tools)

        # Add default tools based on settings
        if self.read_chat_history:
            tools.append(
                Tool(
                    name="get_chat_history",
                    description="Use this function to get the chat history between the user and agent. "
                               "Returns a JSON list of messages.",
                    func=self.get_chat_history,
                )
            )

        if self.read_tool_call_history:
            tools.append(
                Tool(
                    name="get_tool_call_history",
                    description="Use this function to get the tool calls made by the agent. "
                               "Returns a JSON list of tool calls in reverse chronological order.",
                    func=self.get_tool_call_history,
                )
            )

        # Add knowledge base tools if knowledge is configured
        if self.knowledge is not None:
            if self.search_knowledge:
                tools.append(
                    Tool(
                        name="search_knowledge_base",
                        description="Use this function to search the knowledge base for relevant information. "
                                   "Returns relevant documents based on the query.",
                        func=self.search_knowledge_base,
                    )
                )

            if self.update_knowledge:
                tools.append(
                    Tool(
                        name="add_to_knowledge",
                        description="Use this function to add information to the knowledge base for future use.",
                        func=self.add_to_knowledge,
                    )
                )

        # Add memory update tool if user memories are enabled
        if self.enable_user_memories and self.memory is not None and self.memory.create_user_memories:
            tools.append(
                Tool(
                    name="update_memory",
                    description="Use this function to update the Agent's memory with important information. "
                               "Describe the task or information in detail.",
                    func=self.update_memory,
                )
            )

        # Add team transfer functions if team is present and transfer instructions are enabled
        if self.has_team() and self.add_transfer_instructions:
            for member in self.team:
                transfer_func = member.get_transfer_function()
                tools.append(transfer_func)

        return tools if len(tools) > 0 else []
