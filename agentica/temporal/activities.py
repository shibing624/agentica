# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Temporal Activities for Agentica Agent execution.

Activities are non-deterministic operations (LLM calls, tool execution).
They are the building blocks that Workflows orchestrate.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
try:
    from temporalio import activity
except ImportError:
    raise ImportError("Temporal SDK not installed. Please install via: pip install temporalio")

from agentica.agent import Agent
from agentica.model.content import Image
from agentica.utils.log import logger


@dataclass
class AgentActivityInput:
    """Input for Agent Activity.
    
    Attributes:
        message: The message to send to the agent.
        agent_name: Optional name for the agent.
        agent_config: Optional configuration dict for the agent.
        images: Optional list of image URLs or paths.
    """
    message: str
    agent_name: Optional[str] = None
    agent_config: Optional[Dict[str, Any]] = None
    images: Optional[List[str]] = None


@dataclass
class AgentActivityOutput:
    """Output from Agent Activity.
    
    Attributes:
        content: The response content from the agent.
        agent_name: The name of the agent that produced this output.
        run_id: The run ID of the agent execution.
        metrics: Optional metrics from the agent run.
    """
    content: str
    agent_name: Optional[str] = None
    run_id: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


@activity.defn
async def run_agent_activity(input: AgentActivityInput) -> AgentActivityOutput:
    """
    Execute an Agent as a Temporal Activity.
    
    This wraps the non-deterministic LLM call in an Activity,
    allowing Temporal to handle retries and persistence.
    
    Args:
        input: AgentActivityInput containing message and agent configuration.
        
    Returns:
        AgentActivityOutput with the agent's response.
    """
    agent_name = input.agent_name or "default"
    logger.info(f"[Temporal Activity] Running agent: {agent_name}")
    
    # Build agent from config or use defaults
    agent_kwargs: Dict[str, Any] = input.agent_config.copy() if input.agent_config else {}
    if input.agent_name:
        agent_kwargs["name"] = input.agent_name
    
    agent = Agent(**agent_kwargs)
    
    # Convert image strings to Image objects if provided
    images: Optional[List[Image]] = None
    if input.images:
        images = []
        for img_path in input.images:
            if img_path.startswith(("http://", "https://")):
                images.append(Image(url=img_path))
            else:
                images.append(Image(filepath=img_path))
    
    # Execute agent
    response = await agent.arun(
        message=input.message,
        images=images,
    )
    
    logger.info(f"[Temporal Activity] Agent {agent_name} completed")
    
    return AgentActivityOutput(
        content=response.content or "",
        agent_name=input.agent_name,
        run_id=response.run_id,
        metrics=response.metrics,
    )
