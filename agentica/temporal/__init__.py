# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Temporal integration for Agentica.

Temporal provides durable execution for long-running workflows with:
- Workflow state persistence and fault recovery
- Parallel agent execution
- Workflow observability
"""
from agentica.temporal.activities import (
    run_agent_activity,
    AgentActivityInput,
    AgentActivityOutput,
)
from agentica.temporal.workflows import (
    AgentWorkflow,
    SequentialAgentWorkflow,
    ParallelAgentWorkflow,
    ParallelTranslationWorkflow,
    WorkflowInput,
    WorkflowOutput,
    TranslationInput,
)
from agentica.temporal.client import TemporalClient, WorkflowResult

__all__ = [
    # Activities
    "run_agent_activity",
    "AgentActivityInput",
    "AgentActivityOutput",
    # Workflows
    "AgentWorkflow",
    "SequentialAgentWorkflow",
    "ParallelAgentWorkflow",
    "ParallelTranslationWorkflow",
    "WorkflowInput",
    "WorkflowOutput",
    "TranslationInput",
    # Client
    "TemporalClient",
    "WorkflowResult",
]
