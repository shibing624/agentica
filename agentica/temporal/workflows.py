# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Temporal Workflows for Agentica.

Workflows are deterministic orchestration logic that coordinate Agent Activities.
They handle the control flow while Activities handle the actual LLM calls.

Temporal Concepts:
- Workflow: Deterministic orchestration logic (control flow, conditions, waits)
- Activity: Non-deterministic operations (LLM calls, API calls, I/O)

IMPORTANT: This module is designed to work with Temporal's sandbox.
- Workflows must be deterministic
- Activities handle all non-deterministic operations (LLM calls, I/O)
- Use workflow.unsafe.imports_passed_through() for external imports

Usage:
    # In worker code:
    from agentica.temporal.workflows import (
        AgentWorkflow,
        SequentialAgentWorkflow,
        ParallelAgentWorkflow,
        ParallelTranslationWorkflow,
    )
    from agentica.temporal.activities import run_agent_activity
    
    # In client code:
    from agentica.temporal.workflows import WorkflowInput, TranslationInput
"""
from dataclasses import dataclass, field
from datetime import timedelta
from typing import List, Optional, Dict, Any

try:
    from temporalio import workflow
    from temporalio.common import RetryPolicy
except ImportError:
    raise ImportError("Temporal SDK not installed. Please install via: pip install temporalio")

# Define input/output dataclasses here (not imported from activities)
# to avoid triggering agentica imports in the sandbox
@dataclass
class AgentActivityInput:
    """Input for Agent Activity (mirror of activities.AgentActivityInput)."""
    message: str
    agent_name: Optional[str] = None
    agent_config: Optional[Dict[str, Any]] = None
    images: Optional[List[str]] = None


@dataclass
class AgentActivityOutput:
    """Output from Agent Activity (mirror of activities.AgentActivityOutput)."""
    content: str
    agent_name: Optional[str] = None
    run_id: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


@dataclass
class WorkflowInput:
    """Input for Agent Workflows.
    
    Attributes:
        message: The initial message to process.
        agent_configs: Optional list of agent configurations for multi-agent workflows.
    """
    message: str
    agent_configs: Optional[List[Dict[str, Any]]] = None


@dataclass
class TranslationInput:
    """Input for translation workflow.
    
    Attributes:
        text: The text to translate.
        target_language: Target language for translation (default: Chinese).
        num_translations: Number of parallel translations to generate.
    """
    text: str
    target_language: str = "Chinese"
    num_translations: int = 3


@dataclass
class WorkflowOutput:
    """Output from Agent Workflows.
    
    Attributes:
        content: The final output content.
        steps: List of outputs from each agent step.
    """
    content: str
    steps: List[AgentActivityOutput] = field(default_factory=list)


# Default retry policy for activities
DEFAULT_RETRY_POLICY = RetryPolicy(
    maximum_attempts=3,
    initial_interval=timedelta(seconds=1),
    maximum_interval=timedelta(seconds=30),
    backoff_coefficient=2.0,
)

# Activity name constant (must match the activity function name)
RUN_AGENT_ACTIVITY = "run_agent_activity"


@workflow.defn
class AgentWorkflow:
    """
    Basic single-agent workflow.
    
    Executes one agent and returns the result.
    This is the simplest workflow pattern.
    
    Example:
        handle = await client.start_workflow(
            AgentWorkflow.run,
            WorkflowInput(message="What is AI?"),
            id="my-workflow",
            task_queue="agentica-task-queue",
        )
        result = await handle.result()
    """
    
    @workflow.run
    async def run(self, input: WorkflowInput) -> WorkflowOutput:
        workflow.logger.info(f"Starting AgentWorkflow with message: {input.message[:50]}...")
        
        # Get agent config (first one if provided, else empty)
        agent_config = None
        if input.agent_configs and len(input.agent_configs) > 0:
            agent_config = input.agent_configs[0]
        
        # Execute single agent
        result = await workflow.execute_activity(
            RUN_AGENT_ACTIVITY,
            AgentActivityInput(
                message=input.message,
                agent_config=agent_config,
            ),
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=DEFAULT_RETRY_POLICY,
            result_type=AgentActivityOutput,
        )
        
        workflow.logger.info("AgentWorkflow completed")
        
        return WorkflowOutput(
            content=result.content,
            steps=[result],
        )


@workflow.defn
class SequentialAgentWorkflow:
    """
    Sequential multi-agent workflow.
    
    Executes agents one after another, passing output from one to the next.
    This is useful for pipeline-style processing.
    
    Example:
        # Define agent configs for each step
        configs = [
            {"name": "researcher", "instructions": "Research the topic"},
            {"name": "writer", "instructions": "Write based on research"},
            {"name": "editor", "instructions": "Edit and polish"},
        ]
        
        handle = await client.start_workflow(
            SequentialAgentWorkflow.run,
            WorkflowInput(message="Write about AI", agent_configs=configs),
            id="sequential-workflow",
            task_queue="agentica-task-queue",
        )
    """
    
    @workflow.run
    async def run(self, input: WorkflowInput) -> WorkflowOutput:
        if not input.agent_configs:
            raise ValueError("agent_configs required for SequentialAgentWorkflow")
        
        workflow.logger.info(
            f"Starting SequentialAgentWorkflow with {len(input.agent_configs)} agents"
        )
        
        steps: List[AgentActivityOutput] = []
        current_message = input.message
        
        for i, agent_config in enumerate(input.agent_configs):
            agent_name = agent_config.get("name", f"agent_{i+1}")
            workflow.logger.info(
                f"Step {i+1}/{len(input.agent_configs)}: Running {agent_name}"
            )
            
            result = await workflow.execute_activity(
                RUN_AGENT_ACTIVITY,
                AgentActivityInput(
                    message=current_message,
                    agent_name=agent_name,
                    agent_config=agent_config,
                ),
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=DEFAULT_RETRY_POLICY,
                result_type=AgentActivityOutput,
            )
            
            steps.append(result)
            current_message = result.content  # Pass to next agent
        
        workflow.logger.info("SequentialAgentWorkflow completed")
        
        return WorkflowOutput(
            content=steps[-1].content if steps else "",
            steps=steps,
        )


@workflow.defn
class ParallelAgentWorkflow:
    """
    Parallel multi-agent workflow.
    
    Executes multiple agents in parallel and aggregates results.
    This is useful for getting multiple perspectives or parallel processing.
    
    Example:
        # Define agent configs for parallel execution
        configs = [
            {"name": "analyst_1", "instructions": "Analyze from perspective A"},
            {"name": "analyst_2", "instructions": "Analyze from perspective B"},
            {"name": "analyst_3", "instructions": "Analyze from perspective C"},
        ]
        
        handle = await client.start_workflow(
            ParallelAgentWorkflow.run,
            WorkflowInput(message="Analyze this topic", agent_configs=configs),
            id="parallel-workflow",
            task_queue="agentica-task-queue",
        )
    """
    
    @workflow.run
    async def run(self, input: WorkflowInput) -> WorkflowOutput:
        if not input.agent_configs:
            raise ValueError("agent_configs required for ParallelAgentWorkflow")
        
        workflow.logger.info(
            f"Starting ParallelAgentWorkflow with {len(input.agent_configs)} agents"
        )
        
        # Create all activity tasks
        tasks = []
        for i, agent_config in enumerate(input.agent_configs):
            agent_name = agent_config.get("name", f"agent_{i+1}")
            task = workflow.execute_activity(
                RUN_AGENT_ACTIVITY,
                AgentActivityInput(
                    message=input.message,
                    agent_name=agent_name,
                    agent_config=agent_config,
                ),
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=DEFAULT_RETRY_POLICY,
                result_type=AgentActivityOutput,
            )
            tasks.append(task)
        
        # Wait for all to complete
        results: List[AgentActivityOutput] = []
        for task in tasks:
            result = await task
            results.append(result)
        
        workflow.logger.info(f"All {len(results)} agents completed")
        
        # Aggregate results
        contents = []
        for i, r in enumerate(results):
            agent_name = r.agent_name or f"Agent {i+1}"
            contents.append(f"[{agent_name}]\n{r.content}")
        
        aggregated = "\n\n---\n\n".join(contents)
        
        workflow.logger.info("ParallelAgentWorkflow completed")
        
        return WorkflowOutput(
            content=aggregated,
            steps=results,
        )


@workflow.defn
class ParallelTranslationWorkflow:
    """
    Parallel translation workflow with best-pick selection.
    
    Demonstrates the parallelization pattern:
    1. Run multiple translation agents in parallel
    2. Collect all results
    3. Use a picker agent to select the best translation
    
    This provides Temporal's durable execution guarantees for
    translation tasks with automatic retries and fault tolerance.
    
    Example:
        handle = await client.start_workflow(
            ParallelTranslationWorkflow.run,
            TranslationInput(
                text="Hello, how are you?",
                target_language="Chinese",
                num_translations=3,
            ),
            id="translation-workflow",
            task_queue="agentica-task-queue",
        )
        result = await handle.result()  # Returns the best translation
    """
    
    @workflow.run
    async def run(self, input: TranslationInput) -> str:
        workflow.logger.info(
            f"Starting ParallelTranslationWorkflow: {input.num_translations} translations to {input.target_language}"
        )
        
        # Step 1: Run translations in parallel
        translation_tasks = []
        for i in range(input.num_translations):
            task = workflow.execute_activity(
                RUN_AGENT_ACTIVITY,
                AgentActivityInput(
                    message=input.text,
                    agent_name=f"translator_{i+1}",
                    agent_config={
                        "instructions": f"You translate the user's message to {input.target_language}. "
                                       f"Only output the translation, nothing else.",
                    },
                ),
                start_to_close_timeout=timedelta(minutes=3),
                retry_policy=DEFAULT_RETRY_POLICY,
                result_type=AgentActivityOutput,
            )
            translation_tasks.append(task)
        
        # Wait for all translations
        translations: List[AgentActivityOutput] = []
        for task in translation_tasks:
            result = await task
            translations.append(result)
        
        workflow.logger.info(f"Got {len(translations)} translations")
        
        # Step 2: Pick the best translation
        translations_text = "\n".join([
            f"Translation {i+1}: {t.content}" 
            for i, t in enumerate(translations)
        ])
        
        picker_result = await workflow.execute_activity(
            RUN_AGENT_ACTIVITY,
            AgentActivityInput(
                message=f"Original: {input.text}\n\n{translations_text}",
                agent_name="translation_picker",
                agent_config={
                    "instructions": f"Pick the best {input.target_language} translation. "
                                   f"Only output the best translation, nothing else.",
                },
            ),
            start_to_close_timeout=timedelta(minutes=2),
            retry_policy=DEFAULT_RETRY_POLICY,
            result_type=AgentActivityOutput,
        )
        
        workflow.logger.info("ParallelTranslationWorkflow completed")
        return picker_result.content
