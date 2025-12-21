# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Temporal Client for Agentica.

This client starts workflows and retrieves results.

Prerequisites:
    1. Temporal server running: temporal server start-dev
    2. Worker running: python 58_temporal_worker.py

Usage:
    # Simple agent workflow
    python 58_temporal_client.py simple "What is artificial intelligence?"
    
    # Parallel translation workflow (English to Chinese)
    python 58_temporal_client.py translate "Hello, how are you today?"
    
    # Sequential agent workflow (research -> write -> edit)
    python 58_temporal_client.py sequential "Write about machine learning"
    
    # Parallel agent workflow (multiple perspectives)
    python 58_temporal_client.py parallel "Analyze the impact of AI"
    
    # Check workflow status
    python 58_temporal_client.py status <workflow_id>
    
    # Get workflow result (blocks until complete)
    python 58_temporal_client.py result <workflow_id>
"""
import sys
import os
import asyncio
import uuid
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from temporalio.client import Client

# Import from agentica.temporal core module
from agentica.temporal import (
    AgentWorkflow,
    SequentialAgentWorkflow,
    ParallelAgentWorkflow,
    ParallelTranslationWorkflow,
    WorkflowInput,
    TranslationInput,
)

TASK_QUEUE = "agentica-task-queue"


async def run_simple_workflow(client: Client, message: str):
    """Run a simple single-agent workflow."""
    workflow_id = f"simple-{uuid.uuid4().hex[:8]}"
    
    print(f"\n{'=' * 60}")
    print("Starting AgentWorkflow (Simple)")
    print(f"{'=' * 60}")
    print(f"Workflow ID: {workflow_id}")
    print(f"Message: {message}")
    
    start_time = time.time()
    
    handle = await client.start_workflow(
        AgentWorkflow.run,
        WorkflowInput(message=message),
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )
    
    print("\nWaiting for result...")
    result = await handle.result()
    
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Result (took {elapsed:.2f}s):")
    print(f"{'=' * 60}")
    print(result.content)


async def run_translation_workflow(client: Client, text: str):
    """Run parallel translation workflow."""
    workflow_id = f"translate-{uuid.uuid4().hex[:8]}"
    
    print(f"\n{'=' * 60}")
    print("Starting ParallelTranslationWorkflow")
    print(f"{'=' * 60}")
    print(f"Workflow ID: {workflow_id}")
    print(f"Text: {text}")
    
    start_time = time.time()
    
    handle = await client.start_workflow(
        ParallelTranslationWorkflow.run,
        TranslationInput(text=text, target_language="Chinese", num_translations=3),
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )
    
    print("\nRunning 3 translations in parallel...")
    result = await handle.result()
    
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Best Translation (took {elapsed:.2f}s):")
    print(f"{'=' * 60}")
    print(result)


async def run_sequential_workflow(client: Client, topic: str):
    """Run sequential multi-agent workflow."""
    workflow_id = f"sequential-{uuid.uuid4().hex[:8]}"
    
    print(f"\n{'=' * 60}")
    print("Starting SequentialAgentWorkflow")
    print(f"{'=' * 60}")
    print(f"Workflow ID: {workflow_id}")
    print(f"Topic: {topic}")
    
    # Define the pipeline: research -> write -> edit
    agent_configs = [
        {
            "name": "researcher",
            "instructions": "Research the given topic and provide key facts and insights. Be concise.",
        },
        {
            "name": "writer",
            "instructions": "Based on the research provided, write a short article. Keep it under 200 words.",
        },
        {
            "name": "editor",
            "instructions": "Edit and polish the article. Fix any issues and improve clarity. Output the final version.",
        },
    ]
    
    start_time = time.time()
    
    handle = await client.start_workflow(
        SequentialAgentWorkflow.run,
        WorkflowInput(message=topic, agent_configs=agent_configs),
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )
    
    print("\nRunning pipeline: Researcher -> Writer -> Editor...")
    result = await handle.result()
    
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Final Article (took {elapsed:.2f}s):")
    print(f"{'=' * 60}")
    print(result.content)


async def run_parallel_workflow(client: Client, topic: str):
    """Run parallel multi-agent workflow."""
    workflow_id = f"parallel-{uuid.uuid4().hex[:8]}"
    
    print(f"\n{'=' * 60}")
    print("Starting ParallelAgentWorkflow")
    print(f"{'=' * 60}")
    print(f"Workflow ID: {workflow_id}")
    print(f"Topic: {topic}")
    
    # Define agents for parallel analysis
    agent_configs = [
        {
            "name": "technical_analyst",
            "instructions": "Analyze the topic from a technical perspective. Be concise.",
        },
        {
            "name": "business_analyst",
            "instructions": "Analyze the topic from a business/economic perspective. Be concise.",
        },
        {
            "name": "social_analyst",
            "instructions": "Analyze the topic from a social/ethical perspective. Be concise.",
        },
    ]
    
    start_time = time.time()
    
    handle = await client.start_workflow(
        ParallelAgentWorkflow.run,
        WorkflowInput(message=topic, agent_configs=agent_configs),
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )
    
    print("\nRunning 3 analysts in parallel...")
    result = await handle.result()
    
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Analysis Results (took {elapsed:.2f}s):")
    print(f"{'=' * 60}")
    print(result.content)


async def get_workflow_status(client: Client, workflow_id: str):
    """Get workflow status."""
    handle = client.get_workflow_handle(workflow_id)
    desc = await handle.describe()
    print(f"\nWorkflow: {workflow_id}")
    print(f"Status: {desc.status.name}")
    print(f"Start time: {desc.start_time}")
    if desc.close_time:
        print(f"Close time: {desc.close_time}")


async def get_workflow_result(client: Client, workflow_id: str):
    """Get workflow result (blocks until complete)."""
    handle = client.get_workflow_handle(workflow_id)
    print(f"\nWaiting for workflow {workflow_id}...")
    result = await handle.result()
    print(f"\nResult:")
    print(result)


def print_usage():
    """Print usage information."""
    print(__doc__)


async def main():
    if len(sys.argv) < 2:
        print_usage()
        return
    
    # Connect to Temporal
    try:
        client = await Client.connect("localhost:7233")
    except Exception as e:
        print(f"Failed to connect to Temporal server: {e}")
        print("\nMake sure Temporal server is running:")
        print("  temporal server start-dev")
        return
    
    command = sys.argv[1]
    
    if command == "simple":
        message = sys.argv[2] if len(sys.argv) > 2 else "What is artificial intelligence?"
        await run_simple_workflow(client, message)
    
    elif command == "translate":
        text = sys.argv[2] if len(sys.argv) > 2 else "The quick brown fox jumps over the lazy dog."
        await run_translation_workflow(client, text)
    
    elif command == "sequential":
        topic = sys.argv[2] if len(sys.argv) > 2 else "machine learning"
        await run_sequential_workflow(client, topic)
    
    elif command == "parallel":
        topic = sys.argv[2] if len(sys.argv) > 2 else "artificial intelligence"
        await run_parallel_workflow(client, topic)
    
    elif command == "status":
        if len(sys.argv) < 3:
            print("Usage: python 58_temporal_client.py status <workflow_id>")
            return
        await get_workflow_status(client, sys.argv[2])
    
    elif command == "result":
        if len(sys.argv) < 3:
            print("Usage: python 58_temporal_client.py result <workflow_id>")
            return
        await get_workflow_result(client, sys.argv[2])
    
    else:
        print(f"Unknown command: {command}")
        print_usage()


if __name__ == "__main__":
    asyncio.run(main())
