# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Temporal Client for Agentica

This client starts workflows and retrieves results.

Prerequisites:
    1. Temporal server running: temporal server start-dev
    2. Worker running: python 01_worker.py

Usage:
    # Simple agent workflow
    python 02_client.py simple "What is AI?"
    
    # Parallel translation workflow
    python 02_client.py translate "Hello, how are you today?"
    
    # Sequential agent workflow
    python 02_client.py sequential "Write about machine learning"
    
    # Parallel agent workflow
    python 02_client.py parallel "Analyze the impact of AI"
"""
import sys
import os
import asyncio
import uuid
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from temporalio.client import Client

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


def print_usage():
    """Print usage information."""
    print(__doc__)


async def main():
    if len(sys.argv) < 2:
        print_usage()
        return
    
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
    
    else:
        print(f"Unknown command: {command}")
        print_usage()


if __name__ == "__main__":
    asyncio.run(main())
