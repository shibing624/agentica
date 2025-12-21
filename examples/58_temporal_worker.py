# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Temporal Worker for Agentica.

This worker listens for tasks and executes Workflows/Activities.
Run this as a long-running service before using the client.

Usage:
    0. Install Temporal (Mac):
       brew install temporal

    1. Start Temporal server (dev mode):
       temporal server start-dev
       
    2. Start this worker:
       python 58_temporal_worker.py
       
    3. Use the client to start workflows:
       python 58_temporal_client.py simple "What is AI?"

The worker will:
- Connect to Temporal server
- Register workflows and activities
- Poll for tasks and execute them
- Handle retries and failures automatically
"""
import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from temporalio.client import Client
from temporalio.worker import Worker, UnsandboxedWorkflowRunner

# Import workflows from the workflows module directly
from agentica.temporal.workflows import (
    AgentWorkflow,
    SequentialAgentWorkflow,
    ParallelAgentWorkflow,
    ParallelTranslationWorkflow,
)
# Import activity from activities module
from agentica.temporal.activities import run_agent_activity

# Task queue name - must match the client
TASK_QUEUE = "agentica-task-queue"


async def main():
    """Start the Temporal worker."""
    print("=" * 60)
    print("Agentica Temporal Worker")
    print("=" * 60)
    
    # Connect to Temporal server
    print("\nConnecting to Temporal server (localhost:7233)...")
    try:
        client = await Client.connect("localhost:7233")
        print("Connected!")
    except Exception as e:
        print(f"Failed to connect: {e}")
        print("\nMake sure Temporal server is running:")
        print("  temporal server start-dev")
        return
    
    # Create worker with workflows and activities
    # Note: UnsandboxedWorkflowRunner is required because agentica
    # imports modules (httpx, etc.) that are restricted in Temporal's sandbox
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[
            AgentWorkflow,
            SequentialAgentWorkflow,
            ParallelAgentWorkflow,
            ParallelTranslationWorkflow,
        ],
        activities=[
            run_agent_activity,
        ],
        workflow_runner=UnsandboxedWorkflowRunner(),  # Disable sandbox for agentica compatibility
    )
    
    print(f"\nWorker started on task queue: {TASK_QUEUE}")
    print("\nRegistered workflows:")
    print("  - AgentWorkflow (simple single-agent)")
    print("  - SequentialAgentWorkflow (pipeline)")
    print("  - ParallelAgentWorkflow (parallel execution)")
    print("  - ParallelTranslationWorkflow (translation with best-pick)")
    print("\nRegistered activities:")
    print("  - run_agent_activity")
    print("\nPress Ctrl+C to stop")
    print("=" * 60)
    
    # Run worker (blocks until interrupted)
    try:
        await worker.run()
    except asyncio.CancelledError:
        print("\nWorker shutting down...")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nWorker stopped.")
