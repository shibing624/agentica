# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Temporal Parallel Workflow Demo

This example demonstrates parallel workflow execution with Temporal:
1. ParallelAgentWorkflow - Run multiple agents in parallel
2. ParallelTranslationWorkflow - Translate to multiple languages and pick best

Prerequisites:
    1. Install Temporal: brew install temporal
    2. Start Temporal server: temporal server start-dev
    3. Start worker: python 01_worker.py
    4. Run this client: python 03_parallel_workflow.py
"""
import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from temporalio.client import Client

from agentica.temporal.workflows import (
    ParallelAgentWorkflow,
    ParallelTranslationWorkflow,
    WorkflowInput,
    TranslationInput,
)

TASK_QUEUE = "agentica-task-queue"


async def run_parallel_agents():
    """Run multiple agents in parallel."""
    print("=" * 60)
    print("Demo 1: Parallel Agent Execution")
    print("=" * 60)

    client = await Client.connect("localhost:7233")

    # Define multiple agent configurations
    agent_configs = [
        {
            "name": "Researcher",
            "instructions": "You are a researcher. Provide factual information.",
            "model_id": "gpt-4o-mini",
        },
        {
            "name": "Creative Writer",
            "instructions": "You are a creative writer. Provide imaginative content.",
            "model_id": "gpt-4o-mini",
        },
        {
            "name": "Analyst",
            "instructions": "You are an analyst. Provide analytical insights.",
            "model_id": "gpt-4o-mini",
        },
    ]

    prompt = "Describe artificial intelligence in 2-3 sentences."

    print(f"\nPrompt: {prompt}")
    print(f"Running {len(agent_configs)} agents in parallel...")

    result = await client.execute_workflow(
        ParallelAgentWorkflow.run,
        arg=WorkflowInput(message=prompt, agent_configs=agent_configs),
        id=f"parallel-agents-{asyncio.get_event_loop().time()}",
        task_queue=TASK_QUEUE,
    )

    print("\nResults:")
    # result is WorkflowOutput with content and agent_outputs
    if hasattr(result, 'agent_outputs') and result.agent_outputs:
        for i, response in enumerate(result.agent_outputs):
            name = agent_configs[i]['name'] if i < len(agent_configs) else f"Agent {i+1}"
            print(f"\n[{name}]:")
            print(f"  {response[:200]}..." if len(response) > 200 else f"  {response}")
    else:
        print(f"  {result.content}")


async def run_parallel_translation():
    """Run parallel translation to multiple languages."""
    print("\n" + "=" * 60)
    print("Demo 2: Parallel Translation Workflow")
    print("=" * 60)

    client = await Client.connect("localhost:7233")

    text = "Artificial intelligence is transforming how we work and live."
    target_language = "Chinese"
    num_translations = 3

    print(f"\nOriginal text: {text}")
    print(f"Target language: {target_language}")
    print(f"Number of translations: {num_translations}")
    print("Translating in parallel...")

    result = await client.execute_workflow(
        ParallelTranslationWorkflow.run,
        arg=TranslationInput(text=text, target_language=target_language, num_translations=num_translations),
        id=f"parallel-translation-{asyncio.get_event_loop().time()}",
        task_queue=TASK_QUEUE,
    )

    print("\nBest Translation:")
    print(f"  {result}")


async def main():
    """Main function to run all demos."""
    print("Temporal Parallel Workflow Demo")
    print("Make sure Temporal server and worker are running!\n")

    try:
        await run_parallel_agents()
        await run_parallel_translation()

        print("\n" + "=" * 60)
        print("All demos completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("1. Temporal server is running: temporal server start-dev")
        print("2. Worker is running: python 01_worker.py")


if __name__ == "__main__":
    asyncio.run(main())
