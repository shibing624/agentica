# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Simple workflow demo - Introduction to workflow concepts

This example shows basic workflow patterns:
1. Sequential task execution
2. Passing results between agents
3. Simple workflow orchestration
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat
from agentica.workflow import Workflow
from agentica import RunResponse


class SimpleWorkflow(Workflow):
    """A simple workflow that demonstrates basic patterns."""
    
    description: str = "A simple workflow demonstrating sequential agent execution."
    
    # Define agents
    researcher: Agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        name="Researcher",
        instructions="You are a researcher. Gather key facts about the given topic.",
    )
    
    writer: Agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        name="Writer",
        instructions="You are a writer. Write a short summary based on the research provided.",
    )
    
    def run(self, topic: str):
        """Execute the workflow.
        
        Args:
            topic: The topic to research and write about
        """
        # Step 1: Research
        print(f"Step 1: Researching '{topic}'...")
        research_result = self.researcher.run(f"Research the topic: {topic}")
        
        if not research_result or not research_result.content:
            yield RunResponse(content="Research failed.")
            return
        
        print(f"Research completed: {research_result.content[:100]}...")
        
        # Step 2: Write summary
        print("\nStep 2: Writing summary...")
        yield from self.writer.run(
            f"Based on this research, write a brief summary:\n{research_result.content}",
            stream=True
        )


def main():
    print("=" * 60)
    print("Simple Workflow Demo")
    print("=" * 60)
    
    # Create and run workflow
    workflow = SimpleWorkflow()
    
    # Execute workflow
    result = workflow.run("人工智能的发展历史")
    
    # Print result
    print("\n" + "=" * 60)
    print("Final Output:")
    print("=" * 60)
    for chunk in result:
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
