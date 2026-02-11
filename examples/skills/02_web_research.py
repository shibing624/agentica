# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Web Research Skill Demo

This demo shows how to use skills with DeepAgent for web research tasks.
The web-research skill provides structured instructions for conducting
comprehensive research using subagents.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import DeepAgent, OpenAIChat
from agentica.skills import register_skill, get_skill_registry

pwd_path = os.path.dirname(os.path.abspath(__file__))


def demo_web_research_skill():
    """Demo: Load and inspect the web-research skill."""
    print("=" * 60)
    print("Demo 1: Web Research Skill Overview")
    print("=" * 60)

    # Register the web-research skill
    skill_dir = os.path.join(pwd_path, "../data/skills/web-research")
    skill = register_skill(skill_dir, location="project")

    if skill:
        print(f"\nSkill: {skill.name}")
        print(f"Description: {skill.description}")
        print(f"Location: {skill.location}")

        # Show skill prompt (instructions)
        print("\n" + "-" * 40)
        print("Skill Instructions Preview:")
        print("-" * 40)
        prompt = skill.get_prompt()
        # Show first 800 chars
        print(prompt[:800] + "..." if len(prompt) > 800 else prompt)
    else:
        print(f"Failed to load skill from: {skill_dir}")


def demo_research_agent():
    """Demo: Create a research agent with the web-research skill."""
    print("\n" + "=" * 60)
    print("Demo 2: Research Agent with Skill")
    print("=" * 60)

    # Get the web-research skill
    registry = get_skill_registry()
    skill = registry.get("web-research")

    if not skill:
        # Load it if not already loaded
        skill_dir = os.path.join(pwd_path, "../data/skills/web-research")
        skill = register_skill(skill_dir, location="project")

    if skill:
        # Create DeepAgent with skill instructions
        agent = DeepAgent(
            model=OpenAIChat(id="gpt-4o-mini"),
            name="Research-Assistant",
            instructions=[
                "You are a research assistant that conducts thorough web research.",
                "",
                "# Research Skill Instructions",
                skill.get_prompt(),
            ],
            add_datetime_to_instructions=True,
        )

        print(f"\nAgent: {agent.name}")
        print(f"Model: {agent.model}")
        print(f"Tools available: {len(agent.tools)}")

        # Example research question (uncomment to run)
        # question = "帮我调研 2024 年人工智能的主要发展趋势，写出简短的中文报告。"
        # print(f"\nResearch Question: {question}")
        # response = agent.run_sync(question)
        # print(f"\nResearch Report:\n{response.content}")

        print("\nTo run actual research, uncomment the code above.")
    else:
        print("Web-research skill not found")


def demo_research_workflow():
    """Demo: Show the expected research workflow from the skill."""
    print("\n" + "=" * 60)
    print("Demo 3: Research Workflow")
    print("=" * 60)

    print("""
The web-research skill defines a 3-step research process:

Step 1: Create and Save Research Plan
  - Create a research folder: mkdir research_[topic_name]
  - Analyze the research question
  - Write research_plan.md with subtopics

Step 2: Delegate to Research Subagents
  - Use the 'task' tool to spawn subagents
  - Each subagent researches one subtopic
  - Subagents save findings to files

Step 3: Synthesize Findings
  - Read all findings files
  - Create comprehensive response
  - Cite sources with URLs

Best Practices:
  - Plan before delegating (always write research_plan.md first)
  - Clear subtopics (non-overlapping scope)
  - File-based communication (subagents save to files)
  - Systematic synthesis (read all findings before final response)
  - 3-5 searches per subtopic is usually sufficient
""")


if __name__ == "__main__":
    demo_web_research_skill()
    demo_research_agent()
    demo_research_workflow()
