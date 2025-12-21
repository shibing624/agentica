# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Custom Skill Demo - Creating and using custom skills

This example shows how to:
1. Create custom skills from directories
2. Create skills programmatically
3. Use skills with agents
"""
import sys
import os
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat
from agentica.tools.skill_tool import SkillTool


def create_custom_skill_demo():
    """Demo: Create a custom skill from a directory."""
    print("=" * 60)
    print("Demo 1: Create Custom Skill from Directory")
    print("=" * 60)

    # Create a temporary skill directory
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = os.path.join(tmpdir, "code-reviewer")
        os.makedirs(skill_dir)

        # Create SKILL.md file
        skill_md = """---
name: Code Reviewer Skill
description: A skill for reviewing code and providing feedback.
---

# Code Reviewer Skill

## Description
A skill for reviewing code and providing feedback.

## Instructions
When reviewing code:
1. Check for syntax errors
2. Look for potential bugs
3. Suggest improvements for readability
4. Check for security issues
5. Recommend best practices

## Output Format
Provide feedback in the following format:
- **Issues Found**: List of issues
- **Suggestions**: Improvement suggestions
- **Overall Assessment**: Summary
"""
        with open(os.path.join(skill_dir, "SKILL.md"), "w") as f:
            f.write(skill_md)

        # Create SkillTool with custom skill
        skill_tool = SkillTool(custom_skill_dirs=[skill_dir])

        print(f"\nCreated skill directory: {skill_dir}")
        print(f"\nAvailable skills:")
        print(skill_tool.list_skills())

        # Execute the skill
        print("\nExecuting 'code-reviewer' skill:")
        result = skill_tool.execute_skill("code-reviewer")
        print(result)


def programmatic_skill_demo():
    """Demo: Create skills programmatically."""
    print("\n" + "=" * 60)
    print("Demo 2: Create Skills Programmatically")
    print("=" * 60)

    skill_tool = SkillTool()

    # Define skill content
    translator_skill = """
---
name: Translator Skill
description: A skill for translating text between languages.
---

# Translator Skill

## Description
A skill for translating text between languages.

## Instructions
When translating:
1. Identify the source language
2. Preserve the original meaning
3. Use natural expressions in the target language
4. Keep formatting and structure

## Supported Languages
- English
- Chinese (Simplified/Traditional)
- Japanese
- Korean
- French
- German
- Spanish
"""

    # Create a temporary directory and add the skill
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = os.path.join(tmpdir, "translator")
        os.makedirs(skill_dir)
        with open(os.path.join(skill_dir, "SKILL.md"), "w") as f:
            f.write(translator_skill)

        skill = skill_tool.add_skill_dir(skill_dir)
        if skill:
            print(f"\nAdded skill: {skill.name}")
            print(f"Description: {skill.description}")

        print(f"\nAll available skills:")
        print(skill_tool.list_skills())


def skill_with_agent_demo():
    """Demo: Use custom skills with an agent."""
    print("\n" + "=" * 60)
    print("Demo 3: Use Custom Skills with Agent")
    print("=" * 60)

    # Create a data analysis skill
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = os.path.join(tmpdir, "data-analyst")
        os.makedirs(skill_dir)

        skill_content = """---
name: Data Analyst Skill
description: A skill for analyzing data and generating insights.
---

# Data Analyst Skill

## Description
A skill for analyzing data and generating insights.

## Instructions
When analyzing data:
1. Understand the data structure
2. Identify patterns and trends
3. Calculate relevant statistics
4. Generate visualizations if needed
5. Provide actionable insights

## Output Format
- **Data Summary**: Overview of the data
- **Key Findings**: Important patterns
- **Recommendations**: Suggested actions
"""
        with open(os.path.join(skill_dir, "SKILL.md"), "w") as f:
            f.write(skill_content)

        skill_tool = SkillTool(custom_skill_dirs=[skill_dir])

        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini"),
            name="Skill-Enabled Assistant",
            instructions=[
                "You are a helpful assistant with skill capabilities.",
                "Use list_skills() to see available skills.",
                "Use execute_skill(name) to load a skill's instructions.",
            ],
            tools=[skill_tool],
            show_tool_calls=True,
        )

        print("\nAsking agent about available skills...")
        response = agent.run("What skills do you have? List them.")
        print(f"Response: {response.content}")

        print("\nAsking agent to use the data-analyst skill...")
        response = agent.run(
            "Load the data-analyst skill and then analyze this data: "
            "Sales in Q1: $100k, Q2: $150k, Q3: $120k, Q4: $200k"
        )
        print(f"Response: {response.content}")


if __name__ == "__main__":
    create_custom_skill_demo()
    programmatic_skill_demo()
    skill_with_agent_demo()
