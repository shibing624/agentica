# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Custom Skill Demo - Creating and Using Custom Skills

This example shows how to:
1. Create custom skills from directories
2. Create skills programmatically with YAML frontmatter
3. Use SkillTool for runtime skill management
"""
import sys
import os
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, ZhipuAI
from agentica.skills import register_skill, get_skill_registry, reset_skill_registry
from agentica.tools.skill_tool import SkillTool


def demo_create_skill_directory():
    """Demo: Create a custom skill from a directory with SKILL.md."""
    print("=" * 60)
    print("Demo 1: Create Custom Skill from Directory")
    print("=" * 60)

    # Reset registry for clean demo
    reset_skill_registry()

    # Create a temporary skill directory
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = os.path.join(tmpdir, "code-reviewer")
        os.makedirs(skill_dir)

        # Create SKILL.md file with YAML frontmatter
        skill_md = """---
name: code-reviewer
description: A skill for reviewing code and providing feedback.
trigger: /review
requires:
  - python
allowed-tools:
  - shell
  - python
metadata:
  emoji: "🔍"
  version: "1.0"
---

# Code Reviewer Skill

## Description
A skill for reviewing code and providing constructive feedback.

## Instructions
When reviewing code:
1. Check for syntax errors and typos
2. Look for potential bugs and edge cases
3. Suggest improvements for readability
4. Check for security vulnerabilities
5. Recommend best practices and patterns

## Output Format
Provide feedback in the following format:

### Issues Found
- List of issues with severity (critical/warning/info)

### Suggestions
- Improvement recommendations

### Overall Assessment
- Summary and rating (1-5 stars)

## Example
```
### Issues Found
- [WARNING] Variable 'x' is unused (line 10)
- [INFO] Consider using f-strings for formatting

### Suggestions
- Add docstrings to public functions
- Use type hints for better clarity

### Overall Assessment
⭐⭐⭐⭐ Good code with minor improvements needed
```
"""
        with open(os.path.join(skill_dir, "SKILL.md"), "w") as f:
            f.write(skill_md)

        # Register the custom skill
        print(f"\nCreating skill from: {skill_dir}")
        skill = register_skill(skill_dir, location="project")

        if skill:
            print(f"\nSkill registered successfully:")
            print(f"  Name: {skill.name}")
            print(f"  Description: {skill.description}")
            print(f"  Trigger: {skill.trigger}")
            print(f"  Requires: {skill.requires}")
            print(f"  Allowed Tools: {skill.allowed_tools}")

            # Test trigger matching
            registry = get_skill_registry()
            matched = registry.match_trigger("/review check this code")
            if matched:
                print(f"\n  Trigger test: '/review' matches '{matched.name}'")


def demo_programmatic_skill():
    """Demo: Create and register skills programmatically."""
    print("\n" + "=" * 60)
    print("Demo 2: Create Skills Programmatically")
    print("=" * 60)

    reset_skill_registry()

    # Define multiple skill contents
    skills_data = [
        {
            "name": "translator",
            "dir_name": "translator",
            "content": """---
name: translator
description: A skill for translating text between languages.
trigger: /translate
---

# Translator Skill

## Instructions
When translating:
1. Identify source language automatically
2. Preserve original meaning and tone
3. Use natural expressions in target language
4. Keep formatting and structure intact

## Supported Languages
- English, Chinese, Japanese, Korean
- French, German, Spanish, Portuguese
- Russian, Arabic, Hindi

## Usage
/translate [target_lang] [text]
Example: /translate zh Hello, how are you?
"""
        },
        {
            "name": "summarizer",
            "dir_name": "summarizer",
            "content": """---
name: summarizer
description: Summarize long text into concise key points.
trigger: /summarize
---

# Summarizer Skill

## Instructions
When summarizing:
1. Extract main ideas and key points
2. Preserve important details and numbers
3. Organize information logically
4. Keep summary concise (20-30% of original)

## Output Format
- **Key Points**: Bullet list of main ideas
- **Summary**: 2-3 sentence overview
- **Action Items**: Any required follow-ups
"""
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        for skill_data in skills_data:
            skill_dir = os.path.join(tmpdir, skill_data["dir_name"])
            os.makedirs(skill_dir)
            with open(os.path.join(skill_dir, "SKILL.md"), "w") as f:
                f.write(skill_data["content"])

            skill = register_skill(skill_dir, location="project")
            if skill:
                print(f"  Registered: {skill.name} (trigger: {skill.trigger})")

        # List all registered skills
        registry = get_skill_registry()
        print(f"\nTotal skills registered: {len(registry)}")

        # Test trigger matching
        print("\nTesting triggers:")
        test_inputs = ["/translate en こんにちは", "/summarize this long text..."]
        for text in test_inputs:
            matched = registry.match_trigger(text)
            if matched:
                print(f"  '{text[:30]}...' -> {matched.name}")


def demo_skill_tool_usage():
    """Demo: Use SkillTool for runtime skill management."""
    print("\n" + "=" * 60)
    print("Demo 3: SkillTool Usage")
    print("=" * 60)

    # Create SkillTool with custom skill directory
    pwd_path = os.path.dirname(os.path.abspath(__file__))
    custom_skill_dir = os.path.join(pwd_path, "../data/skill/python-lib-analyzer")

    if os.path.exists(custom_skill_dir):
        skill_tool = SkillTool(custom_skill_dirs=[custom_skill_dir])

        print("\nAvailable skills via SkillTool:")
        skills_list = skill_tool.list_skills()
        print(skills_list)

        # Get skill info - use the actual skill name from SKILL.md
        print("\n" + "-" * 40)
        print("Skill Info for 'python-lib-analyzer':")
        print("-" * 40)
        info = skill_tool.get_skill_info("python-lib-analyzer")
        if info:
            print(info[:600] + "..." if len(info) > 600 else info)
    else:
        print(f"Custom skill directory not found: {custom_skill_dir}")


def demo_skill_with_agent():
    """Demo: Use custom skill with Agent."""
    print("\n" + "=" * 60)
    print("Demo 4: Custom Skill with Agent")
    print("=" * 60)

    # Create a data analysis skill
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = os.path.join(tmpdir, "data-analyst")
        os.makedirs(skill_dir)

        skill_content = """---
name: data-analyst
description: A skill for analyzing data and generating insights.
trigger: /analyze
allowed-tools:
  - python
  - calculator
---

# Data Analyst Skill

## Instructions
When analyzing data:
1. Understand the data structure and types
2. Identify patterns, trends, and anomalies
3. Calculate relevant statistics (mean, median, etc.)
4. Generate insights and recommendations

## Output Format
- **Data Summary**: Overview of the data
- **Key Findings**: Important patterns and trends
- **Statistics**: Relevant numerical analysis
- **Recommendations**: Suggested actions
"""
        with open(os.path.join(skill_dir, "SKILL.md"), "w") as f:
            f.write(skill_content)

        # Register and use with Agent
        skill = register_skill(skill_dir, location="project")

        if skill:
            # Create agent with skill prompt injected
            agent = Agent(
                model=ZhipuAI(model="glm-4-flash"),
                name="Data-Analyst-Agent",
                instructions=[
                    "You are a data analysis expert.",
                    "",
                    "# Analysis Skill",
                    skill.get_prompt(),
                ],
            )

            print(f"Created agent: {agent.name}")
            print(f"Skill injected: {skill.name}")

            # Test trigger detection
            registry = get_skill_registry()
            test_input = "/analyze Sales: Q1=100k, Q2=150k, Q3=120k, Q4=200k"
            matched = registry.match_trigger(test_input)
            if matched:
                print(f"\nTrigger detected: {matched.name}")
                print("Ready to analyze data...")

                # Uncomment to run actual analysis
                # response = agent.run("Analyze: Sales Q1=100k, Q2=150k, Q3=120k, Q4=200k")
                # print(response.content)


if __name__ == "__main__":
    demo_create_skill_directory()
    demo_programmatic_skill()
    demo_skill_tool_usage()
    demo_skill_with_agent()
