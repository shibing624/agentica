---
name: skill-creator
description: "Guide for creating effective skills. Use when users want to create a new skill or update an existing skill that extends agent capabilities with specialized knowledge, workflows, or tool integrations."
trigger: /skill
metadata:
  emoji: "🛠️"
---

# Skill Creator

Create effective skills to extend agent capabilities.

## What is a Skill?

Skills are modular packages that provide:
1. **Specialized workflows** - Multi-step procedures for specific domains
2. **Tool integrations** - Instructions for working with specific file formats or APIs
3. **Domain expertise** - Company-specific knowledge, schemas, business logic
4. **Bundled resources** - Scripts, references, and assets for complex tasks

## Skill Directory Structure

```
skill-name/
├── SKILL.md          # Required: skill definition with YAML frontmatter
├── scripts/          # Optional: Python or shell scripts
├── references/       # Optional: reference documents
└── assets/           # Optional: images, data files
```

## SKILL.md Format

```markdown
---
name: my-skill
description: "What the skill does and WHEN to use it. Include trigger conditions here."
trigger: /myskill     # Optional: slash command trigger
requires:             # Optional: required tools
  - shell
allowed-tools:        # Optional: tools this skill can use
  - shell
  - write_file
metadata:             # Optional: additional info
  emoji: "🔧"
---

# Skill Title

Instructions for using this skill...
```

## Core Principles

### 1. Be Concise

Context window is shared. Only add what the agent doesn't already know.

**Bad:** "First, you need to understand that git is a version control system..."
**Good:** "Run `git status` to check changes."

### 2. Set Appropriate Freedom

- **High freedom**: Multiple approaches valid → text instructions
- **Medium freedom**: Preferred pattern exists → pseudocode with parameters  
- **Low freedom**: Operations are fragile → specific scripts

### 3. Write Good Descriptions

The `description` field is the **primary trigger**. Include:
- What the skill does
- When to use it (specific contexts/keywords)

**Example:**
```yaml
description: "Create git commits with conventional format. Use when committing code changes, staging files, or pushing to remote."
```

## Creating a Skill

### Step 1: Define Purpose

What specific task does this skill solve? Keep it focused.

### Step 2: Write SKILL.md

1. Add YAML frontmatter with `name` and `description`
2. Write clear, actionable instructions
3. Include examples over verbose explanations

### Step 3: Add Resources (Optional)

If the skill needs scripts or references, add them:
```
my-skill/
├── SKILL.md
├── scripts/
│   └── helper.py
└── references/
    └── api-docs.md
```

### Step 4: Test

1. Load the skill in Agentica
2. Trigger it with various inputs
3. Iterate based on results

## Skill Locations

Skills are loaded from (in priority order):
1. `.agentica/skills/` - Project-level
2. `~/.agentica/workspace/skills/` - User-level  
3. Built-in skills - Package-level

Project skills override user skills with the same name.

## Examples

### Simple Skill

```markdown
---
name: format-json
description: "Format and validate JSON files. Use when working with .json files or JSON data."
---

# JSON Formatter

Format JSON with 2-space indentation:
\`\`\`bash
python -m json.tool input.json > output.json
\`\`\`

Validate JSON:
\`\`\`python
import json
json.loads(content)  # Raises if invalid
\`\`\`
```

### Skill with Trigger

```markdown
---
name: test-runner
description: "Run project tests with coverage reporting."
trigger: /test
requires:
  - pytest
---

# Test Runner

Run all tests:
\`\`\`bash
pytest -v
\`\`\`

Run with coverage:
\`\`\`bash
pytest --cov=src --cov-report=html
\`\`\`
```
