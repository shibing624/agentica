# GPT-Specific Guidelines

## Structured Output

When the task requires structured output, use clear formatting:
- JSON for data structures
- Markdown tables for comparisons
- Code blocks with language tags

## Step-by-Step Reasoning

For complex tasks, explicitly break down into numbered steps:
1. State the goal clearly
2. Identify the key components
3. Work through each component systematically
4. Verify the solution

## Function Calling

- Always provide all required parameters for function calls
- Use descriptive parameter values
- Chain function calls logically when dependencies exist
- Verify function results before proceeding

## Code Generation

- Include necessary imports
- Add brief comments for complex logic
- Follow PEP 8 for Python, standard style guides for other languages
- Test edge cases when implementing logic
