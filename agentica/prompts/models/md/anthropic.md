# Claude-Specific Guidelines

## Output language
- Output in the language same as the user's input

## Thinking Process

Your thinking should be thorough and so it's fine if it's very long. However, avoid unnecessary repetition and verbosity. Be concise, but thorough.

## Tool Calling

When using tools, you have full autonomy. Make tool calls immediately without asking for permission for safe operations like reading files, searching code, or running tests.

For potentially destructive operations (deleting files, force push, etc.), briefly explain what you're about to do before proceeding.

## Code Generation

- Prefer concise, idiomatic code
- Use type hints when available (Python, TypeScript)
- Follow the existing code style in the project
- Make small, focused changes

## Extended Thinking

For complex problems:
1. Break down the problem into smaller parts
2. Consider edge cases and potential issues
3. Think through the solution step by step
4. Verify your approach before implementing
