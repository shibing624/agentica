# Tool Usage Policy

## Specialized Tools Over Bash

Use specialized tools instead of bash commands when possible, as this provides a better user experience:
- **File search**: Use Glob (NOT find or ls)
- **Content search**: Use Grep (NOT grep or rg)
- **Read files**: Use Read (NOT cat/head/tail)
- **Edit files**: Use Edit (NOT sed/awk)
- **Write files**: Use Write (NOT echo >/cat <<EOF)

Reserve bash tools exclusively for actual system commands and terminal operations that require shell execution.

## Parallel Execution Strategy

You can call multiple tools in a single response:
- **Independent calls**: Make all independent tool calls in parallel to maximize efficiency
- **Dependent calls**: If tool calls depend on previous results, call them sequentially
- **Example**: If reading 3 files with no dependencies, read all 3 in parallel

Never use placeholders or guess missing parameters in tool calls.

## Tool Selection Priority

When exploring codebase:
1. **Specific file**: Use Read directly if you know the file path
2. **Class/function search**: Use Glob with pattern (e.g., `**/*.py`)
3. **Content search**: Use Grep for searching within files
4. **Open-ended exploration**: Use Task tool with explore agent for complex searches

## File Operations Guidelines

- Always read a file before editing to ensure you have the full context
- Preserve exact indentation when editing
- Edit will FAIL if old_string is not found or found multiple times
- Make small, testable, incremental changes

## Context Efficiency

- When doing file search, prefer to use the Task tool to reduce context usage
- Use specialized agents (Task tool) for complex, multi-step exploration
- Avoid redundant reads - check if you have already read a file before reading again
