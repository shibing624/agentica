、# Tool Selection Quick Reference

## Built-in Tools

| Tool | Purpose |
|------|---------|
| `ls` | List directory contents |
| `read_file` | Read file (use offset/limit for large files) |
| `write_file` | Create/overwrite file |
| `edit_file` | Modify file via string replacement |
| `glob` | Find files by pattern (e.g., `*.py`) |
| `grep` | Search file contents with regex |
| `execute` | Run commands (python, git, npm, etc.) |
| `web_search` | Search web |
| `fetch_url` | Fetch webpage content |
| `write_todos` | **Task tracking** - track your own progress |
| `read_todos` | Read task list |
| `task` | **Delegation** - assign work to subagent |
| `save_memory` | Save info to memory |

---

## Critical Distinction: write_todos vs task

### `write_todos` - Self-Organization (Transparent)
- **Purpose**: Track your own work progress
- **Visibility**: User sees the task list and updates
- **Execution**: You do the work yourself
- **Use when**: Complex tasks (3+ steps), user wants progress tracking

### `task` - Delegation (Opaque)
- **Purpose**: Delegate subtasks to specialized subagent
- **Visibility**: User only sees final result, not execution details
- **Execution**: Subagent works independently
- **Use when**: Isolated subtasks, need different expertise, parallel work

**Decision**: Self-organization → `write_todos` | Delegation → `task`

---

## Quick Guidelines

### File Operations
- **Always `ls` first** when path is unknown
- **Batch reads** - call `read_file` on multiple files in parallel
- **Never re-read** - Do NOT call `read_file` on the same file twice in one session. Cache content in your context.
- **Use `edit_file`** for targeted changes (safer than `write_file`)

### Execution
- Use non-interactive flags (`--yes`, `-y`)
- Use `--no-pager` for git commands
- Single-line commands only

### Parallel vs Sequential
- **Parallel**: Independent operations (multiple reads, multiple searches)
- **Sequential**: Dependent operations (result of A needed for B)

### When NOT to Use
- Don't use `write_todos` for simple tasks (< 3 steps)
- Don't use `task` for single-step operations
- Don't use `execute` for file ops (use specialized tools)
