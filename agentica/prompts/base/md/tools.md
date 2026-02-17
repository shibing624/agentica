# Tool Usage Strategy

## Parallel vs Sequential
- **Parallel**: Independent operations (multiple reads, multiple searches)
- **Sequential**: Dependent operations (result of A needed for B)

## File Operations
- **Batch reads** - call `read_file` on multiple files in parallel
- **Use `edit_file`** for targeted changes (safer than `write_file`)
- **Use `multi_edit_file`** when making multiple changes to the SAME file — it applies all edits atomically in one call, avoiding race conditions and saving tokens

## Execution
- Use non-interactive flags (`--yes`, `-y`)
- Use `--no-pager` for git commands

## Avoid Redundancy
- Don't use `execute` for file ops when specialized tools exist
- Don't use `write_todos` for simple tasks (< 3 steps)
- Don't use `task` for single-step operations

## Context Management
- Prefer targeted reads (offset/limit) over full file reads for large files
- Summarize intermediate findings rather than carrying raw output forward
- When context is long, complete current subtask before starting new ones
