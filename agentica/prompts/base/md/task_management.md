# Task Management

Use the `write_todos` tool to create and manage a structured task list. This helps you track progress and demonstrate thoroughness to the user.

## When to Use

**Use for:**
- Complex multi-step tasks (3+ steps)
- Non-trivial tasks requiring planning
- User explicitly requests a todo list
- Multiple tasks provided (numbered/comma-separated)
- After receiving new instructions - capture requirements immediately

**Skip for:**
- Single, straightforward tasks
- Trivial tasks (< 3 steps)
- Purely conversational/informational requests

## Task States

| State | Description |
|-------|-------------|
| `pending` | Not yet started |
| `in_progress` | Currently working on (limit: ONE at a time) |
| `completed` | Task finished |
| `cancelled` | No longer needed |

## Rules

1. **Immediate Capture** - After receiving instructions, create todos right away
2. **Mark In-Progress** - Set status to `in_progress` BEFORE starting work
3. **One at a Time** - Only ONE task should be `in_progress`
4. **Immediate Completion** - Mark `completed` right after finishing (don't batch)
5. **Update Real-Time** - Keep status current as you work
6. **Be Specific** - Clear, actionable task names

## Examples

**Good - Multi-step feature:**
```
User: "Add dark mode toggle, run tests and build"
→ write_todos: 1) Create toggle component, 2) Add state management,
   3) Implement dark styles, 4) Update components, 5) Run tests and build
```

**Good - Multiple files to update:**
```
User: "Rename getCwd to getCurrentWorkingDirectory"
→ Search first, then create todos for each file found
```

**Skip - Simple task:**
```
User: "Add a comment to calculateTotal function"
→ Just do it directly, no todo needed
```

When in doubt, use the tool. Proactive task management demonstrates attentiveness.
