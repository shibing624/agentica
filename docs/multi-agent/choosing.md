# Choosing an Orchestration Pattern

Agentica has several multi-agent patterns. Use the smallest one that gives you the control boundary you need.

There are two layers:

- **SDK / application** — you write Python that composes agents (`as_tool`, Workflow, Subagent, Swarm).
- **Interactive CLI** — the model calls tools mid-session (`task`, `delegate`, peer messaging). Same word “delegate” means different machinery.

## Decision Tree

```text
Do you need more than one agent?
    |
    +-- No
    |     Use Agent.run()
    |
    +-- Yes — are you writing an app, or sitting in the CLI?
          |
          +-- Application / SDK
          |     |
          |     +-- Is the step order fixed and auditable?
          |     |     Use Workflow
          |     |
          |     +-- Should a parent agent choose a helper as a tool?
          |     |     Use Agent.as_tool()
          |     |
          |     +-- Does the helper need isolation, tool permissions, or timeout?
          |     |     Use Subagent (same as the CLI `task` tool)
          |     |
          |     +-- Do multiple peers need to work in parallel or autonomously split work?
          |           Use Swarm as an advanced recipe
          |
          +-- Interactive CLI session
                |
                +-- Short lookup / read-only investigation?
                |     Use `task` (in-process subagent, cheap aux model)
                |
                +-- Large self-contained job needing its own context or cwd?
                |     Use `delegate` (separate `agentica --query --print` process)
                |
                +-- Just tell another live terminal something?
                      Use peer messaging (`list_agents` / `send_message`)
```

## Default Recommendation

Start with `Agent.as_tool()` for lightweight composition and `Workflow` for deterministic pipelines. Use `Subagent` when the child run needs a permission boundary, separate runtime state, or task timeout. Use `Swarm` only when parallel peer collaboration is the product requirement, not just because a task has multiple steps.

In the CLI: default to `task`; reach for `delegate` only when the work deserves its own context window or another checkout; use peer messaging to inform another session, not to hire a worker.

## Comparison

### SDK patterns

| Pattern | Best For | Control | Cost Predictability | Product Risk |
|---------|----------|---------|---------------------|--------------|
| `Agent.as_tool()` | Parent agent calls focused helpers | LLM chooses when to call | Medium | Low |
| `Workflow` | Fixed pipelines and mixed Python/LLM steps | Developer controls order | High | Low |
| `Subagent` | Isolated delegated tasks | Runtime enforces tool/depth/time limits | Medium | Medium |
| `Swarm` | Autonomous or parallel peer work | Coordinator prompt controls split | Lower | Higher |

### CLI session tools

| Tool | Process | Relationship | Result | Best For |
|------|---------|--------------|--------|----------|
| `task` | Same process | Parent → child | Immediate tool result | Cheap, short, usually read-only; several in one message |
| `delegate` | New OS process | Parent → child | Background report / `wait` | Independent context, other `work_dir`, long jobs |
| peer (`send_message`) | Two user terminals | Peers | Receiver decides | Handoffs between live sessions |

`delegate` is **not** a wrapper around `task`. It starts `agentica --query ... --print` through the existing `BackgroundProcessRegistry` (`/ps`, `/stop`, `wait`). Delegated workers do not appear in `list_agents` — that listing is for interactive peers only. Details: [CLI Terminal](../getting-started/terminal.md).

## Rules of Thumb

- If a plain Python function can express the step, keep it in `Workflow` instead of asking an LLM to coordinate it.
- If the only reason for a child agent is specialization, try `Agent.as_tool()` first.
- If a child must not inherit all parent tools, use `Subagent` with explicit `allowed_tools` or `denied_tools`.
- If you cannot describe why workers must coordinate autonomously, do not use `Swarm`.
- Scheduled daily tasks should call a bounded agent preset through the cron scheduler, not a free-form `Swarm`.
- In the CLI: do not `delegate` what a few tool calls or several parallel `task` calls can finish; do not use peer messaging as a substitute for `delegate` when you need a report back.
