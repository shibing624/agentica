# Daily Tasks

Daily Tasks are the product-layer shape for recurring or scheduled agent work. They reuse the SDK cron backend, but add the missing product semantics: user intent, permission profile, run limits, retries, and structured run history.

## Layering

```text
User intent
    |
    v
DailyTaskSpec(name, prompt, schedule, workspace, permission_mode, work_dir)
    |
    v
CronJob storage and scheduler tick
    |
    v
AgentRunner.run(prompt, context)
    |
    v
TaskRun(run_id, task_id, status, timing, error_type)
```

`DailyTaskSpec` is the stable input shape for CLI/Gateway/product surfaces. `CronJob` remains the file-backed execution record. `TaskRun` is the append-only history item written after every execution attempt.

## SDK Usage

```python
from agentica.cron import CronSchedule, DailyTaskSpec, create_daily_task

spec = DailyTaskSpec(
    name="Morning Brief",
    prompt="Summarize overnight incidents and list follow-up actions.",
    schedule=CronSchedule.at_time(8, 30),
    user_id="alice",
    workspace="/srv/agentica/workspaces/alice",
    permission_mode="ask",   # every run: reads and reports; writes/network are denied
    work_dir="/srv/agentica/projects/alice",
    timeout_seconds=60,
    max_retries=2,
)

job = create_daily_task(spec)
```

## Failure Visibility

Every scheduler attempt writes a `TaskRun` record with:

- `status`: `ok`, `failed`, or `timeout`
- `error_type`: named failure class such as `TimeoutError`
- `started_at_ms`, `ended_at_ms`, and `duration_ms`
- `attempt`: retry attempt number
- `result` or `error`

```python
from agentica.cron import list_task_runs

runs = list_task_runs(job_id=job.id)
latest = runs[0]
print(latest.status, latest.error_type, latest.duration_ms)
```

## Product Rules

- Prompts must be self-contained because scheduled runs do not inherit the current chat.
- `permission_mode` / `work_dir` are the job's mandate: every run uses them, regardless of the tier the host process is in when the job fires. A job created through the `cronjob` tool records the calling session's tier and directory; the tool only lets the model pick a stricter tier, never a broader one.
- If `work_dir` no longer exists when the job is due, the scheduler pauses the job instead of running it and sets `paused_reason`. A job's own run can do the same with `cronjob(action="pause", job_id=..., reason=...)` when it cannot go on unattended. A paused job is not retried; `resume_job` clears the reason.
- Use `timeout_seconds` for all unattended tasks. A stuck run should become a visible `timeout`, not a silent hang.
- Use `max_retries` for transient failures. Retries are immediate scheduled attempts with `retry_delay_ms`; after retries are exhausted, recurring jobs wait for their next normal schedule.
