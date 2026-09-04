---
name: cron
description: >-
  Schedule, list, pause, resume, edit, or run the user's agentica cron jobs
  with the cronjob tool. Use when asked to remind, schedule a recurring or
  one-shot task, try a job now, or manage jobs that fire unattended.
metadata:
  version: "1.0"
---

# Scheduled jobs

Use the `cronjob` tool. Do not edit crontab or `~/.agentica/cron/jobs.json` by
hand. The tool schema lists the actions; do not invent them.

A job runs in a **fresh session** with none of this conversation. The prompt
must be self-contained: goal, constraints, where things are, what to do when
blocked. A prompt that needs a follow-up question will stall unattended.

Jobs only fire on their schedule while the scheduler daemon is on. How the user
turns it on depends on the surface:

- **Interactive CLI:** they type `/cron daemon on`. You cannot type slash commands; tell them which one. `/cron` is also their surface for list / add / pause / run.
- **Gateway and other surfaces:** `cron.enabled` in `config.yaml` is a
  deployment-level setting. You cannot toggle it from this conversation; tell
  the user to ask whoever manages this deployment to enable it.

`action='run'` executes a job once now so you can show the result before
waiting for the schedule.
