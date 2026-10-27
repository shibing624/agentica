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

## Waiting on something this session started? Not cron.

To learn when a training run, eval, build, or any `tmux`/`nohup` job finishes,
do not schedule a minute-level poll. Start a read-only watcher with
`execute(background=True)` that loops on the job's exit marker (see the
`execute` tool's tmux/nohup notes). It spends no tokens while it waits, and
its exit comes back to this session as a new message.

A cron monitor costs a full agent run per tick and cannot reach this session:
one production job polled a 37-hour training run every 6 minutes, 400+ times,
and kept polling for four days after the run had ended.

Cron is for work that must recur on a clock or outlive this session.

## If a job watches for an end state, it must stop itself

Write the exit condition into the prompt: "when `<marker>` exists (or the
process is gone), report the result and pause this job (its own id) with the
cronjob tool". Without it the job keeps firing after there is nothing left to
watch.

## A job that checks for changes says so when there are none

For a job that looks for something new (a feed, a page, a queue, a report),
write into its prompt: "if nothing new since the last run, reply with exactly
`[NO_UPDATE]` and nothing else". Each run's result is kept in the job's
history; a fixed marker lets the user (and any surface that shows results)
skip empty runs instead of reading "nothing changed" in a dozen phrasings.

Example prompt: "Check https://example.com/releases for a version newer than
the one in ~/notes/last_release.txt. If there is one, write it to that file
and reply with the version and its headline changes. Otherwise reply with
exactly `[NO_UPDATE]`."

A job's agent has the CLI's file, shell, web, memory and `cronjob` tools, and
no session of its own. It has no `list_agents` / `send_message` /
`ask_user_question`: do not write a prompt that tells it to message a session
or ask the user.

Jobs only fire on their schedule while the scheduler daemon is on. How the user
turns it on depends on the surface:

- **Interactive CLI:** they type `/cron daemon on`. You cannot type slash commands; tell them which one. `/cron` is also their surface for list / add / pause / run.
- **Gateway and other surfaces:** `cron.enabled` in `config.yaml` is a
  deployment-level setting. You cannot toggle it from this conversation; tell
  the user to ask whoever manages this deployment to enable it.

`action='run'` executes a job once now so you can show the result before
waiting for the schedule.
