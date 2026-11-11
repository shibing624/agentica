---
name: long-running-jobs
description: >-
  Run work that outlives a session (hours of training, a server, a remote
  queue) with tmux/nohup/setsid, and track it with `agentica watch` so its
  progress shows and its end is reported. Use before launching anything that
  must keep running after this session exits, or to track a job you did not
  start here.
metadata:
  version: "2.0"
---

# Long-running jobs

## Which tool

| Situation | Do |
|---|---|
| Agentica runs the command, and it may end with the session | `execute(background=True)` |
| The job runs on its own (tmux, nohup/setsid, ssh to another host, a queue) | launch it with a foreground `execute`, then `execute(background=True)` an `agentica watch` |
| The very next step of this turn needs a background id's result | `wait(id)` |

`wait` only blocks on an id that `execute(background=True)` or `delegate`
returned; it cannot reach a pid, a tmux session or another host. Without a next
step that needs the result, do not `wait`: end the turn, the report arrives on
its own.

Do not hand-write a `while ...; sleep` loop for this. `agentica watch` is that
loop, with the failure handling already right.

## agentica watch

```
agentica watch --done CMD [--progress CMD] [--result CMD] [--interval SECONDS]
```

Each option is a shell command, judged by exit code only:

- `--done`: exit 0 = the job has finished, 1 = still running, anything else =
  the check itself failed (ssh down, timeout). A failed check is retried, never
  read as the job failing; after 10 in a row the watch exits 75: lost contact,
  the job may still be running.
- `--progress`: prints recent output; its last line is the progress shown on
  the card.
- `--result`: runs once after `--done`; its output goes into the report and its
  exit code is the job's. Without it, a finished job reports exit 0. If this
  check times out, the watch exits 124 and the log says the job finished but
  its exit code is unknown — that is not lost contact, and not the job's code.

## Launch so the end is checkable

The job must leave a marker even when it crashes:

```
execute(command="tmux new-session -d -s train 'python train.py > train.log 2>&1; echo $? > train.exit'")
execute(
    command="agentica watch --done 'test -f train.exit' --progress 'tail -n 3 train.log' --result 'tail -n 30 train.log; exit $(cat train.exit)'",
    description="训练 Qwen 7B",
    background=True,
)
```

On another machine over ssh, map "running" to exit 1 explicitly (ssh's own
failure is 255, which counts as a failed check and is retried):

```
execute(command="ssh gpu 'cd /w && setsid nohup sh -c \"python train.py > run.log 2>&1; echo \\$? > run.exit\" >/dev/null 2>&1 < /dev/null &'")
execute(
    command="agentica watch --interval 60 --done \"ssh gpu 'test -f /w/run.exit && exit 0; exit 1'\" --progress \"ssh gpu 'tail -n 3 /w/run.log'\" --result \"ssh gpu 'tail -n 30 /w/run.log; exit \\$(cat /w/run.exit)'\"",
    description="GPU 训练",
    background=True,
)
```

A job you did not start: pick whatever marks it done
(`--done 'tmux has-session -t x && exit 1 || exit 0'`, a queue status) and
watch it the same way. With no exit-code marker, leave out `--result`.

## What to tell the user

- Exit 75 means the checks lost contact, not that the job failed. Say the link
  is down and the job may still be running.
- Stopping the watch (the user's Stop, the session ending) never touches the
  job. Stopping the job is its own command (`tmux kill-session -t train`,
  `ssh gpu 'kill -- -<pgid>'`).
- After a restart the job keeps running but its watch is gone: read the marker
  first, then watch it again if it is still running.

If the user keeps a wrapper that launches, re-attaches and stops jobs on their
machines, launch and stop with it, and give its checks to `agentica watch`.
