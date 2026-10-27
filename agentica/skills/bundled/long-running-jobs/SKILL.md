---
name: long-running-jobs
description: >-
  Run work that outlives a session (hours of training, a server, a remote
  queue) with tmux/nohup/setsid and a background watcher that reports when it
  ends. Use before launching anything that must keep running after this
  session exits, or to track a job you did not start here.
metadata:
  version: "1.0"
---

# Long-running jobs

`execute(background=True)` belongs to this session: it is stopped when the
session exits. A job that must outlive the session goes in `tmux` / `nohup` /
`setsid`, and from then on nothing reports its end: `/ps` does not list it and
no exit reaches you.

So launch it to leave a completion marker and, in the same step, start one
background watcher that turns the marker into a completion. Do not wait to be
asked. Polling is right here because there is no event to wait on; the watcher
only reads, and exits non-zero if the job vanished without a marker:

```
execute(command="tmux new-session -d -s train 'python train.py > train.log 2>&1; echo $? > train.exit'")
execute(command="while [ ! -f train.exit ]; do tmux has-session -t train 2>/dev/null || [ -f train.exit ] || { echo 'train gone, no exit file'; tail -20 train.log; exit 2; }; sleep 60; done; tail -20 train.log; exit $(cat train.exit)", background=True)
```

The same watcher fits any job you did not start here (a job queue, a remote
run): loop on whatever condition marks it done.

Tell the user the watcher dies with this session while the job keeps running.
After a restart, read the marker before assuming anything.
