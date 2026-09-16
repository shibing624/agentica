# RFC: External Hook Egress

Status: Implemented

Scope: **agentica side only.** The desktop side (VPet) is specified separately, in
`VPetMac/docs/agentica-hook-adapter.md`. The two documents are one contract; a
change to either side's field names has to change both.

## Problem

agentica can already tell a desktop app what a run is doing, but only through a
channel it invented for exactly one consumer: an HTTP-over-UDS sink whose default
socket is `~/Library/Application Support/VPet/notify.sock`
(`agentica/notify/config.py:31`). Those two lines are the only place VPet appears
in the package — the coupling is not deep, but it is **a private protocol with a
single client**: anyone else (another pet, another notifier, a user's own script)
has to write code against agentica's envelope instead of using a mechanism every
CLI already agrees on.

The alternative is the one every other coding CLI offers: **the user declares a
command, we run it at lifecycle points and hand it JSON on stdin**. Claude Code,
codex, opencode and Vibe Island all work this way. An integration between two
such programs is then a three-line config entry, not a protocol implementation.

This RFC specifies agentica's version of that mechanism. It is **not** a
compatibility layer that mimics Claude Code — see "Naming" below for the reason,
which is a decision, not an oversight.

## Non-goals

- **Not** copying Claude Code's field names. (`hookSpecificOutput.decision.behavior`
  is a Claude Code output encoding; `hook_event_name` values are its vocabulary.)
  We send our own names and document them. A consumer adds one adapter.
- **Not** deleting the notify sink in this change. The sink keeps working exactly
  as it does today while the new egress is verified; see "Phases".
- **Not** an outbound webhook. Remote delivery is a different trust question
  (auth, retries, what leaves the machine) and a separate change.

## Warning: two different things are both called "hooks"

This RFC is about the **executable** kind (`settings.hooks` below). It is not the
Python callback classes already in the package: `AgentHooks` / `RunHooks`
(`agentica/hooks.py:37,62`) are in-process observers and stay as they are. Naming
the config block `settings.hooks` is deliberate (it is what users search for), but
the docstring for `agentica/hooks.py` should point here so the two are not confused.

## Design

### Config

```yaml
# ~/.agentica/config.yaml
settings:
  hooks:
    enabled: false
    consumers:
      - name: desktop
        command: ["/abs/path/to/notifier", "--from-agentica"]
        enabled: true
        events:
          run.started: true
          needs.approval: true
```

- Each consumer has a required unique `name`, its own event subscription, and
  an **argv list** `command`, not a shell string: no quoting rules, no
  accidental shell expansion of the JSON, and no `shell=True` process group. A
  user who wants a shell writes `["/bin/sh","-c","…"]` explicitly.
- Env overrides are `AGENTICA_HOOKS_ENABLED` and the JSON array
  `AGENTICA_HOOKS_CONSUMERS`. Read **once at first use**, like the sink
  (`notify/config.py` docstring), so a mid-run config flip cannot leave a
  half-wired channel.
- `enabled: false` wires nothing — no thread, no process. Same "decided once"
  rule as `install_sink` (`notify/sink.py:412`).

### Events

| event | when | emitted from |
|---|---|---|
| `run.started` | a run begins | `run_events.py` `run_started`, already emitted (`runner/loop.py:605`) |
| `run.completed` | a run ends **and no goal lap is coming** | the sink's existing deferral logic (`notify/sink.py:482-498`) |
| `run.failed` | a run raised | `runner/loop.py:1630` |
| `run.cancelled` | user interrupt | `run_events.py` `run_cancelled` |
| `tool.started` | a tool starts | the Runner's shared tool executor |
| `tool.completed` | a tool ends | the Runner's shared tool executor |
| `session.started` | a CLI logical session becomes active | startup, `/new`, `/clear`, `/resume`, `/fork` |
| `session.ended` | a CLI logical session stops being active | interactive exit / session switch |
| `needs.approval` | a tool call is parked, waiting for the user | the `publish` hook of `make_approve` (`cli/approvals.py:270`) |
| `needs.input` | `ask_user_question` is parked | the armed TUI input slot |
| `needs.resolved` | a pending request is no longer actionable | the approval/question winner |

The schema gate refuses anything else. Runner-owned run/tool names also exist in
`RunEventType`; CLI-owned session/request names have explicit emit sites.

**`goal.*` stays off this wire.** The reason recorded at `runner/core.py:93-103`
holds here too: one request becoming N runs is an agentica implementation detail,
and the `run.completed` deferral already carries the only part a consumer needs.

### Wire: stdin

```json
{
  "hook_event_name": "needs.approval",
  "session_id": "e549ef44-4836-47e4-81e5-92fe60e561f7",
  "prompt": "把这个模块的测试补上",
  "cwd": "/Users/xuming/Documents/Codes/VPetMac",
  "run_id": "…",
  "tool_name": "execute",
  "tool_call_id": "call_00_wNzJ…",
  "request_id": "…",
  "options": ["allow", "allow_prefix", "deny", "deny_prefix"],
  "question": "允许运行 `rm -rf build`？",
  "preview": "rm -rf build",
  "similar_label": "以后同类命令",
  "transport": {"ppid": 1234, "cwd": "/Users/xuming/Documents/Codes/VPetMac", "tty": "/dev/ttys003"}
}
```

Field rules that are decisions, not defaults:

- **`prompt` is the run's anchor text**, exactly as the sink defines it
  (`notify/sink.py:668-690`): the user's message on an ordinary turn, the **goal
  objective** in a goal-driven session. A consumer may display it, but must not
  treat it as "what the user just typed" — the sink already documents that trap and
  a second transport must not reintroduce it. Clipped on our side (`_clip_text`).
- **The event name is in the JSON, not in argv.** Claude Code passes it as an
  argument because it does not put it on stdin; ours is one document with no
  second channel, so a consumer reads one thing and cannot get them out of sync.
- Optional fields are **omitted, not null** — a missing key and an empty string
  are different things, and a consumer that has to test both will eventually test
  one.
- `session_id` is always present. Runs without a persisted session use one
  process-stable synthetic id.
- `transport` is shared with the notify sink and carries process/terminal
  identity plus an attach route when one is active.
- `needs.input` carries `question` + optional `options`, and **no `decision`
  vocabulary**: the reply is a free string. Keeping a question out of the four
  approval words is the same distinction the sink draws at `notify/approvals.py:46`.
- **`options` is what the CLI actually offered**, i.e. `PendingApproval.options`
  (`agent/approvals.py:96`) after the CLI narrowed it via
  `visible_approval_decisions` (`cli/approvals.py:46-51`). Send the subset, not
  the four-word superset: a consumer that renders exactly what it is given cannot
  then show a button the terminal would reject.
- Payload discipline is unchanged from the sink: metadata only (`question`,
  `preview`, `tool`, `answer` clipped by the caller). **No full prompt, no tool
  output, no conversation history, no file contents.** A hook is user config, not
  a licence to widen the payload. Tool previews and failure text are strictly
  credential-redacted before clipping.

### Reply: stdout

Exactly one JSON document, nothing else:

```json
{"request_id": "…", "decision": "allow"}
{"request_id": "…", "answer": "date-fns"}
```

The request id is mandatory and must match the inbound document. Missing or
mismatched ids are not decisions.

The document is the decision. Exit status is not consulted: a wrapper may
print JSON and keep the pipe open, and a process that already answered may
then exit non-zero. Waiting for exit before accepting would stall the race;
gating on `returncode is None or 0` raced with early JSON completion.

| stdout | meaning |
|---|---|
| one valid JSON document | that is the user's answer |
| empty, junk, or unmatched `request_id` | **no decision** — the terminal prompt is still the answer path |

There is no third meaning and no "error" channel: a hook that fails to reply and a
hook that declines to reply are the same thing to us, because in both cases the
user has not answered. A crash that printed nothing is no decision. A crash
after a valid document has already been printed is still that document.

### Two semantics kept from the sink (do not drop these)

1. **No wait cap invented by us.** The harness never imposes a deadline on a
   `needs.*` reply. This is the rule already written at `notify-sink.md:18`
   ("用户还没答不是事件"). A number baked into this layer would mean a desktop
   answer got less time than a typed one, so there is no `settings.hooks.timeout`
   either. A command may set its own internal limit (the VPet bridge does); that
   is the command's business and it is visible in the user's own config.
2. **The terminal and all subscribed hooks race; the first valid answer wins.**
   Processes start in parallel. A valid reply must match `request_id`; silence
   does not win. Once one path resolves the request, every losing process group
   is killed and `needs.resolved` tells observers to dismiss the request.

Fire-and-forget notice consumers are terminated and reaped after 30 seconds.
Blocking `needs.*` requests retain no built-in answer deadline: they race the
terminal until one path answers or the request is cancelled.

## Implementation notes

**Wiring point.** `Runner._emit_event` dispatches run/tool records through
`notify_sink_dispatch`; its `_fan_out_event` is the single shared deferral and
fan-out point for notify and shell hooks. Hook config installs lazily on first
dispatch, so SDK and one-shot runs work without an interactive bootstrap.
The Runner passes a callback into `Model.run_function_calls`: it emits started
immediately before each actual execution and completed from that execution's
finally block, so serial, parallel and cancelled calls reflect real boundaries
rather than the model layer's batch-shaped display stream.

**Process handling:**

```python
proc = subprocess.Popen(argv, stdin=PIPE, stdout=PIPE, stderr=DEVNULL,
                        start_new_session=True)          # own process group
proc.stdin.write(json.dumps(payload).encode()); proc.stdin.close()
```

Each invocation drains stdout on a daemon thread. `needs.*` processes race the
terminal/registry and all losing process groups are killed; notice processes
have a 30-second lifetime cap, and outstanding notice groups are killed when
the agentica process exits (`atexit`) so `start_new_session` children cannot
outlive a CLI / `--query` shutdown. Cleanup addresses the group by the original
leader pid even after that leader exits, then reaps it, so descendants cannot
survive while holding the pipe open.

**No token, and that is not a hole.** The sink needed a bearer token because it
listens on a socket any local process can reach (`notify-sink.md` §6.2). Here the
user supplies the command in their own config: there is no shared endpoint to
forge a request into. This is an escape hatch, not a service.

**Process identity is data, not parent-process inference.** Hook payloads and
notify envelopes share one `transport` builder (`ppid`, `cwd`, optional `tty`
and attach endpoint), so a shell wrapper does not erase the route back to the
owning terminal.

**Failure ladder**: disabled, command missing, spawn error, empty stdout,
unparseable JSON, unknown decision word — all mean "no decision", and the
terminal prompt remains. **Never synthesize `allow`.** Exit status is not a
separate rung: it does not override a valid JSON document, and without one the
stdout rules above already cover a crashed process.

## Explicitly not included

- **Deleting the sink.** See Phases.
- **Mimicking Claude Code's output encoding.** If a consumer wants agentica events
  to arrive in Claude Code's shape, the right place for that translation is an
  adapter on the consumer's side (VPet is adding one), not a lie in our payload.
- **`allow_prefix` downgrade.** The sink accepts four decision words
  (`_ALLOWED_DECISIONS`, `notify/sink.py:80`) and the reply contract keeps all
  four. A consumer that hides two of them is making a UI decision, not a protocol
  change.

## Phases

1. Hook egress and the observe-only notify sink coexist.
2. Named multi-consumer hooks, tool/session events, request ids, resolution
   notices and shared transport identity are implemented.
3. Outbound webhooks remain a separate RFC.

## Acceptance

- With `settings.hooks.enabled: true` and a consumer pointing at the desktop app's
  bridge, approving on the desktop resolves an approval that the terminal is also
  waiting on; answering in the terminal first kills the hook process and leaves no
  zombie (`ps` shows neither the bridge nor a stray process group).
- A hook that prints nothing, or prints junk → **the terminal prompt still
  answers the request**. Same for a bridge that is not installed. Exit status
  does not override a valid JSON document, and does not invent one.
- No deadline in agentica's layer: a reply that takes longer than any number the
  bridge might have chosen still lands, as long as the bridge itself is still
  waiting.
- `enabled: false` spawns nothing (no thread, no process) — verify by process
  count, not by reading the config.
- `run.completed` remains deferred across goal laps and queued CLI work; one
  release reports total busy time and the last answer.
